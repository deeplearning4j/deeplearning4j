/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonGraphBackend.h>
#include <helpers/logger.h>
#include <system/common.h>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

// MLIR core for ModuleOp used in compileToGpuBinary cleanup
#include <mlir/IR/BuiltinOps.h>

// Disk cache for compiled PTX
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <vector>

#include <llvm/Support/raw_ostream.h>

namespace sd {
namespace graph {

namespace {

constexpr uint64_t FNV1A64_OFFSET_BASIS = 1469598103934665603ULL;
constexpr uint64_t FNV1A64_PRIME = 1099511628211ULL;

inline void mixFNV1a(uint64_t& hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; i++) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= FNV1A64_PRIME;
  }
}

bool parseIntValue(const std::string& text, int& value) {
  char* endPtr = nullptr;
  long parsed = std::strtol(text.c_str(), &endPtr, 10);
  if (endPtr == text.c_str()) return false;
  value = static_cast<int>(parsed);
  return true;
}

}  // namespace

// Static member initialization
int TritonGraphBackend::maxParallelCompilations_ = DEFAULT_MAX_PARALLEL_COMPILATIONS;
std::mutex TritonGraphBackend::configMtx_;

// ─── Parallel compilation configuration ─────────────────────────────────────────

int TritonGraphBackend::getMaxParallelCompilations() {
  std::lock_guard<std::mutex> lock(configMtx_);
  
  // Read from environment variable on first call
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    // Check ND4J_TRITON_BUILD_THREADS env var
    const char* envVal = std::getenv("ND4J_TRITON_BUILD_THREADS");
    if (envVal) {
      int envThreads = std::atoi(envVal);
      if (envThreads > 0 && envThreads <= 16) {
        maxParallelCompilations_ = envThreads;
        sd_printf("TritonGraphBackend: Using %d parallel compilation threads (from ND4J_TRITON_BUILD_THREADS)\n",
                  maxParallelCompilations_);
      } else if (envThreads > 0) {
        sd_printf("TritonGraphBackend: ND4J_TRITON_BUILD_THREADS=%d exceeds max (16), using default %d\n",
                  envThreads, DEFAULT_MAX_PARALLEL_COMPILATIONS);
      }
    } else {
      sd_printf("TritonGraphBackend: Using %d parallel compilation threads (default)\n",
                maxParallelCompilations_);
    }
  }
  return maxParallelCompilations_;
}

void TritonGraphBackend::setMaxParallelCompilations(int maxThreads) {
  std::lock_guard<std::mutex> lock(configMtx_);
  if (maxThreads > 0 && maxThreads <= 16) {
    maxParallelCompilations_ = maxThreads;
    sd_printf("TritonGraphBackend: Set max parallel compilations to %d\n", maxThreads);
  } else {
    sd_printf("TritonGraphBackend: Invalid maxThreads=%d (must be 1-16), keeping %d\n",
              maxThreads, maxParallelCompilations_);
  }
}

std::string TritonGraphBackend::getDiskCacheDir() const {
  const char* home = std::getenv("HOME");
  if (home && home[0] != '\0') {
    return std::string(home) + "/.nd4j/triton_cache";
  }
  return ".nd4j/triton_cache";
}

bool TritonGraphBackend::ensureDiskCacheDir(const std::string& cacheDir) const {
  if (cacheDir.empty()) return false;

  std::string currentPath;
  size_t start = 0;
  if (cacheDir[0] == '/') {
    currentPath = "/";
    start = 1;
  }

  while (start <= cacheDir.size()) {
    size_t slashPos = cacheDir.find('/', start);
    std::string part = (slashPos == std::string::npos)
                           ? cacheDir.substr(start)
                           : cacheDir.substr(start, slashPos - start);
    start = (slashPos == std::string::npos) ? (cacheDir.size() + 1) : (slashPos + 1);
    if (part.empty()) continue;

    if (!currentPath.empty() && currentPath.back() != '/') currentPath += "/";
    currentPath += part;

    struct stat st;
    if (stat(currentPath.c_str(), &st) == 0) {
      if (!S_ISDIR(st.st_mode)) {
        sd_printf("TritonGraphBackend: cache path exists but is not a directory: %s\n",
                  currentPath.c_str());
        return false;
      }
      continue;
    }

    if (errno != ENOENT) {
      sd_printf("TritonGraphBackend: stat failed for cache path %s (errno=%d)\n",
                currentPath.c_str(), errno);
      return false;
    }

    if (mkdir(currentPath.c_str(), 0755) != 0 && errno != EEXIST) {
      sd_printf("TritonGraphBackend: mkdir failed for cache path %s (errno=%d)\n",
                currentPath.c_str(), errno);
      return false;
    }
  }

  return true;
}

std::string TritonGraphBackend::computeDiskCacheHash(int startSlot, int endSlot,
                                                     const std::string& ttirText,
                                                     int numWarps, int numStages) const {
  uint64_t hash = FNV1A64_OFFSET_BASIS;
  mixFNV1a(hash, &startSlot, sizeof(startSlot));
  mixFNV1a(hash, &endSlot, sizeof(endSlot));
  mixFNV1a(hash, &numWarps, sizeof(numWarps));
  mixFNV1a(hash, &numStages, sizeof(numStages));
  mixFNV1a(hash, ttirText.data(), ttirText.size());

  const std::string arch = TritonTargetDispatch::getTargetArch();
  if (!arch.empty()) {
    mixFNV1a(hash, arch.data(), arch.size());
  }

  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << hash;
  return oss.str();
}

bool TritonGraphBackend::loadBinaryFromDiskCache(int startSlot, int endSlot,
                                                 const std::string& cacheHash,
                                                 const TritonIRModule& irModule,
                                                 TritonCompiledBinary& binary) const {
  if (cacheHash.empty()) return false;

  const std::string cacheDir = getDiskCacheDir();
  std::ostringstream name;
  name << "seg_" << startSlot << "_" << endSlot << "_" << cacheHash;
  const std::string basePath = cacheDir + "/" + name.str();
  const std::string ptxPath = basePath + ".ptx";
  const std::string metaPath = basePath + ".meta";

  std::ifstream ptxFile(ptxPath, std::ios::binary);
  if (!ptxFile.good()) return false;

  std::ifstream metaFile(metaPath);
  if (!metaFile.good()) return false;

  std::string ptxText((std::istreambuf_iterator<char>(ptxFile)),
                      std::istreambuf_iterator<char>());
  if (ptxText.empty()) return false;
  if (ptxText.back() != '\0') ptxText.push_back('\0');

  int metaNumWarps = irModule.numWarps;
  int metaSharedMem = 0;
  std::string metaKernelName;
  std::string line;
  while (std::getline(metaFile, line)) {
    size_t eqPos = line.find('=');
    if (eqPos == std::string::npos) continue;

    const std::string key = line.substr(0, eqPos);
    const std::string value = line.substr(eqPos + 1);
    if (key == "numWarps") {
      parseIntValue(value, metaNumWarps);
    } else if (key == "sharedMemBytes") {
      parseIntValue(value, metaSharedMem);
    } else if (key == "kernelName") {
      metaKernelName = value;
    }
  }

  if (!metaKernelName.empty() && metaKernelName != irModule.kernelName) {
    return false;
  }

  binary.data = new char[ptxText.size()];
  std::memcpy(binary.data, ptxText.data(), ptxText.size());
  binary.size = ptxText.size() - 1;  // Excludes null terminator
  binary.target = TritonTargetDispatch::detectTarget();
  binary.targetArch = TritonTargetDispatch::getTargetArch();
  binary.numWarps = metaNumWarps;
  binary.sharedMemBytes = metaSharedMem;

  sd_printf("TritonGraphBackend: disk cache HIT for sub-segment [%d-%d] (%zu bytes)\n",
            startSlot, endSlot, binary.size);
  return true;
}

void TritonGraphBackend::writeBinaryToDiskCache(int startSlot, int endSlot,
                                                const std::string& cacheHash,
                                                const TritonIRModule& irModule,
                                                const TritonCompiledBinary& binary) const {
  if (cacheHash.empty() || binary.data == nullptr || binary.size == 0) return;

  const std::string cacheDir = getDiskCacheDir();
  if (!ensureDiskCacheDir(cacheDir)) return;

  std::ostringstream name;
  name << "seg_" << startSlot << "_" << endSlot << "_" << cacheHash;
  const std::string basePath = cacheDir + "/" + name.str();
  const std::string ptxPath = basePath + ".ptx";
  const std::string metaPath = basePath + ".meta";

  const auto tidHash = std::hash<std::thread::id>()(std::this_thread::get_id());
  std::ostringstream suffix;
  suffix << ".tmp." << static_cast<long long>(::getpid()) << "." << tidHash;
  const std::string ptxTmp = ptxPath + suffix.str();
  const std::string metaTmp = metaPath + suffix.str();

  {
    std::ofstream out(ptxTmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
      sd_printf("TritonGraphBackend: failed to open PTX cache temp file %s\n", ptxTmp.c_str());
      return;
    }
    out.write(static_cast<const char*>(binary.data), static_cast<std::streamsize>(binary.size));
    out.flush();
    if (!out.good()) {
      sd_printf("TritonGraphBackend: failed to write PTX cache temp file %s\n", ptxTmp.c_str());
      out.close();
      std::remove(ptxTmp.c_str());
      return;
    }
  }

  if (std::rename(ptxTmp.c_str(), ptxPath.c_str()) != 0) {
    sd_printf("TritonGraphBackend: failed to finalize PTX cache file %s (errno=%d)\n",
              ptxPath.c_str(), errno);
    std::remove(ptxTmp.c_str());
    return;
  }

  std::ostringstream meta;
  meta << "numWarps=" << binary.numWarps << "\n";
  meta << "sharedMemBytes=" << binary.sharedMemBytes << "\n";
  meta << "kernelName=" << irModule.kernelName << "\n";
  meta << "gridX=" << irModule.gridX << "\n";
  meta << "gridY=" << irModule.gridY << "\n";
  meta << "gridZ=" << irModule.gridZ << "\n";
  meta << "blockX=" << irModule.blockX << "\n";
  meta << "blockY=" << irModule.blockY << "\n";
  meta << "blockZ=" << irModule.blockZ << "\n";
  meta << "useIndirectArgs=" << (irModule.useIndirectArgs ? 1 : 0) << "\n";
  meta << "argSlotMapping=";
  for (size_t i = 0; i < irModule.args.size(); i++) {
    if (i > 0) meta << ";";
    const auto& arg = irModule.args[i];
    meta << arg.slotIndex << "," << arg.outputIndex << ","
         << (arg.isOutput ? 1 : 0) << "," << static_cast<int>(arg.dtype);
  }
  meta << "\n";

  {
    std::ofstream out(metaTmp, std::ios::trunc);
    if (!out.good()) {
      sd_printf("TritonGraphBackend: failed to open metadata cache temp file %s\n", metaTmp.c_str());
      std::remove(metaTmp.c_str());
      return;
    }
    out << meta.str();
    out.flush();
    if (!out.good()) {
      sd_printf("TritonGraphBackend: failed to write metadata cache temp file %s\n", metaTmp.c_str());
      out.close();
      std::remove(metaTmp.c_str());
      return;
    }
  }

  if (std::rename(metaTmp.c_str(), metaPath.c_str()) != 0) {
    sd_printf("TritonGraphBackend: failed to finalize metadata cache file %s (errno=%d)\n",
              metaPath.c_str(), errno);
    std::remove(metaTmp.c_str());
    return;
  }

  sd_printf("TritonGraphBackend: disk cache STORED for sub-segment [%d-%d] (%zu bytes)\n",
            startSlot, endSlot, binary.size);
}

// ─── Singleton ──────────────────────────────────────────────────────────────

TritonGraphBackend& TritonGraphBackend::getInstance() {
  static TritonGraphBackend instance;
  return instance;
}


TritonGraphBackend::TritonGraphBackend() = default;

TritonGraphBackend::~TritonGraphBackend() {
  invalidateCache();
}

// ─── Availability ───────────────────────────────────────────────────────────

bool TritonGraphBackend::isAvailable() const {
  return TritonTargetDispatch::isReady();
}

// ─── Check if all ops in a range are Triton-mappable ────────────────────────

bool TritonGraphBackend::areAllOpsMappable(NativeSlot* slots, int start, int end) {
  for (int i = start; i <= end; i++) {
    if (!TritonIRBuilder::isTritonMappable(slots[i].opName)) {
      return false;
    }
  }
  return true;
}

// ─── Segment fusibility check ───────────────────────────────────────────────

bool TritonGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) return false;

  int totalOps = end - start + 1;
  if (totalOps < MIN_MAPPABLE_OPS) return false;

  // ALL ops in segment must be Triton-mappable (not UNSUPPORTED).
  // We now support all categories: element-wise (binary, unary, comparison, logical,
  // ternary, identity, cast), reduction, normalization, and matmul.
  // No size limit here — segments exceeding MAX_COMPILABLE_OPS are automatically
  // split into sub-segments in compileSegment().
  for (int i = start; i <= end; i++) {
    if (!TritonIRBuilder::isTritonMappable(slots[i].opName)) {
      return false;
    }
  }

  return true;
}

// ─── Compilation ────────────────────────────────────────────────────────────

bool TritonGraphBackend::compileSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots,
                                        LongType shapeKey,
                                        int totalSlots,
                                        int* requestedOutputSlotIndices,
                                        int numRequestedOutputs) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey};

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Already compiled for this shape
      lastCompilationAudit_ = it->second.audit;
      totalCacheHits_++;
      return true;
    }
  }

  int segmentOps = seg.endSlot - seg.startSlot + 1;
  CompiledSegment compiledSeg;

  // Use section boundaries for splitting: identify natural boundaries where
  // the op category changes (e.g., element-wise → matmul → element-wise).
  // Each sub-kernel handles one section or a group of compatible sections.
  // This produces correct kernels because each section type needs different
  // grid dimensions, shared memory, and execution patterns.
  auto sections = TritonIRBuilder::identifySections(slots, seg.startSlot, seg.endSlot,
                                                      outputSlots, totalOutputSlots,
                                                      externalInputs, numExternalInputs);

  if (sections.empty()) {
    sd_printf("TritonGraphBackend::compileSegment: no sections found for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    return false;
  }

  sd_printf("TritonGraphBackend: segment [%d-%d] has %d ops, %d sections\n",
            seg.startSlot, seg.endSlot, segmentOps, static_cast<int>(sections.size()));

  // ── Step 1: Collect all sub-segment ranges to compile ──
  struct SubSegmentRange {
    int startSlot;
    int endSlot;
    int opsCount;
  };
  std::vector<SubSegmentRange> subSegmentsToCompile;

  for (int i = 0; i < static_cast<int>(sections.size()); i++) {
    int subStart = sections[i].startSlot;
    int subEnd = sections[i].endSlot;

    // Merge consecutive element-wise-compatible sections into one sub-kernel
    while (i + 1 < static_cast<int>(sections.size())) {
      int mergedOps = subEnd - subStart + 1 + (sections[i + 1].endSlot - sections[i + 1].startSlot + 1);
      if (mergedOps > MAX_COMPILABLE_OPS) break;
      auto nextType = sections[i + 1].type;
      auto curType = sections[i].type;
      bool curMergeable = (curType == KernelSectionType::ELEMENTWISE ||
                           curType == KernelSectionType::IDENTITY ||
                           curType == KernelSectionType::CONSTANT_GENERATION ||
                           curType == KernelSectionType::SHAPE_MANIPULATION ||
                           curType == KernelSectionType::REDUCTION ||
                           curType == KernelSectionType::NORMALIZATION);
      bool nextMergeable = (nextType == KernelSectionType::ELEMENTWISE ||
                            nextType == KernelSectionType::IDENTITY ||
                            nextType == KernelSectionType::CONSTANT_GENERATION ||
                            nextType == KernelSectionType::SHAPE_MANIPULATION ||
                            nextType == KernelSectionType::REDUCTION ||
                            nextType == KernelSectionType::NORMALIZATION);
      if (curMergeable && nextMergeable) {
        subEnd = sections[i + 1].endSlot;
        i++;
      } else {
        break;
      }
    }

    int subOps = subEnd - subStart + 1;
    if (subOps >= MIN_MAPPABLE_OPS) {
      subSegmentsToCompile.push_back({subStart, subEnd, subOps});
    } else {
      // Small segments: just add audit entries without compilation
      for (int s = subStart; s <= subEnd; s++) {
        CompilationAuditEntry entry;
        entry.slotIndex = s;
        entry.opName = slots[s].opName;
        entry.wasCompiled = true;
        compiledSeg.audit.push_back(entry);
      }
    }
  }

  if (subSegmentsToCompile.empty()) {
    sd_printf("TritonGraphBackend: no sub-segments to compile for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    lastCompilationAudit_ = compiledSeg.audit;
    {
      std::lock_guard<std::mutex> lock(cacheMtx_);
      cache_[key] = std::move(compiledSeg);
    }
    return true;
  }

  // ── Step 2: Parallel compilation ──
  int numToCompile = static_cast<int>(subSegmentsToCompile.size());
  int numParallel = std::min(getMaxParallelCompilations(), numToCompile);
  
  sd_printf("TritonGraphBackend: compiling %d sub-segments (%d parallel)\n",
            numToCompile, numParallel);

  // Vector to store compiled kernels
  std::vector<CompiledKernel> compiledKernels(numToCompile);
  std::vector<bool> compileSuccess(numToCompile, false);

  // Lambda for parallel compilation
  auto compileWorker = [&](int workerId) {
    for (int idx = workerId; idx < numToCompile; idx += numParallel) {
      const auto& range = subSegmentsToCompile[idx];
      sd_printf("TritonGraphBackend: worker %d compiling sub-segment [%d-%d] (%d ops)\n",
                workerId, range.startSlot, range.endSlot, range.opsCount);
      
      compiledKernels[idx] = compileToGpuBinary(slots, range.startSlot, range.endSlot,
                                                 totalSlots,
                                                 externalInputs, numExternalInputs,
                                                 outputSlots, totalOutputSlots);
      
      if (compiledKernels[idx].gpuModule && compiledKernels[idx].kernelFunction) {
        compileSuccess[idx] = true;
        compiledKernels[idx].startSlot_ = range.startSlot;
        compiledKernels[idx].endSlot_ = range.endSlot;
      } else {
#ifdef SD_CUDA
        cudaGetLastError();
#endif
        sd_printf("TritonGraphBackend: worker %d sub-segment [%d-%d] compilation FAILED\n",
                  workerId, range.startSlot, range.endSlot);
      }
    }
  };

  // Launch parallel compilation
  if (numParallel > 1 && numToCompile > 1) {
    std::vector<std::thread> threads;
    threads.reserve(numParallel);
    for (int t = 0; t < numParallel; t++) {
      threads.emplace_back(compileWorker, t);
    }
    for (auto& t : threads) {
      t.join();
    }
  } else {
    // Single thread fallback
    compileWorker(0);
  }

  // ── Step 3: Merge results and check for failures ──
  bool anyFailed = false;
  for (int idx = 0; idx < numToCompile; idx++) {
    if (!compileSuccess[idx]) {
      anyFailed = true;
      break;
    }
    compiledSeg.audit.insert(compiledSeg.audit.end(),
                              compiledKernels[idx].audit.begin(),
                              compiledKernels[idx].audit.end());
    compiledSeg.subKernels.push_back(std::move(compiledKernels[idx]));
  }

  if (anyFailed) {
    sd_printf("TritonGraphBackend: one or more sub-segments failed to compile\n");
    for (auto& kernel : compiledSeg.subKernels) {
      if (kernel.gpuModule) {
        TritonTargetDispatch::unloadModule(kernel.gpuModule);
      }
    }
    return false;
  }

  lastCompilationAudit_ = compiledSeg.audit;

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_[key] = std::move(compiledSeg);
  }

  sd_printf("TritonGraphBackend: compiled segment [%d-%d] (%d sub-kernels, shape key %lld)\n",
            seg.startSlot, seg.endSlot, (int)cache_[key].subKernels.size(), shapeKey);
  return true;
}

// ─── Execute a single compiled kernel ───────────────────────────────────────

Status TritonGraphBackend::executeSingleKernel(CompiledKernel& compiled, NativeSlot* slots,
                                                NDArray** externalInputs, int numExternalInputs,
                                                NDArray** outputSlots, int totalOutputSlots,
                                                void* stream) {
  int numBufferArgs = static_cast<int>(compiled.argSlotMapping.size());

  // Resolve all buffer pointers from the arg slot mapping
  std::vector<void*> bufferPtrs;
  bufferPtrs.reserve(numBufferArgs);

  for (auto& argMapping : compiled.argSlotMapping) {
    NDArray* arr = nullptr;
    if (argMapping.slotIndex < 0) {
      int extIdx = -(argMapping.slotIndex + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
    }

    if (!arr) {
      sd_printf("TritonGraphBackend::executeSingleKernel: null array for arg slot %d "
                "(sub-segment [%d-%d])\n",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_);
      return Status::KERNEL_FAILURE;
    }
    void* sbuf = arr->specialBuffer();
    if (!sbuf) {
      sd_printf("TritonGraphBackend::executeSingleKernel: null specialBuffer for arg slot %d "
                "(sub-segment [%d-%d], length=%lld, dtype=%d)\n",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                (long long)arr->lengthOf(), static_cast<int>(arr->dataType()));
      return Status::KERNEL_FAILURE;
    }
    bufferPtrs.push_back(sbuf);
  }

  // Compute n_elements from first output
  LongType nElements = 0;
  for (auto& argMapping : compiled.argSlotMapping) {
    if (argMapping.isOutput) {
      int slotIdx = argMapping.slotIndex;
      if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx]) {
        nElements = outputSlots[slotIdx]->lengthOf();
        break;
      }
    }
  }
  int nElem32 = static_cast<int>(nElements);


  // Compute grid size
  unsigned int actualGridX = (nElements + compiled.blockX - 1) / compiled.blockX;
  unsigned int actualGridY = compiled.gridY;
  unsigned int actualGridZ = compiled.gridZ;
  if (actualGridX == 0) actualGridX = 1;

  // Dereference the stream pointer
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

  // Build kernel args — either direct (each ptr is a separate arg) or indirect
  // (all ptrs packed into a device-side i64 array, kernel receives 1 pointer)
  std::vector<void*> kernelArgs;
  void* argTableDevice = nullptr;

  if (compiled.useIndirectArgs) {
    // Pack all buffer pointers as int64 values into a device-side array.
    // The kernel signature is: @kernel(%argTable: !tt.ptr<i64>, %n_elements: i32)
    // It loads each buffer pointer from argTable[i] and casts via tt.int_to_ptr.
    std::vector<int64_t> argTableHost(numBufferArgs);
    for (int i = 0; i < numBufferArgs; i++) {
      argTableHost[i] = reinterpret_cast<int64_t>(bufferPtrs[i]);
    }

#ifdef SD_CUDA
    // Allocate device buffer for the arg table
    size_t tableBytes = numBufferArgs * sizeof(int64_t);
    auto allocErr = cudaMallocAsync(&argTableDevice, tableBytes,
                                     static_cast<cudaStream_t>(actualStream));
    if (allocErr != cudaSuccess) {
      allocErr = cudaMalloc(&argTableDevice, tableBytes);
      if (allocErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to allocate arg table (%d bytes): %s\n",
                  (int)tableBytes, cudaGetErrorString(allocErr));
        return Status::KERNEL_FAILURE;
      }
    }
    // Copy host → device (async on the execution stream)
    cudaMemcpyAsync(argTableDevice, argTableHost.data(), tableBytes,
                     cudaMemcpyHostToDevice, static_cast<cudaStream_t>(actualStream));
#endif

    // Kernel args: [argTablePtr, n_elements]
    kernelArgs.push_back(&argTableDevice);
    kernelArgs.push_back(&nElem32);
  } else {
    // Direct mode: each buffer pointer is a separate kernel arg + n_elements
    // cuLaunchKernel expects void** where each entry points to the actual param value.
    // bufferPtrs[i] IS the void* value; &bufferPtrs[i] is the pointer-to-pointer.
    for (int i = 0; i < numBufferArgs; i++) {
      kernelArgs.push_back(&bufferPtrs[i]);
    }
    kernelArgs.push_back(&nElem32);
  }

  // Launch
  bool ok;
  if (compiled.useCooperativeLaunch) {
    ok = TritonTargetDispatch::launchCooperativeKernel(
        compiled.kernelFunction,
        actualGridX, actualGridY, actualGridZ,
        compiled.numWarps * 32,
        compiled.blockY, compiled.blockZ,
        compiled.sharedMemBytes,
        actualStream,
        kernelArgs.data(),
        static_cast<int>(kernelArgs.size()));
  } else {
    ok = TritonTargetDispatch::launchKernel(
        compiled.kernelFunction,
        actualGridX, actualGridY, actualGridZ,
        compiled.numWarps * 32,
        compiled.blockY, compiled.blockZ,
        compiled.sharedMemBytes,
        actualStream,
        kernelArgs.data(),
        static_cast<int>(kernelArgs.size()));
  }

  if (!ok) {
    sd_printf("TritonGraphBackend::executeSingleKernel: kernel launch failed for [%d-%d] "
              "(cooperative=%d, grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u)\n",
              compiled.startSlot_, compiled.endSlot_,
              compiled.useCooperativeLaunch ? 1 : 0,
              actualGridX, actualGridY, actualGridZ,
              compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
              compiled.sharedMemBytes);
#ifdef SD_CUDA
    if (argTableDevice) cudaFreeAsync(argTableDevice, static_cast<cudaStream_t>(actualStream));
#endif
    return Status::KERNEL_FAILURE;
  }

#ifdef SD_CUDA
  // Free the indirect arg table after kernel launch (async — kernel reads it before free executes)
  if (argTableDevice) cudaFreeAsync(argTableDevice, static_cast<cudaStream_t>(actualStream));

#endif

  return Status::OK;
}

// ─── Execution ──────────────────────────────────────────────────────────────

Status TritonGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      sd_printf("TritonGraphBackend::executeSegment: no compiled kernel for segment [%d-%d]\n",
                seg.startSlot, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiledSeg = &it->second;
  }

  // Execute all sub-kernels in sequence on the same stream.
  for (int i = 0; i < (int)compiledSeg->subKernels.size(); i++) {
    auto& subKernel = compiledSeg->subKernels[i];
    auto status = executeSingleKernel(subKernel, slots,
                                       externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots,
                                       stream);
    if (status != Status::OK) {
      sd_printf("TritonGraphBackend::executeSegment: sub-kernel %d/%d [%d-%d] failed\n",
                i + 1, (int)compiledSeg->subKernels.size(),
                subKernel.startSlot_, subKernel.endSlot_);
      return status;
    }
    totalKernelLaunches_++;
  }

#ifdef SD_CUDA
  // Synchronize the execution stream to ensure all Triton kernels complete
  // before the caller reads output buffers. Without this, Java-side copyBuffer
  // on a different stream races with the async kernel, producing stale output.
  // NOTE: stream is void** (pointer to cudaStream_t*), must dereference.
  if (stream) {
    void* actualStream = *static_cast<void**>(stream);
    if (actualStream) {
      cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
    }
  }

#endif

  return Status::OK;
}

// ─── Cache invalidation ────────────────────────────────────────────────────

void TritonGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  for (auto& entry : cache_) {
    for (auto& kernel : entry.second.subKernels) {
      if (kernel.gpuModule) {
        TritonTargetDispatch::unloadModule(kernel.gpuModule);
      }
    }
  }
  cache_.clear();
  lastCompilationAudit_.clear();
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> TritonGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

// ─── Internal: compile to GPU binary ────────────────────────────────────────

TritonGraphBackend::CompiledKernel TritonGraphBackend::compileToGpuBinary(
    NativeSlot* slots, int startSlot, int endSlot,
    int totalSlots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  CompiledKernel result;

  // Build Triton IR
  auto irModule = irBuilder_.buildModule(slots, startSlot, endSlot,
                                          totalSlots,
                                          externalInputs, numExternalInputs,
                                          outputSlots, totalOutputSlots);
  auto cleanupModule = [&irModule]() {
    if (irModule.mlirModule) {
      auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
      mod->erase();
      delete mod;
      irModule.mlirModule = nullptr;
    }
  };

  if (!irModule.valid) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed IR build
#endif
    sd_printf("TritonGraphBackend: IR build failed for segment [%d-%d]\n", startSlot, endSlot);
    return result;
  }

  // Build compilation audit
  for (int i = startSlot; i <= endSlot; i++) {
    CompilationAuditEntry entry;
    entry.slotIndex = i;
    entry.opName = slots[i].opName;
    entry.wasCompiled = TritonIRBuilder::isTritonMappable(slots[i].opName);
    if (!entry.wasCompiled) {
      entry.reason = "unmappable op (not in Triton op table)";
    }
    result.audit.push_back(entry);
  }

  // Capture TTIR text for deterministic cache-key generation.
  std::string ttirText;
  {
    auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
    llvm::raw_string_ostream os(ttirText);
    mod->print(os);
  }

  const std::string cacheHash = computeDiskCacheHash(startSlot, endSlot, ttirText,
                                                      irModule.numWarps, irModule.numStages);

  TritonCompiledBinary binary = {nullptr, 0, TritonGpuTarget::UNKNOWN, "", irModule.numWarps, 0};
  bool loadedFromDiskCache = loadBinaryFromDiskCache(startSlot, endSlot, cacheHash, irModule, binary);
  if (!loadedFromDiskCache) {
    binary = TritonTargetDispatch::compile(irModule.mlirModule, irModule.numWarps, irModule.numStages);
    if (binary.data) {
      writeBinaryToDiskCache(startSlot, endSlot, cacheHash, irModule, binary);
    }
  }

  if (!binary.data) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed compilation
#endif
    sd_printf("TritonGraphBackend: Triton compilation failed for segment [%d-%d]\n", startSlot, endSlot);
    cleanupModule();
    return result;
  }

  // Load binary into driver module
  result.gpuModule = TritonTargetDispatch::loadModule(binary);
  if (!result.gpuModule) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed module load
#endif
    sd_printf("TritonGraphBackend: module load failed for segment [%d-%d]\n", startSlot, endSlot);
    delete[] static_cast<char*>(binary.data);
    cleanupModule();
    return result;
  }

  // Get kernel function
  result.kernelFunction = TritonTargetDispatch::getKernelFunction(result.gpuModule, irModule.kernelName);
  if (!result.kernelFunction) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors
#endif
    sd_printf("TritonGraphBackend: kernel function '%s' not found in module\n", irModule.kernelName.c_str());
    TritonTargetDispatch::unloadModule(result.gpuModule);
    result.gpuModule = nullptr;
    delete[] static_cast<char*>(binary.data);
    cleanupModule();
    return result;
  }

  // Set launch config
  result.gridX = irModule.gridX;
  result.gridY = irModule.gridY;
  result.gridZ = irModule.gridZ;
  result.blockX = irModule.blockX;
  result.blockY = irModule.blockY;
  result.blockZ = irModule.blockZ;
  result.sharedMemBytes = binary.sharedMemBytes;
  result.numWarps = binary.numWarps;
  result.argSlotMapping = irModule.args;
  result.useCooperativeLaunch = irModule.useCooperativeLaunch;
  result.useIndirectArgs = irModule.useIndirectArgs;

  // Clean up
  delete[] static_cast<char*>(binary.data);
  cleanupModule();

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
