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
#include <system/Environment.h>
#include <system/common.h>

#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// MLIR core for ModuleOp used in compileToGpuBinary cleanup
#include <mlir/IR/BuiltinOps.h>

// Disk cache for compiled PTX
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <sstream>
#include <utility>
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

bool ptxUsesExternSharedMemory(const std::string& ptxText) {
  return ptxText.find(".extern .shared") != std::string::npos &&
         ptxText.find("global_smem") != std::string::npos;
}

std::string configuredOrDefaultTritonDir(const std::string& configured,
                                         const std::string& home,
                                         const char* defaultLeaf) {
  if (!configured.empty()) {
    return configured;
  }
  if (!home.empty()) {
    return home + "/.nd4j/" + defaultLeaf;
  }
  return std::string(".nd4j/") + defaultLeaf;
}

#ifdef SD_CUDA
inline cudaError_t allocateDeviceBufferAsync(void** ptr, size_t bytes, cudaStream_t stream) {
  if (bytes == 0) bytes = 1;
  return cudaMallocAsync(ptr, bytes, stream);
}

inline cudaError_t freeDeviceBufferAsync(void* ptr, cudaStream_t stream) {
  if (ptr == nullptr) return cudaSuccess;
  return cudaFreeAsync(ptr, stream);
}

inline bool configureCudaKernelSharedMemory(void* kernelFunc, unsigned int sharedMemBytes) {
  if (kernelFunc == nullptr || sharedMemBytes == 0) return true;

  int currentDevice = 0;
  cudaError_t getDeviceErr = cudaGetDevice(&currentDevice);
  if (getDeviceErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: failed to query CUDA device for shared memory setup: %s\n",
              cudaGetErrorString(getDeviceErr));
    cudaGetLastError();
    return false;
  }

  int maxSharedOptIn = 0;
  cudaError_t optInErr = cudaDeviceGetAttribute(
      &maxSharedOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, currentDevice);
  if (optInErr != cudaSuccess || maxSharedOptIn <= 0) {
    cudaGetLastError();
    maxSharedOptIn = 0;
  }

  if (maxSharedOptIn <= 0) {
    int maxSharedDefault = 0;
    cudaError_t defaultErr = cudaDeviceGetAttribute(
        &maxSharedDefault, cudaDevAttrMaxSharedMemoryPerBlock, currentDevice);
    if (defaultErr != cudaSuccess || maxSharedDefault <= 0) {
      sd_printf("TritonGraphBackend: failed to query device shared memory limits on device %d: %s\n",
                currentDevice, cudaGetErrorString(defaultErr));
      cudaGetLastError();
      return false;
    }
    maxSharedOptIn = maxSharedDefault;
  }

  if (sharedMemBytes > static_cast<unsigned int>(maxSharedOptIn)) {
    sd_printf("TritonGraphBackend: kernel shared memory requirement %u exceeds device %d limit %d\n",
              sharedMemBytes, currentDevice, maxSharedOptIn);
    return false;
  }

  if (sharedMemBytes > 49152u && maxSharedOptIn > 49152) {
    CUresult attrRes = cuFuncSetAttribute(
        static_cast<CUfunction>(kernelFunc),
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        static_cast<int>(sharedMemBytes));
    if (attrRes != CUDA_SUCCESS) {
      const char* errStr = nullptr;
      cuGetErrorString(attrRes, &errStr);
      sd_printf("TritonGraphBackend: cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES=%u) "
                "failed: %s (code=%d)\n",
                sharedMemBytes, errStr ? errStr : "unknown", static_cast<int>(attrRes));
      return false;
    }
  }

  return true;
}
#endif

#ifdef SD_CUDA
// Lazy-initialized 8-byte device buffer for zero-length array args.
// Triton kernels won't actually read/write it, but cuLaunchKernel needs
// a valid device pointer in the arg table.
struct DummyDevicePtrCache {
  std::mutex mutex;
  std::unordered_map<int, void*> byDevice;
};

inline DummyDevicePtrCache& dummyDevicePtrCache() {
  static DummyDevicePtrCache cache;
  return cache;
}

inline void* getDummyDevicePtrForDevice(int currentDevice, bool streamIsCapturing) {
  if (currentDevice < 0) return nullptr;

  auto& cache = dummyDevicePtrCache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  auto it = cache.byDevice.find(currentDevice);
  if (it != cache.byDevice.end() && it->second != nullptr) return it->second;

  if (streamIsCapturing) return nullptr;

  void* ptr = nullptr;
  auto err = cudaMalloc(&ptr, 8);
  if (err != cudaSuccess) {
    sd_printf("TritonGraphBackend: failed to allocate dummy device pointer on device %d: %s\n",
              currentDevice, cudaGetErrorString(err));
    return nullptr;
  }
  cache.byDevice[currentDevice] = ptr;
  return ptr;
}
#endif

}  // namespace

// Static member initialization
int TritonGraphBackend::maxParallelCompilations_ = DEFAULT_MAX_PARALLEL_COMPILATIONS;
std::mutex TritonGraphBackend::configMtx_;
thread_local TritonGraphBackend::FallbackRangeExecutor TritonGraphBackend::fallbackRangeExecutor_ = nullptr;

// ─── Parallel compilation configuration ─────────────────────────────────────────

int TritonGraphBackend::getMaxParallelCompilations() {
  std::lock_guard<std::mutex> lock(configMtx_);

  int configuredThreads = sd::Environment::getInstance().tritonBuildThreads();
  if (configuredThreads > 0 && configuredThreads <= 16) {
    maxParallelCompilations_ = configuredThreads;
  } else {
    maxParallelCompilations_ = DEFAULT_MAX_PARALLEL_COMPILATIONS;
  }

  static int lastReported = -1;
  if (lastReported != maxParallelCompilations_) {
    sd_printf("TritonGraphBackend: Using %d parallel compilation threads (Environment)\n",
              maxParallelCompilations_);
    lastReported = maxParallelCompilations_;
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
  const auto& env = sd::Environment::getInstance();
  return configuredOrDefaultTritonDir(env.tritonCacheDir(), env.homeDirectory(), "triton_cache");
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
                                                     LongType segmentShapeKey,
                                                     const std::string& ttirText,
                                                     int numWarps, int numStages) const {
  const auto& env = sd::Environment::getInstance();
  uint64_t hash = FNV1A64_OFFSET_BASIS;
  mixFNV1a(hash, &startSlot, sizeof(startSlot));
  mixFNV1a(hash, &endSlot, sizeof(endSlot));
  mixFNV1a(hash, &segmentShapeKey, sizeof(segmentShapeKey));
  mixFNV1a(hash, &numWarps, sizeof(numWarps));
  mixFNV1a(hash, &numStages, sizeof(numStages));
  int numCTAs = std::max(1, env.tritonNumCTAs());
  int maxNreg = std::max(0, env.tritonMaxNreg());
  int fpFusion = env.tritonEnableFpFusion() ? 1 : 0;
  int disableLineInfo = env.tritonDisableLineInfo() ? 1 : 0;
  mixFNV1a(hash, &numCTAs, sizeof(numCTAs));
  mixFNV1a(hash, &maxNreg, sizeof(maxNreg));
  mixFNV1a(hash, &fpFusion, sizeof(fpFusion));
  mixFNV1a(hash, &disableLineInfo, sizeof(disableLineInfo));
  mixFNV1a(hash, ttirText.data(), ttirText.size());

  std::string arch = TritonTargetDispatch::getTargetArch();
  const std::string archOverride = env.tritonOverrideArch();
  if (!archOverride.empty()) {
    arch = archOverride;
  }
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
  if (!sd::Environment::getInstance().tritonCacheEnabled()) return false;
  if (cacheHash.empty()) return false;

  const std::string cacheDir = getDiskCacheDir();
  std::ostringstream name;
  name << "ttir_" << cacheHash;
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

  // Older cache entries were missing sharedMemBytes metadata. Recompile those
  // kernels if PTX requires extern shared memory; otherwise launches would pass
  // sharedMem=0 and corrupt memory.
  if (metaSharedMem == 0 && ptxUsesExternSharedMemory(ptxText)) {
    sd_printf("TritonGraphBackend: disk cache entry for [%d-%d] is stale "
              "(extern shared PTX with sharedMemBytes=0); forcing recompile\n",
              startSlot, endSlot);
    return false;
  }

  binary.data = new char[ptxText.size()];
  std::memcpy(binary.data, ptxText.data(), ptxText.size());
  binary.size = ptxText.size() - 1;  // Excludes null terminator
  binary.target = TritonTargetDispatch::detectTarget();
  binary.targetArch = TritonTargetDispatch::getTargetArch();
  const std::string archOverride = sd::Environment::getInstance().tritonOverrideArch();
  if (!archOverride.empty()) {
    binary.targetArch = archOverride;
  }
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
  if (!sd::Environment::getInstance().tritonCacheEnabled()) return;
  if (cacheHash.empty() || binary.data == nullptr || binary.size == 0) return;

  const std::string cacheDir = getDiskCacheDir();
  if (!ensureDiskCacheDir(cacheDir)) return;

  std::ostringstream name;
  name << "ttir_" << cacheHash;
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

void TritonGraphBackend::setFallbackRangeExecutor(FallbackRangeExecutor executor) {
  fallbackRangeExecutor_ = std::move(executor);
}

void TritonGraphBackend::clearFallbackRangeExecutor() {
  fallbackRangeExecutor_ = nullptr;
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
  // No hard rejection by size here — oversized segments are split adaptively
  // in compileSegment() before Triton compilation.
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
    if (failedCache_.find(key) != failedCache_.end()) {
      sd_printf("TritonGraphBackend::compileSegment: skipping previously failed segment [%d-%d] "
                "(shapeKey=%lld)\n",
                seg.startSlot, seg.endSlot, shapeKey);
      return false;
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

  // ── Step 1: Build adaptive compile ranges from section graph ──
  struct SubSegmentRange {
    int startSlot;
    int endSlot;
    int opsCount;
    int startSectionIdx;
    int endSectionIdx;
  };
  std::deque<SubSegmentRange> pendingRanges;

  auto isStandaloneSection = [](const KernelSection& section) -> bool {
    if (section.type != KernelSectionType::FUSED_ATTENTION) return false;

    // Mixed sectioned kernels currently launch as a 1D cooperative grid.
    // Attention sections are safe to merge only when they require a single
    // query tile (gridY == 1), i.e. gridRequirement == batchHeads.
    // Multi-tile attention (prefill-style seqQ > tile) must stay standalone.
    int batchHeads = std::max(1, section.batchSize) * std::max(1, section.numHeads);
    bool singleQueryTile = section.gridRequirement <= batchHeads;
    return !singleQueryTile;
  };

  auto makeRange = [&](int startSec, int endSec) -> SubSegmentRange {
    SubSegmentRange r;
    r.startSectionIdx = startSec;
    r.endSectionIdx = endSec;
    r.startSlot = sections[startSec].startSlot;
    r.endSlot = sections[endSec].endSlot;
    r.opsCount = r.endSlot - r.startSlot + 1;
    return r;
  };

  auto makeSlotRange = [&](int startSlot, int endSlot,
                           int startSec, int endSec) -> SubSegmentRange {
    SubSegmentRange r;
    r.startSectionIdx = startSec;
    r.endSectionIdx = endSec;
    r.startSlot = startSlot;
    r.endSlot = endSlot;
    r.opsCount = r.endSlot - r.startSlot + 1;
    return r;
  };

  auto splitRange = [&](const SubSegmentRange& range) {
    const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
    if (sectionCount > 1) {
      int midSec = range.startSectionIdx + (range.endSectionIdx - range.startSectionIdx) / 2;
      auto left = makeRange(range.startSectionIdx, midSec);
      auto right = makeRange(midSec + 1, range.endSectionIdx);
      // Process left first to preserve slot order.
      pendingRanges.push_front(right);
      pendingRanges.push_front(left);
      return;
    }

    if (range.opsCount <= 1) return;
    int midSlot = range.startSlot + (range.endSlot - range.startSlot) / 2;
    auto left = makeSlotRange(range.startSlot, midSlot, range.startSectionIdx, range.endSectionIdx);
    auto right = makeSlotRange(midSlot + 1, range.endSlot, range.startSectionIdx, range.endSectionIdx);
    sd_printf("TritonGraphBackend: splitting single-section range [%d-%d] by slots -> [%d-%d] + [%d-%d]\n",
              range.startSlot, range.endSlot,
              left.startSlot, left.endSlot, right.startSlot, right.endSlot);
    pendingRanges.push_front(right);
    pendingRanges.push_front(left);
  };

  for (int secIdx = 0; secIdx < static_cast<int>(sections.size());) {
    if (isStandaloneSection(sections[secIdx])) {
      pendingRanges.push_back(makeRange(secIdx, secIdx));
      secIdx++;
      continue;
    }

    int runStart = secIdx;
    int runEnd = secIdx;
    while (runEnd + 1 < static_cast<int>(sections.size()) &&
           !isStandaloneSection(sections[runEnd + 1])) {
      runEnd++;
    }
    pendingRanges.push_back(makeRange(runStart, runEnd));
    secIdx = runEnd + 1;
  }

  if (pendingRanges.empty()) {
    sd_printf("TritonGraphBackend: no sub-segments to compile for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    lastCompilationAudit_ = compiledSeg.audit;
    {
      std::lock_guard<std::mutex> lock(cacheMtx_);
      failedCache_.erase(key);
      cache_[key] = std::move(compiledSeg);
    }
    return true;
  }

  auto& env = Environment::getInstance();
  const int maxOpsCap = std::max(0, env.tritonMaxSubsegmentOps());
  const int maxSectionsCap = std::max(0, env.tritonMaxSubsegmentSections());
  const int maxParallelCompiles = std::max(1, getMaxParallelCompilations());
  sd_printf("TritonGraphBackend: adaptive section packing for [%d-%d] "
            "(initialRanges=%d, opsCap=%d, sectionsCap=%d, compileThreads=%d)\n",
            seg.startSlot, seg.endSlot,
            static_cast<int>(pendingRanges.size()),
            maxOpsCap, maxSectionsCap, maxParallelCompiles);
  if (maxOpsCap <= 0) {
    sd_printf("TritonGraphBackend: ops cap disabled for [%d-%d] (runtime control); "
              "set tritonMaxSubsegmentOps>0 to force additional splitting\n",
              seg.startSlot, seg.endSlot);
  }
  if (maxSectionsCap <= 0) {
    sd_printf("TritonGraphBackend: section cap disabled for [%d-%d] (runtime control); "
              "set tritonMaxSubsegmentSections>0 to force additional splitting\n",
              seg.startSlot, seg.endSlot);
  }
  if (maxParallelCompiles > 1 && pendingRanges.size() == 1 &&
      maxOpsCap <= 0 && maxSectionsCap <= 0) {
    sd_printf("TritonGraphBackend: initial work for [%d-%d] is a single range with caps disabled; "
              "compile threads are configured but only one worker can run until the range is split\n",
              seg.startSlot, seg.endSlot);
  }

#ifdef SD_CUDA
  int compileDevice = -1;
  cudaError_t compileDeviceErr = cudaGetDevice(&compileDevice);
  if (compileDeviceErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: cudaGetDevice failed before adaptive compilation "
              "for segment [%d-%d]: %s\n",
              seg.startSlot, seg.endSlot,
              cudaGetErrorString(compileDeviceErr));
    cudaGetLastError();
    return false;
  }
  cudaError_t setDeviceErr = cudaSetDevice(compileDevice);
  if (setDeviceErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: failed to set CUDA device %d before adaptive compilation "
              "for segment [%d-%d]: %s\n",
              compileDevice, seg.startSlot, seg.endSlot, cudaGetErrorString(setDeviceErr));
    cudaGetLastError();
    return false;
  }
#endif

  struct CompileRangeResult {
    SubSegmentRange range;
    CompiledKernel compiled;
  };

  std::atomic<long long> launchedRanges{0};
  std::atomic<long long> completedRanges{0};
  std::atomic<long long> successfulRanges{0};
  std::atomic<long long> failedRanges{0};
  std::atomic<long long> inFlightRanges{0};
  long long splitRetryCount = 0;
  int batchIndex = 0;

  auto compileRange = [&](const SubSegmentRange& range) -> CompileRangeResult {
    CompileRangeResult result;
    result.range = range;
    const auto rangeStart = std::chrono::steady_clock::now();
    const long long launchIndex = launchedRanges.fetch_add(1) + 1;
    const long long inflightNow = inFlightRanges.fetch_add(1) + 1;
#ifdef SD_CUDA
    if (compileDevice >= 0) {
      cudaError_t setDeviceErr = cudaSetDevice(compileDevice);
      if (setDeviceErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to set CUDA device %d in compile worker "
                  "for range [%d-%d]: %s\n",
                  compileDevice, range.startSlot, range.endSlot,
                  cudaGetErrorString(setDeviceErr));
        const long long completedNow = completedRanges.fetch_add(1) + 1;
        const long long failedNow = failedRanges.fetch_add(1) + 1;
        const long long inflightAfter = inFlightRanges.fetch_sub(1) - 1;
        sd_printf("TritonGraphBackend: compile progress seg[%d-%d] launch#%lld range[%d-%d] "
                  "status=FAILED(set-device) completed=%lld success=%lld failed=%lld inflight=%lld\n",
                  seg.startSlot, seg.endSlot, launchIndex, range.startSlot, range.endSlot,
                  completedNow, successfulRanges.load(), failedNow, inflightAfter);
        cudaGetLastError();
        return result;
      }
    }
#endif
    const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
    sd_printf("TritonGraphBackend: compile progress seg[%d-%d] launch#%lld range[%d-%d] "
              "START (ops=%d, sections=%d, inflight=%lld)\n",
              seg.startSlot, seg.endSlot, launchIndex, range.startSlot, range.endSlot,
              range.opsCount, sectionCount, inflightNow);
    result.compiled = compileToGpuBinary(slots, range.startSlot, range.endSlot,
                                         shapeKey,
                                         totalSlots,
                                         externalInputs, numExternalInputs,
                                         outputSlots, totalOutputSlots);
    const bool success = (result.compiled.gpuModule && result.compiled.kernelFunction);
    if (result.compiled.gpuModule && result.compiled.kernelFunction) {
      result.compiled.startSlot_ = range.startSlot;
      result.compiled.endSlot_ = range.endSlot;
    }
    const long long completedNow = completedRanges.fetch_add(1) + 1;
    const long long successNow = success ? (successfulRanges.fetch_add(1) + 1) : successfulRanges.load();
    const long long failedNow = success ? failedRanges.load() : (failedRanges.fetch_add(1) + 1);
    const long long inflightAfter = inFlightRanges.fetch_sub(1) - 1;
    const auto elapsedMs = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - rangeStart).count());
    sd_printf("TritonGraphBackend: compile progress seg[%d-%d] launch#%lld range[%d-%d] "
              "DONE status=%s elapsedMs=%lld completed=%lld success=%lld failed=%lld inflight=%lld\n",
              seg.startSlot, seg.endSlot, launchIndex, range.startSlot, range.endSlot,
              success ? "OK" : "FAILED", elapsedMs,
              completedNow, successNow, failedNow, inflightAfter);
    return result;
  };

  // ── Step 2: Adaptive compile-and-split loop ──
  while (!pendingRanges.empty()) {
    std::vector<SubSegmentRange> readyRanges;
    readyRanges.reserve(static_cast<size_t>(maxParallelCompiles));

    while (!pendingRanges.empty() &&
           readyRanges.size() < static_cast<size_t>(maxParallelCompiles)) {
      SubSegmentRange range = pendingRanges.front();
      pendingRanges.pop_front();

      const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
      const bool canSplit = (sectionCount > 1) || (range.opsCount > 1);
      const bool exceedsOpsCap = (maxOpsCap > 0 && range.opsCount > maxOpsCap);
      const bool exceedsSectionsCap = (maxSectionsCap > 0 && sectionCount > maxSectionsCap);
      if (canSplit && (exceedsOpsCap || exceedsSectionsCap)) {
        sd_printf("TritonGraphBackend: pre-splitting range [%d-%d] (%d ops, %d sections) "
                  "to honor caps (opsCap=%d, sectionsCap=%d)\n",
                  range.startSlot, range.endSlot, range.opsCount, sectionCount,
                  maxOpsCap, maxSectionsCap);
        splitRange(range);
        continue;
      }
      readyRanges.push_back(range);
    }

    if (readyRanges.empty()) {
      continue;
    }

    batchIndex++;
    const int activeWorkers = std::min(maxParallelCompiles, static_cast<int>(readyRanges.size()));
    sd_printf("TritonGraphBackend: compile progress seg[%d-%d] dispatch batch=%d "
              "(ranges=%d, workers=%d/%d, pending=%d, launched=%lld, completed=%lld, inflight=%lld)\n",
              seg.startSlot, seg.endSlot, batchIndex,
              static_cast<int>(readyRanges.size()), activeWorkers, maxParallelCompiles,
              static_cast<int>(pendingRanges.size()),
              launchedRanges.load(), completedRanges.load(), inFlightRanges.load());
    if (maxParallelCompiles > 1 && activeWorkers < maxParallelCompiles) {
      sd_printf("TritonGraphBackend: compile progress seg[%d-%d] batch=%d has %d/%d active workers "
                "(insufficient ready ranges; adjust runtime caps for more parallel fanout)\n",
                seg.startSlot, seg.endSlot, batchIndex, activeWorkers, maxParallelCompiles);
    }

    std::vector<CompileRangeResult> compileResults;
    compileResults.reserve(readyRanges.size());

    if (maxParallelCompiles > 1 && readyRanges.size() > 1) {
      std::vector<std::future<CompileRangeResult>> futures;
      futures.reserve(readyRanges.size());
      for (const auto& range : readyRanges) {
        futures.emplace_back(std::async(std::launch::async, compileRange, range));
      }
      for (auto& f : futures) {
        compileResults.emplace_back(f.get());
      }
    } else {
      for (const auto& range : readyRanges) {
        compileResults.emplace_back(compileRange(range));
      }
    }

    for (auto& compileResult : compileResults) {
      const SubSegmentRange& range = compileResult.range;
      const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
      const bool canSplit = (sectionCount > 1) || (range.opsCount > 1);

      if (compileResult.compiled.gpuModule && compileResult.compiled.kernelFunction) {
        compiledSeg.subKernels.push_back(std::move(compileResult.compiled));
        continue;
      }

#ifdef SD_CUDA
      cudaGetLastError();
#endif
      if (canSplit) {
        sd_printf("TritonGraphBackend: adaptive range [%d-%d] compile failed; splitting by section graph\n",
                  range.startSlot, range.endSlot);
        splitRetryCount++;
        splitRange(range);
        continue;
      }

      // Leaf range failed: all-or-nothing reject entire segment.
      for (auto& kernel : compiledSeg.subKernels) {
        if (kernel.gpuModule) {
          TritonTargetDispatch::unloadModule(kernel.gpuModule);
          kernel.gpuModule = nullptr;
          kernel.kernelFunction = nullptr;
        }
      }
      compiledSeg.subKernels.clear();
      compiledSeg.audit.clear();
      for (int s = seg.startSlot; s <= seg.endSlot; s++) {
        CompilationAuditEntry entry;
        entry.slotIndex = s;
        entry.opName = slots[s].opName;
        entry.wasCompiled = false;
        entry.reason = "segment rejected (all-or-nothing Triton compile failed)";
        compiledSeg.audit.push_back(entry);
      }
      lastCompilationAudit_ = compiledSeg.audit;
      {
        std::lock_guard<std::mutex> lock(cacheMtx_);
        failedCache_.insert(key);
      }
      sd_printf("TritonGraphBackend: all-or-nothing compile FAILED for [%d-%d]; "
                "leaf range [%d-%d] is not Triton-compilable on this graph/device\n",
                seg.startSlot, seg.endSlot, range.startSlot, range.endSlot);
      return false;
    }
  }

  sd_printf("TritonGraphBackend: compile progress seg[%d-%d] summary "
            "(launched=%lld, completed=%lld, success=%lld, failed=%lld, splitRetries=%lld)\n",
            seg.startSlot, seg.endSlot,
            launchedRanges.load(), completedRanges.load(),
            successfulRanges.load(), failedRanges.load(), splitRetryCount);

  std::sort(compiledSeg.subKernels.begin(), compiledSeg.subKernels.end(),
            [](const CompiledKernel& a, const CompiledKernel& b) {
              return a.startSlot_ < b.startSlot_;
            });

  if (compiledSeg.subKernels.empty()) {
    sd_printf("TritonGraphBackend: no compiled sub-kernels for segment [%d-%d]\n",
              seg.startSlot, seg.endSlot);
    {
      std::lock_guard<std::mutex> lock(cacheMtx_);
      failedCache_.insert(key);
    }
    return false;
  }

  for (auto& kernel : compiledSeg.subKernels) {
    compiledSeg.audit.insert(compiledSeg.audit.end(),
                             kernel.audit.begin(),
                             kernel.audit.end());
  }

  sd_printf("TritonGraphBackend: adaptive compilation produced %d sub-segments for [%d-%d]\n",
            static_cast<int>(compiledSeg.subKernels.size()),
            seg.startSlot, seg.endSlot);

#ifdef SD_CUDA
  // Pre-allocate launch workspace outside runtime execution/capture.
  // This ensures the first captured Triton execution does not perform allocations.
  if (compileDevice >= 0) {
    auto setDevErr = cudaSetDevice(compileDevice);
    if (setDevErr != cudaSuccess) {
      sd_printf("TritonGraphBackend: failed to set CUDA device %d for launch workspace pre-allocation: %s\n",
                compileDevice, cudaGetErrorString(setDevErr));
      cudaGetLastError();
      for (auto& kernel : compiledSeg.subKernels) {
        if (kernel.gpuModule) TritonTargetDispatch::unloadModule(kernel.gpuModule);
      }
      return false;
    }
    // Ensure zero-length argument kernels have a valid per-device dummy pointer
    // before any stream capture path executes.
    if (getDummyDevicePtrForDevice(compileDevice, false) == nullptr) {
      sd_printf("TritonGraphBackend: failed pre-allocating dummy arg buffer on device %d for segment [%d-%d]\n",
                compileDevice, seg.startSlot, seg.endSlot);
    }
  }

  cudaStream_t preallocStream = nullptr;
  auto streamErr = cudaStreamCreateWithFlags(&preallocStream, cudaStreamNonBlocking);
  if (streamErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: failed to create pre-allocation stream for segment [%d-%d]: %s\n",
              seg.startSlot, seg.endSlot, cudaGetErrorString(streamErr));
    cudaGetLastError();
    for (auto& kernel : compiledSeg.subKernels) {
      if (kernel.gpuModule) TritonTargetDispatch::unloadModule(kernel.gpuModule);
    }
    return false;
  }

  auto cleanupCompiledWorkspace = [&]() {
    for (auto& k : compiledSeg.subKernels) {
      if (k.cachedArgTableDevice) {
        auto freeErr = freeDeviceBufferAsync(k.cachedArgTableDevice, preallocStream);
        if (freeErr != cudaSuccess) {
          sd_printf("TritonGraphBackend: failed freeing pre-allocated arg table for [%d-%d]: %s\n",
                    k.startSlot_, k.endSlot_, cudaGetErrorString(freeErr));
          cudaGetLastError();
        }
      }
      if (k.cachedSyncCounterDevice) {
        auto freeErr = freeDeviceBufferAsync(k.cachedSyncCounterDevice, preallocStream);
        if (freeErr != cudaSuccess) {
          sd_printf("TritonGraphBackend: failed freeing pre-allocated sync counter for [%d-%d]: %s\n",
                    k.startSlot_, k.endSlot_, cudaGetErrorString(freeErr));
          cudaGetLastError();
        }
      }
      k.cachedArgTableDevice = nullptr;
      k.cachedArgTableBytes = 0;
      k.cachedArgTableDeviceId = -1;
      k.cachedSyncCounterDevice = nullptr;
      k.cachedSyncCounterDeviceId = -1;
      if (k.gpuModule) TritonTargetDispatch::unloadModule(k.gpuModule);
    }
    cudaStreamSynchronize(preallocStream);
    cudaStreamDestroy(preallocStream);
    preallocStream = nullptr;
  };

  for (auto& kernel : compiledSeg.subKernels) {
    if (kernel.useIndirectArgs) {
      size_t tableBytes = kernel.argSlotMapping.size() * sizeof(int64_t);
      if (tableBytes == 0) tableBytes = sizeof(int64_t);
      if (kernel.cachedArgTableDevice == nullptr ||
          kernel.cachedArgTableBytes < tableBytes ||
          kernel.cachedArgTableDeviceId != compileDevice) {
        if (kernel.cachedArgTableDevice != nullptr) {
          auto freeErr = freeDeviceBufferAsync(kernel.cachedArgTableDevice, preallocStream);
          if (freeErr != cudaSuccess) {
            sd_printf("TritonGraphBackend: failed freeing stale arg table for sub-kernel [%d-%d]: %s\n",
                      kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(freeErr));
            cudaGetLastError();
            cleanupCompiledWorkspace();
            return false;
          }
          kernel.cachedArgTableDevice = nullptr;
          kernel.cachedArgTableBytes = 0;
          kernel.cachedArgTableDeviceId = -1;
        }
        auto allocErr = allocateDeviceBufferAsync(&kernel.cachedArgTableDevice, tableBytes, preallocStream);
        if (allocErr != cudaSuccess) {
          sd_printf("TritonGraphBackend: failed pre-allocating indirect arg table (%zu bytes) for sub-kernel [%d-%d]: %s\n",
                    tableBytes, kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(allocErr));
          cudaGetLastError();
          cleanupCompiledWorkspace();
          return false;
        }
        kernel.cachedArgTableBytes = tableBytes;
        kernel.cachedArgTableDeviceId = compileDevice;
      }
    }

    if (kernel.useCooperativeLaunch) {
      if (kernel.cachedSyncCounterDevice == nullptr ||
          kernel.cachedSyncCounterDeviceId != compileDevice) {
        if (kernel.cachedSyncCounterDevice != nullptr) {
          auto freeErr = freeDeviceBufferAsync(kernel.cachedSyncCounterDevice, preallocStream);
          if (freeErr != cudaSuccess) {
            sd_printf("TritonGraphBackend: failed freeing stale sync counter for sub-kernel [%d-%d]: %s\n",
                      kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(freeErr));
            cudaGetLastError();
            cleanupCompiledWorkspace();
            return false;
          }
          kernel.cachedSyncCounterDevice = nullptr;
          kernel.cachedSyncCounterDeviceId = -1;
        }
        auto allocErr = allocateDeviceBufferAsync(&kernel.cachedSyncCounterDevice, sizeof(int), preallocStream);
        if (allocErr != cudaSuccess) {
          sd_printf("TritonGraphBackend: failed pre-allocating cooperative sync counter for sub-kernel [%d-%d]: %s\n",
                    kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(allocErr));
          cudaGetLastError();
          cleanupCompiledWorkspace();
          return false;
        }
        kernel.cachedSyncCounterDeviceId = compileDevice;
      }
    }
  }

  auto preallocSyncErr = cudaStreamSynchronize(preallocStream);
  if (preallocSyncErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: pre-allocation stream sync failed for segment [%d-%d]: %s\n",
              seg.startSlot, seg.endSlot, cudaGetErrorString(preallocSyncErr));
    cudaGetLastError();
    cleanupCompiledWorkspace();
    return false;
  }
  auto preallocDestroyErr = cudaStreamDestroy(preallocStream);
  if (preallocDestroyErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: failed destroying pre-allocation stream for segment [%d-%d]: %s\n",
              seg.startSlot, seg.endSlot, cudaGetErrorString(preallocDestroyErr));
    cudaGetLastError();
    cleanupCompiledWorkspace();
    return false;
  }
#endif

  lastCompilationAudit_ = compiledSeg.audit;
  const int compiledKernelCount = static_cast<int>(compiledSeg.subKernels.size());

  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    failedCache_.erase(key);
    cache_[key] = std::move(compiledSeg);
  }

  sd_printf("TritonGraphBackend: compiled segment [%d-%d] (%d sub-kernels, shape key %lld)\n",
            seg.startSlot, seg.endSlot, compiledKernelCount, shapeKey);
  return true;
}

// ─── Execute a single compiled kernel ───────────────────────────────────────

Status TritonGraphBackend::executeSingleKernel(CompiledKernel& compiled, NativeSlot* slots,
                                                NDArray** externalInputs, int numExternalInputs,
                                                NDArray** outputSlots, int totalOutputSlots,
                                                void* stream) {
  int numBufferArgs = static_cast<int>(compiled.argSlotMapping.size());
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;

#ifdef SD_CUDA
  cudaStream_t cudaExecStream = static_cast<cudaStream_t>(actualStream);
  int currentDevice = -1;
  auto devErr = cudaGetDevice(&currentDevice);
  if (devErr != cudaSuccess) {
    sd_printf("TritonGraphBackend::executeSingleKernel: cudaGetDevice failed: %s\n",
              cudaGetErrorString(devErr));
    return Status::KERNEL_FAILURE;
  }

  bool streamIsCapturing = false;
  if (actualStream != nullptr) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto capErr = cudaStreamIsCapturing(static_cast<cudaStream_t>(actualStream), &captureStatus);
    if (capErr == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone) {
      streamIsCapturing = true;
    }
  }
#endif

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
      // Zero-length arrays (e.g., unused optional attention mask inputs) have no
      // device buffer. Provide a dummy pointer so the arg table has a valid address.
      // The kernel won't actually read/write this slot since the element count is 0.
      if (arr->lengthOf() == 0) {
#ifdef SD_CUDA
        sbuf = getDummyDevicePtrForDevice(currentDevice, streamIsCapturing);
#endif
        if (!sbuf) {
          sd_printf("TritonGraphBackend::executeSingleKernel: null specialBuffer for zero-length arg slot %d "
                    "(sub-segment [%d-%d], dtype=%d, device=%d, capturing=%d) and dummy pointer unavailable\n",
                    argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                    static_cast<int>(arr->dataType())
#ifdef SD_CUDA
                    , currentDevice, streamIsCapturing ? 1 : 0
#else
                    , -1, 0
#endif
                    );
          return Status::KERNEL_FAILURE;
        }
      } else {
        sd_printf("TritonGraphBackend::executeSingleKernel: null specialBuffer for arg slot %d "
                  "(sub-segment [%d-%d], length=%lld, dtype=%d)\n",
                  argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                  (long long)arr->lengthOf(), static_cast<int>(arr->dataType()));
        return Status::KERNEL_FAILURE;
      }
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


  // Compute grid size.
  // Only simple 1D element-wise kernels derive gridX from n_elements at launch.
  // Sectioned/matmul/attention kernels must use the IR-precomputed launch grid.
  unsigned int actualGridX = compiled.gridX;
  unsigned int actualGridY = compiled.gridY;
  unsigned int actualGridZ = compiled.gridZ;
  if (compiled.useDynamicGrid) {
    actualGridX = (nElements + compiled.blockX - 1) / compiled.blockX;
  }
  if (actualGridX == 0) actualGridX = 1;

  // Build kernel args — either direct (each ptr is a separate arg) or indirect
  // (all ptrs packed into a device-side i64 array, kernel receives 1 pointer)
  std::vector<void*> kernelArgs;
  void* argTableDevice = nullptr;
  void* syncCounterDevice = nullptr;

  if (compiled.useIndirectArgs) {
    // Pack all buffer pointers as int64 values into a device-side array.
    // The kernel signature is:
    //   non-cooperative: @kernel(%argTable: !tt.ptr<i64>, %n_elements: i32)
    //   cooperative:     @kernel(%argTable: !tt.ptr<i64>, %n_elements: i32, %sync_counter: !tt.ptr<i32>)
    // It loads each buffer pointer from argTable[i] and casts via tt.int_to_ptr.
    std::vector<int64_t> argTableHost(numBufferArgs);
    for (int i = 0; i < numBufferArgs; i++) {
      argTableHost[i] = reinterpret_cast<int64_t>(bufferPtrs[i]);
    }

#ifdef SD_CUDA
    // Reuse a persistent device arg table per compiled kernel.
    size_t tableBytes = numBufferArgs * sizeof(int64_t);
    bool deviceChanged = (compiled.cachedArgTableDeviceId != currentDevice);
    bool needsAlloc = deviceChanged || compiled.cachedArgTableDevice == nullptr ||
                      compiled.cachedArgTableBytes < tableBytes;
    if (needsAlloc) {
      if (streamIsCapturing) {
        sd_printf("TritonGraphBackend::executeSingleKernel: indirect arg table was not pre-allocated "
                  "for captured launch [%d-%d] (deviceChanged=%d, cachedPtr=%p, cachedBytes=%zu, "
                  "tableBytes=%zu, cachedDeviceId=%d, currentDevice=%d)\n",
                  compiled.startSlot_, compiled.endSlot_,
                  deviceChanged ? 1 : 0, compiled.cachedArgTableDevice,
                  compiled.cachedArgTableBytes, tableBytes,
                  compiled.cachedArgTableDeviceId, currentDevice);
        return Status::KERNEL_FAILURE;
      }
      if (compiled.cachedArgTableDevice != nullptr) {
        auto freeErr = freeDeviceBufferAsync(compiled.cachedArgTableDevice, cudaExecStream);
        if (freeErr != cudaSuccess) {
          sd_printf("TritonGraphBackend: failed to free stale arg table for [%d-%d]: %s\n",
                    compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(freeErr));
          return Status::KERNEL_FAILURE;
        }
        compiled.cachedArgTableDevice = nullptr;
        compiled.cachedArgTableBytes = 0;
        compiled.cachedArgTableDeviceId = -1;
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedArgTableDevice, tableBytes, cudaExecStream);
      if (allocErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to allocate arg table (%d bytes): %s\n",
                  (int)tableBytes, cudaGetErrorString(allocErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedArgTableBytes = tableBytes;
      compiled.cachedArgTableDeviceId = currentDevice;
    }
    argTableDevice = compiled.cachedArgTableDevice;

    // Copy host → device (async on the execution stream)
    auto memcpyErr = cudaMemcpyAsync(argTableDevice, argTableHost.data(), tableBytes,
                                     cudaMemcpyHostToDevice, cudaExecStream);
    if (memcpyErr != cudaSuccess) {
      sd_printf("TritonGraphBackend: failed to copy arg table (%d bytes) for [%d-%d]: %s\n",
                (int)tableBytes, compiled.startSlot_, compiled.endSlot_,
                cudaGetErrorString(memcpyErr));
      return Status::KERNEL_FAILURE;
    }
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

  if (compiled.useCooperativeLaunch) {
#ifdef SD_CUDA
    bool deviceChanged = (compiled.cachedSyncCounterDeviceId != currentDevice);
    if (deviceChanged && compiled.cachedSyncCounterDevice != nullptr) {
      if (streamIsCapturing) {
        sd_printf("TritonGraphBackend::executeSingleKernel: cooperative sync counter device mismatch during capture [%d-%d]\n",
                  compiled.startSlot_, compiled.endSlot_);
        return Status::KERNEL_FAILURE;
      }
      auto freeErr = freeDeviceBufferAsync(compiled.cachedSyncCounterDevice, cudaExecStream);
      if (freeErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to free stale cooperative sync counter for [%d-%d]: %s\n",
                  compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(freeErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedSyncCounterDevice = nullptr;
      compiled.cachedSyncCounterDeviceId = -1;
    }
    if (compiled.cachedSyncCounterDevice == nullptr) {
      if (streamIsCapturing) {
        sd_printf("TritonGraphBackend::executeSingleKernel: cooperative sync counter was not pre-allocated for captured launch [%d-%d]\n",
                  compiled.startSlot_, compiled.endSlot_);
        return Status::KERNEL_FAILURE;
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedSyncCounterDevice, sizeof(int), cudaExecStream);
      if (allocErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to allocate cooperative sync counter: %s\n",
                  cudaGetErrorString(allocErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedSyncCounterDeviceId = currentDevice;
    }
    syncCounterDevice = compiled.cachedSyncCounterDevice;

    auto memsetErr = cudaMemsetAsync(syncCounterDevice, 0, sizeof(int),
                                     cudaExecStream);
    if (memsetErr != cudaSuccess) {
      sd_printf("TritonGraphBackend: failed to initialize cooperative sync counter: %s\n",
                cudaGetErrorString(memsetErr));
      return Status::KERNEL_FAILURE;
    }
    // Cooperative sectioned kernels expect sync counter arg after n_elements.
    kernelArgs.push_back(&syncCounterDevice);
#else
    sd_printf("TritonGraphBackend: cooperative launch requested without CUDA support\n", "");
    return Status::KERNEL_FAILURE;
#endif
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
              "(cooperative=%d, dynamicGrid=%d, grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u)\n",
              compiled.startSlot_, compiled.endSlot_,
              compiled.useCooperativeLaunch ? 1 : 0, compiled.useDynamicGrid ? 1 : 0,
              actualGridX, actualGridY, actualGridZ,
              compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
              compiled.sharedMemBytes);
    return Status::KERNEL_FAILURE;
  }

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

#ifdef SD_CUDA
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;
  bool streamCaptureActive = false;
  if (actualStream != nullptr) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto capErr = cudaStreamIsCapturing(static_cast<cudaStream_t>(actualStream), &captureStatus);
    if (capErr == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone) {
      streamCaptureActive = true;
    }
  }

  if (streamCaptureActive && !compiledSeg->fallbackRanges.empty()) {
    sd_printf("TritonGraphBackend::executeSegment: refusing slot fallback during CUDA graph capture for [%d-%d] (%d fallback ranges)\n",
              seg.startSlot, seg.endSlot, static_cast<int>(compiledSeg->fallbackRanges.size()));
    return Status::KERNEL_FAILURE;
  }
#endif

  // Execute sub-kernels in-order and run uncovered slot gaps via callback.
  sd_printf("TritonGraphBackend::executeSegment: segment [%d-%d] launching %d sub-kernels "
            "(fallbackRanges=%d)\n",
            seg.startSlot, seg.endSlot,
            static_cast<int>(compiledSeg->subKernels.size()),
            static_cast<int>(compiledSeg->fallbackRanges.size()));

  int nextSlotToRun = seg.startSlot;
  for (int i = 0; i < (int)compiledSeg->subKernels.size(); i++) {
    auto& subKernel = compiledSeg->subKernels[i];

    if (nextSlotToRun < subKernel.startSlot_) {
#ifdef SD_CUDA
      if (streamCaptureActive) {
        sd_printf("TritonGraphBackend::executeSegment: refusing leading gap [%d-%d] during CUDA graph capture\n",
                  nextSlotToRun, subKernel.startSlot_ - 1);
        return Status::KERNEL_FAILURE;
      }
#endif
      if (!fallbackRangeExecutor_) {
        sd_printf("TritonGraphBackend::executeSegment: missing fallback executor for gap [%d-%d]\n",
                  nextSlotToRun, subKernel.startSlot_ - 1);
        return Status::KERNEL_FAILURE;
      }
      auto gapStatus = fallbackRangeExecutor_(nextSlotToRun, subKernel.startSlot_ - 1);
      if (gapStatus != Status::OK) {
        sd_printf("TritonGraphBackend::executeSegment: slot-by-slot gap [%d-%d] failed with status=%d\n",
                  nextSlotToRun, subKernel.startSlot_ - 1, static_cast<int>(gapStatus));
        return gapStatus;
      }
    }

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
    if (subKernel.endSlot_ + 1 > nextSlotToRun) {
      nextSlotToRun = subKernel.endSlot_ + 1;
    }
  }

  if (nextSlotToRun <= seg.endSlot) {
#ifdef SD_CUDA
    if (streamCaptureActive) {
      sd_printf("TritonGraphBackend::executeSegment: refusing trailing gap [%d-%d] during CUDA graph capture\n",
                nextSlotToRun, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
#endif
    if (!fallbackRangeExecutor_) {
      sd_printf("TritonGraphBackend::executeSegment: missing fallback executor for trailing gap [%d-%d]\n",
                nextSlotToRun, seg.endSlot);
      return Status::KERNEL_FAILURE;
    }
    auto gapStatus = fallbackRangeExecutor_(nextSlotToRun, seg.endSlot);
    if (gapStatus != Status::OK) {
      sd_printf("TritonGraphBackend::executeSegment: trailing slot-by-slot gap [%d-%d] failed with status=%d\n",
                nextSlotToRun, seg.endSlot, static_cast<int>(gapStatus));
      return gapStatus;
    }
  }

#ifdef SD_CUDA
  // Synchronize the execution stream to ensure all Triton kernels complete
  // before the caller reads output buffers. Without this, Java-side copyBuffer
  // on a different stream races with the async kernel, producing stale output.
  // Sync default stream as well (actualStream == nullptr). Skipping this leaves
  // step-N kernels in flight and can corrupt step-(N+1) execution.
  // Skip synchronize while stream capture is active (capture forbids sync).
  if (!streamCaptureActive) {
    auto syncErr = cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
    if (syncErr != cudaSuccess) {
      sd_printf("TritonGraphBackend::executeSegment: stream sync failed for [%d-%d]: %s\n",
                seg.startSlot, seg.endSlot, cudaGetErrorString(syncErr));
      cudaGetLastError();
      return Status::KERNEL_FAILURE;
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
#ifdef SD_CUDA
      if (kernel.cachedArgTableDevice != nullptr) {
        cudaFree(kernel.cachedArgTableDevice);
        kernel.cachedArgTableDevice = nullptr;
        kernel.cachedArgTableBytes = 0;
        kernel.cachedArgTableDeviceId = -1;
      }
      if (kernel.cachedSyncCounterDevice != nullptr) {
        cudaFree(kernel.cachedSyncCounterDevice);
        kernel.cachedSyncCounterDevice = nullptr;
        kernel.cachedSyncCounterDeviceId = -1;
      }
#endif
      if (kernel.gpuModule) {
        TritonTargetDispatch::unloadModule(kernel.gpuModule);
      }
    }
  }
  cache_.clear();
  failedCache_.clear();
  lastCompilationAudit_.clear();
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> TritonGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

// ─── Internal: compile to GPU binary ────────────────────────────────────────

TritonGraphBackend::CompiledKernel TritonGraphBackend::compileToGpuBinary(
    NativeSlot* slots, int startSlot, int endSlot,
    LongType segmentShapeKey,
    int totalSlots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {
  CompiledKernel result;
  auto now = []() { return std::chrono::steady_clock::now(); };
  auto elapsedMs = [&](const std::chrono::steady_clock::time_point& t0) -> long long {
    return static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now() - t0).count());
  };
  const auto tCompileStart = now();
  sd_printf("TritonGraphBackend: compileToGpuBinary START [%d-%d]\n", startSlot, endSlot);

  // Build Triton IR
  const auto tIrStart = now();
  TritonIRBuilder localBuilder;
  auto irModule = localBuilder.buildModule(slots, startSlot, endSlot,
                                           totalSlots,
                                           externalInputs, numExternalInputs,
                                           outputSlots, totalOutputSlots);
  const long long irBuildMs = elapsedMs(tIrStart);
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
    sd_printf("TritonGraphBackend: IR build FAILED for segment [%d-%d] after %lld ms\n",
              startSlot, endSlot, irBuildMs);
    return result;
  }
  sd_printf("TritonGraphBackend: IR build OK [%d-%d] in %lld ms "
            "(args=%d, indirect=%d, cooperative=%d, grid=%ux%ux%u, block=%ux%ux%u)\n",
            startSlot, endSlot, irBuildMs,
            static_cast<int>(irModule.args.size()),
            irModule.useIndirectArgs ? 1 : 0, irModule.useCooperativeLaunch ? 1 : 0,
            irModule.gridX, irModule.gridY, irModule.gridZ,
            irModule.blockX, irModule.blockY, irModule.blockZ);

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

  auto& env = sd::Environment::getInstance();
  int compileNumWarps = irModule.numWarps;
  int compileNumStages = irModule.numStages;
  if (env.tritonNumWarps() > 0) {
    compileNumWarps = std::max(1, std::min(env.tritonNumWarps(), 32));
  }
  if (env.tritonNumStages() > 0) {
    compileNumStages = std::max(1, std::min(env.tritonNumStages(), 16));
  }
  if (compileNumWarps != irModule.numWarps || compileNumStages != irModule.numStages) {
    sd_printf("TritonGraphBackend: compile option overrides for [%d-%d]: warps %d->%d, stages %d->%d\n",
              startSlot, endSlot,
              irModule.numWarps, compileNumWarps,
              irModule.numStages, compileNumStages);
  }

  const std::string cacheHash = computeDiskCacheHash(startSlot, endSlot, segmentShapeKey, ttirText,
                                                      compileNumWarps, compileNumStages);

  TritonCompiledBinary binary = {nullptr, 0, TritonGpuTarget::UNKNOWN, "", compileNumWarps, 0};
  const std::string archOverride = env.tritonOverrideArch();
  auto loadBinaryFromBasePath = [&](const std::string& basePath,
                                    const char* sourceLabel,
                                    TritonCompiledBinary& out) -> bool {
    const std::string ptxPath = basePath + ".ptx";
    const std::string metaPath = basePath + ".meta";

    std::ifstream ptxFile(ptxPath, std::ios::binary);
    if (!ptxFile.good()) return false;

    std::string ptxText((std::istreambuf_iterator<char>(ptxFile)),
                        std::istreambuf_iterator<char>());
    if (ptxText.empty()) return false;
    if (ptxText.back() != '\0') ptxText.push_back('\0');

    int metaNumWarps = compileNumWarps;
    int metaSharedMem = 0;
    std::string metaKernelName;

    std::ifstream metaFile(metaPath);
    if (metaFile.good()) {
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
    }

    if (!metaKernelName.empty() && metaKernelName != irModule.kernelName) {
      return false;
    }

    if (metaSharedMem == 0 && ptxUsesExternSharedMemory(ptxText)) {
      sd_printf("TritonGraphBackend: %s entry for [%d-%d] is stale "
                "(extern shared PTX with sharedMemBytes=0); ignoring\n",
                sourceLabel, startSlot, endSlot);
      return false;
    }

    out.data = new char[ptxText.size()];
    std::memcpy(out.data, ptxText.data(), ptxText.size());
    out.size = ptxText.size() - 1;  // Excludes null terminator
    out.target = TritonTargetDispatch::detectTarget();
    out.targetArch = TritonTargetDispatch::getTargetArch();
    if (!archOverride.empty()) {
      out.targetArch = archOverride;
    }
    out.numWarps = metaNumWarps;
    out.sharedMemBytes = metaSharedMem;
    sd_printf("TritonGraphBackend: %s HIT for sub-segment [%d-%d] (%zu bytes)\n",
              sourceLabel, startSlot, endSlot, out.size);
    return true;
  };

  auto dumpKernelArtifacts = [&](const TritonCompiledBinary& dumpBinary) {
    if (!env.tritonKernelDump() || dumpBinary.data == nullptr || dumpBinary.size == 0) return;
    const std::string dumpDir = configuredOrDefaultTritonDir(
        env.tritonDumpDir(), env.homeDirectory(), "triton_dump");
    if (!ensureDiskCacheDir(dumpDir)) return;

    const std::string basePath = dumpDir + "/ttir_" + cacheHash;
    {
      std::ofstream ttirOut(basePath + ".ttir", std::ios::trunc);
      if (ttirOut.good()) {
        ttirOut << ttirText;
      }
    }
    {
      std::ofstream ptxOut(basePath + ".ptx", std::ios::binary | std::ios::trunc);
      if (ptxOut.good()) {
        ptxOut.write(static_cast<const char*>(dumpBinary.data),
                     static_cast<std::streamsize>(dumpBinary.size));
      }
    }
    {
      std::ofstream metaOut(basePath + ".meta", std::ios::trunc);
      if (metaOut.good()) {
        metaOut << "numWarps=" << dumpBinary.numWarps << "\n";
        metaOut << "sharedMemBytes=" << dumpBinary.sharedMemBytes << "\n";
        metaOut << "kernelName=" << irModule.kernelName << "\n";
        metaOut << "numStages=" << compileNumStages << "\n";
        metaOut << "numCTAs=" << std::max(1, env.tritonNumCTAs()) << "\n";
        metaOut << "maxNreg=" << std::max(0, env.tritonMaxNreg()) << "\n";
      }
    }
  };

  const auto tBinaryStageStart = now();
  bool loadedFromOverride = false;
  if (env.tritonKernelOverride()) {
    const std::string overrideDir = configuredOrDefaultTritonDir(
        env.tritonOverrideDir(), env.homeDirectory(), "triton_override");
    const std::string basePath = overrideDir + "/ttir_" + cacheHash;
    loadedFromOverride = loadBinaryFromBasePath(basePath, "override", binary);
  }

  const bool alwaysCompile = env.tritonAlwaysCompile();
  bool loadedFromDiskCache = false;
  if (!loadedFromOverride && !alwaysCompile) {
    loadedFromDiskCache = loadBinaryFromDiskCache(startSlot, endSlot, cacheHash, irModule, binary);
  }

  if (!loadedFromOverride && !loadedFromDiskCache) {
    sd_printf("TritonGraphBackend: TTIR->PTX compile START [%d-%d]\n", startSlot, endSlot);
    const auto tCompileStageStart = now();
    binary = TritonTargetDispatch::compile(irModule.mlirModule, compileNumWarps, compileNumStages);
    sd_printf("TritonGraphBackend: TTIR->PTX compile %s [%d-%d] in %lld ms "
              "(ptxBytes=%zu, warps=%d, smem=%d)\n",
              binary.data != nullptr ? "DONE" : "FAILED",
              startSlot, endSlot, elapsedMs(tCompileStageStart),
              binary.size, binary.numWarps, binary.sharedMemBytes);
    if (binary.data && !alwaysCompile) {
      writeBinaryToDiskCache(startSlot, endSlot, cacheHash, irModule, binary);
    }
  } else if (loadedFromOverride) {
    sd_printf("TritonGraphBackend: override load DONE [%d-%d] in %lld ms "
              "(ptxBytes=%zu, warps=%d, smem=%d)\n",
              startSlot, endSlot, elapsedMs(tBinaryStageStart),
              binary.size, binary.numWarps, binary.sharedMemBytes);
  } else {
    sd_printf("TritonGraphBackend: PTX cache load DONE [%d-%d] in %lld ms "
              "(ptxBytes=%zu, warps=%d, smem=%d)\n",
              startSlot, endSlot, elapsedMs(tBinaryStageStart),
              binary.size, binary.numWarps, binary.sharedMemBytes);
  }

  dumpKernelArtifacts(binary);

  if (!binary.data) {
#ifdef SD_CUDA
    cudaGetLastError();  // Clear sticky CUDA errors from failed compilation
#endif
    sd_printf("TritonGraphBackend: Triton compilation FAILED for segment [%d-%d] "
              "(totalElapsed=%lld ms)\n",
              startSlot, endSlot, elapsedMs(tCompileStart));
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

#ifdef SD_CUDA
  if (binary.target == TritonGpuTarget::NVIDIA) {
    unsigned int requestedSharedMem =
        binary.sharedMemBytes > 0 ? static_cast<unsigned int>(binary.sharedMemBytes) : 0u;
    if (!configureCudaKernelSharedMemory(result.kernelFunction, requestedSharedMem)) {
      sd_printf("TritonGraphBackend: shared memory setup failed for segment [%d-%d] "
                "(requested=%u bytes)\n",
                startSlot, endSlot, requestedSharedMem);
      TritonTargetDispatch::unloadModule(result.gpuModule);
      result.gpuModule = nullptr;
      result.kernelFunction = nullptr;
      delete[] static_cast<char*>(binary.data);
      cleanupModule();
      return result;
    }
  }
#endif

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
  result.useDynamicGrid = irModule.useDynamicGrid;
  result.useIndirectArgs = irModule.useIndirectArgs;

  // Clean up
  delete[] static_cast<char*>(binary.data);
  cleanupModule();
  sd_printf("TritonGraphBackend: compileToGpuBinary DONE [%d-%d] total=%lld ms\n",
            startSlot, endSlot, elapsedMs(tCompileStart));

  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
