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
#include <condition_variable>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <thread>
#include <future>
#include <iomanip>
#include <limits>
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

inline bool queryCudaCooperativeLaunchCapacity(void* kernelFunc,
                                               unsigned int blockX, unsigned int blockY, unsigned int blockZ,
                                               unsigned int sharedMemBytes,
                                               bool* cooperativeSupported,
                                               long long* maxBlocks,
                                               int* blocksPerSm,
                                               int* smCount) {
  if (cooperativeSupported) *cooperativeSupported = false;
  if (maxBlocks) *maxBlocks = 0;
  if (blocksPerSm) *blocksPerSm = 0;
  if (smCount) *smCount = 0;
  if (kernelFunc == nullptr) return false;

  int currentDevice = 0;
  cudaError_t getDeviceErr = cudaGetDevice(&currentDevice);
  if (getDeviceErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: cudaGetDevice failed during cooperative capacity query: %s\n",
              cudaGetErrorString(getDeviceErr));
    cudaGetLastError();
    return false;
  }

  CUdevice cuDevice = 0;
  CUresult devRes = cuDeviceGet(&cuDevice, currentDevice);
  if (devRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(devRes, &errStr);
    sd_printf("TritonGraphBackend: cuDeviceGet failed during cooperative capacity query: %s (code=%d)\n",
              errStr ? errStr : "unknown", static_cast<int>(devRes));
    return false;
  }

  int coopLaunchAttr = 0;
  CUresult coopRes =
      cuDeviceGetAttribute(&coopLaunchAttr, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, cuDevice);
  if (coopRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(coopRes, &errStr);
    sd_printf("TritonGraphBackend: cuDeviceGetAttribute(COOPERATIVE_LAUNCH) failed: %s (code=%d)\n",
              errStr ? errStr : "unknown", static_cast<int>(coopRes));
    return false;
  }

  const bool coopSupported = (coopLaunchAttr != 0);
  if (cooperativeSupported) *cooperativeSupported = coopSupported;
  if (!coopSupported) return true;

  int smCountLocal = 0;
  CUresult smRes =
      cuDeviceGetAttribute(&smCountLocal, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
  if (smRes != CUDA_SUCCESS || smCountLocal <= 0) {
    const char* errStr = nullptr;
    cuGetErrorString(smRes, &errStr);
    sd_printf("TritonGraphBackend: cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: %s (code=%d)\n",
              errStr ? errStr : "unknown", static_cast<int>(smRes));
    return false;
  }

  unsigned long long threadsPerBlock64 =
      static_cast<unsigned long long>(blockX) *
      static_cast<unsigned long long>(blockY) *
      static_cast<unsigned long long>(blockZ);
  if (threadsPerBlock64 == 0 ||
      threadsPerBlock64 > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    sd_printf("TritonGraphBackend: invalid launch block size for cooperative capacity query: %llux%ux%u\n",
              static_cast<unsigned long long>(blockX), blockY, blockZ);
    return false;
  }

  int blocksPerSmLocal = 0;
  CUresult occRes = cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocksPerSmLocal,
      static_cast<CUfunction>(kernelFunc),
      static_cast<int>(threadsPerBlock64),
      sharedMemBytes);
  if (occRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(occRes, &errStr);
    sd_printf("TritonGraphBackend: cuOccupancyMaxActiveBlocksPerMultiprocessor failed: %s (code=%d)\n",
              errStr ? errStr : "unknown", static_cast<int>(occRes));
    return false;
  }

  blocksPerSmLocal = std::max(0, blocksPerSmLocal);
  long long capacity = static_cast<long long>(smCountLocal) * static_cast<long long>(blocksPerSmLocal);

  if (smCount) *smCount = smCountLocal;
  if (blocksPerSm) *blocksPerSm = blocksPerSmLocal;
  if (maxBlocks) *maxBlocks = capacity;
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
  int metaGlobalScratchBytes = 0;
  int metaGlobalScratchAlignment = 128;
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
    } else if (key == "globalScratchBytes") {
      parseIntValue(value, metaGlobalScratchBytes);
    } else if (key == "globalScratchAlignment") {
      parseIntValue(value, metaGlobalScratchAlignment);
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
  binary.globalScratchBytes = metaGlobalScratchBytes;
  binary.globalScratchAlignment = metaGlobalScratchAlignment;

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
  meta << "globalScratchBytes=" << binary.globalScratchBytes << "\n";
  meta << "globalScratchAlignment=" << binary.globalScratchAlignment << "\n";
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
#ifdef SD_CUDA
  int activeDevice = -1;
  cudaError_t activeDeviceErr = cudaGetDevice(&activeDevice);
  if (activeDeviceErr != cudaSuccess) {
    sd_printf("TritonGraphBackend::compileSegment: cudaGetDevice failed for segment [%d-%d]: %s\n",
              seg.startSlot, seg.endSlot, cudaGetErrorString(activeDeviceErr));
    cudaGetLastError();
    return false;
  }

  int targetDevice = -1;
  if (seg.startSlot >= 0) {
    targetDevice = slots[seg.startSlot].targetDeviceId;
  }

  int compileDevice = activeDevice;
  if (targetDevice >= 0) {
    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    if (countErr != cudaSuccess || deviceCount <= 0) {
      sd_printf("TritonGraphBackend::compileSegment: failed to query CUDA device count "
                "for segment [%d-%d] targetDeviceId=%d: %s\n",
                seg.startSlot, seg.endSlot, targetDevice, cudaGetErrorString(countErr));
      cudaGetLastError();
      return false;
    }
    if (targetDevice >= deviceCount) {
      sd_printf("TritonGraphBackend::compileSegment: invalid targetDeviceId=%d for segment [%d-%d] "
                "(deviceCount=%d)\n",
                targetDevice, seg.startSlot, seg.endSlot, deviceCount);
      return false;
    }
    compileDevice = targetDevice;
  }

  // Always call cudaSetDevice — not just when compileDevice != activeDevice.
  // On std::async precompilation threads the CUDA primary context may not be
  // initialized at all (cudaGetDevice returns 0 by default, but no context is
  // bound). Without an explicit cudaSetDevice, subsequent driver API calls
  // (cuModuleLoadDataEx, cuFuncSetAttribute) operate on an uninitialized or
  // wrong context, causing misaligned-address and kernel-launch failures.
  {
    cudaError_t setDeviceErr = cudaSetDevice(compileDevice);
    if (setDeviceErr != cudaSuccess) {
      sd_printf("TritonGraphBackend::compileSegment: failed to set CUDA device %d for segment [%d-%d]: %s\n",
                compileDevice, seg.startSlot, seg.endSlot, cudaGetErrorString(setDeviceErr));
      cudaGetLastError();
      return false;
    }
  }
#else
  const int compileDevice = -1;
#endif

  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey, compileDevice};

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
                "(shapeKey=%lld, deviceId=%d)\n",
                seg.startSlot, seg.endSlot, shapeKey, compileDevice);
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

  sd_printf("TritonGraphBackend: segment [%d-%d] has %d ops, %d sections (deviceId=%d)\n",
            seg.startSlot, seg.endSlot, segmentOps, static_cast<int>(sections.size()),
            compileDevice);

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

  // Pre-compute cooperative launch capacity for this device.
  // Only needed when cooperative launch is enabled — when disabled (default),
  // grid sync barriers are suppressed in the IR builder and no splitting is needed.
  const bool cooperativeEnabled = Environment::getInstance().tritonCooperativeLaunch();
#ifdef SD_CUDA
  int preCheckSmCount = 0;
  int preCheckMaxThreadsPerSM = 0;
  int preCheckMaxSharedPerSM = 0;
  if (cooperativeEnabled) {
    int currentDevice = 0;
    cudaError_t devErr = cudaGetDevice(&currentDevice);
    if (devErr == cudaSuccess) {
      CUdevice cuDevice = 0;
      CUresult cuDevErr = cuDeviceGet(&cuDevice, currentDevice);
      if (cuDevErr == CUDA_SUCCESS) {
        cuDeviceGetAttribute(&preCheckSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
        cuDeviceGetAttribute(&preCheckMaxThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice);
        cuDeviceGetAttribute(&preCheckMaxSharedPerSM,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cuDevice);
      }
    }
  }

  // Estimate max cooperative blocks for a typical Triton kernel (4 warps = 128 threads).
  // This is conservative — actual capacity depends on shared memory and register usage.
  auto estimateMaxCooperativeBlocks = [&](int numWarps, int estimatedSharedMem) -> unsigned long long {
    int threadsPerBlock = std::max(1, numWarps) * 32;
    int blocksPerSmByThreads = (preCheckMaxThreadsPerSM > 0 && threadsPerBlock > 0)
        ? (preCheckMaxThreadsPerSM / threadsPerBlock) : 16;
    int blocksPerSmBySmem = 16;
    if (estimatedSharedMem > 0 && preCheckMaxSharedPerSM > 0) {
      blocksPerSmBySmem = preCheckMaxSharedPerSM / estimatedSharedMem;
    }
    int blocksPerSm = std::max(1, std::min(blocksPerSmByThreads, blocksPerSmBySmem));
    return static_cast<unsigned long long>(preCheckSmCount) * blocksPerSm;
  };
#endif

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

    // If this run has multiple sections, check if cooperative launch is feasible.
    // Multi-section kernels need cooperative launch when sections have different
    // grid mappings (e.g., elementwise + matmul). If the max section grid exceeds
    // the device's cooperative capacity, pre-split into individual sections to
    // avoid a compile-fail-split cycle.
    bool needsPreSplit = false;
#ifdef SD_CUDA
    if (cooperativeEnabled && runEnd > runStart && preCheckSmCount > 0) {
      // Find max grid requirement across sections in this run
      int maxSectionGrid = 0;
      bool hasNonElementwise = false;
      for (int s = runStart; s <= runEnd; s++) {
        if (sections[s].gridRequirement > maxSectionGrid) {
          maxSectionGrid = sections[s].gridRequirement;
        }
        if (sections[s].type != KernelSectionType::ELEMENTWISE) {
          hasNonElementwise = true;
        }
      }
      if (hasNonElementwise && maxSectionGrid > 0) {
        // Estimate shared memory: matmul sections typically use 49152+ bytes
        int estimatedSharedMem = 0;
        for (int s = runStart; s <= runEnd; s++) {
          if (sections[s].type == KernelSectionType::MATMUL) {
            estimatedSharedMem = std::max(estimatedSharedMem, 49152);
          } else if (sections[s].type == KernelSectionType::FUSED_ATTENTION) {
            estimatedSharedMem = std::max(estimatedSharedMem, 49152);
          }
        }
        unsigned long long maxCoopBlocks = estimateMaxCooperativeBlocks(4, estimatedSharedMem);
        if (static_cast<unsigned long long>(maxSectionGrid) > maxCoopBlocks) {
          sd_printf("TritonGraphBackend: pre-splitting multi-section run [sec %d-%d, slots %d-%d] "
                    "into individual sections: maxGrid=%d exceeds cooperative capacity=%llu "
                    "(smCount=%d)\n",
                    runStart, runEnd,
                    sections[runStart].startSlot, sections[runEnd].endSlot,
                    maxSectionGrid, maxCoopBlocks, preCheckSmCount);
          needsPreSplit = true;
        }
      }
    }
#endif

    if (needsPreSplit) {
      // Emit each section as its own compile range
      for (int s = runStart; s <= runEnd; s++) {
        pendingRanges.push_back(makeRange(s, s));
      }
    } else {
      pendingRanges.push_back(makeRange(runStart, runEnd));
    }
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
  int activeCompileDevice = -1;
  cudaError_t compileDeviceErr = cudaGetDevice(&activeCompileDevice);
  if (compileDeviceErr != cudaSuccess) {
    sd_printf("TritonGraphBackend: cudaGetDevice failed before adaptive compilation "
              "for segment [%d-%d]: %s\n",
              seg.startSlot, seg.endSlot, cudaGetErrorString(compileDeviceErr));
    cudaGetLastError();
    return false;
  }
  sd_printf("TritonGraphBackend: compile device binding seg[%d-%d] targetDeviceId=%d activeDevice=%d cacheDeviceId=%d\n",
            seg.startSlot, seg.endSlot, targetDevice, activeCompileDevice, compileDevice);
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

  // ── Step 2: Work-stealing compile loop ──
  // Instead of batch-sequential dispatch (launch N, wait ALL, repeat), use a
  // shared work queue with condition variables so workers pick up new work
  // (including split retries) as soon as they finish — eliminating tail latency.

  // Pre-split any ranges that exceed caps before entering the work queue.
  {
    std::deque<SubSegmentRange> preSplit;
    while (!pendingRanges.empty()) {
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
      } else {
        preSplit.push_back(range);
      }
    }
    pendingRanges = std::move(preSplit);
  }

  const int totalInitialRanges = static_cast<int>(pendingRanges.size());
  sd_printf("TritonGraphBackend: work-stealing compile seg[%d-%d] "
            "(ranges=%d, workers=%d)\n",
            seg.startSlot, seg.endSlot, totalInitialRanges, maxParallelCompiles);

  // Shared state for the work-stealing pool
  std::mutex workMtx;
  std::condition_variable workCv;
  std::vector<CompileRangeResult> allResults;
  bool leafFailed = false;       // Set when an unsplittable range fails
  SubSegmentRange failedLeaf{};  // The range that caused the leaf failure
  std::atomic<int> activeWorkers{0};

  auto workerLoop = [&]() {
    while (true) {
      SubSegmentRange range;
      {
        std::unique_lock<std::mutex> lock(workMtx);
        workCv.wait(lock, [&] {
          return !pendingRanges.empty() || leafFailed ||
                 (pendingRanges.empty() && activeWorkers.load() == 0);
        });
        if (leafFailed) return;
        if (pendingRanges.empty()) {
          if (activeWorkers.load() == 0) return;  // All work done
          continue;
        }
        range = pendingRanges.front();
        pendingRanges.pop_front();
        activeWorkers.fetch_add(1);
      }

      auto result = compileRange(range);
      const bool success = (result.compiled.gpuModule && result.compiled.kernelFunction);

      {
        std::lock_guard<std::mutex> lock(workMtx);
        if (success) {
          allResults.push_back(std::move(result));
        } else {
#ifdef SD_CUDA
          cudaGetLastError();
#endif
          const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
          const bool canSplit = (sectionCount > 1) || (range.opsCount > 1);
          if (canSplit) {
            sd_printf("TritonGraphBackend: adaptive range [%d-%d] compile failed; "
                      "splitting by section graph\n",
                      range.startSlot, range.endSlot);
            splitRetryCount++;
            splitRange(range);
            // New sub-ranges are in pendingRanges; wake other workers
          } else {
            // Leaf range failed — signal all workers to stop
            leafFailed = true;
            failedLeaf = range;
          }
        }
        activeWorkers.fetch_sub(1);
      }
      workCv.notify_all();
    }
  };

  if (maxParallelCompiles > 1 && pendingRanges.size() > 1) {
    // Launch worker threads
    const int numWorkers = std::min(maxParallelCompiles,
                                    static_cast<int>(pendingRanges.size()));
    std::vector<std::thread> workers;
    workers.reserve(numWorkers);
    for (int i = 0; i < numWorkers; i++) {
      workers.emplace_back(workerLoop);
    }
    for (auto& w : workers) {
      w.join();
    }
  } else {
    // Single-threaded: drain queue directly (no thread overhead)
    while (!pendingRanges.empty() && !leafFailed) {
      SubSegmentRange range = pendingRanges.front();
      pendingRanges.pop_front();

      auto result = compileRange(range);
      const bool success = (result.compiled.gpuModule && result.compiled.kernelFunction);
      if (success) {
        allResults.push_back(std::move(result));
      } else {
#ifdef SD_CUDA
        cudaGetLastError();
#endif
        const int sectionCount = range.endSectionIdx - range.startSectionIdx + 1;
        const bool canSplit = (sectionCount > 1) || (range.opsCount > 1);
        if (canSplit) {
          sd_printf("TritonGraphBackend: adaptive range [%d-%d] compile failed; "
                    "splitting by section graph\n",
                    range.startSlot, range.endSlot);
          splitRetryCount++;
          splitRange(range);
        } else {
          leafFailed = true;
          failedLeaf = range;
        }
      }
    }
  }

  if (leafFailed) {
    // Leaf range failed: all-or-nothing reject entire segment.
    for (auto& r : allResults) {
      if (r.compiled.gpuModule) {
        TritonTargetDispatch::unloadModule(r.compiled.gpuModule);
        r.compiled.gpuModule = nullptr;
        r.compiled.kernelFunction = nullptr;
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
              "leaf range [%d-%d] is not Triton-compilable on this graph/device (deviceId=%d)\n",
              seg.startSlot, seg.endSlot, failedLeaf.startSlot, failedLeaf.endSlot, compileDevice);
    return false;
  }

  // Move successful results into compiledSeg
  for (auto& r : allResults) {
    compiledSeg.subKernels.push_back(std::move(r.compiled));
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
      if (k.cachedGlobalScratchDevice) {
        auto freeErr = freeDeviceBufferAsync(k.cachedGlobalScratchDevice, preallocStream);
        if (freeErr != cudaSuccess) {
          sd_printf("TritonGraphBackend: failed freeing pre-allocated global scratch for [%d-%d]: %s\n",
                    k.startSlot_, k.endSlot_, cudaGetErrorString(freeErr));
          cudaGetLastError();
        }
      }
      k.cachedArgTableDevice = nullptr;
      k.cachedArgTableBytes = 0;
      k.cachedArgTableDeviceId = -1;
      k.cachedSyncCounterDevice = nullptr;
      k.cachedSyncCounterDeviceId = -1;
      k.cachedGlobalScratchDevice = nullptr;
      k.cachedGlobalScratchBytes = 0;
      k.cachedGlobalScratchDeviceId = -1;
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

  sd_printf("TritonGraphBackend: compiled segment [%d-%d] (%d sub-kernels, shape key %lld, deviceId=%d)\n",
            seg.startSlot, seg.endSlot, compiledKernelCount, shapeKey, compileDevice);
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
  // Clear any sticky CUDA errors left by prior sub-kernel failures.
  // Without this, a device-side error (e.g., misaligned access) from an earlier
  // kernel execution contaminates the CUDA context and causes ALL subsequent
  // operations (memcpy, launch, etc.) to report the same stale error.
  {
    cudaError_t staleErr = cudaGetLastError();
    if (staleErr != cudaSuccess) {
      sd_printf("TritonGraphBackend::executeSingleKernel: cleared stale CUDA error before [%d-%d]: %s\n",
                compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(staleErr));
    }
  }

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

    // Use persistent PINNED host buffer for the arg table source.
    // CUDA graph capture records the cudaMemcpyAsync source address — if we use
    // a stack-local vector, the graph replay reads from dead stack memory → SIGSEGV.
    // Pinned memory survives across graph replays.
    if (compiled.cachedArgTableHostPinned == nullptr ||
        compiled.cachedArgTableHostPinnedBytes < tableBytes) {
      if (compiled.cachedArgTableHostPinned != nullptr) {
        cudaFreeHost(compiled.cachedArgTableHostPinned);
        compiled.cachedArgTableHostPinned = nullptr;
        compiled.cachedArgTableHostPinnedBytes = 0;
      }
      auto pinnedErr = cudaMallocHost(&compiled.cachedArgTableHostPinned, tableBytes);
      if (pinnedErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to allocate pinned arg table host (%d bytes): %s\n",
                  (int)tableBytes, cudaGetErrorString(pinnedErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedArgTableHostPinnedBytes = tableBytes;
    }

    // Write buffer pointers into the persistent pinned host buffer
    auto* argTableHostPinned = static_cast<int64_t*>(compiled.cachedArgTableHostPinned);
    for (int i = 0; i < numBufferArgs; i++) {
      argTableHostPinned[i] = reinterpret_cast<int64_t>(bufferPtrs[i]);
    }

    argTableDevice = compiled.cachedArgTableDevice;

    // Validate arg table pointer before copy
    if (argTableDevice == nullptr) {
      sd_printf("TritonGraphBackend: arg table device pointer is NULL for [%d-%d] "
                "(tableBytes=%d, cachedDeviceId=%d, currentDevice=%d)\n",
                compiled.startSlot_, compiled.endSlot_,
                (int)tableBytes, compiled.cachedArgTableDeviceId, currentDevice);
      return Status::KERNEL_FAILURE;
    }

    // Check pointer alignment (CUDA requires at least 4-byte alignment for memcpy)
    if (reinterpret_cast<uintptr_t>(argTableDevice) % 4 != 0) {
      sd_printf("TritonGraphBackend: arg table device pointer %p is misaligned for [%d-%d] "
                "(alignment=%zu, cachedDeviceId=%d)\n",
                argTableDevice, compiled.startSlot_, compiled.endSlot_,
                reinterpret_cast<uintptr_t>(argTableDevice) % 256,
                compiled.cachedArgTableDeviceId);
      return Status::KERNEL_FAILURE;
    }

    // Copy host → device (async on the execution stream)
    // Uses the persistent pinned host buffer — safe for CUDA graph capture/replay.
    auto memcpyErr = cudaMemcpyAsync(argTableDevice, argTableHostPinned, tableBytes,
                                     cudaMemcpyHostToDevice, cudaExecStream);
    if (memcpyErr != cudaSuccess) {
      sd_printf("TritonGraphBackend: failed to copy arg table (%d bytes) for [%d-%d]: %s "
                "(devicePtr=%p, hostPtr=%p, cachedDeviceId=%d, currentDevice=%d, stream=%p)\n",
                (int)tableBytes, compiled.startSlot_, compiled.endSlot_,
                cudaGetErrorString(memcpyErr),
                argTableDevice, argTableHostPinned,
                compiled.cachedArgTableDeviceId, currentDevice, (void*)cudaExecStream);
      cudaGetLastError();  // Clear the error so subsequent operations aren't poisoned
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

  // Multi-phase launch: phase_id argument (set per-launch, before implicit Triton args)
  int phaseId = 0;  // Default phase 0 (overridden in multi-launch loop)
  if (compiled.useMultiPhaseLaunch) {
    kernelArgs.push_back(&phaseId);  // Will be updated per-phase in the launch loop
  }

  // ── Triton 3.6.0 implicit kernel arguments ──
  // The TritonGPUToLLVM FuncOp conversion adds 2 extra pointer arguments to every
  // kernel function (see FuncOpToLLVM.cpp, NOTE: [Additional Function Arguments]):
  //   1. global_scratch_ptr — pointer to per-program scratch memory
  //   2. profile_ptr — pointer to profiling data (unused, pass nullptr)
  // These MUST be appended to kernelArgs in the same order as the lowering adds them.
  void* globalScratchPtr = nullptr;
  void* profilePtr = nullptr;

#ifdef SD_CUDA
  if (compiled.globalScratchBytes > 0) {
    // Allocate or reuse persistent global scratch buffer
    // Triton partitions scratch by program ID, so total size = scratchBytes * numPrograms
    unsigned int numPrograms = actualGridX * actualGridY * actualGridZ;
    size_t totalScratchBytes = static_cast<size_t>(compiled.globalScratchBytes) * numPrograms;
    bool deviceChanged = (compiled.cachedGlobalScratchDeviceId != currentDevice);
    bool needsAlloc = deviceChanged || compiled.cachedGlobalScratchDevice == nullptr ||
                      compiled.cachedGlobalScratchBytes < totalScratchBytes;
    if (needsAlloc) {
      if (compiled.cachedGlobalScratchDevice != nullptr) {
        freeDeviceBufferAsync(compiled.cachedGlobalScratchDevice, cudaExecStream);
        compiled.cachedGlobalScratchDevice = nullptr;
        compiled.cachedGlobalScratchBytes = 0;
        compiled.cachedGlobalScratchDeviceId = -1;
      }
      auto allocErr = allocateDeviceBufferAsync(&compiled.cachedGlobalScratchDevice,
                                                 totalScratchBytes, cudaExecStream);
      if (allocErr != cudaSuccess) {
        sd_printf("TritonGraphBackend: failed to allocate global scratch (%zu bytes) for [%d-%d]: %s\n",
                  totalScratchBytes, compiled.startSlot_, compiled.endSlot_,
                  cudaGetErrorString(allocErr));
        return Status::KERNEL_FAILURE;
      }
      compiled.cachedGlobalScratchBytes = totalScratchBytes;
      compiled.cachedGlobalScratchDeviceId = currentDevice;
    }
    globalScratchPtr = compiled.cachedGlobalScratchDevice;
  }
#endif

  kernelArgs.push_back(&globalScratchPtr);
  kernelArgs.push_back(&profilePtr);

#ifdef SD_CUDA
  // Re-apply shared memory opt-in at launch time as a safety net.
  // The attribute was set during compilation, but if the CUfunction was compiled
  // on a different thread (parallel compilation pool), the attribute may not have
  // persisted to the execution context. Re-applying is cheap and prevents
  // cuLaunchKernel from failing with CUDA_ERROR_INVALID_VALUE for >48KB shared mem.
  if (compiled.sharedMemBytes > 49152u) {
    if (!configureCudaKernelSharedMemory(compiled.kernelFunction, compiled.sharedMemBytes)) {
      sd_printf("TritonGraphBackend::executeSingleKernel: shared memory re-configuration failed "
                "for [%d-%d] (requested=%u bytes, device=%d)\n",
                compiled.startSlot_, compiled.endSlot_, compiled.sharedMemBytes, currentDevice);
      return Status::KERNEL_FAILURE;
    }
  }

  // Clear any error that might have been set by the shared memory configuration
  cudaGetLastError();
#endif

  // Launch
  bool ok;
  if (compiled.useMultiPhaseLaunch && !compiled.launchPhases.empty()) {
    // Multi-phase launch: launch the SAME kernel once per phase, with different
    // phase_id and grid size. Each kernel launch provides implicit global sync.
    ok = true;
    for (size_t p = 0; p < compiled.launchPhases.size() && ok; p++) {
      phaseId = static_cast<int>(p);  // Update phase_id in kernelArgs (points to &phaseId)
      unsigned int phaseGridX = static_cast<unsigned int>(
          std::max(1, compiled.launchPhases[p].gridX));
      ok = TritonTargetDispatch::launchKernel(
          compiled.kernelFunction,
          phaseGridX, actualGridY, actualGridZ,
          compiled.numWarps * 32,
          compiled.blockY, compiled.blockZ,
          compiled.sharedMemBytes,
          actualStream,
          kernelArgs.data(),
          static_cast<int>(kernelArgs.size()));
    }
  } else if (compiled.useCooperativeLaunch) {
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
#ifdef SD_CUDA
    // Log detailed diagnostic info for launch failures
    int maxSharedOptIn = 0, maxSharedDefault = 0;
    cudaDeviceGetAttribute(&maxSharedOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, currentDevice);
    cudaDeviceGetAttribute(&maxSharedDefault, cudaDevAttrMaxSharedMemoryPerBlock, currentDevice);
    sd_printf("TritonGraphBackend::executeSingleKernel: kernel launch failed for [%d-%d] "
              "(cooperative=%d, dynamicGrid=%d, grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u, "
              "deviceSharedDefault=%d, deviceSharedOptIn=%d, kernelFunc=%p)\n",
              compiled.startSlot_, compiled.endSlot_,
              compiled.useCooperativeLaunch ? 1 : 0, compiled.useDynamicGrid ? 1 : 0,
              actualGridX, actualGridY, actualGridZ,
              compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
              compiled.sharedMemBytes,
              maxSharedDefault, maxSharedOptIn, compiled.kernelFunction);
    cudaGetLastError();  // Clear the error from the failed launch
#else
    sd_printf("TritonGraphBackend::executeSingleKernel: kernel launch failed for [%d-%d] "
              "(cooperative=%d, dynamicGrid=%d, grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u)\n",
              compiled.startSlot_, compiled.endSlot_,
              compiled.useCooperativeLaunch ? 1 : 0, compiled.useDynamicGrid ? 1 : 0,
              actualGridX, actualGridY, actualGridZ,
              compiled.numWarps * 32, compiled.blockY, compiled.blockZ,
              compiled.sharedMemBytes);
#endif
    return Status::KERNEL_FAILURE;
  }

  return Status::OK;
}

// ─── Execution ──────────────────────────────────────────────────────────────

Status TritonGraphBackend::executeSegment(GraphSegment& seg, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots,
                                          void* stream) {
  void* actualStream = (stream != nullptr) ? *static_cast<void**>(stream) : nullptr;
  int execDevice = -1;
  int targetDevice = -1;
  if (seg.startSlot >= 0) {
    targetDevice = slots[seg.startSlot].targetDeviceId;
  }
  bool streamCaptureActive = false;

#ifdef SD_CUDA
  cudaError_t execDeviceErr = cudaGetDevice(&execDevice);
  if (execDeviceErr != cudaSuccess) {
    sd_printf("TritonGraphBackend::executeSegment: cudaGetDevice failed for segment [%d-%d]: %s\n",
              seg.startSlot, seg.endSlot, cudaGetErrorString(execDeviceErr));
    cudaGetLastError();
    return Status::KERNEL_FAILURE;
  }
  if (actualStream != nullptr) {
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    auto capErr = cudaStreamIsCapturing(static_cast<cudaStream_t>(actualStream), &captureStatus);
    if (capErr == cudaSuccess && captureStatus != cudaStreamCaptureStatusNone) {
      streamCaptureActive = true;
    }
  }
#endif

  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, execDevice};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      int cachedDeviceId = -999;
      for (const auto& entry : cache_) {
        if (entry.first.startSlot == seg.startSlot &&
            entry.first.endSlot == seg.endSlot &&
            entry.first.shapeKey == seg.shapeKey) {
          cachedDeviceId = entry.first.deviceId;
          break;
        }
      }
      if (cachedDeviceId != -999) {
        sd_printf("TritonGraphBackend::executeSegment: kernel cache miss for segment [%d-%d] "
                  "(shapeKey=%lld, activeDevice=%d, targetDeviceId=%d). "
                  "Found compiled kernel for deviceId=%d but cross-device module reuse is disallowed.\n",
                  seg.startSlot, seg.endSlot, seg.shapeKey, execDevice, targetDevice, cachedDeviceId);
      } else {
        sd_printf("TritonGraphBackend::executeSegment: no compiled kernel for segment [%d-%d] "
                  "(shapeKey=%lld, deviceId=%d)\n",
                  seg.startSlot, seg.endSlot, seg.shapeKey, execDevice);
      }
      return Status::KERNEL_FAILURE;
    }
    compiledSeg = &it->second;
  }

#ifdef SD_CUDA
  if (streamCaptureActive && !compiledSeg->fallbackRanges.empty()) {
    sd_printf("TritonGraphBackend::executeSegment: refusing slot fallback during CUDA graph capture for [%d-%d] (%d fallback ranges)\n",
              seg.startSlot, seg.endSlot, static_cast<int>(compiledSeg->fallbackRanges.size()));
    return Status::KERNEL_FAILURE;
  }
#endif

  // Execute sub-kernels in-order and run uncovered slot gaps via callback.
  sd_printf("TritonGraphBackend::executeSegment: segment [%d-%d] launching %d sub-kernels "
            "(fallbackRanges=%d, targetDeviceId=%d, activeDevice=%d)\n",
            seg.startSlot, seg.endSlot,
            static_cast<int>(compiledSeg->subKernels.size()),
            static_cast<int>(compiledSeg->fallbackRanges.size()),
            targetDevice, execDevice);

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

#ifdef SD_CUDA
    // Synchronize execution stream before launching the next sub-kernel to ensure
    // any prior async errors are caught and cleared rather than cascading.
    if (i > 0 && !streamCaptureActive) {
      cudaError_t syncErr = cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));
      if (syncErr != cudaSuccess) {
        sd_printf("TritonGraphBackend::executeSegment: stream sync before sub-kernel %d/%d [%d-%d] "
                  "detected prior async error: %s\n",
                  i + 1, (int)compiledSeg->subKernels.size(),
                  subKernel.startSlot_, subKernel.endSlot_,
                  cudaGetErrorString(syncErr));
        cudaGetLastError();  // Clear the sticky error
      }
    }
#endif
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
            "(args=%d, indirect=%d, cooperative=%d, multiPhase=%d(%d phases), grid=%ux%ux%u, block=%ux%ux%u)\n",
            startSlot, endSlot, irBuildMs,
            static_cast<int>(irModule.args.size()),
            irModule.useIndirectArgs ? 1 : 0, irModule.useCooperativeLaunch ? 1 : 0,
            irModule.useMultiPhaseLaunch ? 1 : 0,
            static_cast<int>(irModule.launchPhases.size()),
            irModule.gridX, irModule.gridY, irModule.gridZ,
            irModule.blockX, irModule.blockY, irModule.blockZ);

#ifdef SD_CUDA
  // ── Early cooperative launch capacity check ──
  // Reject BEFORE the expensive TTIR→PTX compilation (which can take 30+ minutes
  // for large fused kernels) if the required grid clearly exceeds what the GPU
  // can support for cooperative launch. We estimate blocks/SM from both thread
  // occupancy (maxThreadsPerSM / threadsPerBlock) and shared memory occupancy
  // (maxSharedPerSM / estimatedSharedMemBytes). The estimate is conservative
  // (may allow some cases that will fail post-compile) but catches the common
  // case of 400+ blocks on 128 SMs with large shared memory per block.
  if (irModule.useCooperativeLaunch) {
    unsigned long long requiredBlocks =
        static_cast<unsigned long long>(std::max(1u, irModule.gridX)) *
        static_cast<unsigned long long>(std::max(1u, irModule.gridY)) *
        static_cast<unsigned long long>(std::max(1u, irModule.gridZ));
    if (irModule.requiredGrid > 0) {
      requiredBlocks = std::max(requiredBlocks,
                                static_cast<unsigned long long>(std::max(1, irModule.requiredGrid)));
    }

    int currentDevice = 0;
    cudaError_t devErr = cudaGetDevice(&currentDevice);
    if (devErr == cudaSuccess) {
      CUdevice cuDevice = 0;
      CUresult cuDevErr = cuDeviceGet(&cuDevice, currentDevice);
      if (cuDevErr == CUDA_SUCCESS) {
        int smCount = 0;
        int maxThreadsPerSM = 0;
        int maxSharedPerSM = 0;
        cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDevice);
        cuDeviceGetAttribute(&maxThreadsPerSM, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, cuDevice);
        cuDeviceGetAttribute(&maxSharedPerSM,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, cuDevice);

        // Compute blocks/SM upper bound from BOTH thread and shared memory occupancy.
        // The actual occupancy is min(thread limit, shared memory limit).
        int threadsPerBlock = std::max(1, irModule.numWarps) * 32;
        int blocksPerSmByThreads = (maxThreadsPerSM > 0 && threadsPerBlock > 0)
            ? (maxThreadsPerSM / threadsPerBlock)
            : 16;

        int blocksPerSmBySmem = 16;  // default if no estimate
        if (irModule.estimatedSharedMemBytes > 0 && maxSharedPerSM > 0) {
          blocksPerSmBySmem = maxSharedPerSM / irModule.estimatedSharedMemBytes;
        }

        int blocksPerSmEstimate = std::max(1, std::min(blocksPerSmByThreads, blocksPerSmBySmem));

        unsigned long long maxPossibleBlocks =
            static_cast<unsigned long long>(smCount) * blocksPerSmEstimate;
        if (smCount > 0 && requiredBlocks > maxPossibleBlocks) {
          sd_printf("TritonGraphBackend: EARLY REJECT cooperative launch for [%d-%d]: "
                    "requiredBlocks=%llu exceeds max=%llu "
                    "(smCount=%d, blocksPerSm<=%d [threads: %d/%d=%d, smem: %d/%d=%d]). "
                    "Skipping expensive compilation.\n",
                    startSlot, endSlot,
                    requiredBlocks, maxPossibleBlocks,
                    smCount, blocksPerSmEstimate,
                    maxThreadsPerSM, threadsPerBlock, blocksPerSmByThreads,
                    maxSharedPerSM, irModule.estimatedSharedMemBytes, blocksPerSmBySmem);
          cleanupModule();
          return result;
        }
        sd_printf("TritonGraphBackend: cooperative launch pre-check OK for [%d-%d]: "
                  "requiredBlocks=%llu, maxPossible=%llu (smCount=%d, blocksPerSm<=%d)\n",
                  startSlot, endSlot, requiredBlocks, maxPossibleBlocks,
                  smCount, blocksPerSmEstimate);
      }
    }
  }
#endif

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
    int metaGlobalScratchBytes = 0;
    int metaGlobalScratchAlignment = 128;
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
        } else if (key == "globalScratchBytes") {
          parseIntValue(value, metaGlobalScratchBytes);
        } else if (key == "globalScratchAlignment") {
          parseIntValue(value, metaGlobalScratchAlignment);
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
    out.globalScratchBytes = metaGlobalScratchBytes;
    out.globalScratchAlignment = metaGlobalScratchAlignment;
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
        metaOut << "globalScratchBytes=" << dumpBinary.globalScratchBytes << "\n";
        metaOut << "globalScratchAlignment=" << dumpBinary.globalScratchAlignment << "\n";
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
  unsigned int requestedSharedMem =
      binary.sharedMemBytes > 0 ? static_cast<unsigned int>(binary.sharedMemBytes) : 0u;

  if (binary.target == TritonGpuTarget::NVIDIA) {
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

  if (binary.target == TritonGpuTarget::NVIDIA && irModule.useCooperativeLaunch) {
    const unsigned int launchBlockX = static_cast<unsigned int>(std::max(1, binary.numWarps) * 32);
    const unsigned int launchBlockY = std::max(1u, irModule.blockY);
    const unsigned int launchBlockZ = std::max(1u, irModule.blockZ);
    unsigned long long requiredBlocks = static_cast<unsigned long long>(std::max(1u, irModule.gridX)) *
                                        static_cast<unsigned long long>(std::max(1u, irModule.gridY)) *
                                        static_cast<unsigned long long>(std::max(1u, irModule.gridZ));
    if (irModule.requiredGrid > 0) {
      requiredBlocks = std::max(requiredBlocks,
                                static_cast<unsigned long long>(std::max(1, irModule.requiredGrid)));
    }

    bool coopSupported = false;
    long long maxCoopBlocks = 0;
    int blocksPerSm = 0;
    int smCount = 0;
    const bool capacityKnown = queryCudaCooperativeLaunchCapacity(
        result.kernelFunction,
        launchBlockX, launchBlockY, launchBlockZ,
        requestedSharedMem,
        &coopSupported, &maxCoopBlocks, &blocksPerSm, &smCount);

    if (!capacityKnown) {
      sd_printf("TritonGraphBackend: cooperative launch capacity check unavailable for [%d-%d]; "
                "continuing with runtime launch validation\n",
                startSlot, endSlot);
    } else if (!coopSupported) {
      sd_printf("TritonGraphBackend: cooperative launch required for [%d-%d], "
                "but current CUDA device does not support cooperative launch\n",
                startSlot, endSlot);
      TritonTargetDispatch::unloadModule(result.gpuModule);
      result.gpuModule = nullptr;
      result.kernelFunction = nullptr;
      delete[] static_cast<char*>(binary.data);
      cleanupModule();
      return result;
    } else if (maxCoopBlocks <= 0 ||
               requiredBlocks > static_cast<unsigned long long>(maxCoopBlocks)) {
      sd_printf("TritonGraphBackend: cooperative launch capacity exceeded for [%d-%d] "
                "(requiredBlocks=%llu, maxBlocks=%lld, smCount=%d, blocksPerSm=%d, "
                "grid=%ux%ux%u, block=%ux%ux%u, sharedMem=%u). "
                "Rejecting this fused range so adaptive splitting can retry.\n",
                startSlot, endSlot,
                static_cast<unsigned long long>(requiredBlocks), maxCoopBlocks,
                smCount, blocksPerSm,
                irModule.gridX, irModule.gridY, irModule.gridZ,
                launchBlockX, launchBlockY, launchBlockZ,
                requestedSharedMem);
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
  // Triton 3.6.0's AllocateWarpGroups pass may change the warp count during compilation.
  // blockX MUST match the actual compiled warp count, not the pre-compilation IR builder value.
  result.blockX = binary.numWarps * 32;
  result.blockY = irModule.blockY;
  result.blockZ = irModule.blockZ;
  result.sharedMemBytes = binary.sharedMemBytes;
  result.globalScratchBytes = binary.globalScratchBytes > 0
      ? static_cast<unsigned int>(binary.globalScratchBytes) : 0u;
  result.globalScratchAlignment = binary.globalScratchAlignment > 0
      ? static_cast<unsigned int>(binary.globalScratchAlignment) : 128u;
  result.numWarps = binary.numWarps;
  result.argSlotMapping = irModule.args;
  result.useCooperativeLaunch = irModule.useCooperativeLaunch;
  result.useDynamicGrid = irModule.useDynamicGrid;
  result.useIndirectArgs = irModule.useIndirectArgs;
  result.useMultiPhaseLaunch = irModule.useMultiPhaseLaunch;
  result.launchPhases = irModule.launchPhases;

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
