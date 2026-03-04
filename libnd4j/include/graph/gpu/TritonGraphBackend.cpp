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
#include <execution/LaunchContext.h>
#include <helpers/logger.h>
#include <system/Environment.h>
#include <system/common.h>

#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

// MLIR core for ModuleOp and MLIRContext used in compileToGpuBinary cleanup
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

// Disk cache for compiled PTX
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
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

    auto& cacheEnv = Environment::getInstance();
  bool cacheCompileAll = cacheEnv.tritonCompileAll();
  size_t cacheExcludeHash = std::hash<std::string>()(cacheEnv.tritonExcludeOps());
  SegmentCacheKey key{seg.startSlot, seg.endSlot, shapeKey, compileDevice, cacheCompileAll, cacheExcludeHash};

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
    // FUSED_ATTENTION: standalone unless single query tile
    if (section.type == KernelSectionType::FUSED_ATTENTION) {
      int batchHeads = std::max(1, section.batchSize) * std::max(1, section.numHeads);
      bool singleQueryTile = section.gridRequirement <= batchHeads;
      return !singleQueryTile;
    }

    // Non-elementwise/identity sections must NOT merge with adjacent elementwise
    // sections — mixing section types in a single kernel can produce incorrect IR.
    // Each non-default section type compiles as its own standalone kernel.
    if (section.type != KernelSectionType::ELEMENTWISE &&
        section.type != KernelSectionType::IDENTITY) {
      return true;
    }

    return false;
  };

  // Determine which sections are compiled as Triton kernels vs native fallback.
  // Default mode: only ELEMENTWISE/IDENTITY compiled, everything else uses cuBLAS/native.
  // When tritonCompileAll=true: compile ALL section types EXCEPT those containing
  // ops in the exclusion list (ND4J_TRITON_EXCLUDE_OPS). This allows fine-grained
  // control, e.g. keeping matmul on cuBLAS while compiling reductions/norms via Triton.
  auto& compileEnv = Environment::getInstance();
  bool compileAll = compileEnv.tritonCompileAll();

  // Parse tritonIncludeTypes into a set of allowed section types for compileAll mode
  std::unordered_set<KernelSectionType> includedTypes;
  {
    std::string includeStr = compileEnv.tritonIncludeTypes();
    if (!includeStr.empty()) {
      std::istringstream iss(includeStr);
      std::string token;
      while (std::getline(iss, token, ',')) {
        // Trim whitespace
        size_t start = token.find_first_not_of(" \t");
        size_t end = token.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        token = token.substr(start, end - start + 1);
        // Map type name to enum
        if (token == "CONST_GEN" || token == "CONSTANT_GENERATION")
          includedTypes.insert(KernelSectionType::CONSTANT_GENERATION);
        else if (token == "SHAPE_MANIP" || token == "SHAPE_MANIPULATION")
          includedTypes.insert(KernelSectionType::SHAPE_MANIPULATION);
        else if (token == "GATHER")
          includedTypes.insert(KernelSectionType::GATHER);
        else if (token == "GATHER_ND")
          includedTypes.insert(KernelSectionType::GATHER_ND);
        else if (token == "CONCAT")
          includedTypes.insert(KernelSectionType::CONCAT);
        else if (token == "SPLIT")
          includedTypes.insert(KernelSectionType::SPLIT);
        else if (token == "SPLIT_V")
          includedTypes.insert(KernelSectionType::SPLIT_V);
        else if (token == "STACK")
          includedTypes.insert(KernelSectionType::STACK);
        else if (token == "REDUCTION")
          includedTypes.insert(KernelSectionType::REDUCTION);
        else if (token == "NORMALIZATION")
          includedTypes.insert(KernelSectionType::NORMALIZATION);
        else if (token == "ATTENTION" || token == "FUSED_ATTENTION")
          includedTypes.insert(KernelSectionType::FUSED_ATTENTION);
        else if (token == "MATMUL")
          includedTypes.insert(KernelSectionType::MATMUL);
        else if (token == "TILE")
          includedTypes.insert(KernelSectionType::TILE);
        else if (token == "STRIDED_SLICE")
          includedTypes.insert(KernelSectionType::STRIDED_SLICE);
        else if (token == "SCATTER_ND")
          includedTypes.insert(KernelSectionType::SCATTER_ND);
        else if (token == "SCATTER_ND_UPDATE")
          includedTypes.insert(KernelSectionType::SCATTER_ND_UPDATE);
        else if (token == "CONVOLUTION")
          includedTypes.insert(KernelSectionType::CONVOLUTION);
        else
          sd_printf("TritonGraphBackend: unknown include type '%s'\n", token.c_str());
      }
      if (!includedTypes.empty()) {
        sd_printf("TritonGraphBackend: compileAll with include types filter (%d types)\n",
                  static_cast<int>(includedTypes.size()));
      }
    }
  }

  auto isFallbackSection = [&](const KernelSection& section) -> bool {
    // SHAPE_MANIPULATION: always fallback to native. Native permute/reshape
    // create zero-cost views (no data copy needed). Triton compilation would
    // do actual data copies with no speedup for seq=1 decode. Additionally,
    // multi-op standalone SHAPE_MANIP sections have cross-block data races
    // (a later op's reordered reads may reference positions written by a
    // different block in an earlier op, with no global barrier).
    if (section.type == KernelSectionType::SHAPE_MANIPULATION) {
      return true;
    }

    if (!compileAll) {
      // Default: only ELEMENTWISE and IDENTITY are compiled
      return section.type != KernelSectionType::ELEMENTWISE &&
             section.type != KernelSectionType::IDENTITY;
    }

    // compileAll mode with include types filter: only compile ELEMENTWISE, IDENTITY,
    // and explicitly listed types
    if (!includedTypes.empty()) {
      if (section.type != KernelSectionType::ELEMENTWISE &&
          section.type != KernelSectionType::IDENTITY &&
          includedTypes.find(section.type) == includedTypes.end()) {
        return true;  // Type not in whitelist → fallback
      }
    }

    // Check op-level exclusion list
    for (int si = section.startSlot; si <= section.endSlot; si++) {
      if (si >= 0 && si < totalSlots && !slots[si].opName.empty()) {
        if (compileEnv.isTritonExcludedOp(slots[si].opName)) {
          return true;  // This section contains an excluded op → fallback
        }
      }
    }
    return false;
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
    // Non-elementwise sections are skipped —
    // they become gaps filled by fallbackRangeExecutor_ (cuBLAS/native).
    if (isFallbackSection(sections[secIdx])) {
      sd_printf("TritonGraphBackend: section %d [%d-%d] type=%d excluded (cuBLAS fallback)\n",
                secIdx, sections[secIdx].startSlot, sections[secIdx].endSlot,
                static_cast<int>(sections[secIdx].type));
      secIdx++;
      continue;
    }

    if (isStandaloneSection(sections[secIdx])) {
      sd_printf("TritonGraphBackend: section %d [%d-%d] type=%d STANDALONE (Triton compile)\n",
                secIdx, sections[secIdx].startSlot, sections[secIdx].endSlot,
                static_cast<int>(sections[secIdx].type));
      pendingRanges.push_back(makeRange(secIdx, secIdx));
      secIdx++;
      continue;
    }

    // Log each compiled section
    sd_printf("TritonGraphBackend: section %d [%d-%d] type=%d COMPILED (Triton)\n",
              secIdx, sections[secIdx].startSlot, sections[secIdx].endSlot,
              static_cast<int>(sections[secIdx].type));

    // Merge consecutive element-wise-compatible sections into one compile range.
    int runStart = secIdx;
    int runEnd = secIdx;
    while (runEnd + 1 < static_cast<int>(sections.size()) &&
           !isStandaloneSection(sections[runEnd + 1]) &&
           !isFallbackSection(sections[runEnd + 1])) {
      runEnd++;
      sd_printf("TritonGraphBackend: section %d [%d-%d] type=%d MERGED into range starting at section %d\n",
                runEnd, sections[runEnd].startSlot, sections[runEnd].endSlot,
                static_cast<int>(sections[runEnd].type), runStart);
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
  const int envOpsCap = std::max(0, env.tritonMaxSubsegmentOps());
  // When the environment cap is 0 (disabled), use MAX_COMPILABLE_OPS as default.
  // Huge IR modules (>512 ops) cause LLVM register pressure explosions (441K virtual
  // regs for 3840 ops) leading to SIGABRT. Set ND4J_TRITON_MAX_SUBSEGMENT_OPS=0
  // to truly disable (not recommended).
  const int maxOpsCap = (envOpsCap > 0) ? envOpsCap : MAX_COMPILABLE_OPS;
  const int maxSectionsCap = std::max(0, env.tritonMaxSubsegmentSections());
  const int maxParallelCompiles = std::max(1, getMaxParallelCompilations());
  sd_printf("TritonGraphBackend: adaptive section packing for [%d-%d] "
            "(initialRanges=%d, opsCap=%d(env=%d), sectionsCap=%d, compileThreads=%d)\n",
            seg.startSlot, seg.endSlot,
            static_cast<int>(pendingRanges.size()),
            maxOpsCap, envOpsCap, maxSectionsCap, maxParallelCompiles);
  if (maxSectionsCap <= 0) {
    sd_printf("TritonGraphBackend: section cap disabled for [%d-%d] (runtime control); "
              "set tritonMaxSubsegmentSections>0 to force additional splitting\n",
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

    // Pre-allocate Triton 3.6 global scratch buffer (per-program scratch memory).
    // Must be done OUTSIDE capture — raw cudaMallocAsync/cudaFreeAsync during capture
    // creates MemAlloc/MemFree graph nodes with addresses that become stale on replay,
    // causing SIGSEGV on cudaGraphLaunch.
    if (kernel.globalScratchBytes > 0) {
      unsigned int numPrograms = std::max(1u, kernel.gridX) *
                                  std::max(1u, kernel.gridY) *
                                  std::max(1u, kernel.gridZ);
      size_t totalScratchBytes = static_cast<size_t>(kernel.globalScratchBytes) * numPrograms;
      if (kernel.cachedGlobalScratchDevice == nullptr ||
          kernel.cachedGlobalScratchBytes < totalScratchBytes ||
          kernel.cachedGlobalScratchDeviceId != compileDevice) {
        if (kernel.cachedGlobalScratchDevice != nullptr) {
          auto freeErr = freeDeviceBufferAsync(kernel.cachedGlobalScratchDevice, preallocStream);
          if (freeErr != cudaSuccess) {
            sd_printf("TritonGraphBackend: failed freeing stale global scratch for sub-kernel [%d-%d]: %s\n",
                      kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(freeErr));
            cudaGetLastError();
            cleanupCompiledWorkspace();
            return false;
          }
          kernel.cachedGlobalScratchDevice = nullptr;
          kernel.cachedGlobalScratchBytes = 0;
          kernel.cachedGlobalScratchDeviceId = -1;
        }
        auto allocErr = allocateDeviceBufferAsync(&kernel.cachedGlobalScratchDevice,
                                                   totalScratchBytes, preallocStream);
        if (allocErr != cudaSuccess) {
          sd_printf("TritonGraphBackend: failed pre-allocating global scratch (%zu bytes) for sub-kernel [%d-%d]: %s\n",
                    totalScratchBytes, kernel.startSlot_, kernel.endSlot_, cudaGetErrorString(allocErr));
          cudaGetLastError();
          cleanupCompiledWorkspace();
          return false;
        }
        kernel.cachedGlobalScratchBytes = totalScratchBytes;
        kernel.cachedGlobalScratchDeviceId = compileDevice;
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

  // Pre-allocate output arrays for slots that don't have arrays yet.
  // In compileAll mode, non-elementwise sections (reductions, gathers, etc.) are
  // compiled into the Triton kernel instead of running via fallback. Their output
  // arrays must exist before the kernel launches since args are resolved by pointer.
  for (auto& argMapping : compiled.argSlotMapping) {
    if (!argMapping.isOutput) continue;
    if (argMapping.slotIndex < 0 || argMapping.slotIndex >= totalOutputSlots) continue;
    if (outputSlots[argMapping.slotIndex] != nullptr) continue;
    // Need to allocate — use shape/dtype from the arg mapping
    if (argMapping.shape.empty()) {
      sd_printf("TritonGraphBackend::executeSingleKernel: cannot pre-allocate slot %d — no shape info "
                "(sub-segment [%d-%d])\n",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_);
      return Status::KERNEL_FAILURE;
    }
    std::vector<LongType> shapeVec(argMapping.shape.begin(), argMapping.shape.end());
    auto* newArr = new NDArray('c', shapeVec, argMapping.dtype, LaunchContext::defaultContext());
    outputSlots[argMapping.slotIndex] = newArr;
  }

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
    // Validate DataBuffer before accessing specialBuffer() — Java close() may have
    // deleted the NDArray or its DataBuffer, leaving outputSlots_ with a dangling pointer.
    // Empty arrays (isEmpty=true, length=0) legitimately have no DataBuffer — they
    // represent optional/unused inputs (e.g., attention mask placeholders). Handle them
    // with a dummy pointer below (same as the zero-length specialBuffer() path).
    auto* db = arr->dataBuffer();
    if ((db == nullptr || !db->isValid()) && !arr->isEmpty() && arr->lengthOf() > 0) {
      sd_printf("TritonGraphBackend::executeSingleKernel: INVALID DataBuffer for arg slot %d "
                "(sub-segment [%d-%d], isOutput=%d, arr=%p, db=%p, dbValid=%d, "
                "rank=%d, length=%lld, dtype=%d, isEmpty=%d)\n",
                argMapping.slotIndex, compiled.startSlot_, compiled.endSlot_,
                argMapping.isOutput ? 1 : 0, (void*)arr, (void*)db,
                db ? (db->isValid() ? 1 : 0) : -1,
                arr->rankOf(), (long long)arr->lengthOf(),
                static_cast<int>(arr->dataType()), arr->isEmpty() ? 1 : 0);
      // Log which slots in this sub-kernel consume this external input
      if (argMapping.slotIndex < 0) {
        for (int si = compiled.startSlot_; si <= compiled.endSlot_; si++) {
          for (int inp = 0; inp < slots[si].numInputs; inp++) {
            if (slots[si].inputSourceIndices[inp] == argMapping.slotIndex) {
              sd_printf("  -> consumed by slot %d op='%s' (input #%d)\n",
                        si, slots[si].opName.c_str(), inp);
            }
          }
        }
      }
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

#ifdef SD_CUDA
  // During CUDA graph capture, log diagnostic info about buffer pointers.
  if (streamIsCapturing) {
    // Clear any sticky error before pointer validation — stale errors can
    // cause cudaPointerGetAttributes to return misleading results.
    cudaGetLastError();
    for (int i = 0; i < static_cast<int>(bufferPtrs.size()); i++) {
      cudaPointerAttributes attrs;
      memset(&attrs, 0, sizeof(attrs));
      auto queryErr = cudaPointerGetAttributes(&attrs, bufferPtrs[i]);
      if (queryErr != cudaSuccess) {
        cudaGetLastError();  // Clear the error
        auto& argMapping = compiled.argSlotMapping[i];
        // Resolve array for extra diagnostics
        NDArray* diagArr = nullptr;
        if (argMapping.slotIndex < 0) {
          int extIdx = -(argMapping.slotIndex + 1);
          if (extIdx < numExternalInputs) diagArr = externalInputs[extIdx];
        } else if (argMapping.slotIndex < totalOutputSlots) {
          diagArr = outputSlots[argMapping.slotIndex];
        }
        sd_printf("TritonGraphBackend: WARNING ptr %p for arg %d (slot=%d, isOutput=%d) "
                  "in [%d-%d]: cudaPointerGetAttributes failed: %s "
                  "(arr=%p, primary=%p, special=%p, len=%lld, dtype=%d, deviceId=%d)\n",
                  bufferPtrs[i], i, argMapping.slotIndex, argMapping.isOutput ? 1 : 0,
                  compiled.startSlot_, compiled.endSlot_, cudaGetErrorString(queryErr),
                  (void*)diagArr,
                  diagArr ? diagArr->buffer() : nullptr,
                  diagArr ? diagArr->specialBuffer() : nullptr,
                  diagArr ? (long long)diagArr->lengthOf() : -1,
                  diagArr ? static_cast<int>(diagArr->dataType()) : -1,
                  diagArr ? diagArr->dataBuffer()->deviceId() : -1);
      } else if (attrs.type == cudaMemoryTypeUnregistered) {
        auto& argMapping = compiled.argSlotMapping[i];
        NDArray* diagArr = nullptr;
        if (argMapping.slotIndex < 0) {
          int extIdx = -(argMapping.slotIndex + 1);
          if (extIdx < numExternalInputs) diagArr = externalInputs[extIdx];
        } else if (argMapping.slotIndex < totalOutputSlots) {
          diagArr = outputSlots[argMapping.slotIndex];
        }
        sd_printf("TritonGraphBackend: WARNING UNREGISTERED ptr %p for arg %d (slot=%d, isOutput=%d) "
                  "in [%d-%d] (arr=%p, primary=%p, special=%p, len=%lld, dtype=%d, deviceId=%d)\n",
                  bufferPtrs[i], i, argMapping.slotIndex, argMapping.isOutput ? 1 : 0,
                  compiled.startSlot_, compiled.endSlot_,
                  (void*)diagArr,
                  diagArr ? diagArr->buffer() : nullptr,
                  diagArr ? diagArr->specialBuffer() : nullptr,
                  diagArr ? (long long)diagArr->lengthOf() : -1,
                  diagArr ? static_cast<int>(diagArr->dataType()) : -1,
                  diagArr ? diagArr->dataBuffer()->deviceId() : -1);
      }
    }
  }
#endif

  // Compute n_elements from the LARGEST output to ensure all elements are computed.
  // When a fused kernel covers multiple independent chains with different output sizes
  // (e.g., main hidden state [1,960] + RoPE frequencies [1,480]), using the first output's
  // count could under-launch threads, leaving larger outputs partially computed.
  // Per-output masks in the IR handle the reverse case (preventing overflow on smaller outputs).
  LongType nElements = 0;
  for (auto& argMapping : compiled.argSlotMapping) {
    if (argMapping.isOutput) {
      int slotIdx = argMapping.slotIndex;
      if (slotIdx >= 0 && slotIdx < totalOutputSlots && outputSlots[slotIdx]) {
        LongType len = outputSlots[slotIdx]->lengthOf();
        if (len > nElements) nElements = len;
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
      if (streamIsCapturing) {
        // During CUDA graph capture, raw cudaMallocAsync/cudaFreeAsync create MemAlloc/MemFree
        // graph nodes with addresses that become stale on replay → SIGSEGV on cudaGraphLaunch.
        // Global scratch must be pre-allocated before capture (in compileSegment prealloc loop).
        sd_printf("TritonGraphBackend::executeSingleKernel: global scratch needs realloc during capture "
                  "[%d-%d] (deviceChanged=%d, ptr=%p, cached=%zu, needed=%zu) — falling back\n",
                  compiled.startSlot_, compiled.endSlot_,
                  deviceChanged ? 1 : 0, compiled.cachedGlobalScratchDevice,
                  compiled.cachedGlobalScratchBytes, totalScratchBytes);
        return Status::KERNEL_FAILURE;
      }
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

// ─── Arg table refresh for CUDA graph replay ───────────────────────────────

Status TritonGraphBackend::refreshArgTablesForReplay(
    GraphSegment& seg,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {
#ifdef SD_CUDA
  int currentDevice = -1;
  cudaGetDevice(&currentDevice);

  auto& refreshEnv = Environment::getInstance();
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, currentDevice,
                      refreshEnv.tritonCompileAll(),
                      std::hash<std::string>()(refreshEnv.tritonExcludeOps())};

  CompiledSegment* compiledSeg = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      sd_printf("TritonGraphBackend::refreshArgTablesForReplay: no compiled segment for [%d-%d] "
                "(shapeKey=%lld, device=%d)\n",
                seg.startSlot, seg.endSlot, seg.shapeKey, currentDevice);
      return Status::KERNEL_FAILURE;
    }
    compiledSeg = &it->second;
  }

  int refreshedCount = 0;
  int skippedCount = 0;
  for (auto& subKernel : compiledSeg->subKernels) {
    if (!subKernel.useIndirectArgs || subKernel.cachedArgTableHostPinned == nullptr) {
      skippedCount++;
      continue;
    }

    auto* argTableHostPinned = static_cast<int64_t*>(subKernel.cachedArgTableHostPinned);
    int numBufferArgs = static_cast<int>(subKernel.argSlotMapping.size());

    for (int i = 0; i < numBufferArgs; i++) {
      auto& argMapping = subKernel.argSlotMapping[i];
      NDArray* arr = nullptr;
      if (argMapping.slotIndex < 0) {
        int extIdx = -(argMapping.slotIndex + 1);
        if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
      } else {
        if (argMapping.slotIndex < totalOutputSlots) arr = outputSlots[argMapping.slotIndex];
      }

      if (arr != nullptr) {
        void* sbuf = arr->specialBuffer();
        if (sbuf != nullptr) {
          argTableHostPinned[i] = reinterpret_cast<int64_t>(sbuf);
        }
      }
    }
    refreshedCount++;
  }

  if (refreshedCount > 0) {
    sd_printf("TritonGraphBackend::refreshArgTablesForReplay: refreshed %d sub-kernels "
              "(skipped %d non-indirect) for seg[%d-%d]\n",
              refreshedCount, skippedCount, seg.startSlot, seg.endSlot);
  }
  return Status::OK;
#else
  return Status::OK;
#endif
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

  auto& execEnv = Environment::getInstance();
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, execDevice,
                      execEnv.tritonCompileAll(),
                      std::hash<std::string>()(execEnv.tritonExcludeOps())};

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
    if (!Environment::getInstance().tritonAllowFallbackCapture()) {
      sd_printf("TritonGraphBackend::executeSegment: refusing slot fallback during CUDA graph capture for [%d-%d] (%d fallback ranges)\n",
                seg.startSlot, seg.endSlot, static_cast<int>(compiledSeg->fallbackRanges.size()));
      return Status::KERNEL_FAILURE;
    }
    sd_printf("TritonGraphBackend::executeSegment: allowing fallback during CUDA graph capture for [%d-%d] (%d fallback ranges) — cuBLAS/native ops will be recorded into the graph\n",
              seg.startSlot, seg.endSlot, static_cast<int>(compiledSeg->fallbackRanges.size()));
  }
#endif

  // Check if Triton kernel execution should be replaced by native fallback for debugging.
  // Environment::tritonSkipKernels(): run all sub-kernel ranges via native slot-by-slot instead of Triton
  // Environment::tritonVerifyKernels(): run Triton, then re-run native and compare outputs
  bool tritonSkipKernels = Environment::getInstance().tritonSkipKernels();
  bool tritonVerifyKernels = Environment::getInstance().tritonVerifyKernels();

  // Execute sub-kernels in-order and run uncovered slot gaps via callback.
  sd_printf("TritonGraphBackend::executeSegment: segment [%d-%d] launching %d sub-kernels "
            "(fallbackRanges=%d, targetDeviceId=%d, activeDevice=%d, skipKernels=%d, verifyKernels=%d)\n",
            seg.startSlot, seg.endSlot,
            static_cast<int>(compiledSeg->subKernels.size()),
            static_cast<int>(compiledSeg->fallbackRanges.size()),
            targetDevice, execDevice,
            tritonSkipKernels ? 1 : 0, tritonVerifyKernels ? 1 : 0);

  int nextSlotToRun = seg.startSlot;
  for (int i = 0; i < (int)compiledSeg->subKernels.size(); i++) {
    auto& subKernel = compiledSeg->subKernels[i];

    if (nextSlotToRun < subKernel.startSlot_) {
#ifdef SD_CUDA
      if (streamCaptureActive && !Environment::getInstance().tritonAllowFallbackCapture()) {
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

    if (tritonSkipKernels) {
      // Skip Triton kernel — run native slot-by-slot instead
      if (fallbackRangeExecutor_) {
        auto skipStatus = fallbackRangeExecutor_(subKernel.startSlot_, subKernel.endSlot_);
        if (skipStatus != Status::OK) {
          sd_printf("TritonGraphBackend::executeSegment: native fallback for skipped kernel [%d-%d] failed with status=%d\n",
                    subKernel.startSlot_, subKernel.endSlot_, static_cast<int>(skipStatus));
          return skipStatus;
        }
      }
    } else {
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

      // Verify mode: save pre-Triton outputs, run Triton, then compare against native
      std::unordered_map<int, NDArray*> savedOutputs;
      if (tritonVerifyKernels && !streamCaptureActive) {
        // Save copies of output arrays that this sub-kernel will write
        for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
          for (int o = 0; o < slots[si].numOutputs; o++) {
            int outIdx = slots[si].outputSlotIndices[o];
            if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
              savedOutputs[outIdx] = new NDArray(outputSlots[outIdx]->dup());
            }
          }
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
#ifdef SD_CUDA
        for (auto& kv : savedOutputs) delete kv.second;
#endif
        return status;
      }

#ifdef SD_CUDA
      // Verify mode: run native slot-by-slot and compare
      if (tritonVerifyKernels && !streamCaptureActive && fallbackRangeExecutor_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));

        // Save Triton outputs
        std::unordered_map<int, NDArray*> tritonOutputs;
        for (int si = subKernel.startSlot_; si <= subKernel.endSlot_; si++) {
          for (int o = 0; o < slots[si].numOutputs; o++) {
            int outIdx = slots[si].outputSlotIndices[o];
            if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
              tritonOutputs[outIdx] = new NDArray(outputSlots[outIdx]->dup());
            }
          }
        }

        // Restore pre-Triton outputs before native execution
        for (auto& kv : savedOutputs) {
          if (kv.first < totalOutputSlots && outputSlots[kv.first]) {
            outputSlots[kv.first]->assign(kv.second);
          }
        }
        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));

        // Run native slot-by-slot
        auto nativeStatus = fallbackRangeExecutor_(subKernel.startSlot_, subKernel.endSlot_);
        cudaStreamSynchronize(static_cast<cudaStream_t>(actualStream));

        if (nativeStatus == Status::OK) {
          // Compare native outputs against Triton outputs
          int mismatches = 0;
          for (auto& kv : tritonOutputs) {
            int outIdx = kv.first;
            NDArray* tritonArr = kv.second;
            NDArray* nativeArr = (outIdx < totalOutputSlots) ? outputSlots[outIdx] : nullptr;
            if (!nativeArr) continue;

            // Compare first few elements
            auto tritonHost = tritonArr->dup('c');
            auto nativeHost = nativeArr->dup('c');
            LongType len = std::min(tritonHost->lengthOf(), nativeHost->lengthOf());
            LongType checkLen = std::min(len, (LongType)16);

            double maxAbsDiff = 0;
            int maxDiffIdx = -1;
            for (int e = 0; e < checkLen; e++) {
              double tVal = tritonHost->e<double>(e);
              double nVal = nativeHost->e<double>(e);
              double diff = std::abs(tVal - nVal);
              if (diff > maxAbsDiff) {
                maxAbsDiff = diff;
                maxDiffIdx = e;
              }
            }

            if (maxAbsDiff > 1e-3) {
              mismatches++;
              double tVal = tritonHost->e<double>(maxDiffIdx);
              double nVal = nativeHost->e<double>(maxDiffIdx);
              sd_printf("TRITON VERIFY MISMATCH: sub-kernel [%d-%d] slot %d: "
                        "maxDiff=%.6f at idx %d (triton=%.6f, native=%.6f, len=%lld, dtype=%d)\n",
                        subKernel.startSlot_, subKernel.endSlot_,
                        outIdx, maxAbsDiff, maxDiffIdx, tVal, nVal,
                        (long long)len, static_cast<int>(nativeArr->dataType()));

              // Print first 8 values from both
              std::string tritonVals = "  triton first8: [";
              std::string nativeVals = "  native first8: [";
              for (int e = 0; e < std::min(checkLen, (LongType)8); e++) {
                if (e > 0) { tritonVals += ", "; nativeVals += ", "; }
                char buf[64];
                snprintf(buf, sizeof(buf), "%.6f", tritonHost->e<double>(e));
                tritonVals += buf;
                snprintf(buf, sizeof(buf), "%.6f", nativeHost->e<double>(e));
                nativeVals += buf;
              }
              tritonVals += "]";
              nativeVals += "]";
              sd_printf("%s\n%s\n", tritonVals.c_str(), nativeVals.c_str());
            }
            delete tritonHost;
            delete nativeHost;
          }
          if (mismatches == 0) {
            sd_printf("TRITON VERIFY OK: sub-kernel [%d-%d] all %d outputs match (tolerance=1e-3)\n",
                      subKernel.startSlot_, subKernel.endSlot_,
                      static_cast<int>(tritonOutputs.size()));
          } else {
            sd_printf("TRITON VERIFY: sub-kernel [%d-%d] %d/%d outputs MISMATCHED\n",
                      subKernel.startSlot_, subKernel.endSlot_,
                      mismatches, static_cast<int>(tritonOutputs.size()));
          }

          // Restore Triton outputs (use Triton results for continued execution)
          for (auto& kv : tritonOutputs) {
            if (kv.first < totalOutputSlots && outputSlots[kv.first]) {
              outputSlots[kv.first]->assign(kv.second);
            }
          }
        } else {
          sd_printf("TRITON VERIFY: native fallback for [%d-%d] failed, skipping comparison\n",
                    subKernel.startSlot_, subKernel.endSlot_);
        }

        for (auto& kv : tritonOutputs) delete kv.second;
        for (auto& kv : savedOutputs) delete kv.second;
      } else {
        for (auto& kv : savedOutputs) delete kv.second;
      }
#endif
    }
    totalKernelLaunches_++;


    if (subKernel.endSlot_ + 1 > nextSlotToRun) {
      nextSlotToRun = subKernel.endSlot_ + 1;
    }
  }

  if (nextSlotToRun <= seg.endSlot) {
#ifdef SD_CUDA
    if (streamCaptureActive && !Environment::getInstance().tritonAllowFallbackCapture()) {
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
  // ── Compose attention present_key / present_value outputs ──
  // The Triton attention kernel only writes output[0] (attention result).
  // output[1] (present_key) = concat(past_key, current_key) along seq dim
  // output[2] (present_value) = concat(past_value, current_value) along seq dim
  // With static KV cache, present output has SAME shape as past input [B,H,maxKvLen,D].
  // We write ONLY the current token's K/V to the LAST position of the present buffer.
  // Then kvScatter reads present[lastPos] → static_buffer[cachePos].
  //
  // NOTE: composePresentKv post-processing runs for ALL attention slots regardless of
  // whether they ran via Triton or via cuBLAS/native fallback. Neither the Triton attention
  // kernel nor the native multi_head_attention op writes present_key/present_value —
  // this scatter of current K/V projections into the present buffer is always needed.

  // File-based diagnostic logging (sd_printf goes to stderr which surefire doesn't capture)
  static int composePresentKvCallCount = 0;
  FILE* kvLog = nullptr;
  if (composePresentKvCallCount < 5) {
    kvLog = fopen("/tmp/triton_compose_kv.log", "a");
  }

  int attnCount = 0;
  for (int si = seg.startSlot; si <= seg.endSlot; si++) {
    if (slots[si].opName.empty()) continue;
    bool isAttn = (slots[si].opName == "onnx_multi_head_attention" ||
                   slots[si].opName == "multi_head_attention");
    if (!isAttn) continue;
    if (slots[si].numInputs <= 4 || slots[si].numOutputs < 2) continue;

    // Inputs: [0]=Q, [1]=K(current), [2]=V(current), [3]=bias, [4]=past_key, [5]=past_value
    int currentKeySrc = slots[si].inputSourceIndices[1];
    int currentValueSrc = (slots[si].numInputs > 2) ? slots[si].inputSourceIndices[2] : -1;

    int presentKeyOut = slots[si].outputSlotIndices[1];
    int presentValueOut = (slots[si].numOutputs >= 3) ? slots[si].outputSlotIndices[2] : -1;

    if (kvLog && attnCount == 0) {
      fprintf(kvLog, "=== composePresentKv call #%d (seg[%d-%d], capture=%d) ===\n",
              composePresentKvCallCount, seg.startSlot, seg.endSlot, streamCaptureActive ? 1 : 0);
      fprintf(kvLog, "  attn slot=%d opName='%s' numInputs=%d numOutputs=%d\n",
              si, slots[si].opName.c_str(), slots[si].numInputs, slots[si].numOutputs);
      fprintf(kvLog, "  currentKeySrc=%d currentValueSrc=%d presentKeyOut=%d presentValueOut=%d\n",
              currentKeySrc, currentValueSrc, presentKeyOut, presentValueOut);
    }

    // Lambda: scatter current K/V into present output at the LAST seq position.
    // kvScatter then reads present[lastPos] → static_buffer[cachePos].
    auto scatterCurrentToPresent = [&](int currentSlot, int presentSlot, const char* label) {
      // Resolve current K/V array (positive = output slot, negative = external input)
      NDArray* currentArr = nullptr;
      if (currentSlot < 0) {
        int extIdx = -(currentSlot + 1);
        if (extIdx >= 0 && extIdx < numExternalInputs && externalInputs[extIdx])
          currentArr = externalInputs[extIdx];
        if (kvLog) fprintf(kvLog, "  %s: currentSlot=%d → extIdx=%d arr=%p\n", label, currentSlot, extIdx, (void*)currentArr);
      } else if (currentSlot >= 0 && currentSlot < totalOutputSlots && outputSlots[currentSlot]) {
        currentArr = outputSlots[currentSlot];
        if (kvLog) fprintf(kvLog, "  %s: currentSlot=%d → outputSlot arr=%p\n", label, currentSlot, (void*)currentArr);
      } else {
        if (kvLog) fprintf(kvLog, "  %s: currentSlot=%d INVALID (total=%d, arr=%p)\n", label, currentSlot, totalOutputSlots,
                           currentSlot >= 0 && currentSlot < totalOutputSlots ? (void*)outputSlots[currentSlot] : nullptr);
      }
      if (!currentArr) {
        if (kvLog) fprintf(kvLog, "  %s: SKIP - no current array\n", label);
        return;
      }

      if (presentSlot < 0 || presentSlot >= totalOutputSlots || !outputSlots[presentSlot]) {
        if (kvLog) fprintf(kvLog, "  %s: SKIP - presentSlot=%d invalid (total=%d, arr=%p)\n", label, presentSlot,
                           totalOutputSlots, presentSlot >= 0 && presentSlot < totalOutputSlots ? (void*)outputSlots[presentSlot] : nullptr);
        return;
      }
      auto* presentArr = outputSlots[presentSlot];

      auto currentBuf = currentArr->dataBuffer();
      auto presentBuf = presentArr->dataBuffer();
      if (!currentBuf || !presentBuf || !currentBuf->special() || !presentBuf->special()) {
        if (kvLog) fprintf(kvLog, "  %s: SKIP - null buffer (curBuf=%p curSpecial=%p presBuf=%p presSpecial=%p)\n",
                           label, (void*)currentBuf, currentBuf ? currentBuf->special() : nullptr,
                           (void*)presentBuf, presentBuf ? presentBuf->special() : nullptr);
        return;
      }

      if (kvLog && attnCount == 0) {
        fprintf(kvLog, "  %s: currentRank=%d currentShape=[", label, currentArr->rankOf());
        for (int d = 0; d < currentArr->rankOf(); d++) fprintf(kvLog, "%s%lld", d?",":"", currentArr->sizeAt(d));
        fprintf(kvLog, "] presentRank=%d presentShape=[", presentArr->rankOf());
        for (int d = 0; d < presentArr->rankOf(); d++) fprintf(kvLog, "%s%lld", d?",":"", presentArr->sizeAt(d));
        fprintf(kvLog, "]\n");
      }

      // Present is [B, H, seqLen, D] (4D BHSD)
      if (presentArr->rankOf() != 4) {
        if (kvLog) fprintf(kvLog, "  %s: SKIP - presentRank=%d != 4\n", label, presentArr->rankOf());
        return;
      }
      int numHeads = static_cast<int>(presentArr->sizeAt(1));
      int seqLen = static_cast<int>(presentArr->sizeAt(2));
      int headDim = static_cast<int>(presentArr->sizeAt(3));
      int lastPos = seqLen - 1;

      size_t elemSize = presentArr->sizeOfT();
      char* dstBase = static_cast<char*>(presentBuf->special());
      char* srcBase = static_cast<char*>(currentBuf->special());

      // Scatter per-head: src[h*headDim..] → dst[h*seqLen*headDim + lastPos*headDim..]
      for (int h = 0; h < numHeads; h++) {
        size_t dstOffset = static_cast<size_t>(h * seqLen + lastPos) * headDim * elemSize;
        size_t srcOffset = static_cast<size_t>(h) * headDim * elemSize;
        cudaMemcpyAsync(dstBase + dstOffset, srcBase + srcOffset, headDim * elemSize,
                        cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(actualStream));
      }
      if (kvLog && attnCount == 0) {
        fprintf(kvLog, "  %s: DONE scatter %d heads × %d headDim at lastPos=%d (seqLen=%d)\n",
                label, numHeads, headDim, lastPos, seqLen);
      }
    };

    scatterCurrentToPresent(currentKeySrc, presentKeyOut, "KEY");
    scatterCurrentToPresent(currentValueSrc, presentValueOut, "VAL");
    attnCount++;
  }

  if (kvLog) {
    if (attnCount > 0) {
      fprintf(kvLog, "composePresentKv: processed %d attention layers\n", attnCount);
    } else {
      // Debug: dump first few op names to see what's in the segment
      fprintf(kvLog, "composePresentKv: NO attention ops found in seg[%d-%d]\n", seg.startSlot, seg.endSlot);
      int dumped = 0;
      for (int si = seg.startSlot; si <= seg.endSlot && dumped < 20; si++) {
        if (!slots[si].opName.empty()) {
          fprintf(kvLog, "  slot[%d] opName='%s' numIn=%d numOut=%d\n",
                  si, slots[si].opName.c_str(), slots[si].numInputs, slots[si].numOutputs);
          dumped++;
        }
      }
    }
    fclose(kvLog);
    composePresentKvCallCount++;
  }

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

std::unordered_set<int> TritonGraphBackend::getGapSlots(const GraphSegment& seg, NativeSlot* slots) const {
  std::unordered_set<int> gapSlots;

  // Find the cached compiled segment for this segment's current shape key
  int activeDevice = 0;
#ifdef SD_CUDA
  cudaGetDevice(&activeDevice);
#endif
  auto& gapEnv = sd::Environment::getInstance();
  bool compileAll = gapEnv.tritonCompileAll();
  size_t excludeOpsHash = std::hash<std::string>()(gapEnv.tritonExcludeOps());
  SegmentCacheKey key{seg.startSlot, seg.endSlot, seg.shapeKey, activeDevice, compileAll, excludeOpsHash};

  std::lock_guard<std::mutex> lock(cacheMtx_);
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    // No compiled segment — all slots are gaps
    for (int s = seg.startSlot; s <= seg.endSlot; s++) {
      gapSlots.insert(s);
    }
    return gapSlots;
  }

  // Build set of slots covered by sub-kernels
  std::unordered_set<int> coveredSlots;
  for (const auto& sk : it->second.subKernels) {
    for (int s = sk.startSlot_; s <= sk.endSlot_; s++) {
      coveredSlots.insert(s);
    }
  }

  // Gap slots = all segment slots NOT covered by any sub-kernel
  for (int s = seg.startSlot; s <= seg.endSlot; s++) {
    if (coveredSlots.find(s) == coveredSlots.end()) {
      gapSlots.insert(s);
    }
  }

  sd_printf("NativeDSP: getGapSlots: seg[%d-%d] %d subKernels, %d covered, %d gap slots (of %d total)\n",
            seg.startSlot, seg.endSlot,
            static_cast<int>(it->second.subKernels.size()),
            static_cast<int>(coveredSlots.size()),
            static_cast<int>(gapSlots.size()),
            seg.endSlot - seg.startSlot + 1);

  return gapSlots;
}

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
    // Free the MLIRContext that owns all MLIR memory for this compilation.
    // Each sub-segment creates a new MLIRContext (~10-100MB for large kernels);
    // failing to free it causes unbounded memory growth during multi-sub-segment
    // compilation of VLM-scale graphs (3840 ops → many sub-segments).
    if (irModule.mlirContext) {
      delete static_cast<mlir::MLIRContext*>(irModule.mlirContext);
      irModule.mlirContext = nullptr;
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

  // Early MLIR verification to catch type mismatches before expensive compilation
  {
    auto* mod = static_cast<mlir::ModuleOp*>(irModule.mlirModule);
    if (mlir::failed(mlir::verify(*mod))) {
      std::string irDump;
      llvm::raw_string_ostream os(irDump);
      mod->print(os, mlir::OpPrintingFlags().enableDebugInfo());
      FILE* f = fopen("/tmp/triton_ir_verify_fail.mlir", "w");
      if (f) { fprintf(f, "%s", irDump.c_str()); fclose(f); }
      sd_printf("TritonGraphBackend: MLIR verification FAILED for [%d-%d]. "
                "IR dumped to /tmp/triton_ir_verify_fail.mlir (%d bytes)\n",
                startSlot, endSlot, static_cast<int>(irDump.size()));
      cleanupModule();
      return result;
    }

    // Dump MLIR IR of first few compiled kernels for debugging
    static int irDumpCount = 0;
    if (irDumpCount < 10) {
      std::string irDump;
      llvm::raw_string_ostream os(irDump);
      mod->print(os);
      char fname[256];
      snprintf(fname, sizeof(fname), "/tmp/triton_ir_dump_%03d_slots_%d_%d.mlir",
               irDumpCount, startSlot, endSlot);
      FILE* f = fopen(fname, "w");
      if (f) {
        fprintf(f, "// Kernel: %s\n", irModule.kernelName.c_str());
        fprintf(f, "// Slots: [%d-%d]\n", startSlot, endSlot);
        fprintf(f, "// Args: %d (indirect=%d)\n",
                static_cast<int>(irModule.args.size()), irModule.useIndirectArgs ? 1 : 0);
        fprintf(f, "// Grid: %ux%ux%u Block: %ux%ux%u\n",
                irModule.gridX, irModule.gridY, irModule.gridZ,
                irModule.blockX, irModule.blockY, irModule.blockZ);
        fprintf(f, "// Args detail:\n");
        for (int a = 0; a < static_cast<int>(irModule.args.size()); a++) {
          auto& arg = irModule.args[a];
          fprintf(f, "//   [%d] slot=%d output=%d dtype=%d shape=[",
                  a, arg.slotIndex, arg.isOutput ? 1 : 0, static_cast<int>(arg.dtype));
          for (size_t d = 0; d < arg.shape.size(); d++) {
            if (d > 0) fprintf(f, ",");
            fprintf(f, "%lld", (long long)arg.shape[d]);
          }
          fprintf(f, "]\n");
        }
        fprintf(f, "\n%s", irDump.c_str());
        fclose(f);
        sd_printf("TritonGraphBackend: dumped MLIR IR to %s (%d bytes)\n",
                  fname, static_cast<int>(irDump.size()));
      }
      irDumpCount++;
    }
  }

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
