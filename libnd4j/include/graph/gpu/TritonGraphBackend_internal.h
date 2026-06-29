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

//
// Internal helpers shared across TritonGraphBackend split compilation units.
// NOT a public header — only included by TritonGraphBackend_*.cpp files.
//

#ifndef LIBND4J_TRITON_GRAPH_BACKEND_INTERNAL_H
#define LIBND4J_TRITON_GRAPH_BACKEND_INTERNAL_H

#include <config.h>

#if HAVE_TRITON

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspHashUtils.h>
#include <helpers/logger.h>
#include <system/Environment.h>

#ifdef SD_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory/cuda/CudaMemoryPool.h>
#endif

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace sd {
namespace graph {
namespace triton_internal {

// ─── FNV-1a hashing for disk cache keys ─���────────────────────────────────────
// Canonical definitions in <graph/DspHashUtils.h>; aliased here for convenience.

constexpr uint64_t FNV1A64_OFFSET_BASIS = dsp::FNV1A64_OFFSET_BASIS;
constexpr uint64_t FNV1A64_PRIME        = dsp::FNV1A64_PRIME;

inline void mixFNV1a(uint64_t& hash, const void* data, size_t size) {
  dsp::fnv1aMix(hash, data, size);
}

// ─── Metadata parsing ────────────────────────────────────────────────────────

inline bool parseIntValue(const std::string& text, int& value) {
  char* endPtr = nullptr;
  long parsed = std::strtol(text.c_str(), &endPtr, 10);
  if (endPtr == text.c_str()) return false;
  value = static_cast<int>(parsed);
  return true;
}

// ─── PTX inspection ──────────────────────────────────────────────────────────

inline bool ptxUsesExternSharedMemory(const std::string& ptxText) {
  return ptxText.find(".extern .shared") != std::string::npos &&
         ptxText.find("global_smem") != std::string::npos;
}

// ─── Directory path helpers ──────────────────────────────────────────────────

inline std::string configuredOrDefaultTritonDir(const std::string& configured,
                                                const std::string& home,
                                                const char* defaultLeaf) {
  // Priority 1: explicit caller-supplied path (from ND4J_TRITON_CACHE_DIR /
  // ND4J_TRITON_DUMP_DIR env vars read by TritonConfig).
  if (!configured.empty()) {
    return configured;
  }
  // Priority 2: fall back to the cache dir from the subsystem config
  // (TritonConfig::initFromEnvironment reads ND4J_TRITON_CACHE_DIR).
  std::string cfgCacheDir = sd::Environment::getInstance().triton().cacheDir();
  if (!cfgCacheDir.empty()) {
    if (cfgCacheDir.back() != '/' && cfgCacheDir.back() != '\\') cfgCacheDir += '/';
    return cfgCacheDir + defaultLeaf;
  }
  // Priority 3: default to ~/.kompile/cache/triton/<leaf>.
  if (!home.empty()) {
    return home + "/.kompile/cache/triton/" + defaultLeaf;
  }
  return std::string(".kompile/cache/triton/") + defaultLeaf;
}

// ─── Slot resolution helpers ─────────────────────────────────────────────────

inline NDArray* resolveRangeArray(int slotIndex,
                                  NDArray** externalInputs, int numExternalInputs,
                                  NDArray** outputSlots, int totalOutputSlots) {
  if (slotIndex < 0) {
    int extIdx = -(slotIndex + 1);
    return (extIdx >= 0 && extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
  }

  return (slotIndex >= 0 && slotIndex < totalOutputSlots) ? outputSlots[slotIndex] : nullptr;
}

inline void markOrderedRangeDeviceCurrent(int startSlot, int endSlot, NativeSlot* slots,
                                          NDArray** externalInputs, int numExternalInputs,
                                          NDArray** outputSlots, int totalOutputSlots) {
  if (startSlot > endSlot) return;

  std::unordered_set<DataBuffer*> seenInputs;
  std::unordered_set<DataBuffer*> seenOutputs;

  for (int si = startSlot; si <= endSlot; si++) {
    auto& slot = slots[si];

    for (int i = 0; i < slot.wiring.numInputs; i++) {
      NDArray* arr = resolveRangeArray(slot.wiring.inputSourceIndices[i],
                                       externalInputs, numExternalInputs,
                                       outputSlots, totalOutputSlots);
      if (arr == nullptr || arr->dataBuffer() == nullptr) continue;

      auto* db = arr->dataBuffer();
      if (seenInputs.insert(db).second) {
        db->readSpecial();
      }
    }

    for (int o = 0; o < slot.wiring.numOutputs; o++) {
      int outIdx = slot.wiring.outputSlotIndices[o];
      if (outIdx < 0 || outIdx >= totalOutputSlots) continue;
      NDArray* arr = outputSlots[outIdx];
      if (arr == nullptr || arr->dataBuffer() == nullptr) continue;

      auto* db = arr->dataBuffer();
      if (seenOutputs.insert(db).second) {
        db->writeSpecial();
      }
    }
  }
}

// ─── CUDA device buffer helpers ──────────────────────────────────────────────

#ifdef SD_CUDA

inline cudaError_t allocateDeviceBufferAsync(void** ptr, size_t bytes, cudaStream_t stream) {
  if (bytes == 0) bytes = 1;
  return cudaMallocAsync(ptr, bytes, stream);
}

inline cudaError_t freeDeviceBufferAsync(void* ptr, cudaStream_t stream) {
  if (ptr == nullptr) return cudaSuccess;
  return cudaFreeAsync(ptr, stream);
}

#endif  // SD_CUDA

// configureCudaKernelSharedMemory and queryCudaCooperativeLaunchCapacity are
// always declared so they compile on CPU (the caller guards with dspIsCudaBuild()).
// On CPU builds the bodies are no-op stubs.  On CUDA builds the bodies use the
// real CUDA Driver + Runtime APIs.

#ifdef SD_CUDA
inline bool configureCudaKernelSharedMemory(void* kernelFunc, unsigned int sharedMemBytes) {
  if (kernelFunc == nullptr || sharedMemBytes == 0) return true;

  int currentDevice = 0;
  cudaError_t getDeviceErr = cudaGetDevice(&currentDevice);
  if (getDeviceErr != cudaSuccess) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: failed to query CUDA device for shared memory setup: %s",
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
      DSP_DIAG(COMPILE, "TritonGraphBackend: failed to query device shared memory limits on device %d: %s",
               currentDevice, cudaGetErrorString(defaultErr));
      cudaGetLastError();
      return false;
    }
    maxSharedOptIn = maxSharedDefault;
  }

  if (sharedMemBytes > static_cast<unsigned int>(maxSharedOptIn)) {
    DSP_DIAG(COMPILE, "TritonGraphBackend: kernel shared memory requirement %u exceeds device %d limit %d",
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
      DSP_DIAG(COMPILE, "TritonGraphBackend: cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES=%u) "
               "failed: %s (code=%d)",
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
    DSP_DIAG(BACKEND, "TritonGraphBackend: cudaGetDevice failed during cooperative capacity query: %s",
             cudaGetErrorString(getDeviceErr));
    cudaGetLastError();
    return false;
  }

  CUdevice cuDevice = 0;
  CUresult devRes = cuDeviceGet(&cuDevice, currentDevice);
  if (devRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(devRes, &errStr);
    DSP_DIAG(BACKEND, "TritonGraphBackend: cuDeviceGet failed during cooperative capacity query: %s (code=%d)",
             errStr ? errStr : "unknown", static_cast<int>(devRes));
    return false;
  }

  int coopLaunchAttr = 0;
  CUresult coopRes =
      cuDeviceGetAttribute(&coopLaunchAttr, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, cuDevice);
  if (coopRes != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(coopRes, &errStr);
    DSP_DIAG(BACKEND, "TritonGraphBackend: cuDeviceGetAttribute(COOPERATIVE_LAUNCH) failed: %s (code=%d)",
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
    DSP_DIAG(BACKEND, "TritonGraphBackend: cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: %s (code=%d)",
             errStr ? errStr : "unknown", static_cast<int>(smRes));
    return false;
  }

  unsigned long long threadsPerBlock64 =
      static_cast<unsigned long long>(blockX) *
      static_cast<unsigned long long>(blockY) *
      static_cast<unsigned long long>(blockZ);
  if (threadsPerBlock64 == 0 ||
      threadsPerBlock64 > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    DSP_DIAG(BACKEND, "TritonGraphBackend: invalid launch block size for cooperative capacity query: %llux%ux%u",
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
    DSP_DIAG(BACKEND, "TritonGraphBackend: cuOccupancyMaxActiveBlocksPerMultiprocessor failed: %s (code=%d)",
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

// ─── Dummy device pointer cache ──────────────────────────────────────────────
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

  void* ptr = sd::memory::CudaMemoryPool::getInstance().allocateDirect(8, currentDevice);
  if (ptr == nullptr) {
    DSP_DIAG(MEMORY, "TritonGraphBackend: failed to allocate dummy device pointer on device %d",
             currentDevice);
    return nullptr;
  }
  cache.byDevice[currentDevice] = ptr;
  return ptr;
}

#else  // !SD_CUDA — CPU stubs so callers compile without CUDA

inline bool configureCudaKernelSharedMemory(void* /*kernelFunc*/, unsigned int /*sharedMemBytes*/) {
  return false;
}

inline bool queryCudaCooperativeLaunchCapacity(void* /*kernelFunc*/,
                                               unsigned int /*blockX*/,
                                               unsigned int /*blockY*/,
                                               unsigned int /*blockZ*/,
                                               unsigned int /*sharedMemBytes*/,
                                               bool* cooperativeSupported,
                                               long long* maxBlocks,
                                               int* blocksPerSm,
                                               int* smCount) {
  if (cooperativeSupported) *cooperativeSupported = false;
  if (maxBlocks) *maxBlocks = 0;
  if (blocksPerSm) *blocksPerSm = 0;
  if (smCount) *smCount = 0;
  return false;
}

inline void* getDummyDevicePtrForDevice(int /*currentDevice*/, bool /*streamIsCapturing*/) {
  return nullptr;
}

#endif  // SD_CUDA

}  // namespace triton_internal
}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_GRAPH_BACKEND_INTERNAL_H
