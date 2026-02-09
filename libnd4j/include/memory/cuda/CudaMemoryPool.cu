/* ******************************************************************************
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
// @author Adam Gibson
//

#include <memory/cuda/CudaMemoryPool.h>
#include <system/Environment.h>
#include <helpers/DebugHelper.h>
#include <execution/LaunchContext.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#if !defined(_WIN32) && !defined(_WIN64)
#include <malloc.h>
#endif

namespace sd {
namespace memory {

CudaMemoryPool& CudaMemoryPool::getInstance() {
  static CudaMemoryPool instance;
  return instance;
}

CudaMemoryPool::CudaMemoryPool() {
  std::memset(pools_, 0, sizeof(pools_));
  std::memset(poolInitialized_, 0, sizeof(poolInitialized_));
  std::memset(peerAccessEnabled_, 0, sizeof(peerAccessEnabled_));
  supported_ = checkSupport();

  if (supported_) {
    sd_debug("CUDA Memory Pools: Supported and enabled\n", "");
  } else {
    sd_debug("CUDA Memory Pools: Not supported (CUDA < 11.2), using fallback\n", "");
  }

  // Initialize pinned host memory limit from Environment.
  // Configurable via SD_CUDA_PINNED_HOST_LIMIT env var (in MB) or
  // Environment::setCudaPinnedHostLimit() from Java. Default is 8 GB.
  // Pinned host memory is a last-resort fallback when all GPUs are exhausted.
  // A large limit causes excessive host memory consumption that counts toward
  // JavaCPP's maxPhysicalBytes, triggering OOM during long-running workloads.
  int64_t limitMB = Environment::getInstance().cudaPinnedHostLimit();
  size_t limit = static_cast<size_t>(limitMB) * 1024ULL * 1024ULL;
  pinnedHostBytesLimit_.store(limit);
  sd_printf("CudaMemoryPool: Pinned host memory limit: %zu bytes (%.1f GB)\n",
            limit, limit / (1024.0 * 1024.0 * 1024.0));

  initializePeerAccess();
}

void CudaMemoryPool::initializePeerAccess() {
  if (peerAccessInitialized_) return;
  peerAccessInitialized_ = true;

  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount <= 1) return;

  int prevDev = -1;
  cudaGetDevice(&prevDev);

  for (int i = 0; i < deviceCount && i < MAX_DEVICES; i++) {
    peerAccessEnabled_[i][i] = true;  // device can always access its own memory
    for (int j = 0; j < deviceCount && j < MAX_DEVICES; j++) {
      if (i == j) continue;
      int canAccess = 0;
      cudaDeviceCanAccessPeer(&canAccess, i, j);
      if (canAccess) {
        cudaSetDevice(i);
        cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
        if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
          peerAccessEnabled_[i][j] = true;
          sd_printf("CudaMemoryPool: Enabled peer access from device %d to device %d\n", i, j);
        } else {
          cudaGetLastError();  // clear error
          sd_debug("CudaMemoryPool: Failed to enable peer access from device %d to device %d\n", i, j);
        }
      }
    }
  }

  cudaSetDevice(prevDev);
}

CudaMemoryPool::~CudaMemoryPool() {
  releaseAll();
}

bool CudaMemoryPool::checkSupport() {
  // Memory pools require CUDA 11.2+
  int driverVersion = 0;
  cudaDriverGetVersion(&driverVersion);

  // CUDA version is encoded as 1000*major + 10*minor
  // 11.2 = 11020
  if (driverVersion < 11020) {
    return false;
  }

  // Also check that the device supports memory pools
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount == 0) {
    return false;
  }

  // Check first device for memory pool support
  int supportsMemoryPools = 0;
  cudaDeviceGetAttribute(&supportsMemoryPools, cudaDevAttrMemoryPoolsSupported, 0);

  return supportsMemoryPools != 0;
}

bool CudaMemoryPool::initializeForDevice(int deviceId) {
  if (!supported_ || deviceId < 0 || deviceId >= MAX_DEVICES) {
    return false;
  }

  std::lock_guard<std::mutex> lock(initMutex_);

  if (poolInitialized_[deviceId]) {
    return true;  // Already initialized
  }

  // Get the device's default memory pool
  cudaError_t err = cudaDeviceGetDefaultMemPool(&pools_[deviceId], deviceId);
  if (err != cudaSuccess) {
    sd_debug("Failed to get default memory pool for device %d: %s\n", deviceId, cudaGetErrorString(err));
    return false;
  }

  // Configure the pool release threshold.
  // Use 0 so the pool returns freed memory to the driver immediately when not in use.
  // This prevents the pool from holding onto all GPU memory indefinitely,
  // which would starve cudaMalloc fallback paths and other devices.
  // cudaMallocAsync still reuses pool memory for same-stream allocations
  // even with threshold=0, so the fast path is unaffected.
  uint64_t threshold = 0;
  err = cudaMemPoolSetAttribute(pools_[deviceId], cudaMemPoolAttrReleaseThreshold, &threshold);
  if (err != cudaSuccess) {
    sd_debug("Warning: Could not set pool release threshold: %s\n", cudaGetErrorString(err), "");
  }

  poolInitialized_[deviceId] = true;
  sd_debug("CUDA Memory Pool initialized for device %d\n", deviceId, "");

  return true;
}


void* CudaMemoryPool::allocate(size_t size, int deviceId, cudaStream_t stream, int* actualDeviceId) {
  // Clear any previous CUDA errors to ensure clean state for allocation
  cudaError_t prevErr = cudaGetLastError();
  if (prevErr != cudaSuccess) {
    sd_debug("CudaMemoryPool::allocate: Cleared previous CUDA error before allocation: %s\n",
             cudaGetErrorString(prevErr), "");
  }

  if (actualDeviceId) *actualDeviceId = deviceId;

  // Ensure we're on the requested device. cudaMallocAsync and cudaMalloc both
  // allocate on the CURRENT device, not on the deviceId parameter. Without this
  // check, callers that forgot to cudaSetDevice() before allocate() would silently
  // allocate on the wrong GPU, causing "invalid argument" errors during cross-device
  // copies when the pointer doesn't belong to the expected device.
  // We save and restore the original device to avoid side effects on the caller.
  int savedDev = -1;
  cudaGetDevice(&savedDev);
  bool needDeviceRestore = (savedDev != deviceId);
  if (needDeviceRestore) {
    cudaSetDevice(deviceId);
  }

  // Helper to restore device before returning
  auto restoreDevice = [needDeviceRestore, savedDev]() {
    if (needDeviceRestore) cudaSetDevice(savedDev);
  };

  // If pools not enabled or not supported, fall back to regular cudaMalloc
  if (!enabled_.load() || !supported_) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      sd_debug("cudaMalloc failed: %s\n", cudaGetErrorString(err), "");
      auto result = allocateFailover(size, deviceId, actualDeviceId);
      restoreDevice();
      return result;
    }

    restoreDevice();
    return ptr;
  }

  // Initialize pool for this device if needed
  if (!poolInitialized_[deviceId]) {
    if (!initializeForDevice(deviceId)) {
      // Fall back to regular cudaMalloc
      void* ptr = nullptr;
      cudaError_t err = cudaMalloc(&ptr, size);
      if (err != cudaSuccess) {
        sd_debug("cudaMalloc fallback failed: %s\n", cudaGetErrorString(err), "");
        auto result = allocateFailover(size, deviceId, actualDeviceId);
        restoreDevice();
        return result;
      }

      restoreDevice();
      return ptr;
    }
  }

  // Use async allocation from pool - THIS IS THE FAST PATH (no tracking needed)
  void* ptr = nullptr;
  // Use the stream provided by the caller. When nullptr (default CUDA stream) is passed,
  // all allocations share stream 0, which allows the pool to reuse memory across them.
  // Callers that need a specific compute stream (e.g., Workspace.cu) should resolve
  // it themselves before calling allocate(). We intentionally do NOT auto-detect the
  // stream here because it would cause recursive ContextBuffers initialization:
  // CudaMemoryPool::allocate() -> LaunchContext::defaultContext() -> getCudaStream()
  // -> ContextBuffers::initialize() -> CudaMemoryPool::allocate() -> ...
  cudaError_t err = cudaMallocAsync(&ptr, size, stream);

  if (err != cudaSuccess) {
    sd_printf("CudaMemoryPool::allocate: cudaMallocAsync failed on device %d (size=%zu): %s\n",
              deviceId, size, cudaGetErrorString(err));
    // cudaMallocAsync failure places an error both on the host-side sticky state
    // AND on the stream.  We must clear both, otherwise subsequent operations
    // on the same stream (or cudaStreamSynchronize) will pick up this stale error.
    cudaGetLastError();  // clear host-side sticky error
    if (stream != nullptr) {
      cudaStreamSynchronize(stream);
    }
    cudaGetLastError();  // clear any error surfaced by the sync
    err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
      sd_printf("CudaMemoryPool::allocate: cudaMalloc also failed on device %d (size=%zu): %s\n",
                deviceId, size, cudaGetErrorString(err));
      auto result = allocateFailover(size, deviceId, actualDeviceId);
      restoreDevice();
      return result;
    }

    restoreDevice();
    return ptr;
  }

  // Pool allocation succeeded - no tracking needed, will use cudaFreeAsync
  restoreDevice();
  return ptr;
}

void* CudaMemoryPool::allocateFailover(size_t size, int currentDeviceId, int* actualDeviceId) {
  sd_debug("CudaMemoryPool::allocateFailover: Primary allocation failed on device %d for %zu bytes\n", currentDeviceId, size);

  // Step 1: Sync device, trim pool, then retry.
  // Use cudaDeviceSynchronize instead of stream-based sync to avoid
  // LaunchContext::defaultContext() which can trigger recursive ContextBuffers
  // initialization that calls CudaMemoryPool::allocate() → infinite recursion.
  if (supported_ && poolInitialized_[currentDeviceId]) {
    cudaDeviceSynchronize();
    cudaGetLastError();  // clear errors after sync
    trimPool(currentDeviceId);

    // Try cudaMallocAsync first (reuses pool memory directly)
    // Use nullptr (default stream) to avoid LaunchContext recursion.
    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, size, nullptr);
    if (err == cudaSuccess && ptr != nullptr) {
      sd_debug("CudaMemoryPool::allocateFailover: Succeeded via pool after trim on device %d\n", currentDeviceId, "");
      if (actualDeviceId) *actualDeviceId = currentDeviceId;
      return ptr;
    }
    cudaGetLastError();  // clear error

    // Also try cudaMalloc (uses driver-level free memory released by trim)
    err = cudaMalloc(&ptr, size);
    if (err == cudaSuccess && ptr != nullptr) {
      sd_debug("CudaMemoryPool::allocateFailover: Succeeded via cudaMalloc after trim on device %d\n", currentDeviceId, "");
      if (actualDeviceId) *actualDeviceId = currentDeviceId;
      return ptr;
    }
    cudaGetLastError();  // clear error
  }

  // Step 2: Try other GPU devices. Prefer peer-accessible devices first (faster data
  // sharing), then non-peer devices (requires device switch for ops, but CudaExecutioner
  // already handles this by checking array deviceId and calling cudaSetDevice).
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  int prevDev = -1;
  cudaGetDevice(&prevDev);

  // Build sorted device list: peer devices first, then non-peer, each sorted by free memory
  struct DeviceInfo { int id; size_t freeMem; bool isPeer; };
  std::vector<DeviceInfo> candidates;
  for (int d = 0; d < deviceCount && d < MAX_DEVICES; d++) {
    if (d == currentDeviceId) continue;
    cudaSetDevice(d);
    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (freeMem > size * 1.1) {  // 10% margin
      bool isPeer = peerAccessEnabled_[currentDeviceId][d];
      candidates.push_back({d, freeMem, isPeer});
    }
  }
  cudaSetDevice(prevDev);

  // Sort: peer devices first, then by free memory descending
  std::sort(candidates.begin(), candidates.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
    if (a.isPeer != b.isPeer) return a.isPeer > b.isPeer;  // peer first
    return a.freeMem > b.freeMem;  // then by free memory
  });

  for (const auto& candidate : candidates) {
    int d = candidate.id;
    sd_printf("CudaMemoryPool::allocateFailover: Trying %s device %d (free: %zu bytes) for %zu bytes\n",
              candidate.isPeer ? "peer" : "non-peer", d, candidate.freeMem, size);
    cudaSetDevice(d);

    // Try pool allocation first on this device
    void* ptr = nullptr;
    if (supported_ && poolInitialized_[d]) {
      cudaError_t err = cudaMallocAsync(&ptr, size, nullptr);
      if (err == cudaSuccess && ptr != nullptr) {
        sd_printf("CudaMemoryPool::allocateFailover: Succeeded via pool on %s device %d for %zu bytes\n",
                  candidate.isPeer ? "peer" : "non-peer", d, size);
        if (actualDeviceId) *actualDeviceId = d;
        cudaSetDevice(prevDev);
        return ptr;
      }
      cudaGetLastError();  // clear error
      ptr = nullptr;
    }

    // Fall back to cudaMalloc on this device
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err == cudaSuccess && ptr != nullptr) {
      sd_printf("CudaMemoryPool::allocateFailover: Succeeded via cudaMalloc on %s device %d for %zu bytes\n",
                candidate.isPeer ? "peer" : "non-peer", d, size);
      if (actualDeviceId) *actualDeviceId = d;
      cudaSetDevice(prevDev);
      return ptr;
    }
    cudaGetLastError();  // clear error
  }
  cudaSetDevice(prevDev);

  // Step 3: Fall back to pinned host memory
  // WARNING: Pinned host memory is accessible from GPU via UVA but at PCIe bandwidth.
  // The actualDeviceId is NOT updated here - it stays as currentDeviceId since CUDA
  // operations from that device can still access pinned host memory. The hostAllocations_
  // map tracks these pointers and sizes for correct deallocation via cudaFreeHost.

  // Check pinned host memory budget before allocating
  size_t limit = pinnedHostBytesLimit_.load();
  size_t currentUsage = pinnedHostBytesUsed_.load();
  if (limit > 0 && (currentUsage + size) > limit) {
    sd_printf("CudaMemoryPool::allocateFailover: Pinned host memory limit exceeded (%zu + %zu > %zu bytes, limit=%.1f GB). "
              "Increase via SD_CUDA_PINNED_HOST_LIMIT env var (in MB) or Environment::setCudaPinnedHostLimit().\n",
              currentUsage, size, limit, limit / (1024.0 * 1024.0 * 1024.0));
    THROW_EXCEPTION("CUDA out of memory: all GPU devices exhausted and pinned host memory limit exceeded. "
                    "Set SD_CUDA_PINNED_HOST_LIMIT (in MB) to increase the limit.");
  }

  sd_printf("CudaMemoryPool::allocateFailover: WARNING - All GPUs exhausted, falling back to pinned host memory for %zu bytes on device %d (pinned used: %zu)\n", size, currentDeviceId, currentUsage);
  cudaGetLastError();  // clear error state
  void* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size);
  if (err == cudaSuccess && ptr != nullptr) {
    {
      std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
      hostAllocations_[ptr] = size;
    }
    pinnedHostBytesUsed_.fetch_add(size);
    sd_printf("CudaMemoryPool::allocateFailover: Pinned host fallback succeeded for %zu bytes (ptr=%p, total pinned: %zu)\n", size, ptr, pinnedHostBytesUsed_.load());
    return ptr;
  }

  sd_debug("CudaMemoryPool::allocateFailover: All allocation attempts failed for %zu bytes\n", size, "");
  return nullptr;
}

void CudaMemoryPool::free(void* ptr, int deviceId, cudaStream_t stream) {
  if (ptr == nullptr) {
    return;
  }

  // Check host allocations (very rare - only from last-resort failover)
  {
    std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
    auto hostIt = hostAllocations_.find(ptr);
    if (hostIt != hostAllocations_.end()) {
      size_t freedSize = hostIt->second;
      hostAllocations_.erase(hostIt);
      pinnedHostBytesUsed_.fetch_sub(freedSize);
      cudaFreeHost(ptr);
      return;
    }
  }

  // Ensure we're on the correct device for the free. cudaFreeAsync with a stream
  // from a different device than the allocation can fail silently for non-P2P GPUs.
  // This mirrors the save/restore pattern in allocate().
  int savedDev = -1;
  cudaGetDevice(&savedDev);
  bool needDeviceRestore = (deviceId >= 0 && savedDev != deviceId);
  if (needDeviceRestore) {
    cudaSetDevice(deviceId);
  }

  // Device memory: use cudaFreeAsync for stream-ordered deallocation.
  // Works for both pool and non-pool allocations since CUDA 11.2.
  if (enabled_.load() && supported_) {
    cudaError_t err = cudaFreeAsync(ptr, stream);
    if (err == cudaSuccess) {
      if (needDeviceRestore) cudaSetDevice(savedDev);
      return;
    }
    cudaGetLastError();  // clear error
  }
  // Fallback for unsupported or error cases
  cudaFree(ptr);
  if (needDeviceRestore) cudaSetDevice(savedDev);
}

void CudaMemoryPool::getStats(int deviceId, size_t& usedBytes, size_t& reservedBytes) {
  usedBytes = 0;
  reservedBytes = 0;

  if (!supported_ || !poolInitialized_[deviceId]) {
    return;
  }

  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &usedBytes);
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &reservedBytes);
}

void CudaMemoryPool::trimPool(int deviceId) {
  if (!supported_ || deviceId < 0 || deviceId >= MAX_DEVICES || !poolInitialized_[deviceId]) {
    return;
  }

  int prevDevice = 0;
  cudaGetDevice(&prevDevice);

  // Switch to the target device so we sync and trim the correct device's pool
  if (prevDevice != deviceId) {
    cudaSetDevice(deviceId);
  }

  // Sync the default stream on this device where cudaFreeAsync calls were issued.
  // Until the stream reaches the free point, memory remains "in use" and can't be trimmed.
  cudaStreamSynchronize(0);

  // Trim to release unused reserved memory back to the device
  cudaMemPoolTrimTo(pools_[deviceId], 0);

  if (prevDevice != deviceId) {
    cudaSetDevice(prevDevice);
  }

#if !defined(_WIN32) && !defined(_WIN64)
  // Release freed host memory pages back to the OS. glibc's malloc retains freed
  // pages in its arena, which inflates RSS (physicalBytes). malloc_trim releases
  // unused pages so JavaCPP's maxPhysicalBytes check doesn't falsely OOM.
  malloc_trim(0);
#endif
}

void CudaMemoryPool::trimPoolOnStream(int deviceId, cudaStream_t stream) {
  if (!supported_ || deviceId < 0 || deviceId >= MAX_DEVICES || !poolInitialized_[deviceId]) {
    return;
  }

  int prevDevice = 0;
  cudaGetDevice(&prevDevice);

  if (prevDevice != deviceId) {
    cudaSetDevice(deviceId);
  }

  // Sync the actual execution stream where cudaFreeAsync calls were issued.
  cudaStreamSynchronize(stream);

  // Trim to release unused reserved memory back to the device
  cudaMemPoolTrimTo(pools_[deviceId], 0);

  if (prevDevice != deviceId) {
    cudaSetDevice(prevDevice);
  }

  // NOTE: Do NOT call malloc_trim(0) here. malloc_trim walks the glibc heap
  // metadata, and if any C++ op has overrun a host buffer (corrupting adjacent
  // malloc chunk headers), malloc_trim discovers the corruption and triggers
  // "double free or corruption" SIGABRT. Since this is called ~20 times per
  // vision encoder execution, it amplifies the chance of hitting corruption.
  // The madvise(MADV_DONTNEED) in freeGpuOnStream already releases physical
  // pages without touching glibc metadata. RSS stays under control.
}

void CudaMemoryPool::releaseAll() {
  if (!supported_) {
    return;
  }

  for (int i = 0; i < MAX_DEVICES; i++) {
    if (poolInitialized_[i]) {
      trimPool(i);
      // Note: We don't destroy the default pool, just trim it
      poolInitialized_[i] = false;
    }
  }

  // Free and clear fallback pinned host allocations
  {
    std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
    for (auto& pair : hostAllocations_) {
      cudaFreeHost(pair.first);
    }
    hostAllocations_.clear();
    pinnedHostBytesUsed_.store(0);
  }
}

}  // namespace memory
}  // namespace sd
