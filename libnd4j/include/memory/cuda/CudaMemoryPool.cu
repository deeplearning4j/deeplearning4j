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
#include <array/DataBuffer.h>
#include <memory/MemoryCounter.h>
#include <system/Environment.h>
#include <helpers/DebugHelper.h>
#include <execution/LaunchContext.h>
#include <graph/DspDiagnostics.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>

namespace sd {
namespace memory {

SD_INLINE cudaStream_t resolveCaptureStream(cudaStream_t stream) {
  if (tl_graphExecutionActive && stream == nullptr && tl_graphCaptureStream != nullptr) {
    return tl_graphCaptureStream;
  }
  return stream;
}

// Guard against re-entrancy when resolving LaunchContext stream.
// LaunchContext::defaultContext()->getCudaStream() can trigger ContextBuffers::initialize()
// which calls CudaMemoryPool::allocate() → infinite recursion.
static thread_local bool tl_resolvingContextStream = false;

// Resolve nullptr stream to a valid CUDA stream. Priority:
// 1. DSP execution stream (set during DSP plan execution)
// 2. LaunchContext default stream (if not re-entrant)
// 3. nullptr (stream 0) as last resort
SD_INLINE cudaStream_t resolveNullStream(cudaStream_t stream) {
  if (stream != nullptr) return stream;
  if (tl_dspExecutionStream != nullptr) return tl_dspExecutionStream;
  if (!tl_resolvingContextStream) {
    tl_resolvingContextStream = true;
    auto* ctxStream = sd::LaunchContext::defaultContext()->getCudaStream();
    tl_resolvingContextStream = false;
    if (ctxStream != nullptr) return *ctxStream;
  }
  return nullptr;  // Last resort — first-time context init
}

CudaMemoryPool& CudaMemoryPool::getInstance() {
  static CudaMemoryPool instance;
  return instance;
}

void CudaMemoryPool::setMemoryPressureCallback(MemoryPressureCallback callback) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  memoryPressureCallback_ = callback;
}

CudaMemoryPool::CudaMemoryPool() {
  try {
    if (sd::Environment::getInstance().isDebug() || sd::Environment::getInstance().isVerbose()) {
      sd_debug("CudaMemoryPool: Beginning construction...\n", "");
    }

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

    if (sd::Environment::getInstance().isDebug() || sd::Environment::getInstance().isVerbose()) {
      sd_debug("CudaMemoryPool: Construction completed successfully\n", "");
    }
  } catch (...) {
    supported_ = false;
    sd_debug("CudaMemoryPool: Exception during construction - disabling CUDA memory pools\n", "");
  }
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

  // This API is public, so it must not assume the caller already switched to
  // the target device before querying device-local pool and memory attributes.
  int savedDev = -1;
  cudaError_t getDevErr = cudaGetDevice(&savedDev);
  bool needDeviceRestore = (getDevErr == cudaSuccess && savedDev != deviceId);
  if (needDeviceRestore) {
    cudaError_t setDevErr = cudaSetDevice(deviceId);
    if (setDevErr != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
  }

  auto restoreDevice = [needDeviceRestore, savedDev]() {
    if (needDeviceRestore) cudaSetDevice(savedDev);
  };

  // Get the device's default memory pool
  cudaError_t err = cudaDeviceGetDefaultMemPool(&pools_[deviceId], deviceId);
  if (err != cudaSuccess) {
    sd_debug("Failed to get default memory pool for device %d: %s\n", deviceId, cudaGetErrorString(err));
    restoreDevice();
    return false;
  }

  // Configure the pool release threshold based on device memory ratio.
  // The pool holds reserved memory for reuse. Setting this to a fraction of total
  // GPU memory ensures the pool returns excess memory to the driver, leaving headroom
  // for CUDA contexts, streams, cuDNN workspaces, and other non-pool allocations.
  // Ratio-based: 75% of total GPU memory. The remaining 25% stays available for
  // non-pool uses (stream creation, cuDNN, display, etc.).
  size_t devFree = 0, devTotal = 0;
  cudaMemGetInfo(&devFree, &devTotal);
  uint64_t threshold = static_cast<uint64_t>(devTotal * 0.75);
  err = cudaMemPoolSetAttribute(pools_[deviceId], cudaMemPoolAttrReleaseThreshold, &threshold);
  if (err != cudaSuccess) {
    sd_debug("Warning: Could not set pool release threshold: %s\n", cudaGetErrorString(err), "");
  } else {
    sd_printf("CudaMemoryPool: Device %d pool release threshold set to %zu MB (75%% of %zu MB total)\n",
              deviceId, threshold / (1024*1024), devTotal / (1024*1024));
  }

  poolInitialized_[deviceId] = true;
  sd_debug("CUDA Memory Pool initialized for device %d\n", deviceId, "");
  restoreDevice();

  return true;
}


void* CudaMemoryPool::allocate(size_t size, int deviceId, cudaStream_t stream, int* actualDeviceId) {
  // During CUDA graph capture, use pre-allocated workspace instead of cudaMallocAsync.
  // This eliminates cudaGraphMemAllocNode from the captured graph, preventing
  // "invalid argument" on cudaGraphLaunch from unpaired alloc/free nodes.
  if (tl_graphExecutionActive && tl_captureWorkspace != nullptr) {
    // Align to 256 bytes for GPU memory alignment
    size_t aligned = (size + 255) & ~255ULL;
    if (tl_captureWorkspaceOffset + aligned <= tl_captureWorkspaceSize) {
      void* ptr = static_cast<char*>(tl_captureWorkspace) + tl_captureWorkspaceOffset;
      tl_captureWorkspaceOffset += aligned;
      if (actualDeviceId) *actualDeviceId = deviceId;
      return ptr;
    }
    // Workspace exhausted — return nullptr to abort the current op gracefully.
    // Falling through to cudaMallocAsync during graph capture corrupts the capture
    // stream (error 901 / invalid argument), making the entire capture invalid.
    // Returning nullptr causes the op to fail, which sets compilationFailed=true on the
    // segment and falls back to slot-by-slot execution for this segment.
    sd_printf("CudaMemoryPool: capture workspace exhausted (%zu + %zu > %zu), "
              "returning nullptr (aborting capture) for %zu bytes\n",
              tl_captureWorkspaceOffset, aligned, tl_captureWorkspaceSize, size);
    return nullptr;
  }
  if (tl_graphExecutionActive && tl_captureWorkspace == nullptr) {
    static int captureAllocLogCount = 0;
    if (captureAllocLogCount < 5) {
      sd_printf("CudaMemoryPool: during capture but NO capture workspace! size=%zu\n", size);
      captureAllocLogCount++;
    }
  }

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
    cudaError_t setDevErr = cudaSetDevice(deviceId);
    if (setDevErr != cudaSuccess) {
      cudaGetLastError();
      return nullptr;
    }
  }

  // Helper to restore device before returning
  auto restoreDevice = [needDeviceRestore, savedDev]() {
    if (needDeviceRestore) cudaSetDevice(savedDev);
  };

  cudaStream_t allocStream = resolveCaptureStream(stream);

  // Resolve nullptr to a valid stream — prevents cross-stream pool fragmentation.
  allocStream = resolveNullStream(allocStream);

  // If pools not enabled or not supported, fall back to regular cudaMalloc
  if (!enabled_.load() || !supported_) {
    void* ptr = nullptr;
    if (tl_graphExecutionActive) {
      // During CUDA graph capture, cudaMalloc (synchronous) breaks capture.
      // Use cudaMallocAsync with the caller-provided stream (the captured stream).
      cudaError_t err = cudaMallocAsync(&ptr, size, allocStream);
      if (err != cudaSuccess) {
        cudaGetLastError();
        restoreDevice();
        return nullptr;
      }
      restoreDevice();
      return ptr;
    }
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
      if (tl_graphExecutionActive) {
        cudaError_t err = cudaMallocAsync(&ptr, size, allocStream);
        if (err != cudaSuccess) {
          cudaGetLastError();
          restoreDevice();
          return nullptr;
        }
        restoreDevice();
        return ptr;
      }
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
  // Use the stream provided by the caller. During graph capture, nullptr is resolved
  // to the active capture stream to avoid cross-stream capture invalidation.
  // Outside capture, nullptr means default stream 0 and allows broad pool reuse.
  // Callers that need a specific compute stream (e.g., Workspace.cu) should resolve
  // it themselves before calling allocate(). We intentionally do NOT auto-detect the
  // stream here because it would cause recursive ContextBuffers initialization:
  // CudaMemoryPool::allocate() -> LaunchContext::defaultContext() -> getCudaStream()
  // -> ContextBuffers::initialize() -> CudaMemoryPool::allocate() -> ...
  cudaError_t err = cudaMallocAsync(&ptr, size, allocStream);

  if (err != cudaSuccess) {
    // During CUDA graph capture, error recovery (stream sync, failover) would break
    // capture. Return nullptr — the caller must handle allocation failure during capture.
    if (tl_graphExecutionActive) {
      cudaGetLastError();  // clear sticky error
      restoreDevice();
      return nullptr;
    }

    // cudaMallocAsync failure places an error both on the host-side sticky state
    // AND on the stream.  We must clear both before retrying.
    cudaGetLastError();  // clear host-side sticky error
    if (allocStream != nullptr) {
      cudaStreamSynchronize(allocStream);
    }
    cudaGetLastError();  // clear any error surfaced by the sync

    // STREAM MISMATCH FIX: The most common cause of cudaMallocAsync failure is
    // that memory was freed on a different stream (e.g., execStream) than the
    // allocation stream (nullptr/stream 0). The pool can't reuse freed memory
    // across streams without synchronization. Sync dirty free streams and retry
    // BEFORE going to the heavyweight allocateFailover path.
    {
      bool syncedAny = false;
      std::lock_guard<std::mutex> lock(dirtyStreamsMutex_[deviceId]);
      for (auto s : dirtyFreeStreams_[deviceId]) {
        if (s != allocStream) {  // only sync OTHER streams (our stream already synced above)
          cudaStreamSynchronize(s);
          syncedAny = true;
        }
      }
      if (syncedAny) {
        dirtyFreeStreams_[deviceId].clear();
        cudaGetLastError();  // clear any error from syncs
        ptr = nullptr;
        err = cudaMallocAsync(&ptr, size, allocStream);
        if (err == cudaSuccess && ptr != nullptr) {
          restoreDevice();
          return ptr;  // Success after stream sync — no failover needed
        }
        cudaGetLastError();
      }
    }

    // Still failed after stream sync. Log and go to full failover.
    size_t poolUsed = 0, poolReserved = 0;
    getStats(deviceId, poolUsed, poolReserved);
    sd_printf("CudaMemoryPool::allocate: cudaMallocAsync failed on device %d (size=%zu): %s\n"
              "  Pool: used=%zu MB, reserved=%zu MB (reclaimable=%zu MB)\n",
              deviceId, size, cudaGetErrorString(err),
              poolUsed / (1024*1024), poolReserved / (1024*1024),
              (poolReserved > poolUsed ? (poolReserved - poolUsed) : 0) / (1024*1024));

    //  Do NOT fall back to cudaMalloc here. Direct cudaMalloc allocations
    // bypass pool stats (cudaMemPoolAttrUsedMemCurrent won't track them), causing
    // getStats() to underreport memory usage. Go straight to allocateFailover()
    // which does trimPool + retry in a controlled way.
    auto result = allocateFailover(size, deviceId, actualDeviceId);
    restoreDevice();
    return result;
  }

  // Pool allocation succeeded - no tracking needed, will use cudaFreeAsync
  restoreDevice();
  return ptr;
}

void* CudaMemoryPool::allocateFailover(size_t size, int currentDeviceId, int* actualDeviceId) {
  sd_debug("CudaMemoryPool::allocateFailover: Primary allocation failed on device %d for %zu bytes\n", currentDeviceId, size);

  // Get available memory on current device for pressure event
  size_t currentFreeMem = 0, currentTotalMem = 0;
  int prevDev = -1;
  cudaGetDevice(&prevDev);
  if (prevDev != currentDeviceId) {
    cudaSetDevice(currentDeviceId);
  }
  // Clear any sticky CUDA error before querying memory — a pending error causes
  // cudaMemGetInfo to fail and return total=0, masking the real memory state.
  cudaGetLastError();
  auto memInfoErr = cudaMemGetInfo(&currentFreeMem, &currentTotalMem);
  if (memInfoErr != cudaSuccess) {
    sd_printf("CudaMemoryPool::allocateFailover: cudaMemGetInfo FAILED on device %d: %s (error %d)\n",
              currentDeviceId, cudaGetErrorString(memInfoErr), static_cast<int>(memInfoErr));
    cudaGetLastError();  // Clear the error from the failed cudaMemGetInfo call itself
  }

  // Diagnostic: show pool stats alongside cudaMemGetInfo so we can distinguish
  // "pool holds reserved memory" from "GPU genuinely full"
  size_t poolUsed = 0, poolReserved = 0;
  getStats(currentDeviceId, poolUsed, poolReserved);

  // Also check MemoryCounter (C++ side tracking) and host memory
  size_t mcDevice = MemoryCounter::getInstance().allocatedDevice(currentDeviceId);
  size_t mcHost = MemoryCounter::getInstance().allocatedGroup(MemoryType::HOST);
  size_t mcDeviceGroup = MemoryCounter::getInstance().allocatedGroup(MemoryType::DEVICE);
  size_t pinnedUsed = pinnedHostBytesUsed_.load();

  sd_printf("CudaMemoryPool::allocateFailover: MEMORY DIAGNOSTIC on device %d:\n"
            "  cudaMemGetInfo: free=%zu MB, total=%zu MB\n"
            "  Pool: used=%zu MB, reserved=%zu MB (reclaimable=%zu MB)\n"
            "  MemoryCounter: device[%d]=%zu MB, deviceGroup=%zu MB, hostGroup=%zu MB\n"
            "  Pinned host fallback: %zu MB\n"
            "  Requested: %zu bytes (%.2f MB)\n",
            currentDeviceId,
            currentFreeMem / (1024*1024), currentTotalMem / (1024*1024),
            poolUsed / (1024*1024), poolReserved / (1024*1024), (poolReserved - poolUsed) / (1024*1024),
            currentDeviceId, mcDevice / (1024*1024), mcDeviceGroup / (1024*1024), mcHost / (1024*1024),
            pinnedUsed / (1024*1024),
            size, size / (1024.0*1024.0));

  if (prevDev != currentDeviceId) {
    cudaSetDevice(prevDev);
  }

  // Step 1: Trim pool and retry.
  // trimPool() syncs only streams with pending cudaFreeAsync (tracked in dirtyFreeStreams_),
  // then releases pool-reserved memory back to the driver.
  if (supported_ && poolInitialized_[currentDeviceId]) {
    trimPool(currentDeviceId);

    // Log post-trim state to see how much memory was actually recovered
    size_t postTrimFree = 0, postTrimTotal = 0;
    size_t postTrimPoolUsed = 0, postTrimPoolReserved = 0;
    {
      int prevDev2 = -1;
      cudaGetDevice(&prevDev2);
      if (prevDev2 != currentDeviceId) cudaSetDevice(currentDeviceId);
      cudaMemGetInfo(&postTrimFree, &postTrimTotal);
      if (prevDev2 != currentDeviceId) cudaSetDevice(prevDev2);
    }
    getStats(currentDeviceId, postTrimPoolUsed, postTrimPoolReserved);
    sd_printf("CudaMemoryPool::allocateFailover: After trim on device %d: "
              "cudaFree=%zu MB (was %zu MB, recovered %zu MB), pool used=%zu MB, reserved=%zu MB\n",
              currentDeviceId,
              postTrimFree / (1024*1024), currentFreeMem / (1024*1024),
              (postTrimFree > currentFreeMem ? (postTrimFree - currentFreeMem) : 0) / (1024*1024),
              postTrimPoolUsed / (1024*1024), postTrimPoolReserved / (1024*1024));

    // trimPool() restores the caller's original device. Switch back to the
    // requested device before retrying so the retry allocation lands where the
    // diagnostics and actualDeviceId expect.
    int retryPrevDev = -1;
    cudaGetDevice(&retryPrevDev);
    bool retryNeedRestore = (retryPrevDev != currentDeviceId);
    if (retryNeedRestore) {
      cudaSetDevice(currentDeviceId);
    }

    // Try cudaMallocAsync first (reuses pool memory directly)
    // Use nullptr (default stream) to avoid LaunchContext recursion.
    void* ptr = nullptr;
    cudaError_t err = cudaMallocAsync(&ptr, size, nullptr);
    if (err == cudaSuccess && ptr != nullptr) {
      sd_printf("CudaMemoryPool::allocateFailover: Succeeded via pool after trim on device %d\n", currentDeviceId);
      if (actualDeviceId) *actualDeviceId = currentDeviceId;
      if (retryNeedRestore) cudaSetDevice(retryPrevDev);
      return ptr;
    }
    cudaGetLastError();  // clear error

    // Also try cudaMalloc (uses driver-level free memory released by trim)
    err = cudaMalloc(&ptr, size);
    if (err == cudaSuccess && ptr != nullptr) {
      sd_printf("CudaMemoryPool::allocateFailover: Succeeded via cudaMalloc after trim on device %d\n", currentDeviceId);
      if (actualDeviceId) *actualDeviceId = currentDeviceId;
      if (retryNeedRestore) cudaSetDevice(retryPrevDev);
      return ptr;
    }
    if (retryNeedRestore) cudaSetDevice(retryPrevDev);
    cudaGetLastError();  // clear error
  }

  // Step 2: Try other GPU devices (peer first, then non-peer).
  // Peer devices allow direct GPU-GPU access. Non-peer devices require staged
  // D2H+H2D transfers — the higher-level migration code (DataBuffer::migrate,
  // replicateToDevice) is responsible for using the correct copy path based on
  // whether peer access is enabled between the source and target devices.
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  struct DeviceInfo { int id; size_t freeMem; bool isPeer; };
  std::vector<DeviceInfo> candidates;
  for (int d = 0; d < deviceCount && d < MAX_DEVICES; d++) {
    if (d == currentDeviceId) continue;

    bool isPeer = peerAccessEnabled_[currentDeviceId][d];
    cudaSetDevice(d);

    // Trim this device's pool before checking free memory. Without trimming,
    // cudaMemGetInfo reports free memory MINUS pool reserved, which can be
    // much lower than actual available memory. This ensures candidates are
    // accurately evaluated rather than incorrectly excluded.
    if (supported_ && poolInitialized_[d]) {
      trimPool(d);
    }

    size_t freeMem = 0, totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    if (freeMem > size * 1.1) {  // 10% margin
      candidates.push_back({d, freeMem, isPeer});
    }
  }
  cudaSetDevice(prevDev);

  // Sort: peer devices first (direct access), then by free memory descending
  std::sort(candidates.begin(), candidates.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
    if (a.isPeer != b.isPeer) return a.isPeer;  // peer first
    return a.freeMem > b.freeMem;
  });

  // MEMORY PRESSURE EVENT: Build and report to callback
  MemoryPressureEvent event;
  event.requestedDeviceId = currentDeviceId;
  event.requestedSize = size;
  event.availableMemory = currentFreeMem;
  event.alternativeDeviceId = candidates.empty() ? -1 : candidates[0].id;
  event.isPeerAccessible = !candidates.empty() && candidates[0].isPeer;
  event.recommendedAction = candidates.empty() ?
    MemoryPressureEvent::Action::USE_PINNED_HOST :
    MemoryPressureEvent::Action::FAILOVER;

  // Store the event and set flag
  {
    std::lock_guard<std::mutex> lock(pressureEventMutex_);
    lastPressureEvent_ = event;
    memoryPressureDetected_.store(true);
  }

  // Call registered callback if any
  bool allowAllocation = true;
  {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    if (memoryPressureCallback_) {
      allowAllocation = memoryPressureCallback_(event);
      if (!allowAllocation) {
        sd_printf("CudaMemoryPool::allocateFailover: Callback rejected allocation on device %d\n", currentDeviceId);
        cudaSetDevice(prevDev);
        return nullptr;  // Callback wants us to fail
      }
    }
  }

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

  // Also check host-side memory: MemoryCounter HOST group tracks primary buffer allocations.
  // Pinned host memory from cudaMallocHost is ADDITIONAL host consumption on top of those
  // primary buffers. JavaCPP's maxPhysicalBytes caps total physical memory — if primary
  // buffers already use most of that budget, adding pinned host memory will cause OOM.
  size_t hostGroupUsed = MemoryCounter::getInstance().allocatedGroup(MemoryType::HOST);
  int64_t hostGroupLimit = MemoryCounter::getInstance().groupLimit(MemoryType::HOST);
  size_t totalHostUsage = hostGroupUsed + currentUsage + size;

  if (hostGroupLimit > 0 && totalHostUsage > static_cast<size_t>(hostGroupLimit)) {
    sd_printf("CudaMemoryPool::allocateFailover: HOST memory limit would be exceeded: "
              "primary=%zu MB + pinned=%zu MB + new=%zu MB = %zu MB > limit=%lld MB\n",
              hostGroupUsed / (1024*1024), currentUsage / (1024*1024), size / (1024*1024),
              totalHostUsage / (1024*1024), hostGroupLimit / (1024LL*1024LL));
    THROW_EXCEPTION("CUDA out of memory: all GPU devices exhausted and total host memory "
                    "(primary buffers + pinned fallback) would exceed host memory limit. "
                    "Set SD_MAX_PRIMARY_BYTES to increase the host limit.");
  }

  if (limit > 0 && (currentUsage + size) > limit) {
    sd_printf("CudaMemoryPool::allocateFailover: Pinned host memory limit exceeded (%zu + %zu > %zu bytes, limit=%.1f GB). "
              "Increase via SD_CUDA_PINNED_HOST_LIMIT env var (in MB) or Environment::setCudaPinnedHostLimit().\n",
              currentUsage, size, limit, limit / (1024.0 * 1024.0 * 1024.0));
    THROW_EXCEPTION("CUDA out of memory: all GPU devices exhausted and pinned host memory limit exceeded. "
                    "Set SD_CUDA_PINNED_HOST_LIMIT (in MB) to increase the limit.");
  }

  sd_printf("CudaMemoryPool::allocateFailover: WARNING - All GPUs exhausted, falling back to pinned host memory "
            "for %zu bytes on device %d (pinned used: %zu MB, host primary: %zu MB, total host: %zu MB)\n",
            size, currentDeviceId, currentUsage / (1024*1024), hostGroupUsed / (1024*1024),
            (currentUsage + hostGroupUsed) / (1024*1024));
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

  // During CUDA graph capture, skip ALL frees to avoid recording MemFree graph nodes.
  // Workspace addresses: managed by the workspace buffer lifecycle (bump allocator).
  // Non-workspace (graph-external) addresses: cudaFreeAsync records a MemFree graph node
  // for memory allocated OUTSIDE the capture. On cudaGraphLaunch, the MemFree node
  // references a stale/invalid address → SIGSEGV. DataBuffer::deleteSpecial() is the
  // primary guard (returns early during capture for non-workspace memory), but this
  // serves as defense-in-depth for any code path that bypasses DataBuffer.
  if (tl_graphExecutionActive && tl_captureWorkspace != nullptr) {
    char* wsStart = static_cast<char*>(tl_captureWorkspace);
    char* wsEnd = wsStart + tl_captureWorkspaceSize;
    char* p = static_cast<char*>(ptr);
    if (p >= wsStart && p < wsEnd) {
      return;  // Within workspace — no-op (managed by workspace lifecycle)
    }
    // Non-workspace memory during capture — also skip to prevent MemFree graph nodes
    return;
  }
  // Note: tl_captureSkipFrees is NOT checked here — gap op temps allocated during
  // capture via cudaMallocAsync need paired MemFree nodes (cudaFreeAsync).
  // External memory is protected by DataBuffer::deleteSpecial() which checks
  // tl_graphExecutionActive and returns early.

  // Check host allocations with exception handling
  try {
    std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
    auto hostIt = hostAllocations_.find(ptr);
    if (hostIt != hostAllocations_.end()) {
      size_t freedSize = hostIt->second;
      hostAllocations_.erase(hostIt);
      pinnedHostBytesUsed_.fetch_sub(freedSize);

      if (sd::Environment::getInstance().isDebug() || sd::Environment::getInstance().isVerbose()) {
        sd_printf("CudaMemoryPool::free: Freed pinned host memory %p (%zu bytes)\n", ptr, freedSize);
      }

      cudaFreeHost(ptr);
      return;
    }
  } catch (...) {
    // DO NOT call cudaFreeHost(ptr) here — if the map lookup threw, we don't know
    // whether ptr is pinned (cudaMallocHost) or GPU (cudaMallocAsync) or regular heap (malloc).
    // Calling cudaFreeHost on a malloc'd pointer would munmap the pages, corrupting the
    // glibc heap and causing SIGSEGV on the next malloc/free. Just log and fall through
    // to the GPU free path below, which handles non-pinned pointers correctly.
    sd_printf("CudaMemoryPool::free: WARNING - Exception accessing hostAllocations_ for ptr=%p. "
              "Treating as GPU memory (NOT calling cudaFreeHost).\n", ptr);
  }

  // Check if CUDA context is still alive before calling any CUDA APIs.
  // During JVM shutdown, the CUDA runtime may destroy the context before GC
  // finalizes remaining DataBuffers. Any CUDA call on a destroyed context
  // causes SIGSEGV (the VM Thread has no CUDA context).
  int savedDev = -1;
  cudaError_t contextCheck = cudaGetDevice(&savedDev);
  if (contextCheck != cudaSuccess) {
    // Context is destroyed — skip cleanup. The OS reclaims all GPU memory
    // when the process exits; attempting cudaFree/cudaFreeAsync here would crash.
    cudaGetLastError();  // clear the sticky error
    return;
  }

  // Ensure we're on the correct device for the free. cudaFreeAsync with a stream
  // from a different device than the allocation can fail silently for non-P2P GPUs.
  // This mirrors the save/restore pattern in allocate().
  bool needDeviceRestore = (deviceId >= 0 && savedDev != deviceId);
  if (needDeviceRestore) {
    cudaError_t setDevErr = cudaSetDevice(deviceId);
    if (setDevErr != cudaSuccess) {
      cudaGetLastError();
      return;
    }
  }

  // Resolve nullptr streams only AFTER switching to the allocation device.
  // Otherwise LaunchContext/defaultContext can hand us a stream handle from the
  // caller's device, and cudaFreeAsync would run on a foreign stream/device pair.
  cudaStream_t freeStream = resolveCaptureStream(stream);
  freeStream = resolveNullStream(freeStream);

  // Device memory: use cudaFreeAsync for stream-ordered deallocation.
  // Works for both pool and non-pool allocations since CUDA 11.2.
  if (enabled_.load() && supported_) {
    cudaError_t err = cudaFreeAsync(ptr, freeStream);
    if (err == cudaSuccess) {
      // Track which stream this free was issued on so trimPool() can sync
      // only the relevant streams instead of blocking the entire device.
      int trackDevice = (deviceId >= 0 && deviceId < MAX_DEVICES) ? deviceId : 0;
      {
        std::lock_guard<std::mutex> lock(dirtyStreamsMutex_[trackDevice]);
        dirtyFreeStreams_[trackDevice].insert(freeStream);  // nullptr (stream 0) is valid
      }
      if (needDeviceRestore) cudaSetDevice(savedDev);
      return;
    }
    // Log failed cudaFreeAsync for debugging — this path means memory is LEAKED
    sd_printf("CudaMemoryPool::free: cudaFreeAsync FAILED for ptr=%p dev=%d stream=%p: %s\n",
              ptr, deviceId, (void*)freeStream, cudaGetErrorString(err));
    cudaGetLastError();  // clear error
  }
  // Fallback for unsupported or error cases
  cudaFree(ptr);
  if (needDeviceRestore) cudaSetDevice(savedDev);
}

void CudaMemoryPool::removeDirtyStream(int deviceId, cudaStream_t stream) {
  if (deviceId >= 0 && deviceId < MAX_DEVICES) {
    std::lock_guard<std::mutex> lock(dirtyStreamsMutex_[deviceId]);
    dirtyFreeStreams_[deviceId].erase(stream);
  }
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
  cudaError_t getDevErr = cudaGetDevice(&prevDevice);
  if (getDevErr != cudaSuccess) {
    cudaGetLastError();
    return;
  }
  if (prevDevice != deviceId) {
    cudaError_t setDevErr = cudaSetDevice(deviceId);
    if (setDevErr != cudaSuccess) {
      cudaGetLastError();
      return;
    }
  }

  // Sync only the streams that have had cudaFreeAsync issued on them.
  // free() records each stream into dirtyFreeStreams_[deviceId].
  // We drain that set here so only streams with pending frees are synced,
  // leaving unrelated compute work on other streams unblocked.
  {
    std::vector<cudaStream_t> streamsToSync;
    {
      std::lock_guard<std::mutex> lock(dirtyStreamsMutex_[deviceId]);
      streamsToSync.assign(dirtyFreeStreams_[deviceId].begin(),
                           dirtyFreeStreams_[deviceId].end());
      dirtyFreeStreams_[deviceId].clear();
    }
    for (auto s : streamsToSync) {
      cudaError_t err = cudaStreamSynchronize(s);
      if (err != cudaSuccess) {
        // Stream may have been destroyed — CUDA guarantees all ops on a
        // destroyed stream complete before destroy returns, so this is safe.
        cudaGetLastError();  // clear error
      }
    }
  }
  cudaError_t trimErr = cudaMemPoolTrimTo(pools_[deviceId], 0);
  if (trimErr != cudaSuccess) {
    // Clear the sticky error so callers (e.g. cudaStreamCreate) don't pick it up.
    cudaGetLastError();
  }

  if (prevDevice != deviceId) {
    cudaSetDevice(prevDevice);
  }
}

void CudaMemoryPool::trimPoolOnStream(int deviceId, cudaStream_t stream) {
  if (!supported_ || deviceId < 0 || deviceId >= MAX_DEVICES || !poolInitialized_[deviceId]) {
    return;
  }

  int prevDevice = 0;
  cudaError_t getDevErr = cudaGetDevice(&prevDevice);
  if (getDevErr != cudaSuccess) {
    cudaGetLastError();
    return;
  }

  if (prevDevice != deviceId) {
    cudaError_t setDevErr = cudaSetDevice(deviceId);
    if (setDevErr != cudaSuccess) {
      cudaGetLastError();
      return;
    }
  }

  // Diagnostic: pool state BEFORE trim
  size_t preUsed = 0, preReserved = 0;
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &preUsed);
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &preReserved);

  // Sync the provided stream first (caller expects this stream's work to complete).
  if (stream != nullptr) {
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
      cudaGetLastError();  // clear error from stale/destroyed stream
    }
  }

  //  Also drain ALL dirty streams tracked by free(). Without this,
  // cudaFreeAsync calls from previous execution streams (which change each chunk
  // when ContextBuffers reinitializes) remain unsynced. The pool can't reuse their
  // memory for new allocations on stream 0, causing OOM despite having enough total
  // memory. This matches trimPool()'s behavior but is called more frequently
  // (every periodic flush vs only on allocation failure).
  int numDirtySynced = 0;
  {
    std::vector<cudaStream_t> streamsToSync;
    {
      std::lock_guard<std::mutex> lock(dirtyStreamsMutex_[deviceId]);
      streamsToSync.assign(dirtyFreeStreams_[deviceId].begin(),
                           dirtyFreeStreams_[deviceId].end());
      dirtyFreeStreams_[deviceId].clear();
    }
    for (auto s : streamsToSync) {
      // Only skip if the stream was actually synced above (non-null).
      // When stream==nullptr, the initial sync block is skipped, so we must
      // still sync nullptr entries here (default stream frees from destructors).
      if (s == stream && stream != nullptr) continue;
      cudaError_t err = cudaStreamSynchronize(s);
      if (err != cudaSuccess) {
        cudaGetLastError();  // clear error from stale/destroyed stream
      }
      numDirtySynced++;
    }
  }

  // Diagnostic: pool state AFTER sync, BEFORE trim
  size_t postSyncUsed = 0, postSyncReserved = 0;
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &postSyncUsed);
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &postSyncReserved);

  // Trim to release unused reserved memory back to the device
  cudaError_t trimErr = cudaMemPoolTrimTo(pools_[deviceId], 0);
  if (trimErr != cudaSuccess) {
    cudaGetLastError();  // clear sticky error
  }

  // Diagnostic: pool state AFTER trim
  size_t postTrimUsed = 0, postTrimReserved = 0;
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &postTrimUsed);
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &postTrimReserved);

  DSP_DIAG(MEMORY, "trimPoolOnStream(dev=%d stream=%p): "
            "dirtyStreams=%d | "
            "pre: used=%zu MB reserved=%zu MB | "
            "postSync: used=%zu MB (freed %zu MB) | "
            "postTrim: used=%zu MB reserved=%zu MB (released %zu MB)",
            deviceId, (void*)stream,
            numDirtySynced,
            preUsed/(1024*1024), preReserved/(1024*1024),
            postSyncUsed/(1024*1024),
            (preUsed > postSyncUsed) ? (preUsed - postSyncUsed)/(1024*1024) : (size_t)0,
            postTrimUsed/(1024*1024), postTrimReserved/(1024*1024),
            (postSyncReserved > postTrimReserved) ? (postSyncReserved - postTrimReserved)/(1024*1024) : (size_t)0);

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

  // SAFETY: Wrap cleanup in try-catch to prevent destructor crashes
  try {
    // Free and clear fallback pinned host allocations
    {
      std::lock_guard<std::mutex> lock(fallbackAllocMutex_);

      // Create a copy of pointers to avoid iterator invalidation
      std::vector<void*> pointersToFree;
      pointersToFree.reserve(hostAllocations_.size());

      for (const auto& pair : hostAllocations_) {
        if (pair.first != nullptr) {
          pointersToFree.push_back(pair.first);
        }
      }

      if (sd::Environment::getInstance().isDebug() || sd::Environment::getInstance().isVerbose()) {
        sd_printf("CudaMemoryPool::releaseAll: Freeing %zu pinned host allocations\n", pointersToFree.size());
      }

      // Clear the map before freeing
      hostAllocations_.clear();
      pinnedHostBytesUsed_.store(0);

      // Now free the pointers
      for (void* ptr : pointersToFree) {
        cudaFreeHost(ptr);
      }
    }

    for (int i = 0; i < MAX_DEVICES; i++) {
      if (poolInitialized_[i]) {
        trimPool(i);
        poolInitialized_[i] = false;
      }
    }
  } catch (...) {
    if (sd::Environment::getInstance().isDebug() || sd::Environment::getInstance().isVerbose()) {
      sd_debug("CudaMemoryPool::releaseAll: Exception during cleanup - possible heap corruption\n", "");
    }
  }
}

}  // namespace memory
}  // namespace sd
