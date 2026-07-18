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
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace sd {
namespace memory {

SD_INLINE cudaStream_t resolveCaptureStream(cudaStream_t stream) {
  if (tl_graphExecutionActive && stream == nullptr && tl_graphCaptureStream != nullptr) {
    return reinterpret_cast<cudaStream_t>(tl_graphCaptureStream);
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
  if (tl_dspExecutionStream != nullptr) return reinterpret_cast<cudaStream_t>(tl_dspExecutionStream);
  if (!tl_resolvingContextStream) {
    tl_resolvingContextStream = true;
    auto* ctxStream = sd::LaunchContext::defaultContext()->getCudaStream();
    tl_resolvingContextStream = false;
    if (ctxStream != nullptr) return *ctxStream;
  }
  return nullptr;  // Last resort — first-time context init
}

CudaMemoryPool& CudaMemoryPool::getInstance() {
  static CudaMemoryPool* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new CudaMemoryPool();
  });
  return *instance;
}

void CudaMemoryPool::setMemoryPressureCallback(MemoryPressureCallback callback) {
  std::lock_guard<std::mutex> lock(callbackMutex_);
  memoryPressureCallback_ = callback;
}

void CudaMemoryPool::setSoftLimitPercent(int percent) {
  if (percent < 0) percent = 0;
  if (percent > 100) percent = 100;
  softLimitPercent_.store(percent, std::memory_order_relaxed);
  if (sd::Environment::getInstance().isVerbose()) {
    sd_printf("CudaMemoryPool: Soft limit set to %d%% (0=disabled)\n", percent);
  }
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
    if (sd::Environment::getInstance().isVerbose()) {
      sd_printf("CudaMemoryPool: Pinned host memory limit: %zu bytes (%.1f GB)\n",
                limit, limit / (1024.0 * 1024.0 * 1024.0));
    }

    // Initialize proactive soft-limit from Environment.
    // Configurable via SD_CUDA_SOFT_LIMIT_PERCENT env var or
    // Environment::setCudaSoftLimitPercent() from Java. Default is 0 (disabled).
    // Set to e.g. 70 to proactively route allocations to other devices when
    // GPU usage exceeds 70%, preventing cumulative exhaustion from many small
    // allocations (DSP warmup intermediates) before the watchdog kills at 92%.
    int softLimit = Environment::getInstance().cudaSoftLimitPercent();
    if (softLimit > 0 && softLimit <= 100) {
      softLimitPercent_.store(softLimit, std::memory_order_relaxed);
      if (sd::Environment::getInstance().isVerbose()) {
        sd_printf("CudaMemoryPool: Proactive soft limit: %d%% (from Environment)\n", softLimit);
      }
    }

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
        cudaError_t setDeviceErr = cudaSetDevice(i);
        if (setDeviceErr != cudaSuccess) {
          cudaGetLastError();
          continue;
        }
        cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
        if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
          peerAccessEnabled_[i][j] = true;
          sd_debug("CudaMemoryPool: Enabled peer access from device %d to device %d\n", i, j);
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
  size_t devFree = 0, devTotal = 0;
  cudaMemGetInfo(&devFree, &devTotal);
  int releasePercent = Environment::getInstance().memory().poolReleaseThresholdPercent();
  uint64_t threshold = static_cast<uint64_t>(devTotal * (releasePercent / 100.0));
  err = cudaMemPoolSetAttribute(pools_[deviceId], cudaMemPoolAttrReleaseThreshold, &threshold);
  if (err != cudaSuccess) {
    sd_debug("Warning: Could not set pool release threshold: %s\n", cudaGetErrorString(err), "");
  } else {
    sd_debug("CudaMemoryPool: Device %d pool release threshold set to %zu MB (%d%% of %zu MB total)\n",
              deviceId, threshold / (1024*1024), releasePercent, devTotal / (1024*1024));
  }

  poolInitialized_[deviceId] = true;
  sd_debug("CUDA Memory Pool initialized for device %d\n", deviceId, "");
  restoreDevice();

  return true;
}


void* CudaMemoryPool::allocate(size_t size, int deviceId, cudaStream_t stream, int* actualDeviceId) {
  // After releaseAll(), the pool is torn down. Return nullptr.
  if (released_.load(std::memory_order_acquire)) {
    return nullptr;
  }

  // Reap event-deferred direct frees whose consumers completed (task #57).
  // The atomic gate makes this a no-op load in the common (empty) case.
  if (deferredFreeCount_.load(std::memory_order_acquire) != 0) {
    drainDeferredDirectFrees();
  }

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
    // Workspace exhausted — THROW so the caller gets a stack trace.
    // Falling through to cudaMallocAsync during graph capture corrupts the capture
    // stream (error 901 / invalid argument), making the entire capture invalid.
    // Previously returned nullptr (silent failure) — the caller got a null buffer
    // and crashed with a misleading KERNEL_FAILURE. Now we throw with actionable
    // diagnostic info so the user can increase the workspace size.
    {
      std::string errMsg =
          "CudaMemoryPool: CAPTURE WORKSPACE OOM — workspace exhausted during CUDA graph capture.\n"
          "  workspace_used=" + std::to_string(tl_captureWorkspaceOffset / (1024*1024)) + "MB"
          "  requested=" + std::to_string(aligned / (1024*1024)) + "MB"
          "  workspace_total=" + std::to_string(tl_captureWorkspaceSize / (1024*1024)) + "MB"
          "  (used+requested=" + std::to_string((tl_captureWorkspaceOffset + aligned) / (1024*1024)) + "MB"
          " > total=" + std::to_string(tl_captureWorkspaceSize / (1024*1024)) + "MB)\n"
          "  FIX: Increase capture workspace via Environment property:\n"
          "    -Dnd4j.dsp.captureWorkspaceMb=512  (or higher)\n"
          "  or env var: ND4J_DSP_CAPTURE_WORKSPACE_MB=512";
      THROW_EXCEPTION(errMsg.c_str());
    }
  }
  if (tl_graphExecutionActive && tl_captureWorkspace == nullptr) {
    // This should be unreachable: the capture code in NativeDynamicShapePlan_gpubackend.cpp
    // aborts capture if workspace allocation fails. If we get here, it means
    // capture is active without a workspace — this will corrupt the capture stream.
    // Fail loudly per design: capture failures must not be silently recovered from.
    std::string errMsg = "CudaMemoryPool::allocate called during CUDA graph capture but NO capture workspace is set! "
                         "size=" + std::to_string(size) + " bytes. "
                         "This indicates a bug in the capture setup — capture should have been aborted "
                         "when workspace allocation failed.";
    THROW_EXCEPTION(errMsg.c_str());
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

  // Resolve the consumer/allocation stream up front — the soft-limit failover
  // below needs it for managed-fallback prefetch ordering (task #57).
  cudaStream_t allocStream = resolveCaptureStream(stream);
  allocStream = resolveNullStream(allocStream);

  // CROSS-DEVICE PLACEMENT: cudaMallocAsync allocates from the mem pool of the STREAM'S
  // device, NOT the current device. The resolved allocStream (tl_dspExecutionStream / the
  // LaunchContext stream) is bound to the PRIMARY DSP device (0). For a secondary-device
  // allocation (multi-GPU op-segment sharding) using it silently places the buffer on device
  // 0 even though we cudaSetDevice(deviceId) above — and a later device-`deviceId` H2D onto
  // that device-0 buffer then fails with cudaErrorInvalidValue. We are already on deviceId
  // here, so use its per-thread stream to guarantee the allocation lands in device
  // `deviceId`'s VRAM. Device 0 keeps the resolved DSP stream (single-GPU hot path untouched).
  if (deviceId != 0) {
    allocStream = cudaStreamPerThread;
  }

  // ─── Proactive soft-limit check ───────────────────────────────────────────
  // When enabled, check device usage BEFORE attempting local allocation.
  // If the device is above the soft-limit threshold, route to allocateFailover()
  // proactively. This prevents cumulative exhaustion from many small allocations
  // (e.g., DSP slot-by-slot warmup intermediates) that individually succeed
  // but collectively fill the GPU, racing past the watchdog's kill threshold.
  //
  // Skipped during CUDA graph capture (tl_graphExecutionActive) because
  // failover is impossible during capture — stream sync would break it.
  int softLimit = softLimitPercent_.load(std::memory_order_relaxed);
  if (softLimit > 0 && !tl_graphExecutionActive) {
    size_t freeMem = 0, totalMem = 0;
    cudaGetLastError();  // clear any sticky error before querying
    cudaError_t memInfoErr = cudaMemGetInfo(&freeMem, &totalMem);
    if (memInfoErr == cudaSuccess && totalMem > 0) {
      double usagePercent = 100.0 * (1.0 - static_cast<double>(freeMem) / static_cast<double>(totalMem));
      if (usagePercent >= static_cast<double>(softLimit)) {
        if (sd::Environment::getInstance().isVerbose()) {
          sd_printf("CudaMemoryPool::allocate: Proactive failover — device %d at %.1f%% usage "
                    "(soft limit %d%%), routing %zu bytes to another device\n",
                    deviceId, usagePercent, softLimit, size);
        }
        // skipSameDeviceRetry=true: the whole point of the soft limit is to route
        // to another device. If we let Step 1 trim-and-retry on the same device,
        // it succeeds (pool's own usage is low — the high cudaMemGetInfo reading
        // is from other processes), keeping all allocations on device 0 while
        // device 1 sits idle with gigabytes free.
        auto result = allocateFailover(size, deviceId, actualDeviceId,
                                       /*skipSameDeviceRetry=*/true, allocStream);
        if (result != nullptr) {
          restoreDevice();
          return result;
        }
        // All other devices also full — fall through to attempt local allocation.
        // This is the correct behavior: better to try and potentially succeed
        // (the soft limit is conservative) than to fail without trying.
        if (sd::Environment::getInstance().isVerbose()) {
          sd_printf("CudaMemoryPool::allocate: Proactive failover found no alternative, "
                    "attempting local allocation on device %d\n", deviceId);
        }
      }
    }
  }

  // (allocStream resolved above, before the soft-limit block.)

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
    // Pools disabled/unsupported: use cudaMallocAsync on the resolved stream (the
    // device's default mempool). No raw cudaMalloc — all allocation stays pool-routed.
    cudaError_t err = cudaMallocAsync(&ptr, size, allocStream);
    if (err != cudaSuccess) {
      sd_debug("cudaMallocAsync (pools-off path) failed: %s\n", cudaGetErrorString(err), "");
      auto result = allocateFailover(size, deviceId, actualDeviceId,
                                   /*skipSameDeviceRetry=*/false, allocStream);
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
      // initializeForDevice failed: use cudaMallocAsync on the resolved stream
      // (device default mempool) rather than a raw cudaMalloc.
      cudaError_t err = cudaMallocAsync(&ptr, size, allocStream);
      if (err != cudaSuccess) {
        sd_debug("cudaMallocAsync (uninit-device path) failed: %s\n", cudaGetErrorString(err), "");
        auto result = allocateFailover(size, deviceId, actualDeviceId,
                                   /*skipSameDeviceRetry=*/false, allocStream);
        restoreDevice();
        return result;
      }

      restoreDevice();
      return ptr;
    }
  }

  // ─── Per-process device budget (Environment::maxDeviceMemory) ─────────────
  // When a bound is set (>0), cap THIS process's device pool usage: if serving
  // `size` locally would push pool-used past the budget, route to
  // allocateFailover (host-resident managed / peer — safe post-#57) instead of
  // growing device reservation. Uses the pool's OWN usage (cudaMemPoolAttr-
  // UsedMemCurrent via getStats), NOT cudaMemGetInfo, so co-tenant processes on
  // the same GPU don't distort the accounting — this is the bound a scheduler
  // packs against (kompile embedding lane: "this process gets N GB of the GPU").
  // Skipped during capture (failover would break it) and when unbounded
  // (-1 default: a single atomic load, zero behavioral change for every existing
  // caller — the whole regression surface is off unless a bound is explicitly set
  // via Nd4j.getEnvironment().setMaxDeviceMemory() or SD_MAX_DEVICE_BYTES).
  const int64_t deviceBudget = Environment::getInstance().maxDeviceMemory();
  if (deviceBudget > 0 && !tl_graphExecutionActive && poolInitialized_[deviceId]) {
    size_t poolUsed = 0, poolReserved = 0;
    getStats(deviceId, poolUsed, poolReserved);
    if (poolUsed + size > static_cast<size_t>(deviceBudget)) {
      DSP_DIAG(MEMORY,
               "DEVICE_BUDGET_EXCEEDED: dev=%d poolUsed=%zu + req=%zu > budget=%lld — failover to host/peer",
               deviceId, poolUsed, size, static_cast<long long>(deviceBudget));
      auto result = allocateFailover(size, deviceId, actualDeviceId,
                                     /*skipSameDeviceRetry=*/true, allocStream);
      if (result != nullptr) {
        restoreDevice();
        return result;
      }
      // Failover exhausted every device + host — fall through to a local attempt
      // rather than hard-failing (the caller still handles a null return). A bound
      // that cannot be met even with host spill is a genuine, unavoidable OOM.
      DSP_DIAG(MEMORY, "DEVICE_BUDGET_EXCEEDED: failover found no home — attempting local (device %d)",
               deviceId);
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

    // CROSS-DEVICE STREAM FIX: the resolved allocStream (e.g. tl_dspExecutionStream, the DSP
    // execution stream) can belong to a DIFFERENT device than `deviceId`. During multi-GPU
    // op-segment sharding a secondary-device segment allocates its constants while that device
    // is current, but tl_dspExecutionStream is still device 0's stream — and cudaMallocAsync
    // requires a stream on the allocating device, so it fails. Retry on the target device's OWN
    // per-thread stream (valid: we are on `deviceId` here) BEFORE spilling to managed host
    // memory, so the constant lands in device `deviceId`'s VRAM (where an async H2D is valid)
    // instead of host-resident managed pages (which force slow UVA faults for every device op).
    {
      cudaGetLastError();
      ptr = nullptr;
      cudaError_t perThreadErr = cudaMallocAsync(&ptr, size, cudaStreamPerThread);
      if (perThreadErr == cudaSuccess && ptr != nullptr) {
        restoreDevice();
        return ptr;
      }
      cudaGetLastError();
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
    auto result = allocateFailover(size, deviceId, actualDeviceId,
                                   /*skipSameDeviceRetry=*/false, allocStream);
    restoreDevice();
    return result;
  }

  // Pool allocation succeeded - no tracking needed, will use cudaFreeAsync
  restoreDevice();
  return ptr;
}

void* CudaMemoryPool::allocateFailover(size_t size, int currentDeviceId, int* actualDeviceId,
                                       bool skipSameDeviceRetry, cudaStream_t consumerStream) {
  const char* failoverReason = skipSameDeviceRetry ? "proactive-or-budget" : "primary-allocation-failed";
  DSP_DIAG(MEMORY,
           "ALLOCATE_FAILOVER_BEGIN: reason=%s currentDevice=%d requestedBytes=%zu skipSameDeviceRetry=%d consumerStream=%p",
           failoverReason, currentDeviceId, size, (int)skipSameDeviceRetry, (void*)consumerStream);
  if (DSP_DIAG_ENABLED(MEMORY) || sd::Environment::getInstance().isVerbose()) {
    sd_printf("CudaMemoryPool::allocateFailover: BEGIN reason=%s currentDevice=%d requested=%zu bytes "
              "skipSameDeviceRetry=%d consumerStream=%p\n",
              failoverReason, currentDeviceId, size, (int)skipSameDeviceRetry, (void*)consumerStream);
  } else {
    sd_debug("CudaMemoryPool::allocateFailover: %s on device %d for %zu bytes\n",
             skipSameDeviceRetry ? "Proactive soft-limit failover" : "Primary allocation failed",
             currentDeviceId, size);
  }

  // OOM path: reap any completed deferred direct frees first — their memory may
  // be exactly what this allocation needs (task #57).
  drainDeferredDirectFrees();

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
  if (DSP_DIAG_ENABLED(MEMORY)) {
    size_t poolUsed = 0, poolReserved = 0;
    getStats(currentDeviceId, poolUsed, poolReserved);

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
  }

  if (prevDev != currentDeviceId) {
    cudaSetDevice(prevDev);
  }

  // Step 1: Trim pool and retry on the SAME device.
  // trimPool() syncs only streams with pending cudaFreeAsync (tracked in dirtyFreeStreams_),
  // then releases pool-reserved memory back to the driver.
  //
  // SKIP when skipSameDeviceRetry is true (proactive soft-limit path). The soft limit's
  // purpose is to route allocations to ANOTHER device when cudaMemGetInfo reports high
  // overall GPU usage. But trim-and-retry succeeds when the pool's own usage is low
  // (the high usage is from other CUDA processes sharing the GPU). Succeeding here
  // keeps all allocations on the overloaded device, defeating the soft limit entirely.
  if (!skipSameDeviceRetry && supported_ && poolInitialized_[currentDeviceId]) {
    trimPool(currentDeviceId);

    // Log post-trim state to see how much memory was actually recovered
    if (DSP_DIAG_ENABLED(MEMORY)) {
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
    }

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
      sd_debug("CudaMemoryPool::allocateFailover: Succeeded via pool after trim on device %d\n", currentDeviceId);
      if (actualDeviceId) *actualDeviceId = currentDeviceId;
      if (retryNeedRestore) cudaSetDevice(retryPrevDev);
      return ptr;
    }
    cudaGetLastError();  // clear error

    // No raw cudaMalloc fallback here: the cudaMallocAsync above already reuses the
    // driver memory released by trimPool(). If it still failed, fall through to other
    // devices rather than introducing a non-pool (cudaMalloc) allocation that bypasses
    // pool stats and needs special cudaFree routing.
    if (retryNeedRestore) cudaSetDevice(retryPrevDev);
    cudaGetLastError();  // clear error
  }

  // Step 2: Try ALL other GPU devices with free memory.
  // IMPORTANT: Do NOT skip non-peer devices. On multi-GPU systems without NVLink
  // (e.g. RTX 3070 Ti + RTX 4090 on separate PCIe), peer access is unavailable
  // but cudaMallocManaged provides transparent UVA page migration that works
  // correctly. Blocking non-peer devices causes unnecessary OOM crashes when
  // there are gigabytes of free memory on other GPUs.
  // Peer devices: use cudaMallocAsync/cudaMalloc (fastest, direct P2P).
  // Non-peer devices: use cudaMallocManaged (UVA page migration, still GPU-speed
  //   after first touch — far better than pinned host memory).
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  struct DeviceInfo { int id; size_t freeMem; bool isPeer; };
  std::vector<DeviceInfo> candidates;
  for (int d = 0; d < deviceCount && d < MAX_DEVICES; d++) {
    if (d == currentDeviceId) {
      DSP_DIAG(MEMORY, "ALLOCATE_FAILOVER_CANDIDATE_SKIP: device=%d reason=current-device", d);
      continue;
    }
    if (isDeviceExcludedFromFailover(d)) {
      DSP_DIAG(MEMORY, "ALLOCATE_FAILOVER_CANDIDATE_SKIP: device=%d reason=excluded", d);
      continue;
    }

    bool isPeer = peerAccessEnabled_[currentDeviceId][d];
    cudaError_t setDeviceErr = cudaSetDevice(d);
    if (setDeviceErr != cudaSuccess) {
      DSP_DIAG(MEMORY,
               "ALLOCATE_FAILOVER_CANDIDATE_SKIP: device=%d reason=cudaSetDevice-failed error=%d message=%s",
               d, (int)setDeviceErr, cudaGetErrorString(setDeviceErr));
      cudaGetLastError();
      continue;
    }

    // Trim this device's pool before checking free memory. Without trimming,
    // cudaMemGetInfo reports free memory MINUS pool reserved, which can be
    // much lower than actual available memory.
    if (supported_ && poolInitialized_[d]) {
      trimPool(d);
    }

    size_t freeMem = 0, totalMem = 0;
    cudaGetLastError();
    cudaError_t infoErr = cudaMemGetInfo(&freeMem, &totalMem);
    if (infoErr != cudaSuccess) {
      DSP_DIAG(MEMORY,
               "ALLOCATE_FAILOVER_CANDIDATE_SKIP: device=%d reason=cudaMemGetInfo-failed error=%d message=%s",
               d, (int)infoErr, cudaGetErrorString(infoErr));
      cudaGetLastError();
      continue;
    }
    if (freeMem > size * 1.1) {  // 10% margin
      candidates.push_back({d, freeMem, isPeer});
    } else {
      DSP_DIAG(MEMORY,
               "ALLOCATE_FAILOVER_CANDIDATE_SKIP: device=%d reason=insufficient-free freeMB=%zu totalMB=%zu requestedBytes=%zu peer=%d",
               d, freeMem / (1024*1024), totalMem / (1024*1024), size, (int)isPeer);
    }
  }
  cudaSetDevice(prevDev);

  // Sort: peer devices first (faster access), then by free memory descending
  std::sort(candidates.begin(), candidates.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
    if (a.isPeer != b.isPeer) return a.isPeer > b.isPeer;  // peers first
    return a.freeMem > b.freeMem;
  });
  DSP_DIAG(MEMORY,
           "ALLOCATE_FAILOVER_CANDIDATES: reason=%s currentDevice=%d deviceCount=%d candidateCount=%zu requestedBytes=%zu",
           failoverReason, currentDeviceId, deviceCount, candidates.size(), size);
  if (DSP_DIAG_ENABLED(MEMORY) || sd::Environment::getInstance().isVerbose()) {
    sd_printf("CudaMemoryPool::allocateFailover: candidates reason=%s currentDevice=%d "
              "deviceCount=%d candidateCount=%zu requested=%zu bytes\n",
              failoverReason, currentDeviceId, deviceCount, candidates.size(), size);
  }
  for (size_t rank = 0; rank < candidates.size(); rank++) {
    const auto& c = candidates[rank];
    DSP_DIAG(MEMORY,
             "ALLOCATE_FAILOVER_CANDIDATE: rank=%zu device=%d peer=%d freeMB=%zu requestedBytes=%zu",
             rank, c.id, (int)c.isPeer, c.freeMem / (1024*1024), size);
    if (DSP_DIAG_ENABLED(MEMORY) || sd::Environment::getInstance().isVerbose()) {
      sd_printf("CudaMemoryPool::allocateFailover: candidate rank=%zu device=%d %s free=%zu MB requested=%zu bytes\n",
                rank, c.id, c.isPeer ? "peer" : "non-peer/managed", c.freeMem / (1024*1024), size);
    }
  }

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
        sd_debug("CudaMemoryPool::allocateFailover: Callback rejected allocation on device %d\n", currentDeviceId);
        cudaSetDevice(prevDev);
        return nullptr;
      }
    }
  }

  for (const auto& candidate : candidates) {
    int d = candidate.id;
    bool isPeer = candidate.isPeer;
    sd_debug("CudaMemoryPool::allocateFailover: Trying device %d (%s, free: %zu MB) for %zu bytes\n",
              d, isPeer ? "peer" : "non-peer/managed", candidate.freeMem / (1024*1024), size);
    cudaError_t setDeviceErr = cudaSetDevice(d);
    if (setDeviceErr != cudaSuccess) {
      cudaGetLastError();
      continue;
    }

    if (isPeer) {
      // Peer device: allocate from the async pool. cudaMallocAsync uses device d's
      // default memory pool even when our wrapper pool isn't explicitly initialized
      // for d, so no raw cudaMalloc fallback is needed. If it fails, fall through to
      // the next device.
      void* ptr = nullptr;
      cudaError_t err = cudaMallocAsync(&ptr, size, nullptr);
      if (err == cudaSuccess && ptr != nullptr) {
        sd_debug("CudaMemoryPool::allocateFailover: Succeeded via pool on peer device %d for %zu bytes\n", d, size);
        if (actualDeviceId) *actualDeviceId = d;
        DSP_DIAG(MEMORY,
                 "ALLOCATE_FAILOVER_CHOSEN: reason=%s kind=peer-device requestedDevice=%d actualDevice=%d bytes=%zu freeBeforeMB=%zu",
                 failoverReason, currentDeviceId, d, size, candidate.freeMem / (1024*1024));
        if (DSP_DIAG_ENABLED(MEMORY) || sd::Environment::getInstance().isVerbose()) {
          sd_printf("CudaMemoryPool::allocateFailover: CHOSEN kind=peer-device requestedDevice=%d actualDeviceId=%d "
                    "bytes=%zu freeBefore=%zu MB\n",
                    currentDeviceId, d, size, candidate.freeMem / (1024*1024));
        }
        cudaSetDevice(prevDev);
        return ptr;
      }
      DSP_DIAG(MEMORY,
               "ALLOCATE_FAILOVER_CANDIDATE_FAILED: device=%d kind=peer-device error=%d message=%s bytes=%zu",
               d, (int)err, cudaGetErrorString(err), size);
      cudaGetLastError();
    } else {
      // Non-peer device: use cudaMallocManaged for transparent UVA access.
      // Without P2P/NVLink, currentDeviceId's kernels cannot read device d's
      // memory directly, and migrating pages INTO currentDeviceId cannot work
      // either — this path only runs when currentDeviceId is FULL, so a
      // demand-paging fault has no evictable frames to migrate into (the
      // device is packed with unevictable cudaMallocAsync pool memory). The
      // driver then fails the fault and kills the context with error 700
      // (the bge [32x512] warmup OOM cascade, task #57 / WS-O3).
      // The only residency BOTH devices can stably use is HOST: pin pages
      // there (preferred location CPU) and pre-establish the GPU mapping
      // (SetAccessedBy), so consumer kernels do PCIe reads with NO page
      // faults — pinned-host semantics with managed bookkeeping.
      void* ptr = nullptr;
      cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
      if (err == cudaSuccess && ptr != nullptr) {
        cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
        // Map the host-resident pages into both GPUs' page tables up front so
        // access never takes a fault-and-migrate path.
        cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, currentDeviceId);
        cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, d);
        // Populate host residency ordered on the CONSUMER stream. A
        // default-stream prefetch gives the consuming kernels (DSP/LC exec
        // stream) no ordering against population — the kernel races the
        // residency setup. Enqueue-order on the consumer stream is sufficient
        // (consumers on that stream run strictly after it); NO host sync here
        // (WS-N mandate). Only an ENQUEUE failure is checked — a bad stream /
        // bad range surfaces synchronously from the async call itself.
        cudaError_t prefetchErr = cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, consumerStream);
        if (prefetchErr != cudaSuccess) {
          cudaGetLastError();
          sd_printf("CudaMemoryPool::allocateFailover: prefetch enqueue failed on managed "
                    "fallback (host-resident, for device %d): %s — freeing and trying next candidate\n",
                    currentDeviceId, cudaGetErrorString(prefetchErr));
          cudaFree(ptr);  // fresh allocation, never handed out — no consumers to order against
          continue;
        }
        sd_debug("CudaMemoryPool::allocateFailover: Succeeded via cudaMallocManaged "
                  "(host-resident, mapped for device %d) for %zu bytes\n",
                  currentDeviceId, size);
        registerDirectAllocation(ptr, size);
        if (actualDeviceId) *actualDeviceId = currentDeviceId;
        DSP_DIAG(MEMORY,
                 "ALLOCATE_FAILOVER_CHOSEN: reason=%s kind=managed-host-resident requestedDevice=%d candidateDevice=%d actualDevice=%d bytes=%zu freeBeforeMB=%zu",
                 failoverReason, currentDeviceId, d, currentDeviceId, size, candidate.freeMem / (1024*1024));
        if (DSP_DIAG_ENABLED(MEMORY) || sd::Environment::getInstance().isVerbose()) {
          sd_printf("CudaMemoryPool::allocateFailover: CHOSEN kind=managed-host-resident requestedDevice=%d "
                    "candidateDevice=%d actualDeviceId=%d bytes=%zu freeBefore=%zu MB\n",
                    currentDeviceId, d, currentDeviceId, size, candidate.freeMem / (1024*1024));
        }
        cudaSetDevice(prevDev);
        return ptr;
      }
      DSP_DIAG(MEMORY,
               "ALLOCATE_FAILOVER_CANDIDATE_FAILED: device=%d kind=managed-host-resident error=%d message=%s bytes=%zu",
               d, (int)err, cudaGetErrorString(err), size);
      cudaGetLastError();
    }
  }
  cudaSetDevice(prevDev);

  // Step 3: Fall back to pinned host memory
  // WARNING: Pinned host memory is accessible from GPU via UVA but at PCIe bandwidth.
  // actualDeviceId is set to currentDeviceId because CUDA operations from that device
  // can still access pinned host memory. The hostAllocations_ map tracks these pointers
  // and sizes for correct deallocation via cudaFreeHost.

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
    sd_debug("CudaMemoryPool::allocateFailover: Pinned host fallback succeeded for %zu bytes (ptr=%p, total pinned: %zu)\n", size, ptr, pinnedHostBytesUsed_.load());
    if (actualDeviceId) *actualDeviceId = currentDeviceId;
    DSP_DIAG(MEMORY,
             "ALLOCATE_FAILOVER_CHOSEN: reason=%s kind=pinned-host requestedDevice=%d actualDevice=%d bytes=%zu pinnedUsedMB=%zu hostGroupMB=%zu",
             failoverReason, currentDeviceId, currentDeviceId, size,
             pinnedHostBytesUsed_.load() / (1024*1024), hostGroupUsed / (1024*1024));
    if (DSP_DIAG_ENABLED(MEMORY) || sd::Environment::getInstance().isVerbose()) {
      sd_printf("CudaMemoryPool::allocateFailover: CHOSEN kind=pinned-host requestedDevice=%d actualDeviceId=%d "
                "bytes=%zu pinnedUsed=%zu MB hostGroup=%zu MB\n",
                currentDeviceId, currentDeviceId, size,
                pinnedHostBytesUsed_.load() / (1024*1024), hostGroupUsed / (1024*1024));
    }
    return ptr;
  }

  DSP_DIAG(MEMORY,
           "ALLOCATE_FAILOVER_FAILED: reason=%s currentDevice=%d requestedBytes=%zu cudaHostAllocError=%d message=%s",
           failoverReason, currentDeviceId, size, (int)err, cudaGetErrorString(err));
  sd_debug("CudaMemoryPool::allocateFailover: All allocation attempts failed for %zu bytes\n", size, "");
  return nullptr;
}

void CudaMemoryPool::free(void* ptr, int deviceId, cudaStream_t stream) {
  if (ptr == nullptr) {
    return;
  }

  // After releaseAll() (called from ~CudaMemoryPool during shutdown), all internal
  // maps are cleared. Any free() call at this point would walk freed map nodes →
  // SIGSEGV. This happens when GC finalizer threads race with C++ static destruction.
  // The OS reclaims all GPU memory when the process exits, so skipping is safe.
  if (released_.load(std::memory_order_acquire)) {
    return;
  }

  // Check if this pointer falls within an active capture workspace.
  // Capture workspaces are single cudaMalloc blocks used as bump allocators during
  // CUDA graph capture. Interior pointers (sub-allocations) cannot be freed individually —
  // cudaFreeAsync on them returns "invalid argument", and the cudaFree fallback
  // corrupts the CUDA context. The workspace block is freed as a whole when the
  // replay handle is destroyed or releaseGpuIntermediates() is called.
  // This guard catches frees AFTER capture ends (when tl_graphExecutionActive is false).
  if (isInCaptureWorkspace(ptr)) {
    return;  // No-op — managed by workspace lifecycle
  }

  // Capture-constant arena interior pointer: a process-lifetime shape/TAD constant
  // bump-allocated by allocateFromCaptureArena(). It is never freed individually —
  // the whole arena is released as a block in releaseAll(). A cudaFree(Async) on an
  // arena-interior pointer would return "invalid argument" and leak/corrupt.
  if (isInCaptureArena(ptr)) {
    return;  // No-op — arena freed as a block at releaseAll()
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

  // Check host allocations with exception handling.
  // Two checks: (1) exact match for base pointers, (2) range check for interior
  // pointers (views/sub-allocations into a pinned host block). Interior pointers
  // cannot be freed individually — only the base allocation can be cudaFreeHost'd.
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

      // cudaFreeHost is synchronous and stream-oblivious — same in-flight-consumer
      // hazard as the direct-allocation cudaFree (task #57): a pinned-host failover
      // buffer read by a still-running kernel must not be unmapped under it.
      // NO host sync (WS-N mandate): event-deferred, reaped when the consumer
      // stream's work at free time has provably completed.
      deferDirectFree(ptr, freedSize, /*deviceId=*/-1, /*isHostAlloc=*/true,
                      resolveNullStream(resolveCaptureStream(stream)));
      return;
    }

    // Range check: if ptr is an interior pointer into any pinned host block,
    // skip the free. This happens when NDArray views (subarray, reshape) create
    // DataBuffers that point into the middle of a pinned host allocation.
    // cudaFreeAsync on such pointers returns "invalid argument" and the memory
    // leaks. By detecting them here, we avoid the failed cudaFreeAsync entirely.
    char* p = static_cast<char*>(ptr);
    for (const auto& entry : hostAllocations_) {
      char* base = static_cast<char*>(entry.first);
      size_t sz = entry.second;
      if (p > base && p < base + sz) {
        // Interior pointer — no-op. The base allocation will be freed when
        // the parent DataBuffer is destroyed.
        return;
      }
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

  // CROSS-DEVICE PLACEMENT (mirror of allocate): cudaFreeAsync must run on a stream that
  // belongs to the buffer's device. The resolved freeStream is bound to the primary DSP
  // device (0); freeing a secondary-device buffer on it fails silently / leaks. We are on
  // deviceId here (cudaSetDevice above), so use its per-thread stream. deviceId <= 0 (primary
  // or unknown) keeps the resolved stream so the single-GPU path is untouched.
  if (deviceId > 0) {
    freeStream = cudaStreamPerThread;
  }

  // Persistent capture-safe allocations from allocateDirect() are pool memory
  // (cudaMallocAsync) bound to a dedicated non-capturing allocation stream. Free
  // them with cudaFreeAsync on THAT stream so the alloc/free ordering stays on the
  // same stream and no graph mem-node is recorded. Checked before the direct/cudaFree
  // path below.
  {
    std::lock_guard<std::mutex> lock(directAllocMutex_);
    auto asyncIt = directAsyncAllocations_.find(ptr);
    if (asyncIt != directAsyncAllocations_.end()) {
      int dev = asyncIt->second.deviceId;
      directAsyncAllocations_.erase(asyncIt);
      cudaStream_t s = (dev >= 0 && dev < MAX_DEVICES) ? directAllocStreams_[dev] : nullptr;
      cudaError_t err = cudaFreeAsync(ptr, s);
      if (err != cudaSuccess) {
        // Clear and leak rather than risk a context-corrupting cudaFree on pool memory.
        cudaGetLastError();
        if (sd::Environment::getInstance().isDebug()) {
          sd_printf("CudaMemoryPool::free: cudaFreeAsync failed for allocateDirect ptr=%p dev=%d: %s (leaked)\n",
                    ptr, dev, cudaGetErrorString(err));
        }
      }
      if (needDeviceRestore) cudaSetDevice(savedDev);
      return;
    }
  }

  // Check if this is a direct (cudaMalloc/cudaMallocManaged) allocation — these
  // MUST use cudaFree, NOT cudaFreeAsync. Direct allocations are weight buffers
  // migrated out of the async pool, and OOM-failover fallbacks (managed/host-
  // resident) from allocateFailover. Using cudaFreeAsync on a cudaMalloc pointer
  // returns "invalid argument" and leaks the memory.
  {
    bool isDirect = false;
    size_t freedSize = 0;
    {
      std::lock_guard<std::mutex> lock(directAllocMutex_);
      auto directIt = directAllocations_.find(ptr);
      if (directIt != directAllocations_.end()) {
        isDirect = true;
        freedSize = directIt->second;
        directAllocations_.erase(directIt);
      }
    }
    if (isDirect) {
      // cudaFree is SYNCHRONOUS and ignores stream ordering: freeing a managed
      // failover buffer unmaps its UVM range IMMEDIATELY, even while an enqueued
      // kernel that reads it is still in flight (async cuBLAS GEMMs inside
      // dot_product_attention_v2 deliberately don't sync). Pool buffers survive
      // this dtor-after-enqueue pattern because cudaFreeAsync is stream-ordered
      // behind the consumer — direct allocations got no such protection, so the
      // first OOM-failover temp consumed by an async GEMM died with a driver-
      // level fault: error 700 poisoned the context (bge [32x512] warmup OOM
      // cascade, task #57). NO host sync here (WS-N mandate): defer the free
      // behind an event recorded on the consumer stream; it is reaped by
      // drainDeferredDirectFrees() once cudaEventQuery reports completion.
      deferDirectFree(ptr, freedSize, deviceId, /*isHostAlloc=*/false, freeStream);
      if (needDeviceRestore) cudaSetDevice(savedDev);
      return;
    }
  }

  // Graph-baked address protection: this pointer is still referenced by a live
  // segment (baked into a CUDA graph's captured nodes, or cached in a frozen
  // slot-by-slot slot). Freeing it now would allow pool reuse at the same address
  // while the segment is live — causing data corruption or CUDA err700 on the next
  // replay/re-exec. DEFER the free and RECORD that the owner requested it; the free
  // is issued by unpinGraphBakedAddress() at refCount==0 ONLY because freeRequested
  // is set here. A pinned buffer that is never free()'d (a SameDiff weight/constant
  // that outlives the plan) is thus NOT freed at unpin — it is externally owned.
  {
    std::lock_guard<std::mutex> lock(graphBakedMutex_);
    auto it = graphBakedPins_.find(ptr);
    if (it != graphBakedPins_.end()) {
      it->second.freeRequested = true;
      DSP_DIAG(MEMORY,
               "GRAPH_PIN defer-free ptr=%p dev=%d refCount=%d freeRequested=1",
               ptr, deviceId, it->second.refCount);
      if (needDeviceRestore) cudaSetDevice(savedDev);
      return;
    }
  }

  // Device memory: use cudaFreeAsync for stream-ordered deallocation.
  // Works for both pool and non-pool allocations since CUDA 11.2.
  if (enabled_.load() && supported_) {
    // CAPTURE-SAFE DEFER: if a CUDA graph capture is ACTIVELY in progress on freeStream,
    // issuing cudaFreeAsync now corrupts it. For a capture-time (workspace-interior) buffer
    // the call fails "invalid argument", and any cudaFreeAsync that references a pointer not
    // owned by this capture's mempool poisons the capture — so the subsequent
    // cudaStreamEndCapture fails ("operation failed due to a previous error during capture")
    // and the segment reports KERNEL_FAILURE (status=50). The tl_captureWorkspace skip above
    // misses the window after unregisterCaptureWorkspace() while the capture is still open, so
    // detect the live capture state directly here (cudaStreamIsCapturing is a pure query — no
    // sync, capture-safe). The buffer is owned by the capture workspace and released as a unit
    // after the capture completes, so deferring its individual free is correct and mirrors the
    // tl_captureWorkspace skip above.
    cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
    cudaError_t capErr = cudaStreamIsCapturing(freeStream, &capStatus);
    if (capErr == cudaSuccess && capStatus != cudaStreamCaptureStatusNone) {
      if (sd::Environment::getInstance().isDebug()) {
        sd_printf("CudaMemoryPool::free: DEFER ptr=%p dev=%d stream=%p — CUDA capture active "
                  "(status=%d); individual free owned by capture workspace, released after capture\n",
                  ptr, deviceId, (void*)freeStream, (int)capStatus);
      }
      if (needDeviceRestore) cudaSetDevice(savedDev);
      return;
    }
    if (capErr != cudaSuccess) cudaGetLastError();  // clear benign capture-query error
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
    // Log failed cudaFreeAsync for debugging — this path means memory is LEAKED.
    //  Do NOT call cudaFree() as fallback. cudaFreeAsync failures typically
    // mean the pointer was allocated by cudaMallocAsync on a different stream or during
    // graph capture. Calling cudaFree() on such a pointer corrupts the CUDA context
    // permanently (error 700 "illegal memory access" on all subsequent CUDA calls).
    // Leaking the memory is vastly preferable to corrupting the entire GPU context.
    static int cudaFreeAsyncFailCount = 0;
    if (cudaFreeAsyncFailCount < 10) {
      sd_printf("CudaMemoryPool::free: cudaFreeAsync FAILED for ptr=%p dev=%d stream=%p: %s (LEAKED, no cudaFree fallback)\n",
                ptr, deviceId, (void*)freeStream, cudaGetErrorString(err));
      // Check if this is an interior pointer within a pinned host allocation
      {
        std::lock_guard<std::mutex> lock2(fallbackAllocMutex_);
        char* p = static_cast<char*>(ptr);
        for (const auto& entry : hostAllocations_) {
          char* base = static_cast<char*>(entry.first);
          size_t sz = entry.second;
          if (p >= base && p < base + sz) {
            sd_printf("CudaMemoryPool::free: ptr=%p is INTERIOR POINTER into pinned host block base=%p size=%zu (offset=%lld). "
                      "This DataBuffer is a sub-allocation that was never independently tracked.\n",
                      ptr, (void*)base, sz, (long long)(p - base));
            break;
          }
        }
      }
      cudaFreeAsyncFailCount++;
      if (cudaFreeAsyncFailCount == 10) {
        sd_printf("CudaMemoryPool::free: suppressing further cudaFreeAsync failure messages\n", "");
      }
    }
    cudaGetLastError();  // clear error
    if (needDeviceRestore) cudaSetDevice(savedDev);
    return;
  }
  // Fallback for pools not supported — use synchronous cudaFree.
  // This path is only reached when CudaMemoryPool is disabled (supported_ == false
  // or enabled_ == false). The pointer was allocated via cudaMalloc (not cudaMallocAsync),
  // so cudaFree is the correct deallocator.
  cudaFree(ptr);
  if (needDeviceRestore) cudaSetDevice(savedDev);
}

void CudaMemoryPool::removeDirtyStream(int deviceId, cudaStream_t stream) {
  if (deviceId >= 0 && deviceId < MAX_DEVICES) {
    std::lock_guard<std::mutex> lock(dirtyStreamsMutex_[deviceId]);
    dirtyFreeStreams_[deviceId].erase(stream);
  }
}

// ─── Pinned Host Memory Management ─────────────────────────────────────

void* CudaMemoryPool::allocatePinnedHost(size_t size) {
  if (released_.load(std::memory_order_acquire)) {
    return nullptr;
  }
  // Enforce limit if set
  size_t limit = pinnedHostBytesLimit_.load();
  size_t currentUsage = pinnedHostBytesUsed_.load();
  if (limit > 0 && currentUsage + size > limit) {
    // Check if we can free some old allocations first (LRU not implemented — just reject)
    return nullptr;
  }

  void* ptr = nullptr;
  cudaError_t err = cudaMallocHost(&ptr, size);
  if (err != cudaSuccess || ptr == nullptr) {
    return nullptr;
  }

  // Track this allocation
  std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
  hostAllocations_[ptr] = size;
  pinnedHostBytesUsed_.fetch_add(size);
  return ptr;
}

bool CudaMemoryPool::freePinnedHost(void* ptr) {
  if (ptr == nullptr) return true;

  std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
  auto it = hostAllocations_.find(ptr);
  if (it != hostAllocations_.end()) {
    size_t freedSize = it->second;
    // Lifecycle log: pinned blocks can be BAKED as H2D sources inside captured
    // CUDA graphs (Triton arg tables). Freeing one while such a graph is live
    // kills the driver's host registration — the next cudaGraphLaunch of that
    // graph SIGSEGVs host-side. This line + the NODE[] srcHost dump pointer-match
    // the killer to the victim.
    DSP_DIAG(EXECUTE, "CudaMemoryPool::freePinnedHost ptr=%p bytes=%zu", ptr, freedSize);
    hostAllocations_.erase(it);
    pinnedHostBytesUsed_.fetch_sub(freedSize);
    cudaFreeHost(ptr);
    return true;
  }

  // Not tracked — fall back to direct cudaFreeHost.
  // This handles pointers from raw cudaMallocHost calls in legacy code.
  cudaFreeHost(ptr);
  return false;
}

bool CudaMemoryPool::relinquishPinnedHost(void* ptr) {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
  auto it = hostAllocations_.find(ptr);
  if (it == hostAllocations_.end()) return false;
  // Ownership handed to an external holder (e.g. a CUDA graph replay handle's
  // capturedHostPtrs, which cudaFreeHost's at handle death). Drop bookkeeping
  // WITHOUT freeing so a later freePinnedHost of this ptr is a no-op instead of
  // a double free.
  DSP_DIAG(EXECUTE, "CudaMemoryPool::relinquishPinnedHost ptr=%p bytes=%zu (ownership -> external)",
           ptr, it->second);
  pinnedHostBytesUsed_.fetch_sub(it->second);
  hostAllocations_.erase(it);
  return true;
}

bool CudaMemoryPool::isPinnedHostAllocation(void* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
  return hostAllocations_.count(ptr) > 0;
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
        cudaGetLastError();  // clear error
      }
    }
  }
  // Log pool state before trim when verbose/debug
  size_t preUsed = 0, preReserved = 0;
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &preUsed);
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &preReserved);

  cudaError_t trimErr = cudaMemPoolTrimTo(pools_[deviceId], 0);
  if (trimErr != cudaSuccess) {
    // Clear the sticky error so callers (e.g. cudaStreamCreate) don't pick it up.
    cudaGetLastError();
  }

  size_t postUsed = 0, postReserved = 0;
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &postUsed);
  cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &postReserved);

  if (sd::Environment::getInstance().isDebug() || sd::Environment::getInstance().isVerbose()) {
    if (preReserved > 0 || preUsed > 0) {
      sd_printf("CudaMemoryPool::trimPool dev=%d: BEFORE used=%zuMB reserved=%zuMB AFTER used=%zuMB reserved=%zuMB freed=%zuMB\n",
                deviceId, preUsed/(1024*1024), preReserved/(1024*1024),
                postUsed/(1024*1024), postReserved/(1024*1024),
                (preReserved > postReserved ? preReserved - postReserved : 0)/(1024*1024));
    }
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
  if (DSP_DIAG_ENABLED(MEMORY)) {
    cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &preUsed);
    cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &preReserved);
  }

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
  if (DSP_DIAG_ENABLED(MEMORY)) {
    cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &postSyncUsed);
    cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &postSyncReserved);
  }

  // Trim to release unused reserved memory back to the device
  cudaError_t trimErr = cudaMemPoolTrimTo(pools_[deviceId], 0);
  if (trimErr != cudaSuccess) {
    cudaGetLastError();  // clear sticky error
  }

  // Diagnostic: pool state AFTER trim
  size_t postTrimUsed = 0, postTrimReserved = 0;
  if (DSP_DIAG_ENABLED(MEMORY)) {
    cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrUsedMemCurrent, &postTrimUsed);
    cudaMemPoolGetAttribute(pools_[deviceId], cudaMemPoolAttrReservedMemCurrent, &postTrimReserved);
  }

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

void CudaMemoryPool::registerCaptureWorkspace(void* basePtr, size_t bytes) {
  if (basePtr == nullptr || bytes == 0) return;
  std::lock_guard<std::mutex> lock(captureWorkspaceMutex_);
  captureWorkspaceRanges_[basePtr] = bytes;
  sd_debug("CudaMemoryPool: registered capture workspace %p (%zu bytes, %zu total ranges)\n",
           basePtr, bytes, captureWorkspaceRanges_.size());
}

void CudaMemoryPool::unregisterCaptureWorkspace(void* basePtr) {
  if (basePtr == nullptr) return;
  std::lock_guard<std::mutex> lock(captureWorkspaceMutex_);
  auto erased = captureWorkspaceRanges_.erase(basePtr);
  if (erased > 0) {
    sd_debug("CudaMemoryPool: unregistered capture workspace %p (%zu remaining ranges)\n",
             basePtr, captureWorkspaceRanges_.size());
  }
}

// ─── Graph-Baked Address Protection ────────────────────────────────────────

void CudaMemoryPool::pinGraphBakedAddress(void* ptr, int deviceId) {
  if (ptr == nullptr) return;
  std::lock_guard<std::mutex> lock(graphBakedMutex_);
  auto& info = graphBakedPins_[ptr];
  info.refCount++;
  info.deviceId = deviceId;
  DSP_DIAG(MEMORY, "GRAPH_PIN pin ptr=%p dev=%d refCount=%d",
           ptr, deviceId, info.refCount);
}

void CudaMemoryPool::unpinGraphBakedAddress(void* ptr, int deviceId, cudaStream_t stream) {
  if (ptr == nullptr) return;
  bool shouldFree = false;
  bool freeRequested = false;
  int remainingRefCount = -1;
  {
    std::lock_guard<std::mutex> lock(graphBakedMutex_);
    auto it = graphBakedPins_.find(ptr);
    if (it == graphBakedPins_.end()) {
      DSP_DIAG(MEMORY, "GRAPH_PIN unpin-missing ptr=%p dev=%d", ptr, deviceId);
      return;
    }
    it->second.refCount--;
    remainingRefCount = it->second.refCount;
    freeRequested = it->second.freeRequested;
    if (it->second.refCount <= 0) {
      // Free ONLY if the owner actually requested a free() while the buffer was pinned.
      // A buffer that was pinned for protection but never free()'d (a SameDiff weight/
      // constant that outlives the plan) must NOT be freed here — it is externally owned;
      // freeing it would double-free a live weight (→ err700) or fail for a non-pool
      // constant (cudaFreeAsync "invalid argument" → leak).
      shouldFree = freeRequested;
      graphBakedPins_.erase(it);
    }
  }
  DSP_DIAG(MEMORY,
           "GRAPH_PIN unpin ptr=%p dev=%d remainingRefCount=%d freeRequested=%d release=%d",
           ptr, deviceId, remainingRefCount, freeRequested ? 1 : 0, shouldFree ? 1 : 0);
  if (shouldFree) {
    // Deferred-free path: free() was skipped while pinned (freeRequested); now that no
    // live segment holds the address, release it to the pool on the provided stream.
    free(ptr, deviceId, stream);
  }
}

bool CudaMemoryPool::isGraphBakedPinned(void* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lock(graphBakedMutex_);
  return graphBakedPins_.count(ptr) > 0;
}

bool CudaMemoryPool::isInCaptureWorkspace(void* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lock(captureWorkspaceMutex_);
  char* p = static_cast<char*>(ptr);
  for (const auto& entry : captureWorkspaceRanges_) {
    char* wsStart = static_cast<char*>(entry.first);
    size_t wsSize = entry.second;
    if (p >= wsStart && p < wsStart + wsSize) {
      return true;
    }
  }
  return false;
}

cudaStream_t CudaMemoryPool::ensureDirectAllocStream(int deviceId) {
  if (deviceId < 0 || deviceId >= MAX_DEVICES) return nullptr;
  std::lock_guard<std::mutex> lock(directAllocStreamMutex_);
  if (directAllocStreams_[deviceId] == nullptr) {
    int prevDev = -1;
    cudaGetDevice(&prevDev);
    bool restore = (prevDev != deviceId && cudaSetDevice(deviceId) == cudaSuccess);
    cudaStream_t s = nullptr;
    cudaError_t err = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    if (err == cudaSuccess && s != nullptr) {
      directAllocStreams_[deviceId] = s;
    } else {
      cudaGetLastError();
    }
    if (restore) cudaSetDevice(prevDev);
  }
  return directAllocStreams_[deviceId];
}

void* CudaMemoryPool::allocateDirect(size_t size, int deviceId) {
  if (released_.load(std::memory_order_acquire) || size == 0) return nullptr;
  if (deviceId < 0) {
    cudaGetDevice(&deviceId);
  }

  // Dedicated non-capturing stream → cudaMallocAsync produces a standalone pool
  // allocation with NO cudaGraphMemAllocNode, safe to bake as a captured-graph kernel
  // arg and persistent across capture-workspace release / graph teardown.
  cudaStream_t s = ensureDirectAllocStream(deviceId);
  if (s == nullptr) {
    // Cannot guarantee a capture-safe standalone allocation without a dedicated
    // stream — fail rather than risk a workspace-interior or graph-node allocation.
    static int noStreamFailCount = 0;
    if (noStreamFailCount < 15) {
      noStreamFailCount++;
      sd_printf("CudaMemoryPool::allocateDirect: NO dedicated alloc stream for dev=%d size=%zu "
                "graphExecActive=%d (ensureDirectAllocStream returned null)\n",
                deviceId, size, (int)tl_graphExecutionActive);
    }
    return nullptr;
  }

  int prevDev = -1;
  cudaGetDevice(&prevDev);
  bool restore = (prevDev != deviceId && cudaSetDevice(deviceId) == cudaSuccess);

  void* ptr = nullptr;
  cudaError_t err = cudaMallocAsync(&ptr, size, s);
  if (err == cudaSuccess && ptr != nullptr) {
    // Materialize the allocation so the buffer is valid for cross-stream reads by the
    // captured graph at replay. cudaStreamSynchronize is a HOST-side wait — and under a
    // CUDA-graph capture (all beginCapture() calls use cudaStreamCaptureModeThreadLocal)
    // ANY host sync issued ON THE CAPTURING THREAD is illegal, regardless of which stream
    // it targets. allocateDirect is reached on the capturing thread during capture (e.g.
    // ConstantHelper::replicatePointer materializing a TAD-offset/constant for a slice
    // strided-copy segment), so syncing here invalidates the capture → CudaGraphHandle::
    // endCapture reports "previous error during capture" → seg KERNEL_FAILURE/status=50.
    // During the DSP capture/replay region we MUST skip the host sync: the dedicated
    // NON-capturing stream's tiny allocation completes long before the first
    // cudaGraphLaunch, and replicatePointer's H2D fill is already an async cudaMemcpyAsync
    // node on the captured stream (ordered before the reader kernel), so the buffer is
    // valid at replay without a host sync. Outside capture, keep the sync so eager
    // cross-stream callers see an immediately-materialized buffer.
    if (!tl_graphExecutionActive) {
      cudaStreamSynchronize(s);
      cudaGetLastError();
    }
    {
      std::lock_guard<std::mutex> lock(directAllocMutex_);
      directAsyncAllocations_[ptr] = DirectAsyncInfo{size, deviceId};
    }
    if (sd::Environment::getInstance().isDebug()) {
      sd_printf("CudaMemoryPool::allocateDirect: persistent capture-safe alloc ptr=%p size=%zu dev=%d "
                "(non-capturing stream, survives workspace/graph teardown)\n",
                ptr, size, deviceId);
    }
  } else {
    // DIAGNOSTIC (bounded, failure-only): allocateDirect failing mid-capture is the
    // status=50 root for ops that replicate a constant during capture (e.g. slice's output
    // shape buffer via CudaShapeBufferCreator → replicatePointer). Surface the cudaError +
    // capture state so the reason (capture-illegal alloc vs real OOM) is visible WITHOUT
    // global setDebug (which corrupts the capture). Bounded like the cudaFreeAsync-LEAKED log.
    static int allocDirectFailCount = 0;
    if (allocDirectFailCount < 15) {
      allocDirectFailCount++;
      sd_printf("CudaMemoryPool::allocateDirect: cudaMallocAsync FAILED dev=%d size=%zu "
                "graphExecActive=%d err=%d (%s)\n",
                deviceId, size, (int)tl_graphExecutionActive, (int)err, cudaGetErrorString(err));
    }
    cudaGetLastError();
    ptr = nullptr;
  }

  if (restore) cudaSetDevice(prevDev);
  return ptr;
}

void CudaMemoryPool::ensureCaptureArena(int deviceId) {
  if (released_.load(std::memory_order_acquire)) return;
  if (deviceId < 0) cudaGetDevice(&deviceId);
  if (deviceId < 0 || deviceId >= MAX_DEVICES) return;

  const size_t ARENA_BYTES = 64ull * 1024ull * 1024ull;  // 64MB per block ≈ 1800 padded shape buffers

  std::lock_guard<std::mutex> lock(captureArenaMutex_);
  // Headroom check: if the last existing block still has a full ARENA_BYTES of free space,
  // there is sufficient headroom for the upcoming capture — no new block needed.
  // (A single allocation is capped by ARENA_BYTES in allocateFromCaptureArena, so this
  // guarantees at least one arena-sized alloc will succeed without growing mid-capture.)
  if (!captureArenaBlocks_[deviceId].empty()) {
    const ArenaBlock& last = captureArenaBlocks_[deviceId].back();
    size_t freeBytes = last.capacity - last.offset;
    if (freeBytes >= ARENA_BYTES) return;  // last block has sufficient headroom
    // Otherwise fall through to append a fresh block below.
  }

  // A new block can only be allocated when NO capture is ACTIVELY in progress on this
  // thread (cudaMallocAsync on the capturing thread = err900 cudaErrorStreamCaptureUnsupported).
  // tl_graphExecutionActive spans the whole DSP region incl. warmup AND the pre-beginCapture
  // window (CaptureStateGuard sets it before beginCapture) — where no capture is yet active and
  // alloc is legal — so we MUST check the REAL capture state, not the flag. Both call sites
  // (_gpubackend.cu:2745 + _cudagraph.cu:904) are BEFORE beginCapture, so the alloc is legal.
  bool captureActive = false;
  if (tl_graphCaptureStream != nullptr) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(tl_graphCaptureStream), &cs) == cudaSuccess) {
      captureActive = (cs != cudaStreamCaptureStatusNone);
    }
    cudaGetLastError();
  }
  if (captureActive) return;  // cannot materialize mid-capture; allocateFromCaptureArena reports

  int prevDev = -1; cudaGetDevice(&prevDev);
  bool restore = (prevDev != deviceId && cudaSetDevice(deviceId) == cudaSuccess);
  cudaStream_t s = ensureDirectAllocStream(deviceId);  // dedicated NON-capturing stream
  void* base = nullptr;
  cudaError_t err = (s != nullptr) ? cudaMallocAsync(&base, ARENA_BYTES, s) : cudaErrorMemoryAllocation;
  if (err == cudaSuccess && base != nullptr) {
    cudaStreamSynchronize(s);  // materialize — legal here (not capturing)
    cudaGetLastError();
    size_t blockIdx = captureArenaBlocks_[deviceId].size();
    ArenaBlock blk;
    blk.base     = base;
    blk.capacity = ARENA_BYTES;
    blk.offset   = 0;
    captureArenaBlocks_[deviceId].push_back(blk);
    if (sd::Environment::getInstance().isDebug()) {
      sd_printf("CudaMemoryPool::ensureCaptureArena: materialized block[%zu] %zuMB capture-constant arena "
                "for dev=%d base=%p\n", blockIdx, ARENA_BYTES >> 20, deviceId, base);
    }
  } else {
    cudaGetLastError();
  }
  if (restore) cudaSetDevice(prevDev);
}

void* CudaMemoryPool::allocateFromCaptureArena(size_t size, int deviceId) {
  if (released_.load(std::memory_order_acquire) || size == 0) return nullptr;
  if (deviceId < 0) cudaGetDevice(&deviceId);
  if (deviceId < 0 || deviceId >= MAX_DEVICES) return nullptr;

  // 256B-align: shape buffers (+ canary padding) are read as LongType arrays.
  const size_t ALIGN = 256;
  const size_t alignedSize = (size + (ALIGN - 1)) & ~static_cast<size_t>(ALIGN - 1);

  // Materialize the first block if not present — no-op if it already exists, or if a capture
  // is active (then the bump below falls through to the growth path, which also checks capture
  // state and returns nullptr if active). The pre-capture ensureCaptureArena() call normally
  // allocates block[0] before any capture-time bump.
  ensureCaptureArena(deviceId);

  std::lock_guard<std::mutex> lock(captureArenaMutex_);

  if (captureArenaBlocks_[deviceId].empty()) {
    // No block at all: either capture is active (ensureCaptureArena bailed) or alloc failed.
    return nullptr;
  }

  // Try to bump from the last block (common path).
  ArenaBlock& last = captureArenaBlocks_[deviceId].back();
  if (last.offset + alignedSize <= last.capacity) {
    void* ptr = static_cast<char*>(last.base) + last.offset;
    last.offset += alignedSize;
    return ptr;
  }

  // Last block is full. Attempt to grow by appending a new block — BUT only when no CUDA
  // graph capture is currently active on this thread. Mid-capture cudaMallocAsync is illegal
  // (err900). In that very rare case (capture started before pre-warm finished, or an
  // unusually large single plan) we return nullptr so the caller falls back gracefully.
  bool captureActive = false;
  if (tl_graphCaptureStream != nullptr) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(tl_graphCaptureStream), &cs) == cudaSuccess) {
      captureActive = (cs != cudaStreamCaptureStatusNone);
    }
    cudaGetLastError();
  }
  if (captureActive) {
    // Cannot grow mid-capture. Log once and return nullptr (caller falls back, same as
    // the old "arena FULL" path — i.e. slot-by-slot for this particular constant).
    static int growBlockCount = 0;
    if (growBlockCount < 15) {
      growBlockCount++;
      size_t totalBlocks = captureArenaBlocks_[deviceId].size();
      size_t totalCapacity = 0;
      size_t totalUsed = 0;
      for (const auto& blk : captureArenaBlocks_[deviceId]) {
        totalCapacity += blk.capacity;
        totalUsed += blk.offset;
      }
      if (sd::Environment::getInstance().isDebug()) {
        sd_printf("CudaMemoryPool::allocateFromCaptureArena: capture ACTIVE, cannot grow dev=%d "
                  "need=%zu blocks=%zu cap=%zuMB used=%zuMB\n",
                  deviceId, alignedSize, totalBlocks, totalCapacity >> 20, totalUsed >> 20);
      }
    }
    return nullptr;
  }

  // Not capturing: allocate a new block on the dedicated non-capturing stream and push it.
  // Block size: at least ARENA_BYTES, or larger if a single request exceeds it.
  const size_t ARENA_BYTES = 64ull * 1024ull * 1024ull;
  size_t newBlockSize = (alignedSize > ARENA_BYTES) ? alignedSize : ARENA_BYTES;

  int prevDev = -1; cudaGetDevice(&prevDev);
  bool restore = (prevDev != deviceId && cudaSetDevice(deviceId) == cudaSuccess);
  cudaStream_t s = ensureDirectAllocStream(deviceId);  // must NOT be holding captureArenaMutex_ when calling ...
  // NOTE: ensureDirectAllocStream acquires directAllocStreamMutex_, not captureArenaMutex_,
  // so calling it under captureArenaMutex_ is deadlock-free (distinct mutexes, consistent order).
  void* newBase = nullptr;
  cudaError_t err = (s != nullptr) ? cudaMallocAsync(&newBase, newBlockSize, s) : cudaErrorMemoryAllocation;
  if (err == cudaSuccess && newBase != nullptr) {
    cudaStreamSynchronize(s);   // materialize — legal here (not capturing)
    cudaGetLastError();
    size_t blockIdx = captureArenaBlocks_[deviceId].size();
    ArenaBlock blk;
    blk.base     = newBase;
    blk.capacity = newBlockSize;
    blk.offset   = alignedSize;  // immediately consume the request
    captureArenaBlocks_[deviceId].push_back(blk);
    if (sd::Environment::getInstance().isDebug()) {
      sd_printf("CudaMemoryPool::allocateFromCaptureArena: GREW arena dev=%d block[%zu] "
                "%zuMB base=%p (total blocks=%zu)\n",
                deviceId, blockIdx, newBlockSize >> 20, newBase,
                captureArenaBlocks_[deviceId].size());
    }
    if (restore) cudaSetDevice(prevDev);
    return newBase;  // ptr = base of new block (offset was 0, now alignedSize)
  }

  // Growth failed (unlikely OOM). Log and return nullptr.
  cudaGetLastError();
  if (restore) cudaSetDevice(prevDev);
  static int growFailCount = 0;
  if (growFailCount < 15) {
    growFailCount++;
    sd_printf("CudaMemoryPool::allocateFromCaptureArena: failed to grow arena dev=%d "
              "need=%zu err=%d (arena exhausted and alloc failed)\n",
              deviceId, newBlockSize, (int)err);
  }
  return nullptr;
}

bool CudaMemoryPool::isInCaptureArena(void* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(captureArenaMutex_));
  char* p = static_cast<char*>(ptr);
  for (int d = 0; d < MAX_DEVICES; d++) {
    for (const auto& blk : captureArenaBlocks_[d]) {
      char* base = static_cast<char*>(blk.base);
      if (base != nullptr && p >= base && p < base + blk.capacity) return true;
    }
  }
  return false;
}

void CudaMemoryPool::immediateDirectFree(void* ptr, size_t size, int deviceId, bool isHostAlloc) {
  DSP_DIAG(MEMORY, "IMMEDIATE_DIRECT_FREE: ptr=%p size=%zu dev=%d host=%d",
           ptr, size, deviceId, (int)isHostAlloc);
  if (isHostAlloc) {
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
      sd_printf("CudaMemoryPool::immediateDirectFree: cudaFreeHost failed ptr=%p size=%zu: %s\n",
                ptr, size, cudaGetErrorString(err));
      cudaGetLastError();
    }
    return;
  }
  int savedDev = -1;
  cudaGetDevice(&savedDev);
  bool restore = (deviceId >= 0 && savedDev != deviceId && cudaSetDevice(deviceId) == cudaSuccess);
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    sd_printf("CudaMemoryPool::immediateDirectFree: cudaFree failed for direct allocation ptr=%p size=%zu: %s\n",
              ptr, size, cudaGetErrorString(err));
    cudaGetLastError();
  }
  if (restore) cudaSetDevice(savedDev);
}

void CudaMemoryPool::deferDirectFree(void* ptr, size_t size, int deviceId, bool isHostAlloc,
                                     cudaStream_t orderStream) {
  // Async-only ordering (no host sync): record an event on the consumer stream
  // at free time. The synchronous cudaFree/cudaFreeHost only happens after
  // cudaEventQuery reports that everything enqueued before the free completed —
  // so an in-flight consumer (async cuBLAS GEMM reading a failover temp) can
  // never have its memory unmapped underneath it (task #57).
  cudaEvent_t evt = nullptr;
  if (orderStream != nullptr && !tl_graphExecutionActive) {
    if (cudaEventCreateWithFlags(&evt, cudaEventDisableTiming) != cudaSuccess) {
      cudaGetLastError();
      evt = nullptr;
    } else if (cudaEventRecord(evt, orderStream) != cudaSuccess) {
      cudaGetLastError();
      cudaEventDestroy(evt);
      evt = nullptr;
    }
  }
  if (evt == nullptr) {
    // No orderable stream (teardown edge, capture guard) — legacy immediate free.
    DSP_DIAG(MEMORY, "DEFER_DIRECT_FREE: no orderable stream (capture=%d) — immediate free ptr=%p size=%zu",
             (int)tl_graphExecutionActive, ptr, size);
    immediateDirectFree(ptr, size, deviceId, isHostAlloc);
    return;
  }
  int pending;
  {
    std::lock_guard<std::mutex> lock(deferredFreeMutex_);
    deferredDirectFrees_.push_back({ptr, size, deviceId, static_cast<void*>(evt), isHostAlloc});
    pending = static_cast<int>(deferredDirectFrees_.size());
  }
  deferredFreeCount_.fetch_add(1, std::memory_order_release);
  DSP_DIAG(MEMORY, "DEFER_DIRECT_FREE: ptr=%p size=%zu dev=%d host=%d orderStream=%p pending=%d",
           ptr, size, deviceId, (int)isHostAlloc, (void*)orderStream, pending);
}

void CudaMemoryPool::drainDeferredDirectFrees(bool force) {
  if (deferredFreeCount_.load(std::memory_order_acquire) == 0) return;
  std::vector<DeferredDirectFree> ready;
  {
    std::lock_guard<std::mutex> lock(deferredFreeMutex_);
    auto it = deferredDirectFrees_.begin();
    while (it != deferredDirectFrees_.end()) {
      bool reap = force;
      if (!reap) {
        cudaError_t q = cudaEventQuery(reinterpret_cast<cudaEvent_t>(it->readyEvent));
        if (q == cudaSuccess) {
          reap = true;
        } else if (q != cudaErrorNotReady) {
          // Event unusable (poisoned/destroyed context) — reap anyway; leaking
          // here would compound an already-fatal state.
          DSP_DIAG(MEMORY, "DRAIN_DEFERRED_FREES: event query error %d for ptr=%p — reaping anyway",
                   (int)q, it->ptr);
          cudaGetLastError();
          reap = true;
        }
      }
      if (reap) {
        ready.push_back(*it);
        it = deferredDirectFrees_.erase(it);
      } else {
        ++it;
      }
    }
    deferredFreeCount_.store(static_cast<int>(deferredDirectFrees_.size()), std::memory_order_release);
  }
  if (!ready.empty()) {
    DSP_DIAG(MEMORY, "DRAIN_DEFERRED_FREES: reaped=%zu remaining=%d force=%d",
             ready.size(), deferredFreeCount_.load(std::memory_order_acquire), (int)force);
  }
  for (auto& e : ready) {
    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.readyEvent));
    cudaGetLastError();
    immediateDirectFree(e.ptr, e.size, e.deviceId, e.isHostAlloc);
  }
}

void CudaMemoryPool::registerDirectAllocation(void* ptr, size_t size) {
  if (ptr == nullptr || size == 0) return;
  std::lock_guard<std::mutex> lock(directAllocMutex_);
  directAllocations_[ptr] = size;
}

bool CudaMemoryPool::isDirectAllocation(void* ptr) const {
  if (ptr == nullptr) return false;
  std::lock_guard<std::mutex> lock(directAllocMutex_);
  return directAllocations_.count(ptr) > 0;
}

void CudaMemoryPool::releaseAll() {
  // Signal all concurrent free() calls to become no-ops BEFORE touching any maps.
  // GC finalizer threads may be in CudaMemoryPool::free() right now, walking the
  // hostAllocations_ or directAllocations_ hash maps. If we clear() those maps
  // while another thread is iterating, the iterator dereferences freed bucket nodes
  // → SIGSEGV on the VMThread. Setting released_ first makes concurrent free()
  // callers bail out at the top, so by the time we acquire the mutexes below,
  // no other thread is inside the critical sections.
  released_.store(true, std::memory_order_release);

  // Force-reap all event-deferred direct frees — at teardown nothing consumes
  // these buffers anymore, so event state no longer matters (task #57).
  drainDeferredDirectFrees(/*force=*/true);

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

    // Free direct (cudaMalloc) allocations — weight buffers migrated out of pool
    {
      std::lock_guard<std::mutex> lock(directAllocMutex_);
      for (const auto& entry : directAllocations_) {
        if (entry.first != nullptr) {
          cudaFree(entry.first);
        }
      }
      directAllocations_.clear();

      // Free persistent capture-safe allocations from allocateDirect() on their
      // dedicated streams (pool memory → cudaFreeAsync), then destroy the streams.
      for (const auto& entry : directAsyncAllocations_) {
        if (entry.first != nullptr) {
          int dev = entry.second.deviceId;
          cudaStream_t s = (dev >= 0 && dev < MAX_DEVICES) ? directAllocStreams_[dev] : nullptr;
          cudaFreeAsync(entry.first, s);
        }
      }
      directAsyncAllocations_.clear();
    }
    {
      std::lock_guard<std::mutex> lock(directAllocStreamMutex_);
      // Free all per-device capture-constant arena blocks (allocateFromCaptureArena backing)
      // on their dedicated streams BEFORE the streams are destroyed below.
      {
        std::lock_guard<std::mutex> alock(captureArenaMutex_);
        for (int d = 0; d < MAX_DEVICES; d++) {
          for (auto& blk : captureArenaBlocks_[d]) {
            if (blk.base != nullptr) {
              cudaFreeAsync(blk.base, directAllocStreams_[d]);
              blk.base = nullptr;
            }
          }
          captureArenaBlocks_[d].clear();
        }
      }
      for (int d = 0; d < MAX_DEVICES; d++) {
        if (directAllocStreams_[d] != nullptr) {
          cudaStreamSynchronize(directAllocStreams_[d]);
          cudaStreamDestroy(directAllocStreams_[d]);
          directAllocStreams_[d] = nullptr;
        }
      }
    }
  } catch (...) {
    if (sd::Environment::getInstance().isDebug() || sd::Environment::getInstance().isVerbose()) {
      sd_debug("CudaMemoryPool::releaseAll: Exception during cleanup - possible heap corruption\n", "");
    }
  }
}

// ─── Device exclusion list for failover ───────────────────────────────────────

void CudaMemoryPool::addExcludedFailoverDevice(int deviceId) {
  std::lock_guard<std::mutex> lock(exclusionMutex_);
  excludedFailoverDevices_.insert(deviceId);
  sd_debug("CudaMemoryPool: Device %d added to failover exclusion list\n", deviceId);
}

void CudaMemoryPool::removeExcludedFailoverDevice(int deviceId) {
  std::lock_guard<std::mutex> lock(exclusionMutex_);
  excludedFailoverDevices_.erase(deviceId);
  sd_debug("CudaMemoryPool: Device %d removed from failover exclusion list\n", deviceId);
}

void CudaMemoryPool::clearExcludedFailoverDevices() {
  std::lock_guard<std::mutex> lock(exclusionMutex_);
  excludedFailoverDevices_.clear();
  sd_debug("CudaMemoryPool: Failover exclusion list cleared\n", "");
}

bool CudaMemoryPool::isDeviceExcludedFromFailover(int deviceId) const {
  std::lock_guard<std::mutex> lock(exclusionMutex_);
  return excludedFailoverDevices_.count(deviceId) > 0;
}

}  // namespace memory
}  // namespace sd
