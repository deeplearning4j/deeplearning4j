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
// CUDA Memory Pool Manager - Uses cudaMallocAsync for efficient memory reuse
//

#ifndef LIBND4J_CUDA_MEMORY_POOL_H
#define LIBND4J_CUDA_MEMORY_POOL_H

#include <cuda_runtime.h>
#include <system/common.h>
#include <mutex>
#include <vector>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <functional>

namespace sd {
namespace memory {

/**
 * Memory pressure event information
 */
struct SD_LIB_EXPORT MemoryPressureEvent {
  int requestedDeviceId;           // Device where allocation was requested
  size_t requestedSize;            // Size of allocation that failed
  size_t availableMemory;          // Available memory on requested device
  int alternativeDeviceId;         // Alternative device that has memory (-1 if none)
  bool isPeerAccessible;           // Whether alternative device is peer-accessible
  enum class Action {
    FAIL,           // Fail the allocation
    FAILOVER,       // Failover to alternative device (if peer)
    USE_PINNED_HOST // Use pinned host memory
  } recommendedAction;
};

/**
 * Callback type for memory pressure events
 * Return true to allow the allocation to proceed, false to fail
 */
using MemoryPressureCallback = std::function<bool(const MemoryPressureEvent& event)>;

/**
 * CudaMemoryPool - Manages CUDA memory pools for efficient allocation
 *
 * Uses CUDA 11.2+ memory pool APIs (cudaMallocAsync/cudaFreeAsync) to
 * eliminate allocation overhead by reusing memory from a pool.
 *
 * Benefits:
 * - Allocations return almost instantly (no driver call)
 * - Memory is automatically reused without explicit caching
 * - No memory fragmentation issues
 * - Thread-safe by design
 */
class SD_LIB_EXPORT CudaMemoryPool {
 public:
  /**
   * Get the singleton instance
   */
  static CudaMemoryPool& getInstance();

  /**
   * Check if peer-to-peer access is enabled between two devices.
   * Returns true if srcDevice can directly access dstDevice memory.
   */
  bool isPeerAccessEnabled(int srcDevice, int dstDevice) const {
    if (srcDevice < 0 || srcDevice >= MAX_DEVICES || dstDevice < 0 || dstDevice >= MAX_DEVICES)
      return false;
    return peerAccessEnabled_[srcDevice][dstDevice];
  }

  /**
   * Register an externally-allocated pinned host pointer so that free() uses cudaFreeHost.
   * Used by DataBuffer::migrate when falling back to pinned host after non-peer failover.
   */
  void registerHostAllocation(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(fallbackAllocMutex_);
    hostAllocations_[ptr] = size;
    pinnedHostBytesUsed_.fetch_add(size);
  }

  /**
   * Initialize the memory pool for a specific device
   * Called automatically on first allocation, but can be called explicitly
   */
  bool initializeForDevice(int deviceId);

  /**
   * Allocate memory from the pool
   * Falls back to cudaMalloc if pools are not supported
   *
   * @param size Size in bytes to allocate
   * @param deviceId Device to allocate on
   * @param stream CUDA stream for async allocation (can be nullptr for default)
   * @param actualDeviceId If non-null, receives the device where memory was actually allocated
   *                       (may differ from deviceId if failover occurred)
   * @return Pointer to allocated memory, or nullptr on failure
   */
  void* allocate(size_t size, int deviceId, cudaStream_t stream = nullptr, int* actualDeviceId = nullptr);

  /**
   * Allocate a PERSISTENT device buffer whose lifetime is independent of the async
   * pool's per-stream ordering, the CUDA graph capture workspace, and any CUDA graph.
   *
   * Motivation: during CUDA graph capture, allocate() bump-allocates from the capture
   * workspace (tl_captureWorkspace) — an interior pointer of a single block that is
   * freed by releaseWorkspace() whenever a segment is invalidated for rebuild. Buffers
   * that are baked as kernel arguments into the captured graph (e.g. TAD-offset buffers
   * read by softMaxCuda) MUST outlive the workspace, or replay reads freed memory →
   * CUDA err700. This method gives such buffers a standalone lifetime.
   *
   * Implementation: cudaMallocAsync on a dedicated, per-device, NON-capturing allocation
   * stream (never passed into a capture), then a one-time materializing sync. This
   * produces a real pool-backed allocation with NO cudaGraphMemAllocNode (which would
   * fail cudaGraphLaunch with "invalid argument" if issued on the capturing stream),
   * and NO synchronous cudaMalloc (which is illegal during capture). The buffer is
   * tracked so free() routes to cudaFreeAsync on the same allocation stream. It is freed
   * only when the owner releases it (e.g. TAD cache eviction), never by workspace or
   * graph teardown.
   *
   * @param size     bytes to allocate
   * @param deviceId device to allocate on
   * @return pointer, or nullptr on failure
   */
  void* allocateDirect(size_t size, int deviceId);

  // ── Capture-constant arena (state is private; see captureArenaBlocks_ in the private section) ──
  // Materialize the per-device capture-constant arena backing if not present. MUST be called
  // PRE-capture (no capture yet active) — it cudaMallocAsync's the 64MB backing, illegal once the
  // thread is capturing. Idempotent / no-op if already materialized or if a capture is active.
  void ensureCaptureArena(int deviceId);
  // Bump-allocate `size` bytes of capture-safe persistent constant memory from the per-device
  // arena (ensures backing, then pointer-arithmetic bump). Returns nullptr if exhausted or not
  // pre-materialized while capturing, so the caller can fall back/throw. Used by
  // ConstantHelper::replicatePointer's capture branch.
  void* allocateFromCaptureArena(size_t size, int deviceId);
  // True if ptr lies within any device's capture arena (so free() must NOT cudaFree it).
  bool isInCaptureArena(void* ptr) const;

  /**
   * Free memory back to the pool
   * Falls back to cudaFree if pools are not supported
   *
   * @param ptr Pointer to free
   * @param deviceId Device the pointer was allocated on
   * @param stream CUDA stream for async free (can be nullptr for default)
   */
  void free(void* ptr, int deviceId, cudaStream_t stream = nullptr);

  /**
   * Remove a stream from the dirty free tracking set for a device.
   * MUST be called before cudaStreamDestroy() to prevent trimPool()
   * from syncing a destroyed stream handle (which causes SIGSEGV).
   */
  void removeDirtyStream(int deviceId, cudaStream_t stream);

  /**
   * Check if memory pools are enabled and supported
   */
  bool isEnabled() const { return enabled_.load(); }

  /**
   * Enable or disable memory pools
   * When disabled, falls back to regular cudaMalloc/cudaFree
   */
  void setEnabled(bool enabled) { enabled_.store(enabled); }

  /**
   * Get pool statistics for a device
   */
  void getStats(int deviceId, size_t& usedBytes, size_t& reservedBytes);

  /**
   * Trim unused memory from the pool.
   * Syncs only the streams that have had cudaFreeAsync issued on them
   * (tracked via dirtyFreeStreams_), then trims pool reserved memory.
   * This avoids blocking unrelated compute work on other streams.
   */
  void trimPool(int deviceId);

  /**
   * Trim unused memory from the pool, syncing the specified stream.
   * Use this when frees were issued on a specific execution stream
   * rather than the default stream.
   */
  void trimPoolOnStream(int deviceId, cudaStream_t stream);

  /**
   * Get the current pinned host memory usage from failover allocations.
   * @return bytes currently allocated in pinned host memory
   */
  size_t getPinnedHostBytesUsed() const { return pinnedHostBytesUsed_.load(); }

  /**
   * Get the maximum allowed pinned host memory for failover.
   * @return max bytes, or 0 for unlimited
   */
  size_t getPinnedHostBytesLimit() const { return pinnedHostBytesLimit_.load(); }

  /**
   * Set the maximum allowed pinned host memory for failover allocations.
   * When exceeded, allocateFailover will return nullptr instead of allocating more.
   * Set to 0 for unlimited (default).
   * @param maxBytes maximum bytes allowed for pinned host failover
   */
  void setPinnedHostBytesLimit(size_t maxBytes) { pinnedHostBytesLimit_.store(maxBytes); }

  // =========================================================================
  // Pinned Host Memory Management
  // =========================================================================

  /**
   * Allocate pinned host memory with pool tracking.
   * Replaces raw cudaMallocHost calls so the pool can track and limit pinned host memory.
   * @param size bytes to allocate
   * @return pointer, or nullptr on failure
   */
  void* allocatePinnedHost(size_t size);

  /**
   * Free pinned host memory that was allocated via allocatePinnedHost().
   * If the pointer was NOT allocated via allocatePinnedHost() (e.g., raw cudaMallocHost
   * from older code), it falls back to cudaFreeHost directly.
   * @param ptr pointer to free
   * @return true if pointer was found in tracked allocations, false otherwise
   */
  bool freePinnedHost(void* ptr);

  /**
   * Hand ownership of a pinned host allocation to an external holder (e.g. a
   * CUDA graph replay handle that baked it as an H2D source and frees it at
   * handle death). Drops pool bookkeeping WITHOUT calling cudaFreeHost, so a
   * later freePinnedHost on the same pointer becomes a safe no-op.
   */
  bool relinquishPinnedHost(void* ptr);

  /**
   * Check if a pointer is a tracked pinned host allocation.
   */
  bool isPinnedHostAllocation(void* ptr) const;

  /**
   * Release all pools (call on shutdown)
   */
  void releaseAll();

  // =========================================================================
  // Memory Pressure Callbacks
  // =========================================================================

  /**
   * Register a callback to be invoked when memory pressure is detected.
   * The callback receives detailed information about the allocation failure
   * and can decide whether to allow failover or fail the allocation.
   *
   * @param callback Function to call when memory pressure is detected.
   *                 Return true to allow allocation to proceed with recommended action,
   *                 false to fail the allocation.
   */
  void setMemoryPressureCallback(MemoryPressureCallback callback);

  /**
   * Check if memory pressure occurred during last allocation.
   * This allows higher-level code to query if over-allocation happened.
   */
  bool wasMemoryPressureDetected() const { return memoryPressureDetected_.load(); }

  /**
   * Clear the memory pressure flag.
   */
  void clearMemoryPressureFlag() { memoryPressureDetected_.store(false); }

  /**
   * Get details of the last memory pressure event.
   * Valid only if wasMemoryPressureDetected() returns true.
   */
  const MemoryPressureEvent& getLastMemoryPressureEvent() const { return lastPressureEvent_; }

  // =========================================================================
  // Proactive Soft Limit
  // =========================================================================

  /**
   * Set the proactive soft-limit percentage for device memory usage.
   * When a device's usage exceeds this threshold, new allocations are
   * routed to other devices via allocateFailover() BEFORE individual
   * allocations fail. This prevents cumulative exhaustion from many
   * small allocations (e.g., DSP slot-by-slot warmup intermediates)
   * that individually succeed but collectively fill the GPU.
   *
   * Set below the SubprocessMemoryWatchdog's gpuStopPercent (typically 75%)
   * so the pool proactively migrates before the watchdog kills the process.
   *
   * @param percent 0 = disabled (default), 1-100 = proactive failover threshold.
   *                Typical value: 70 (routes at 70% usage, before watchdog's 75%).
   */
  void setSoftLimitPercent(int percent);

  /**
   * Get the current proactive soft-limit percentage.
   * @return 0 if disabled, otherwise the threshold percentage (1-100).
   */
  int getSoftLimitPercent() const { return softLimitPercent_.load(std::memory_order_relaxed); }

  // =========================================================================
  // Capture Workspace Range Tracking
  // =========================================================================

  /**
   * Register an active capture workspace range. CudaMemoryPool::free() will
   * silently skip any pointer that falls within a registered workspace range.
   * Call when a capture workspace is allocated and persists for graph lifetime.
   *
   * During CUDA graph capture, intermediate allocations are bump-allocated from
   * a pre-allocated workspace (interior pointers within a single cudaMalloc block).
   * After capture ends, tl_graphExecutionActive is cleared, so the in-capture
   * guards in free()/deleteSpecial() no longer fire. When these intermediates
   * are later destroyed (GC, slot cleanup), cudaFreeAsync on interior pointers
   * returns "invalid argument", and the cudaFree fallback corrupts the CUDA context.
   * This registry catches those frees regardless of capture state.
   *
   * @param basePtr  Base pointer of the workspace allocation
   * @param bytes    Size of the workspace in bytes
   */
  void registerCaptureWorkspace(void* basePtr, size_t bytes);

  /**
   * Unregister a capture workspace range (when graph/replay handle is destroyed
   * or releaseGpuIntermediates frees the workspace).
   * @param basePtr  Base pointer previously registered
   */
  void unregisterCaptureWorkspace(void* basePtr);

  // =========================================================================
  // Graph-Baked Address Protection
  // =========================================================================

  /**
   * Pin a GPU address that is baked into a live CUDA graph's captured nodes.
   * While pinned, CudaMemoryPool::free() will skip cudaFreeAsync on this address,
   * preventing pool reuse while the CUDA graph that baked it is still live.
   * Safe to call multiple times for the same address (reference-counted).
   *
   * Background: when a slot's old buffer is freed via tl_deferredSlotDeletes and
   * the pool re-hands that address to a subsequent allocation (e.g., ref-SD weights),
   * and the CUDA graph replays using the baked-in address, the graph reads the new
   * allocation's data — causing data corruption or CUDA err700 if re-freed.
   *
   * @param ptr      GPU pointer baked into the CUDA graph
   * @param deviceId Device the pointer was allocated on
   */
  void pinGraphBakedAddress(void* ptr, int deviceId);

  /**
   * Unpin a GPU address previously pinned via pinGraphBakedAddress().
   * When the reference count reaches zero, immediately issues cudaFreeAsync
   * on the saved stream (or stream 0 if not recorded) to release the memory
   * back to the pool. This is the deferred-free path for graph-baked buffers.
   *
   * @param ptr      GPU pointer to unpin
   * @param deviceId Device the pointer was allocated on (used for cudaFreeAsync)
   * @param stream   CUDA stream for the deferred cudaFreeAsync (nullptr = stream 0)
   */
  void unpinGraphBakedAddress(void* ptr, int deviceId, cudaStream_t stream = nullptr);

  /**
   * Check if a pointer is currently pinned as graph-baked.
   * Called from CudaMemoryPool::free() to skip freeing pinned addresses.
   */
  bool isGraphBakedPinned(void* ptr) const;

  // =========================================================================
  // Direct (non-pool) Allocation Tracking
  // =========================================================================

  /**
   * Register a pointer as a direct cudaMalloc allocation (outside the async pool).
   * free() will route these to cudaFree instead of cudaFreeAsync.
   * Used for weight buffer migration: after migrating weight DataBuffers from the
   * async pool to direct cudaMalloc, we register them here so subsequent frees
   * use the correct deallocation path.
   *
   * @param ptr   Pointer allocated via cudaMalloc
   * @param size  Size of the allocation in bytes
   */
  void registerDirectAllocation(void* ptr, size_t size);

  /**
   * Check if a pointer was registered as a direct (cudaMalloc) allocation.
   * @param ptr  Pointer to check
   * @return true if this pointer was registered via registerDirectAllocation()
   */
  bool isDirectAllocation(void* ptr) const;

  /**
   * Check if a pointer falls within any registered capture workspace.
   * @param ptr  Pointer to check
   * @return true if ptr is an interior pointer of an active workspace
   */
  bool isInCaptureWorkspace(void* ptr) const;

  /**
   * Add a device to the failover exclusion list.
   * Excluded devices will be skipped during allocateFailover() Steps 2 and 2b.
   * Used to isolate subprocess memory: e.g., the staging process excludes
   * device 1 so it never displaces the embedding subprocess's resident pages.
   * @param deviceId The device to exclude from failover allocation
   */
  void addExcludedFailoverDevice(int deviceId);

  /**
   * Remove a device from the failover exclusion list.
   * @param deviceId The device to remove from exclusion
   */
  void removeExcludedFailoverDevice(int deviceId);

  /**
   * Clear all device exclusions from the failover list.
   */
  void clearExcludedFailoverDevices();

  /**
   * Check if a device is excluded from failover allocation.
   * @param deviceId The device to check
   * @return true if the device is excluded
   */
  bool isDeviceExcludedFromFailover(int deviceId) const;

  ~CudaMemoryPool();

 private:
  CudaMemoryPool();
  CudaMemoryPool(const CudaMemoryPool&) = delete;
  CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

  /**
   * Failover allocation when primary allocation fails.
   * Tries: trim pool + retry, other GPU devices, pinned host memory.
   * @param actualDeviceId If non-null, receives the device where memory was actually allocated
   * @param skipSameDeviceRetry When true, skip Step 1 (trim-and-retry on currentDeviceId).
   *        Set by the proactive soft-limit path — if the soft limit triggered failover to
   *        route allocations to another device, retrying on the same device defeats the purpose
   *        (the pool's own usage is low, so trim+retry always succeeds, keeping all allocations
   *        on the overloaded device while other devices sit idle).
   * @param consumerStream Stream that will consume the allocation (the caller's alloc/exec
   *        stream). Non-peer managed fallbacks prefetch on THIS stream so residency is
   *        ordered before the consuming kernels — prefetching on the default stream races
   *        demand paging on non-peer pairs and produces error 700 (bge warmup OOM cascade).
   */
  void* allocateFailover(size_t size, int currentDeviceId, int* actualDeviceId = nullptr,
                         bool skipSameDeviceRetry = false, cudaStream_t consumerStream = nullptr);

  static constexpr int MAX_DEVICES = 16;

  // Memory pools per device (using cudaMemPool_t)
  cudaMemPool_t pools_[MAX_DEVICES];
  bool poolInitialized_[MAX_DEVICES];
  std::mutex initMutex_;

  // Track host pinned allocations with sizes (very rare - only from last-resort failover).
  // Device allocations (both pool and non-pool) use cudaFree which works for both since CUDA 11.2.
  std::unordered_map<void*, size_t> hostAllocations_;  // ptr -> size, from cudaMallocHost failover
  mutable std::mutex fallbackAllocMutex_;  // mutable for isPinnedHostAllocation() const
  std::atomic<size_t> pinnedHostBytesUsed_{0};   // cumulative pinned host bytes currently allocated
  std::atomic<size_t> pinnedHostBytesLimit_{0};   // max allowed pinned host bytes (0 = unlimited)

  // Track streams that have had cudaFreeAsync issued on them, per device.
  // trimPool() syncs only these streams instead of the entire device, so
  // unrelated GPU work on other streams is never blocked.
  std::unordered_set<cudaStream_t> dirtyFreeStreams_[MAX_DEVICES];
  std::mutex dirtyStreamsMutex_[MAX_DEVICES];

  // Peer access tracking: peerAccessEnabled_[i][j] means device i can access device j's memory
  bool peerAccessEnabled_[MAX_DEVICES][MAX_DEVICES]{};
  bool peerAccessInitialized_{false};
  void initializePeerAccess();

  // Whether pools are enabled (can be disabled for debugging)
  std::atomic<bool> enabled_{true};

  // Whether the system supports memory pools (CUDA 11.2+)
  bool supported_{false};

  // Set by releaseAll() during shutdown. Once true, free() is a no-op.
  // Prevents SIGSEGV from GC/finalizer threads calling free() after the
  // pool's internal maps have been cleared by the destructor.
  std::atomic<bool> released_{false};

  // Check if current CUDA version supports memory pools
  bool checkSupport();

  // Memory pressure callback and tracking
  MemoryPressureCallback memoryPressureCallback_;
  std::mutex callbackMutex_;
  std::atomic<bool> memoryPressureDetected_{false};
  MemoryPressureEvent lastPressureEvent_;
  std::mutex pressureEventMutex_;

  // Proactive soft-limit: when device usage exceeds this percentage,
  // allocate() routes to allocateFailover() before the local allocation
  // is attempted. 0 = disabled (default).
  std::atomic<int> softLimitPercent_{0};

  // Active capture workspace ranges: basePtr → size.
  // Protected by captureWorkspaceMutex_. Typically 1-4 entries (one per captured segment).
  // Used by free() to skip cudaFreeAsync on interior pointers of capture workspaces.
  mutable std::mutex captureWorkspaceMutex_;
  std::unordered_map<void*, size_t> captureWorkspaceRanges_;

  // Direct (non-pool) allocations: ptr → size.
  // Pointers allocated via cudaMalloc instead of cudaMallocAsync, registered by
  // registerDirectAllocation(). free() checks this before cudaFreeAsync and routes
  // to cudaFree instead. Used for weight buffers migrated out of the async pool
  // to prevent pool fragmentation (weights pinning pool blocks prevents trimPool
  // from reclaiming freed intermediate memory).
  mutable std::mutex directAllocMutex_;
  std::unordered_map<void*, size_t> directAllocations_;

  // Persistent capture-safe allocations from allocateDirect(): ptr → {size, deviceId}.
  // Allocated via cudaMallocAsync on a dedicated non-capturing stream so they survive
  // capture-workspace release and CUDA-graph teardown (e.g. TAD-offset buffers baked
  // as kernel args). free() routes these to cudaFreeAsync on directAllocStreams_[dev].
  // Shares directAllocMutex_ with directAllocations_.
  struct DirectAsyncInfo { size_t size; int deviceId; };
  std::unordered_map<void*, DirectAsyncInfo> directAsyncAllocations_;

  // ── Event-deferred direct frees (task #57) ──────────────────────────────
  // cudaFree/cudaFreeHost are SYNCHRONOUS and stream-oblivious: freeing an
  // OOM-failover buffer (managed / pinned-host) unmaps it IMMEDIATELY, even
  // while an enqueued consumer kernel is still in flight (async cuBLAS GEMMs
  // don't sync before the owning NDArray destructs). Pool buffers survive the
  // same dtor-after-enqueue pattern because cudaFreeAsync is stream-ordered.
  // NO host sync is allowed here (WS-N mandate) — instead the free records an
  // event on the consumer stream and is REAPED later, once cudaEventQuery
  // (non-blocking) reports completion, at pool choke points (allocate entry,
  // failover entry, releaseAll).
  struct DeferredDirectFree {
    void* ptr;
    size_t size;
    int deviceId;
    void* readyEvent;   // cudaEvent_t recorded on the consumer stream
    bool isHostAlloc;   // true → cudaFreeHost, false → cudaFree
  };
  std::vector<DeferredDirectFree> deferredDirectFrees_;
  std::mutex deferredFreeMutex_;
  std::atomic<int> deferredFreeCount_{0};
  // Enqueue a deferred free ordered after orderStream's current work; frees
  // immediately (legacy behavior) when no orderable stream exists.
  void deferDirectFree(void* ptr, size_t size, int deviceId, bool isHostAlloc, cudaStream_t orderStream);
  // Reap deferred entries whose events completed; force=true frees everything
  // regardless of event state (teardown only).
  void drainDeferredDirectFrees(bool force = false);
  // The raw synchronous free (device-switch guard + error handling).
  void immediateDirectFree(void* ptr, size_t size, int deviceId, bool isHostAlloc);

  // Dedicated per-device allocation streams used ONLY by allocateDirect()/its frees.
  // Created lazily, NEVER passed into a CUDA graph capture, so cudaMallocAsync on them
  // produces standalone pool allocations with no graph mem-nodes. Guarded by
  // directAllocStreamMutex_.
  cudaStream_t directAllocStreams_[MAX_DEVICES]{};
  std::mutex directAllocStreamMutex_;
  // Lazily create (once) and return the dedicated non-capturing allocation stream for a device.
  cudaStream_t ensureDirectAllocStream(int deviceId);

  // ── Capture-constant arena ──────────────────────────────────────────────
  // Persistent per-device device-memory arena for CONSTANTS replicated DURING a CUDA-graph
  // capture (shape buffers / TAD offsets via ConstantHelper::replicatePointer). On the
  // capturing thread cudaMallocAsync is illegal (err900 cudaErrorStreamCaptureUnsupported)
  // and the capture workspace is released after capture (→ err700 dangling) — so neither a
  // fresh async alloc nor the workspace works. The arena backing is allocated/grown ONLY
  // when NOT capturing (pre-capture or between plans); during capture we BUMP-allocate (pure
  // pointer arithmetic, no CUDA call). Replicated constants are process-lifetime (cached
  // forever in ConstantShapeHelper), so the arena only grows — blocks are NEVER freed or
  // reset individually because their device addresses are baked into live captured CUDA graphs
  // (resetting = err700). free() recognizes arena-interior pointers and skips them; all blocks
  // are freed at releaseAll(). Guarded by captureArenaMutex_.
  //
  // Multi-block growable design: a single 64 MB block holds ~1800 padded shape buffers, but
  // across many plan create/destroy cycles the cumulative distinct constants can exceed that.
  // We therefore keep a per-device VECTOR of blocks. allocateFromCaptureArena bumps from the
  // LAST block; if it is full and no capture is active, a NEW block is pushed (growth).
  // ensureCaptureArena is called PRE-CAPTURE (before beginCapture) to guarantee at least one
  // full ARENA_BYTES of free headroom in the last block: if the last block has < ARENA_BYTES
  // free (or no blocks exist), it appends a fresh 64 MB block so capture-time bump-allocs
  // never need to grow. This prevents the status=50 failure where block[0] fills across
  // decode steps and mid-capture growth is illegal (cudaMallocAsync err900 during capture).
  // Blocks are raw pointers (no smart pointers per repo rule) and freed as a batch in releaseAll.
  struct ArenaBlock {
    void*  base;      // device pointer (nullptr = unused sentinel)
    size_t capacity;  // bytes allocated
    size_t offset;    // bytes consumed (bump cursor)
  };
  std::vector<ArenaBlock> captureArenaBlocks_[MAX_DEVICES];  // per-device block list
  std::mutex captureArenaMutex_;
  // (ensureCaptureArena / allocateFromCaptureArena / isInCaptureArena are declared PUBLIC near
  //  allocateDirect — ConstantHelper::replicatePointer and the DSP backends call them.)

  // Device exclusion list for failover — prevents allocateFailover() from
  // placing memory on these devices. Used to isolate subprocess memory:
  // the staging process excludes device 1 so it never displaces the
  // embedding subprocess's resident pages.
  std::unordered_set<int> excludedFailoverDevices_;
  mutable std::mutex exclusionMutex_;

  // Graph-baked address pins: ptr → {refCount, deviceId, freeRequested}.
  // A pin is PURE PROTECTION: addresses a live segment references (baked into a CUDA
  // graph OR cached in a frozen slot-by-slot slot) must not be returned to the pool
  // while the segment is live. free() on a pinned ptr is DEFERRED and records
  // freeRequested=true; unpinGraphBakedAddress() executes the free at refCount==0 ONLY
  // if it was requested. A pinned buffer that is never free()'d (a SameDiff weight/
  // constant that outlives the plan) is therefore released by unpin WITHOUT a free —
  // it is externally owned and must not be freed by the plan.
  struct GraphBakedInfo { int refCount; int deviceId; bool freeRequested; };
  mutable std::mutex graphBakedMutex_;
  std::unordered_map<void*, GraphBakedInfo> graphBakedPins_;
};

}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_CUDA_MEMORY_POOL_H
