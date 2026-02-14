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

  ~CudaMemoryPool();

 private:
  CudaMemoryPool();
  CudaMemoryPool(const CudaMemoryPool&) = delete;
  CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

  /**
   * Failover allocation when primary allocation fails.
   * Tries: trim pool + retry, other GPU devices, pinned host memory.
   * @param actualDeviceId If non-null, receives the device where memory was actually allocated
   */
  void* allocateFailover(size_t size, int currentDeviceId, int* actualDeviceId = nullptr);

  static constexpr int MAX_DEVICES = 16;

  // Memory pools per device (using cudaMemPool_t)
  cudaMemPool_t pools_[MAX_DEVICES];
  bool poolInitialized_[MAX_DEVICES];
  std::mutex initMutex_;

  // Track host pinned allocations with sizes (very rare - only from last-resort failover).
  // Device allocations (both pool and non-pool) use cudaFree which works for both since CUDA 11.2.
  std::unordered_map<void*, size_t> hostAllocations_;  // ptr -> size, from cudaMallocHost failover
  std::mutex fallbackAllocMutex_;
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

  // Check if current CUDA version supports memory pools
  bool checkSupport();

  // Memory pressure callback and tracking
  MemoryPressureCallback memoryPressureCallback_;
  std::mutex callbackMutex_;
  std::atomic<bool> memoryPressureDetected_{false};
  MemoryPressureEvent lastPressureEvent_;
  std::mutex pressureEventMutex_;
};

}  // namespace memory
}  // namespace sd

#endif  // LIBND4J_CUDA_MEMORY_POOL_H
