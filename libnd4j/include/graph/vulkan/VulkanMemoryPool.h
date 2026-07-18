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

#ifndef LIBND4J_VULKAN_MEMORY_POOL_H
#define LIBND4J_VULKAN_MEMORY_POOL_H

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <vulkan/vulkan.h>
#include <graph/vulkan/VulkanDeviceManager.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Forward declaration
// ---------------------------------------------------------------------------
namespace sd { namespace graph {
class VulkanDeviceContext;
class VulkanExecutionStream;
} }

namespace sd {
namespace graph {

// ---------------------------------------------------------------------------
// Allocation record — maps sd::Pointer → allocation details.
// Host-visible suballocations return real mapped pointers.
// DEVICE_LOCAL-only allocations return a stable registry token that must never
// be dereferenced on the host (non-null, but points into pool sentinel space).
// ---------------------------------------------------------------------------
struct VulkanAllocRecord {
  int            deviceId    = -1;
  uint32_t       memTypeIdx  = UINT32_MAX;   // VkDeviceMemory type index
  VkMemoryPropertyFlags memoryPropertyFlags = 0;
  bool           dedicated   = false;        // standalone vkAllocateMemory
  bool           hostVisible = false;        // can be host-mapped
  void*          mappedPtr   = nullptr;      // non-null when host-visible
  // Block-suballocation fields (used when !dedicated)
  void*          blockKey    = nullptr;      // ptr to the Block this belongs to
  VkDeviceSize   offsetInBlock = 0;
  // Every allocation owns one transfer/storage-capable VkBuffer.
  VkBuffer       buffer        = VK_NULL_HANDLE;
  VkDevice       logicalDevice = VK_NULL_HANDLE;
  VkDeviceSize   logicalSize   = 0;  // bytes requested by the caller
  VkDeviceSize   reservedSize  = 0;  // VkMemoryRequirements::size
  // Backing allocation metadata required by Vulkan external-memory import.
  VkDeviceMemory backingMemory = VK_NULL_HANDLE;
  VkDeviceSize   backingSize   = 0;
  bool           externalShareable = false;
};

// ---------------------------------------------------------------------------
// VulkanMemoryPool
//
// Mandatory block suballocator for Vulkan device memory.
//
// Design constraints from ADR-0111 §3 + §9:
//
//  - maxMemoryAllocationCount on commodity Vulkan hardware is ~4096.
//    One vkAllocateMemory per NDArray exhausts it immediately. All
//    allocations are therefore suballocated from large blocks (default
//    BLOCK_SIZE = 64 MiB, tunable at construction), with dedicated
//    allocations only above DEDICATED_THRESHOLD (default = block/2).
//
//  - Allocation selects one compatible DEVICE_LOCAL memory type. On unified
//    memory systems that type may naturally also be host visible. Exhaustion,
//    budget violations, or the absence of compatible device-local memory are
//    hard failures; placement never changes tier to make an allocation succeed.
//
//  - Fence-deferred frees: free() enqueues onto a per-device retire list
//    stamped with a timeline value.  sweep(device, completedTimeline) reclaims
//    entries whose stamp has been passed.  Until VulkanDeviceContext provides
//    real timeline semaphores the caller passes epoch values from
//    vkDeviceWaitIdle-backed accounting — replaced transparently in Task 2.
//
//  - Pointer registry: every returned sd::Pointer maps to a VulkanAllocRecord.
//    Host-visible suballocations return the actual mapped pointer.
//    DEVICE_LOCAL-only returns a unique sentinel token from the sentinel table
//    (never dereferenced on the host).
//
// Thread safety: all public methods are thread-safe (mutex per device + global
// alloc-registry mutex).
// ---------------------------------------------------------------------------
class SD_LIB_EXPORT VulkanMemoryPool {
 public:
  // Default block size: 64 MiB.
  static constexpr VkDeviceSize kDefaultBlockSize = 64ull * 1024 * 1024;

  // Dedicated allocation threshold: allocations larger than this bypass
  // block suballocation and get their own vkAllocateMemory call.
  // Default = block / 2 = 32 MiB.
  static constexpr VkDeviceSize kDefaultDedicatedThreshold = kDefaultBlockSize / 2;

  /**
   * Construct the pool.
   * @param blockSize          VkDeviceMemory block size for suballocation.
   * @param dedicatedThreshold Allocations above this use dedicated memory.
   */
  explicit VulkanMemoryPool(VkDeviceSize blockSize          = kDefaultBlockSize,
                            VkDeviceSize dedicatedThreshold = kDefaultDedicatedThreshold);
  ~VulkanMemoryPool();

  /**
   * Wait for pool devices to become idle and release all pool allocations.
   * Safe to call repeatedly and used by the backend lifecycle before DSO unload.
   */
  void shutdown();

  // Non-copyable
  VulkanMemoryPool(const VulkanMemoryPool&) = delete;
  VulkanMemoryPool& operator=(const VulkanMemoryPool&) = delete;

  /**
   * Allocate `bytes` bytes for `deviceId`.
   *
   * Selects a compatible DEVICE_LOCAL memory type and returns nullptr on
   * budget exhaustion, OOM, or an incompatible device memory topology.
   * The returned pointer is either a real mapped address (host-visible) or
   * a unique sentinel token (device-local only).
   *
   * @param deviceId  Target device (0-based).
   * @param bytes     Requested size in bytes.
   * @param alignment Required alignment; 0 defaults to minStorageBufferAlignment
   *                  queried from the physical device.
   */
  void* allocate(int deviceId, VkDeviceSize bytes, VkDeviceSize alignment = 0);

  /**
   * Allocate a persistently mapped transfer buffer from a memory type that is
   * explicitly HOST_VISIBLE and HOST_COHERENT. This is the Vulkan equivalent
   * of CUDA pinned host memory; it never changes placement to DEVICE_LOCAL.
   */
  void* allocateHostVisible(int deviceId, VkDeviceSize bytes);

  /**
   * Enqueue ptr for deferred release at timeline value `timelineValue`.
   * The actual vkFreeMemory / suballocator reclaim happens in sweep().
   * Returns false if ptr is not known to this pool.
   */
  bool free(void* ptr, uint64_t timelineValue = 0);

  /**
   * Reclaim all retire-list entries for `deviceId` whose stamp <= completedTimeline.
   * Call after confirming the GPU has reached that timeline value
   * (initially via vkDeviceWaitIdle epoch accounting; later via real timeline
   * semaphore signals from VulkanDeviceContext).
   */
  void sweep(int deviceId, uint64_t completedTimeline);

  /**
   * Convenience: reclaim immediately after the caller has proven the owning
   * device is idle. Used internally by freeSynchronized and by shutdown code.
   */
  void freeImmediate(void* ptr);

  /**
   * Wait for all work on the owning logical device, then reclaim ptr.
   * This is the correctness-preserving release path until every Vulkan
   * submission carries a per-allocation timeline stamp.
   */
  bool freeSynchronized(void* ptr);

  /**
   * Copy the allocation record into caller-owned storage.
   * Returns false if the pointer is not known to this pool.
   */
  bool queryRecord(void* ptr, VulkanAllocRecord& record) const;

  /** Checked synchronous transfers matching NativeOps memcpy directions. */
  bool copyHostToDevice(void* dstDevice, const void* srcHost, VkDeviceSize bytes);
  bool copyDeviceToHost(void* dstHost, void* srcDevice, VkDeviceSize bytes);
  bool copyDeviceToDevice(void* dstDevice, void* srcDevice, VkDeviceSize bytes);

  /** Stream-ordered transfer primitives used by NativeOps and DataBuffer. */
  bool copyHostToDeviceAsync(void* dstDevice, const void* srcHost,
                             VkDeviceSize bytes, VulkanExecutionStream* stream,
                             VkDeviceSize dstOffset = 0,
                             VkDeviceSize srcOffset = 0);
  bool copyDeviceToHostAsync(void* dstHost, void* srcDevice,
                             VkDeviceSize bytes, VulkanExecutionStream* stream,
                             VkDeviceSize dstOffset = 0,
                             VkDeviceSize srcOffset = 0);
  bool copyDeviceToDeviceAsync(void* dstDevice, void* srcDevice,
                               VkDeviceSize bytes, VulkanExecutionStream* stream,
                               VkDeviceSize dstOffset = 0,
                               VkDeviceSize srcOffset = 0);

  /** Fill a device allocation with the low byte of value. */
  bool fill(void* dstDevice, int value, VkDeviceSize bytes);
  bool fillAsync(void* dstDevice, int value, VkDeviceSize bytes,
                 VulkanExecutionStream* stream);

  /** Active allocation bytes and reserved Vulkan block/dedicated bytes. */
  bool getMemoryPoolStats(int deviceId, uint64_t& usedBytes,
                          uint64_t& reservedBytes) const;

  /** Bytes and free spans immediately reusable by the block suballocator. */
  bool getReusablePoolStats(int deviceId, uint64_t& reusableBytes,
                            uint64_t& reusableSpanCount) const;

  /** Lifetime successful allocation requests and requests served by an existing block. */
  bool getLifetimeAllocationStats(int deviceId, uint64_t& totalAcquired,
                                  uint64_t& totalReused) const;

  /** Release completely unused blocks on a device; never waits for the queue. */
  uint64_t trim(int deviceId, uint64_t minimumBytesToKeep = 0);

  /**
   * Approximate free memory for `deviceId`.
   * Uses VK_EXT_memory_budget when available on the device, otherwise
   * total – pool-tracked usage.
   */
  uint64_t getFreeMemory(int deviceId) const;

  /**
   * Return device id for a pointer previously allocated by this pool.
   * Returns -1 if not found.
   */
  int getDeviceId(void* ptr) const;

  /** Exact VkMemoryPropertyFlags of the allocation's owning memory type. */
  int getAllocationMemoryPropertyFlags(void* ptr, int deviceId) const;

  /**
   * Expose the transfer/storage VkBuffer owned by an allocation.
   * Returns VK_NULL_HANDLE for an unrecognized pointer.
   */
  VkBuffer getBuffer(void* ptr) const;

  // ── Allocation observability API ───────────────────────────────────────

  /**
   * Gap M1: Return the block id for a suballocated pointer (ptr must be a
   * value previously returned by allocate()).  Block ids are stable integers
   * assigned in creation order within each device's pool.  Returns -1 when:
   *   - the allocation is dedicated (not suballocated from a block),
   *   - ptr is not known to this pool.
   */
  int getBlockId(void* ptr) const;

  /**
   * Gap M4: Return the number of retire-list entries pending reclaim for
   * deviceId.  Returns 0 if deviceId is out of range or no frees are pending.
   */
  int getRetireListPendingCount(int deviceId) const;

  /** Singleton accessor (one pool per process). */
  static VulkanMemoryPool& getInstance();

 private:
  // ── Free-list block suballocator ──────────────────────────────────────
  // Each Block holds one vkAllocateMemory call.  Free spans within the block
  // are tracked as {offset, size} in a sorted free-list.
  struct FreeSpan {
    VkDeviceSize offset;
    VkDeviceSize size;
  };

  struct Block {
    VkDevice       logicalDevice  = VK_NULL_HANDLE;
    VkDeviceMemory memory         = VK_NULL_HANDLE;
    VkDeviceSize   blockSize      = 0;
    uint32_t       memTypeIdx     = UINT32_MAX;
    int            deviceId       = -1;
    bool           hostVisible    = false;
    bool           externalShareable = false;
    void*          mappedBase     = nullptr;   // persistent map if host-visible
    std::vector<FreeSpan> freeList;            // sorted by offset
    size_t         activeAllocs   = 0;         // suballocations alive in this block

    // Try to suballocate `size` bytes with `align` from this block's free-list.
    // Returns UINT64_MAX on failure, else offset within block.
    VkDeviceSize tryAllocate(VkDeviceSize size, VkDeviceSize align);

    // Return a span back to the free-list (merge adjacent spans).
    void reclaim(VkDeviceSize offset, VkDeviceSize size);
  };

  // All blocks for one (device, memTypeIdx) pair.
  struct MemTypePool {
    std::vector<std::unique_ptr<Block>> blocks;
  };

  // Per-device state (mutex-protected).
  struct DeviceState {
    mutable std::mutex              mtx;
    // key = memTypeIdx
    std::unordered_map<uint32_t, MemTypePool> pools;
    // Active bytes and bytes reserved from the Vulkan allocator.
    std::atomic<uint64_t>           trackedBytes{0};
    std::atomic<uint64_t>           reservedBytes{0};
    std::atomic<uint64_t>           totalAcquired{0};
    std::atomic<uint64_t>           totalReused{0};

    // ── Retire list for deferred frees ──────────────────────────────────
    struct RetireEntry {
      void*    ptr;          // pointer returned to caller
      uint64_t timelineStamp;
    };
    std::deque<RetireEntry> retireList;

    // ── Dedicated-allocation memory map ─────────────────────────────────
    // For dedicated (non-suballocated) allocations, the VkDeviceMemory
    // handle is stored here keyed by the pointer returned to the caller.
    std::unordered_map<void*, VkDeviceMemory> dedicatedMemMap;

    // ── T2 observability: block-id registry (Gap M1) ─────────────────────
    // Maps blockKey (Block*) → stable integer block id assigned at creation.
    std::unordered_map<void*, int>  blockIds;
    int                             nextBlockId{0};
  };

  // ── Internal helpers ────────────────────────────────────────────────────

  // Return the lazily-created state for deviceId. Caller must hold devicesMtx_.
  DeviceState* ensureDeviceState(int deviceId);

  // Find or create a block for (deviceId, memTypeIdx) capable of
  // suballocating `size` bytes with `alignment`.
  // Returns {block_ptr, offset} or {nullptr, 0} on OOM.
  std::pair<Block*, VkDeviceSize> findOrCreateBlock(
      DeviceState& state, int deviceId, uint32_t memTypeIdx,
      VkDeviceSize size, VkDeviceSize alignment);

  // Create a brand-new block of blockSize_ for (device, memTypeIdx).
  std::unique_ptr<Block> makeBlock(int deviceId, VkDevice logDev,
                                   VkPhysicalDevice phys,
                                   uint32_t memTypeIdx, VkDeviceSize blockSize);

  // Allocate and bind dedicated memory for a large logical buffer.
  void* allocateDedicated(int deviceId, VkDevice logDev, VkBuffer buffer,
                          uint32_t memTypeIdx,
                          VkMemoryPropertyFlags memoryPropertyFlags,
                          VkDeviceSize logicalSize,
                          VkDeviceSize reservedSize, bool hostVisible,
                          bool externalShareable);

  // Find the best compatible memory type for a VkMemoryRequirements bit mask.
  static uint32_t findMemoryType(VkPhysicalDevice physDev, uint32_t typeFilter,
                                 VkMemoryPropertyFlags required,
                                 VkMemoryPropertyFlags preferred = 0);

  // Legacy one-shot helpers retained only for source compatibility while all
  // public transfers route through VulkanExecutionStream.
  struct StagingBuffer {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
  };
  bool createStagingBuffer(int deviceId, VkDeviceSize bytes,
                           VkBufferUsageFlags usage, StagingBuffer& staging);
  static void destroyStagingBuffer(StagingBuffer& staging);
  bool submitCopy(int deviceId, VkBuffer src, VkBuffer dst, VkDeviceSize bytes);
  bool submitFill(int deviceId, VkBuffer dst, VkDeviceSize bytes, uint32_t pattern);
  bool waitForDevice(int deviceId);

  // Query default alignment for storage buffers on physDev.
  static VkDeviceSize defaultAlignment(VkPhysicalDevice physDev);

  // Emit DSP_DIAG(MEMORY, ...) for a placement decision.
  static void diagPlacement(int deviceId, VkDeviceSize bytes,
                            const char* memoryType);

  // Perform actual reclaim of one pointer (called from sweep/freeImmediate).
  void doReclaim(void* ptr, bool updateManagerAccounting = true);

  // ── State ───────────────────────────────────────────────────────────────
  std::once_flag shutdownFlag_;
  VkDeviceSize blockSize_;
  VkDeviceSize dedicatedThreshold_;

  // Per-device pools (indexed by deviceId, lazily grown).
  mutable std::mutex devicesMtx_;
  std::vector<std::unique_ptr<DeviceState>> deviceStates_;

  // Global pointer → record registry (separate lock from per-device state
  // to avoid lock-ordering issues with block operations).
  mutable std::mutex registryMtx_;
  std::unordered_map<void*, VulkanAllocRecord> registry_;

  // Sentinel table for DEVICE_LOCAL-only tokens.
  // Each entry is a unique non-null address that will never be read as data.
  mutable std::mutex sentinelMtx_;
  std::vector<std::unique_ptr<uint8_t>> sentinelStore_;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
#endif  // LIBND4J_VULKAN_MEMORY_POOL_H
