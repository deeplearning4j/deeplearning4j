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

#ifndef LIBND4J_VULKAN_DEVICE_CONTEXT_H
#define LIBND4J_VULKAN_DEVICE_CONTEXT_H

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <vulkan/vulkan.h>
#include <graph/vulkan/VulkanDeviceManager.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {

class VulkanPipelineCache;

// ---------------------------------------------------------------------------
// Device capability flags queried once at context creation time.
// ---------------------------------------------------------------------------
struct VulkanDeviceCaps {
  bool fp16           = false;  // shaderFloat16 enabled on the logical device
  bool fp64           = false;  // shaderFloat64 enabled on the logical device
  bool int64          = false;  // shaderInt64 enabled on the logical device
  bool int8           = false;  // shaderInt8 enabled on the logical device
  bool storage16      = false;  // storageBuffer16BitAccess enabled on the logical device
  bool timelineSem    = false;  // VK_KHR_timeline_semaphore or Vulkan 1.2 core
  bool memoryBudget   = false;  // VK_EXT_memory_budget
  bool extMemHost     = false;  // VK_EXT_external_memory_host (zero-copy staging)
  bool externalMemoryFd = false;  // VK_KHR_external_memory_fd enabled and usable
  bool externalSemaphoreFd = false;  // VK_KHR_external_semaphore_fd enabled and usable
  bool externalMemoryDedicatedOnly = false;  // exported buffers require dedicated allocation
  uint32_t subgroupSize = 0;    // physical subgroup size
  uint32_t apiVersion = 0;      // negotiated API version (VK_API_VERSION_*)
  uint32_t computeQueueFamily  = UINT32_MAX;
  uint32_t transferQueueFamily = UINT32_MAX;  // dedicated transfer queue, or UINT32_MAX
};

// ---------------------------------------------------------------------------
// VulkanDeviceContext — the LaunchContext analog for Vulkan (ADR-0111 §2)
//
// Lazily created per-device singleton bundle owning:
//   - VkDevice with features enabled per VulkanDeviceCaps.
//   - Compute and optional dedicated transfer VkQueue.
//   - Per-thread VkCommandPool registry (command pools are externally
//     synchronized — may NOT be shared across threads, unlike CUDA streams).
//   - Per-device descriptor-pool arena and VkPipelineCache.
//   - Timeline semaphore (when apiVersion >= 1.2 or VK_KHR_timeline_semaphore
//     is available) as the device's ordering spine.  This semaphore drives the
//     VulkanMemoryPool retire-list sweeps (Task 1 integration).
//
// VulkanDeviceContext is the canonical logical-device owner. Replay handles
// borrow its device and queue, and own only replay-local command/fence state.
// Capability checks therefore describe features enabled on the exact device
// that executes the recorded commands.
//
// Thread safety: the context itself is thread-safe (mutex on lazy init).
// Per-thread command pools are accessed only from their owning thread.
//
// Device loss: VK_ERROR_DEVICE_LOST marks the context lost and hard-fails
// ongoing work on that device.  No silent re-init.
// ---------------------------------------------------------------------------
class SD_LIB_EXPORT VulkanDeviceContext {
 public:
  /**
   * Get the per-device context for deviceId, lazily creating it.
   * Returns nullptr if Vulkan is not initialized or deviceId is invalid.
   */
  static VulkanDeviceContext* getContext(int deviceId);

  /**
   * Destroy all contexts (call at process exit / unit test teardown).
   * Waits for device idle before destroying.
   */
  static void destroyAll();

  /** True while any canonical logical-device context is still owned. */
  static bool hasLiveContexts();

  ~VulkanDeviceContext();

  // Non-copyable, non-movable
  VulkanDeviceContext(const VulkanDeviceContext&) = delete;
  VulkanDeviceContext& operator=(const VulkanDeviceContext&) = delete;

  // ── Device identity ──────────────────────────────────────────────────

  int deviceId() const { return deviceId_; }
  VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
  VkDevice device() const { return device_; }
  const VulkanDeviceCaps& caps() const { return caps_; }

  /** True only when this logical device can export/import both memory and semaphores. */
  bool hasExternalDeviceSharing() const {
    return caps_.externalMemoryFd && caps_.externalSemaphoreFd &&
           getMemoryFdFn_ != nullptr && getMemoryFdPropertiesFn_ != nullptr &&
           getSemaphoreFdFn_ != nullptr && importSemaphoreFdFn_ != nullptr;
  }

  PFN_vkGetMemoryFdKHR getMemoryFdFn() const { return getMemoryFdFn_; }
  PFN_vkGetMemoryFdPropertiesKHR getMemoryFdPropertiesFn() const {
    return getMemoryFdPropertiesFn_;
  }
  PFN_vkGetSemaphoreFdKHR getSemaphoreFdFn() const { return getSemaphoreFdFn_; }
  PFN_vkImportSemaphoreFdKHR importSemaphoreFdFn() const {
    return importSemaphoreFdFn_;
  }

  // ── Queues ────────────────────────────────────────────────────────────

  /**
   * Submit to the canonical compute queue with Vulkan-required host external
   * synchronization shared by replay, transfers, and DSP execution.
   */
  VkResult submitCompute(uint32_t submitCount,
                         const VkSubmitInfo* submitInfos, VkFence fence);

  /** Wait for the canonical compute queue through the same synchronization. */
  VkResult waitComputeIdle();

  // ── Per-thread command pools ──────────────────────────────────────────

  /**
   * Obtain (or lazily create) the command pool for the calling thread on
   * this device.  May NOT be shared across threads.
   */
  VkCommandPool getThreadCommandPool();

  /** Release the command pool for the calling thread (call on thread exit). */
  void releaseThreadCommandPool();

  // ── Descriptor pool ──────────────────────────────────────────────────

  /** The per-device descriptor pool arena. */
  VkDescriptorPool descriptorPool() const { return descriptorPool_; }

  /**
   * Allocate/free descriptor sets through the device context so all users
   * obey Vulkan's external-synchronization rule for a shared descriptor pool.
   */
  VkResult allocateDescriptorSets(const VkDescriptorSetAllocateInfo* allocateInfo,
                                  VkDescriptorSet* descriptorSets);
  VkResult freeDescriptorSets(uint32_t descriptorSetCount,
                              const VkDescriptorSet* descriptorSets);

  // ── Pipeline cache ────────────────────────────────────────────────────

  /**
   * The per-device VkPipelineCache.  Pipeline/JIT caches MUST be keyed by
   * deviceId — never share across devices (the JIT/ZLUDA lesson).
   */
  VkPipelineCache pipelineCache() const { return pipelineCache_; }

  /**
   * Persist the driver VkPipelineCache blob to disk (ADR 0115 Tier 2).
   * One blob per physical device, keyed by vendorID/deviceID/driverVersion/
   * pipelineCacheUUID. Safe no-op when persistence is disabled, the context
   * is uninitialized or lost, or the blob size is unchanged since the last
   * flush. Called after replay finalization (kill-safety on mobile) and at
   * context destroy.
   */
  bool savePipelineCacheBlob();

  /** Device-lifetime MLIR/SPIR-V shader cache shared by eager native kernels. */
  VulkanPipelineCache* shaderPipelineCache() const {
    return shaderPipelineCache_.get();
  }

  // ── Timeline semaphore (ordering spine) ───────────────────────────────

  /**
   * Whether this device supports timeline semaphores (Vulkan 1.2 core or
   * VK_KHR_timeline_semaphore extension).
   */
  bool hasTimelineSemaphore() const { return timelineSemaphore_ != VK_NULL_HANDLE; }

  /**
   * Advance the device timeline by signalling `value` via host.
   * Used to drive VulkanMemoryPool::sweep() after confirmed submission.
   */
  void signalTimeline(uint64_t value);

  /**
   * Block the host until the device timeline reaches `value`.
   * Timeout in nanoseconds (default: 5 seconds).
   */
  bool waitTimeline(uint64_t value, uint64_t timeoutNs = 5000000000ULL);

  /**
   * Query the current device timeline value without blocking.
   */
  uint64_t queryTimeline() const;

  /**
   * Increment and return the next timeline value to use for a submission.
   * Thread-safe.
   */
  uint64_t nextTimelineValue() { return nextTimelineValue_.fetch_add(1, std::memory_order_relaxed) + 1; }

  /**
   * Convenience: submit work and advance the device timeline, then sweep the
   * memory pool for entries whose timeline has passed.
   */
  void submitAndSweep(VkSubmitInfo submitInfo, uint64_t timelineValue);

  // ── Device loss ───────────────────────────────────────────────────────

  bool isLost() const { return lost_.load(std::memory_order_acquire); }
  void markLost() { lost_.store(true, std::memory_order_release); }

 private:
  explicit VulkanDeviceContext(int deviceId);

  bool initialize();
  void destroy();

  // Find queue families on the physical device.
  uint32_t findComputeQueueFamily() const;
  uint32_t findTransferQueueFamily(uint32_t computeFamily) const;

  // Build the required/optional extension lists based on device caps.
  std::vector<const char*> selectExtensions(
      const std::vector<VkExtensionProperties>& available);

  // ── State ────────────────────────────────────────────────────────────

  int              deviceId_       = -1;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice         device_         = VK_NULL_HANDLE;
  VulkanDeviceCaps caps_;

  VkQueue  computeQueue_  = VK_NULL_HANDLE;
  VkQueue  transferQueue_ = VK_NULL_HANDLE;
  mutable std::mutex computeQueueMtx_;

  // Per-thread command pool registry.
  mutable std::mutex          cmdPoolMtx_;
  std::unordered_map<std::thread::id, VkCommandPool> cmdPools_;

  // Fences whose wait failed are retained until device teardown proves idle.
  mutable std::mutex          unresolvedFenceMtx_;
  std::vector<VkFence>        unresolvedFences_;

  VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
  mutable std::mutex descriptorPoolMtx_;
  VkPipelineCache  pipelineCache_  = VK_NULL_HANDLE;
  // Tier-2 blob persistence state (ADR 0115): serializes flushes and skips
  // rewrites when vkGetPipelineCacheData reports an unchanged size.
  mutable std::mutex pipelineCacheBlobMtx_;
  size_t lastSavedPipelineCacheBytes_ = 0;
  std::unique_ptr<VulkanPipelineCache> shaderPipelineCache_;

  // Timeline semaphore (VK_NULL_HANDLE when not supported). Entry points
  // are resolved from the exact core-or-KHR capability path enabled on VkDevice.
  VkSemaphore                       timelineSemaphore_ = VK_NULL_HANDLE;
  PFN_vkSignalSemaphore             signalSemaphoreFn_ = nullptr;
  PFN_vkWaitSemaphores              waitSemaphoresFn_ = nullptr;
  PFN_vkGetSemaphoreCounterValue    getSemaphoreCounterValueFn_ = nullptr;

  // Cross-device Vulkan sharing. These entry points are resolved only when the
  // exact logical device enabled the corresponding opaque-FD extensions.
  PFN_vkGetMemoryFdKHR              getMemoryFdFn_ = nullptr;
  PFN_vkGetMemoryFdPropertiesKHR    getMemoryFdPropertiesFn_ = nullptr;
  PFN_vkGetSemaphoreFdKHR           getSemaphoreFdFn_ = nullptr;
  PFN_vkImportSemaphoreFdKHR        importSemaphoreFdFn_ = nullptr;

  std::mutex                        timelineSignalMtx_;
  std::atomic<uint64_t>             nextTimelineValue_{0};

  std::atomic<bool>         lost_{false};

  // Global context registry (indexed by deviceId).
  static std::mutex                                contextsMtx_;
  static std::vector<std::unique_ptr<VulkanDeviceContext>> contexts_;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
#endif  // LIBND4J_VULKAN_DEVICE_CONTEXT_H
