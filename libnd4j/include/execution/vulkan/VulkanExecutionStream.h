/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_EXECUTION_STREAM_H
#define LIBND4J_VULKAN_EXECUTION_STREAM_H

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sd {
namespace graph {

class VulkanDeviceContext;

/**
 * A real Vulkan execution stream.
 *
 * Every stream owns a command pool and an ordered list of fence-backed queue
 * submissions. All streams for a device submit through that device context's
 * canonical queue, so Vulkan queue order provides CUDA-style same-device stream
 * ordering without host-side queue-idle calls. Async transfer staging and
 * deferred frees live until their submission fence completes.
 */
class SD_LIB_EXPORT VulkanExecutionStream {
 public:
  static VulkanExecutionStream* create(int deviceId);
  static bool destroy(VulkanExecutionStream* stream);
  static VulkanExecutionStream* fromOpaque(void* stream, bool allowDefault = true);
  static VulkanExecutionStream* defaultExecution(int deviceId);
  static VulkanExecutionStream* defaultCopy(int deviceId);

  /**
   * Return the stream bound to this thread, or the device's execution stream
   * when no explicit binding exists. This is the capture-safe stream lookup
   * used by Vulkan helpers whose legacy interface does not carry a stream.
   */
  static VulkanExecutionStream* currentOrDefault(int deviceId = -1);

  /** Bind an active stream to this thread and return the previous binding. */
  static VulkanExecutionStream* setCurrent(VulkanExecutionStream* stream);

  static bool synchronizeDevice(int deviceId);
  static void destroyAll();

  /**
   * Query whether a direct Vulkan copy is legal for the two device contexts.
   * Opaque-FD import/export is restricted to logical devices backed by the same
   * underlying physical device. Distinct physical devices require a shared
   * device-group VkDevice with heap-specific peer-memory support.
   */
  static bool isCrossDeviceCopySupported(int srcDeviceId, int dstDeviceId,
                                         std::string* reason = nullptr);

  ~VulkanExecutionStream();

  VulkanExecutionStream(const VulkanExecutionStream&) = delete;
  VulkanExecutionStream& operator=(const VulkanExecutionStream&) = delete;

  int deviceId() const { return deviceId_; }
  bool isActive() const;
  uint64_t lastSequence() const;

  /** NativeOps copy flags: 0 H2H, 1 H2D, 2 D2H, 3 D2D. */
  bool enqueueCopy(void* dst, const void* src, VkDeviceSize bytes, int direction,
                   VkDeviceSize dstOffset = 0, VkDeviceSize srcOffset = 0);

  /** Byte-wise device memset, preserving stream ordering for every byte count. */
  bool enqueueFill(void* dst, int value, VkDeviceSize bytes, VkDeviceSize dstOffset = 0);

  /**
   * Submit an externally owned, executable command buffer. The caller must keep
   * it alive and must not reset it before waitThrough(sequence) completes.
   */
  uint64_t submitExternal(VkCommandBuffer commandBuffer);

  /**
   * Record and submit one stream-owned command buffer. Cleanup callbacks run
   * after its fence completes (or immediately if recording/submission fails).
   */
  uint64_t enqueueCommands(
      const std::function<bool(VkCommandBuffer)>& recorder,
      std::vector<std::function<void()>> cleanupCallbacks = {});

  /** Enqueue a host callback after all submissions previously queued here. */
  uint64_t enqueueHostCallback(std::function<void()> callback);

  bool waitThrough(uint64_t sequence);
  bool synchronize();
  bool waitEvent(const class VulkanExecutionEvent& event);

  /**
   * Release a pool allocation only after all work currently queued on this
   * stream completes. This is the Vulkan equivalent of cudaFreeAsync ordering.
   */
  bool retireAllocation(void* ptr);

 private:
  struct StagingAllocation {
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    bool coherent = false;
  };

  struct HostCopy {
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
  };

  struct PendingSubmission {
    uint64_t sequence = 0;
    VkFence fence = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    bool ownsCommandBuffer = false;
    std::vector<StagingAllocation> staging;
    std::vector<HostCopy> hostCopies;
    std::vector<std::function<void()>> completionCallbacks;
    std::vector<std::function<void()>> cleanupCallbacks;
  };

  explicit VulkanExecutionStream(int deviceId, bool backendOwned);

  bool initialize();
  bool createStaging(VkDeviceSize bytes, VkBufferUsageFlags usage,
                     StagingAllocation& staging);
  bool flushStaging(const StagingAllocation& staging);
  bool invalidateStaging(const StagingAllocation& staging);
  VkCommandBuffer beginOneShot();
  bool endOneShot(VkCommandBuffer commandBuffer);
  uint64_t submit(VkCommandBuffer commandBuffer, bool ownsCommandBuffer,
                  std::vector<StagingAllocation>&& staging,
                  std::vector<HostCopy>&& hostCopies,
                  std::vector<std::function<void()>>&& callbacks,
                  VkSemaphore waitSemaphore = VK_NULL_HANDLE,
                  VkPipelineStageFlags waitStage =
                      VK_PIPELINE_STAGE_TRANSFER_BIT,
                  std::vector<std::function<void()>>&& cleanupCallbacks = {});
    bool collectCompleted(bool wait, uint64_t throughSequence);
    void completeAndDestroy(PendingSubmission& pending);
    void discardAndDestroy(PendingSubmission& pending);
    void destroyStaging(StagingAllocation& staging);
  static VulkanExecutionStream* createBackendOwned(int deviceId);
  static VulkanExecutionStream* lookup(void* stream, bool includeDestroyed);

  int deviceId_ = -1;
  bool backendOwned_ = false;
  bool active_ = false;
  VulkanDeviceContext* context_ = nullptr;
  VkDevice device_ = VK_NULL_HANDLE;
  VkCommandPool commandPool_ = VK_NULL_HANDLE;

  mutable std::mutex mutex_;
  uint64_t nextSequence_ = 1;
  uint64_t completedSequence_ = 0;
  std::deque<PendingSubmission> pending_;

  static std::mutex registryMutex_;
  static std::unordered_set<VulkanExecutionStream*> allStreams_;
  static std::unordered_set<VulkanExecutionStream*> activeStreams_;
  static std::vector<VulkanExecutionStream*> ownedStreams_;
  static std::unordered_map<int, VulkanExecutionStream*> defaultExecutionStreams_;
  static std::unordered_map<int, VulkanExecutionStream*> defaultCopyStreams_;
};

/** Scope a legacy stream-less helper to the caller's Vulkan execution stream. */
class SD_LIB_EXPORT VulkanExecutionStreamGuard {
 public:
  explicit VulkanExecutionStreamGuard(VulkanExecutionStream* stream)
      : previous_(VulkanExecutionStream::setCurrent(stream)) {}
  ~VulkanExecutionStreamGuard() { VulkanExecutionStream::setCurrent(previous_); }

  VulkanExecutionStreamGuard(const VulkanExecutionStreamGuard&) = delete;
  VulkanExecutionStreamGuard& operator=(const VulkanExecutionStreamGuard&) = delete;

 private:
  VulkanExecutionStream* previous_;
};

/** Fence position recorded from a VulkanExecutionStream. */
class SD_LIB_EXPORT VulkanExecutionEvent {
 public:
  static VulkanExecutionEvent* create();
  static bool destroy(VulkanExecutionEvent* event);
  static void destroyAll();
  static VulkanExecutionEvent* fromOpaque(void* event);

  bool record(VulkanExecutionStream* stream);
  bool synchronize();
  int deviceId() const { return deviceId_; }
  bool isRecorded() const { return stream_ != nullptr || completed_; }

 private:
  VulkanExecutionEvent() = default;
  ~VulkanExecutionEvent() = default;

  mutable std::mutex mutex_;
  VulkanExecutionStream* stream_ = nullptr;
  uint64_t sequence_ = 0;
  int deviceId_ = -1;
  bool completed_ = false;

  static std::mutex registryMutex_;
  static std::unordered_set<VulkanExecutionEvent*> events_;

  friend class VulkanExecutionStream;
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
#endif  // LIBND4J_VULKAN_EXECUTION_STREAM_H
