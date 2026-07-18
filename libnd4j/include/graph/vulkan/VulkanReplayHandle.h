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

#ifndef LIBND4J_VULKAN_REPLAY_HANDLE_H
#define LIBND4J_VULKAN_REPLAY_HANDLE_H

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/GraphReplayHandle.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanPipelineCache.h>
#include <vulkan/vulkan.h>

#include <memory>
#include <string>
#include <vector>

// Include the full recorder header so unique_ptr<VulkanSegmentRecorder> can
// generate a complete destructor without an incomplete-type error.
#include <graph/vulkan/VulkanSegmentRecorder.h>

namespace sd {
namespace graph {

/**
 * Vulkan compute implementation of GraphReplayHandle.
 *
 * Records Vulkan compute commands into a command buffer once, then replays
 * by resubmitting the same command buffer on subsequent calls. Vulkan
 * command buffers are explicitly designed for record-once, submit-many
 * usage, making this a natural fit for the replay abstraction.
 *
 * Lifecycle:
 *   1. beginCapture() - resets and begins recording the command buffer
 *   2. [caller records compute dispatches via recordDispatch() or directly]
 *   3. endCapture()   - ends command buffer recording
 *   4. finalize()     - transitions to READY (no-op for Vulkan; buffer is
 *                       already resubmittable after endCapture)
 *   5. replay()       - submits the command buffer, waits via fence
 *
 * The handle borrows the canonical VkDevice and compute queue from
 * VulkanDeviceContext. It owns only replay-local command, fence, workspace,
 * descriptor, and compiled-pipeline state.
 */
class SD_LIB_EXPORT VulkanReplayHandle : public GraphReplayHandle {
 public:
  explicit VulkanReplayHandle(int deviceId = 0);
  ~VulkanReplayHandle() override;

  // Non-copyable
  VulkanReplayHandle(const VulkanReplayHandle&) = delete;
  VulkanReplayHandle& operator=(const VulkanReplayHandle&) = delete;

  // -- GraphReplayHandle interface ------------------------------------------

  bool beginCapture(void* stream) override;
  bool endCapture(void* stream) override;
  bool finalize() override;
  bool replay(void* stream) override;

  ReplayState getState() const override;
  ReplayStatistics getStatistics() const override;
  const char* backendName() const override { return "Vulkan Compute"; }

  // -- Workspace management -------------------------------------------------

  bool allocateWorkspace(size_t bytes, int deviceId = 0,
                         void* registryPtr = nullptr, int segIdx = 0) override;
  void releaseWorkspace(void* registryPtr = nullptr, int segIdx = 0) override;
  void freeHostPointers() override;

  // -- Vulkan-specific accessors --------------------------------------------

  /** Get the recorded command buffer (valid after beginCapture). */
  VkCommandBuffer getCommandBuffer() const { return cmdBuffer_; }

  /** Get the logical device for creating pipelines/descriptors. */
  VkDevice getDevice() const { return device_; }

  /** Get the physical device for querying memory properties. */
  VkPhysicalDevice getPhysicalDevice() const { return physicalDevice_; }

  /** Get the compute queue family index. */
  uint32_t getComputeQueueFamily() const { return computeQueueFamily_; }

  /** Get device ID this handle was created for. */
  int getDeviceId() const override { return deviceId_; }

  /** Wait only for this handle's most recent replay submission. */
  VkResult waitForReplayIdle() const {
    return device_ != VK_NULL_HANDLE && fence_ != VK_NULL_HANDLE
               ? vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX)
               : VK_ERROR_INITIALIZATION_FAILED;
  }

  /** Capabilities enabled on the exact logical device used for replay. */
  const VulkanDeviceCaps* getDeviceCaps() const {
    return deviceContext_ != nullptr ? &deviceContext_->caps() : nullptr;
  }

  bool isFp16Supported() const {
    return deviceContext_ != nullptr && deviceContext_->caps().fp16 &&
           deviceContext_->caps().storage16;
  }

  /** Get the device's subgroup size (Adreno: 64/128, Mali: 16). */
  uint32_t getSubgroupSize() const {
    return deviceContext_ != nullptr ? deviceContext_->caps().subgroupSize : 0;
  }

  /**
   * Check if the device supports Unified Memory Architecture (UMA).
   * UMA means HOST_VISIBLE and DEVICE_LOCAL share the same physical memory,
   * enabling zero-copy access without staging buffers on mobile GPUs.
   * Valid only after initVulkan() has been called (i.e., after the first
   * beginCapture() or allocateWorkspace()).
   */
  bool isUmaAvailable() const { return umaDetected_; }

  /**
   * Record a compute dispatch into the command buffer during capture.
   *
   * Binds the given compute pipeline and descriptor set, then records a
   * vkCmdDispatch with the specified work group counts.
   *
   * @param pipeline       Compute pipeline to bind
   * @param layout         Pipeline layout matching the descriptor set layout
   * @param descriptorSet  Descriptor set with buffer bindings
   * @param groupCountX    Number of work groups in X dimension
   * @param groupCountY    Number of work groups in Y dimension
   * @param groupCountZ    Number of work groups in Z dimension
   */
  void recordDispatch(VkPipeline pipeline, VkPipelineLayout layout,
                      VkDescriptorSet descriptorSet,
                      uint32_t groupCountX, uint32_t groupCountY,
                      uint32_t groupCountZ);

  /**
   * Record a compute dispatch by compiling (or retrieving a cached) pipeline
   * from the given MLIR module string.
   *
   * On the first call for a (opName, mlirModuleStr) pair the MLIR module is
   * lowered to SPIR-V, a VkPipeline is created, and the result is cached in
   * pipelineCache_. Subsequent calls for the same pair reuse the cached
   * pipeline without recompilation.
   *
   * @param opName         Human-readable op name used for the cache key and
   *                       diagnostic messages.
   * @param mlirModuleStr  Textual MLIR module containing the compute kernel.
   * @param descriptorSet  Descriptor set with buffer bindings (must match the
   *                       layout created by VulkanPipelineCache).
   * @param groupCountX    Number of work groups in X dimension.
   * @param groupCountY    Number of work groups in Y dimension.
   * @param groupCountZ    Number of work groups in Z dimension.
   * @return               true on success, false if pipeline compilation fails.
   */
  bool recordDispatch(const std::string& opName,
                      const std::string& mlirModuleStr,
                      VkDescriptorSet descriptorSet,
                      uint32_t groupCountX, uint32_t groupCountY,
                      uint32_t groupCountZ);

  /**
   * Record a memory barrier for compute-to-compute synchronization.
   *
   * Inserts a pipeline barrier between dispatches to ensure writes from
   * one dispatch are visible to the next.
   */
  void recordComputeBarrier();

  /** Number of dispatches recorded so far (valid after endCapture). */
  int getNumDispatches() const { return numDispatches_; }

  /**
   * True when this handle has at least one recorded compute dispatch and
   * therefore replay() will execute GPU work. A false result is valid for a
   * captured segment containing only view/identity aliases: replay remains a
   * zero-compute metadata operation, matching CUDA DSP. Callers must reject an
   * empty command buffer only when the segment itself contains compute slots.
   */
  bool replayDoesCompute() const { return numDispatches_ > 0; }

  /**
   * Return the MLIR pipeline cache (created lazily in initVulkan()).
   * Used by VulkanSegmentRecorder to compile and cache op pipelines.
   */
  VulkanPipelineCache* getPipelineCache() const { return pipelineCache_.get(); }

  /** Attach a recorder for Vulkan-native segment dispatch. */
  void setRecorder(std::unique_ptr<VulkanSegmentRecorder> recorder) {
    recorder_ = std::move(recorder);
  }

  /** Return the attached recorder, or nullptr if none. */
  VulkanSegmentRecorder* getRecorder() const { return recorder_.get(); }

  /** Release and return ownership of the recorder (caller takes it). */
  std::unique_ptr<VulkanSegmentRecorder> releaseRecorder() {
    return std::move(recorder_);
  }

  // -- Android lifecycle management -----------------------------------------

  /**
   * Suspend Vulkan resources on device loss (pause event).
   * Invalidates command buffers and cleans up device-specific state.
   */
  void suspend();

  /**
   * Resume Vulkan execution after device loss (resume event).
   * Reinitializes Vulkan if necessary.
   */
  void resume();

  /**
   * Check if device has been lost.
   * @return true if device loss was detected, false otherwise
   */
  bool isDeviceLost() const;

 private:
  ReplayState state_;
  int deviceId_;

  // Vulkan handles
  // instance_, physicalDevice_, and device_ are borrowed from the canonical
  // VulkanDeviceContext/manager and are never destroyed here.
  // cmdPool_, cmdBuffer_, and fence_ are replay-local and owned by this handle.
  VulkanDeviceContext* deviceContext_ = nullptr;
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  uint32_t computeQueueFamily_ = 0;
  VkCommandPool cmdPool_ = VK_NULL_HANDLE;
  VkCommandBuffer cmdBuffer_ = VK_NULL_HANDLE;
  VkFence fence_ = VK_NULL_HANDLE;

  // Workspace memory
  VkBuffer workspaceBuffer_ = VK_NULL_HANDLE;
  VkDeviceMemory workspaceMemory_ = VK_NULL_HANDLE;
  size_t workspaceSize_ = 0;

  // Host-visible staging buffers tracked for freeHostPointers
  std::vector<VkBuffer> hostBuffers_;
  std::vector<VkDeviceMemory> hostMemoryAllocations_;

  // Statistics
  int replayCount_ = 0;
  int numDispatches_ = 0;
  double captureStartTimeMs_ = 0.0;
  double captureTimeMs_ = 0.0;
  double lastReplayTimeMs_ = 0.0;

  // Device info for diagnostics
  std::string deviceName_;
  uint32_t apiVersion_ = 0;

  bool initialized_ = false;
  bool umaDetected_ = false;
  bool deviceLost_ = false;

  // Pipeline cache: compiles MLIR→SPIR-V→VkPipeline and caches results.
  // Created lazily in initVulkan() after the device is ready.
  std::unique_ptr<VulkanPipelineCache> pipelineCache_;

  // Optional segment recorder: when set, Vulkan commands for each op have
  // been recorded and replay() will execute them (no slot-by-slot fallback).
  std::unique_ptr<VulkanSegmentRecorder> recorder_;

  /**
   * Initialize the Vulkan instance, device, queue, command pool, command
   * buffer, and fence. Called lazily on first beginCapture().
   * @return true if all Vulkan objects were created successfully
   */
  bool initVulkan();

  /**
   * Find a memory type matching the given filter and property flags.
   * @param typeFilter  Bitmask from VkMemoryRequirements::memoryTypeBits
   * @param properties  Required memory property flags
   * @return Memory type index, or UINT32_MAX if none found
   */
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

  /**
   * Detect if device supports Unified Memory Architecture (UMA).
   * UMA means HOST_VISIBLE and DEVICE_LOCAL are in the same memory heap,
   * avoiding the need for staging buffers on mobile GPUs.
   */
  void detectUMA();

  /**
   * Destroy all Vulkan objects in reverse creation order.
   * Safe to call multiple times and with VK_NULL_HANDLE values.
   */
  void cleanup();
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
#endif  // LIBND4J_VULKAN_REPLAY_HANDLE_H
