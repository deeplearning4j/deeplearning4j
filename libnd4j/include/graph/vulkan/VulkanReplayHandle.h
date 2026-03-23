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

#if defined(HAVE_VULKAN)

#include <graph/GraphReplayHandle.h>
#include <vulkan/vulkan.h>

#include <memory>
#include <vector>

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
 * The handle owns its own VkInstance, VkDevice, VkCommandPool, and fence.
 * A single compute queue family is selected at initialization. Workspace
 * memory is allocated as device-local VkBuffer + VkDeviceMemory.
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
  int getDeviceId() const { return deviceId_; }

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
   * Record a memory barrier for compute-to-compute synchronization.
   *
   * Inserts a pipeline barrier between dispatches to ensure writes from
   * one dispatch are visible to the next.
   */
  void recordComputeBarrier();

 private:
  ReplayState state_;
  int deviceId_;

  // Vulkan handles (owned)
  VkInstance instance_ = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
  VkDevice device_ = VK_NULL_HANDLE;
  VkQueue computeQueue_ = VK_NULL_HANDLE;
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

  bool initialized_ = false;

  /**
   * Initialize the Vulkan instance, device, queue, command pool, command
   * buffer, and fence. Called lazily on first beginCapture().
   * @return true if all Vulkan objects were created successfully
   */
  bool initVulkan();

  /**
   * Find a queue family that supports compute operations.
   * @return Queue family index, or UINT32_MAX if none found
   */
  uint32_t findComputeQueueFamily();

  /**
   * Find a memory type matching the given filter and property flags.
   * @param typeFilter  Bitmask from VkMemoryRequirements::memoryTypeBits
   * @param properties  Required memory property flags
   * @return Memory type index, or UINT32_MAX if none found
   */
  uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

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
