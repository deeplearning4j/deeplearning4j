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

#if defined(HAVE_VULKAN)

#include <graph/vulkan/VulkanReplayHandle.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace sd {
namespace graph {

// ── Timing helper ────────────────────────────────────────────────────────────

static double nowMs() {
  auto tp = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(tp.time_since_epoch()).count();
}

// ── Constructor / Destructor ─────────────────────────────────────────────────

VulkanReplayHandle::VulkanReplayHandle(int deviceId)
    : state_(ReplayState::EMPTY),
      deviceId_(deviceId) {
}

VulkanReplayHandle::~VulkanReplayHandle() {
  freeHostPointers();
  releaseWorkspace();
  cleanup();
}

// ── Vulkan Initialization ────────────────────────────────────────────────────

bool VulkanReplayHandle::initVulkan() {
  if (initialized_) return true;

  // --- Create VkInstance (no validation layers for production perf) ---
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "nd4j-vulkan-replay";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "libnd4j";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo instanceInfo = {};
  instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceInfo.pApplicationInfo = &appInfo;
  instanceInfo.enabledLayerCount = 0;
  instanceInfo.enabledExtensionCount = 0;

  VkResult result = vkCreateInstance(&instanceInfo, nullptr, &instance_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkCreateInstance failed (result=%d)\n", static_cast<int>(result));
    return false;
  }

  // --- Enumerate physical devices and select by deviceId ---
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
  if (deviceCount == 0) {
    sd_printf("VulkanReplayHandle: no Vulkan-capable physical devices found\n", 0);
    cleanup();
    return false;
  }

  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

  uint32_t selectedIdx = (deviceId_ >= 0 && static_cast<uint32_t>(deviceId_) < deviceCount)
                             ? static_cast<uint32_t>(deviceId_)
                             : 0;
  physicalDevice_ = devices[selectedIdx];

  VkPhysicalDeviceProperties devProps;
  vkGetPhysicalDeviceProperties(physicalDevice_, &devProps);
  sd_printf("VulkanReplayHandle: using device %s (index %u)\n", devProps.deviceName, selectedIdx);

  // --- Find a compute queue family ---
  computeQueueFamily_ = findComputeQueueFamily();
  if (computeQueueFamily_ == UINT32_MAX) {
    sd_printf("VulkanReplayHandle: no compute queue family found\n", 0);
    cleanup();
    return false;
  }

  // --- Create logical device with one compute queue ---
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo = {};
  queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueInfo.queueFamilyIndex = computeQueueFamily_;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceInfo = {};
  deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  deviceInfo.enabledExtensionCount = 0;
  deviceInfo.enabledLayerCount = 0;

  result = vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkCreateDevice failed (result=%d)\n", static_cast<int>(result));
    cleanup();
    return false;
  }

  vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);

  // --- Create command pool with reset capability ---
  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = computeQueueFamily_;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  result = vkCreateCommandPool(device_, &poolInfo, nullptr, &cmdPool_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkCreateCommandPool failed (result=%d)\n", static_cast<int>(result));
    cleanup();
    return false;
  }

  // --- Allocate a single primary command buffer ---
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = cmdPool_;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  result = vkAllocateCommandBuffers(device_, &allocInfo, &cmdBuffer_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkAllocateCommandBuffers failed (result=%d)\n", static_cast<int>(result));
    cleanup();
    return false;
  }

  // --- Create fence for CPU-GPU synchronization ---
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled so first wait succeeds

  result = vkCreateFence(device_, &fenceInfo, nullptr, &fence_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkCreateFence failed (result=%d)\n", static_cast<int>(result));
    cleanup();
    return false;
  }

  initialized_ = true;
  return true;
}

uint32_t VulkanReplayHandle::findComputeQueueFamily() {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyCount, families.data());

  // Prefer a compute-only queue family (no graphics) for dedicated compute work
  uint32_t computeOnly = UINT32_MAX;
  uint32_t computeAny = UINT32_MAX;

  for (uint32_t i = 0; i < queueFamilyCount; ++i) {
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      if (computeAny == UINT32_MAX) {
        computeAny = i;
      }
      // Compute-only: has compute but NOT graphics
      if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && computeOnly == UINT32_MAX) {
        computeOnly = i;
      }
    }
  }

  return (computeOnly != UINT32_MAX) ? computeOnly : computeAny;
}

uint32_t VulkanReplayHandle::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeFilter & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  return UINT32_MAX;
}

// ── Capture Lifecycle ────────────────────────────────────────────────────────

bool VulkanReplayHandle::beginCapture(void* /*stream*/) {
  // Lazy initialization on first capture
  if (!initialized_ && !initVulkan()) {
    state_ = ReplayState::ERROR;
    return false;
  }

  // Wait for any in-flight replay to complete before resetting the command buffer
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);

  // Reset the command buffer for fresh recording
  VkResult result = vkResetCommandBuffer(cmdBuffer_, 0);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkResetCommandBuffer failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERROR;
    return false;
  }

  // Begin recording with SIMULTANEOUS_USE_BIT so the same buffer can be
  // submitted multiple times without re-recording
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  result = vkBeginCommandBuffer(cmdBuffer_, &beginInfo);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkBeginCommandBuffer failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERROR;
    return false;
  }

  numDispatches_ = 0;
  captureStartTimeMs_ = nowMs();
  state_ = ReplayState::CAPTURING;
  return true;
}

bool VulkanReplayHandle::endCapture(void* /*stream*/) {
  if (state_ != ReplayState::CAPTURING) {
    sd_printf("VulkanReplayHandle: endCapture called but state is not CAPTURING (state=%d)\n",
              static_cast<int>(state_));
    return false;
  }

  VkResult result = vkEndCommandBuffer(cmdBuffer_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkEndCommandBuffer failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERROR;
    return false;
  }

  captureTimeMs_ = nowMs() - captureStartTimeMs_;
  state_ = ReplayState::CAPTURED;
  return true;
}

bool VulkanReplayHandle::finalize() {
  if (state_ != ReplayState::CAPTURED) {
    sd_printf("VulkanReplayHandle: finalize called but state is not CAPTURED (state=%d)\n",
              static_cast<int>(state_));
    return false;
  }

  // For Vulkan, the command buffer is already resubmittable after
  // vkEndCommandBuffer. No additional instantiation step is needed
  // (unlike CUDA which requires cudaGraphInstantiate). This method
  // exists to satisfy the GraphReplayHandle lifecycle contract.
  state_ = ReplayState::READY;
  return true;
}

// ── Replay ───────────────────────────────────────────────────────────────────

bool VulkanReplayHandle::replay(void* /*stream*/) {
  if (state_ != ReplayState::READY) {
    sd_printf("VulkanReplayHandle: replay called but state is not READY (state=%d)\n",
              static_cast<int>(state_));
    return false;
  }

  double replayStart = nowMs();

  // Wait for the previous submission to complete
  VkResult result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkWaitForFences failed before submit (result=%d)\n",
              static_cast<int>(result));
    state_ = ReplayState::ERROR;
    return false;
  }

  // Reset fence for this submission
  result = vkResetFences(device_, 1, &fence_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkResetFences failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERROR;
    return false;
  }

  // Submit the recorded command buffer
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuffer_;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.signalSemaphoreCount = 0;

  result = vkQueueSubmit(computeQueue_, 1, &submitInfo, fence_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkQueueSubmit failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERROR;
    return false;
  }

  // Wait for completion (synchronous replay, matching CUDA graph behavior)
  result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkWaitForFences failed after submit (result=%d)\n",
              static_cast<int>(result));
    state_ = ReplayState::ERROR;
    return false;
  }

  lastReplayTimeMs_ = nowMs() - replayStart;
  replayCount_++;
  return true;
}

// ── Compute Dispatch Recording ───────────────────────────────────────────────

void VulkanReplayHandle::recordDispatch(VkPipeline pipeline, VkPipelineLayout layout,
                                         VkDescriptorSet descriptorSet,
                                         uint32_t groupCountX, uint32_t groupCountY,
                                         uint32_t groupCountZ) {
  if (state_ != ReplayState::CAPTURING) {
    sd_printf("VulkanReplayHandle: recordDispatch called outside of capture\n", 0);
    return;
  }

  vkCmdBindPipeline(cmdBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(cmdBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                          0, 1, &descriptorSet, 0, nullptr);
  vkCmdDispatch(cmdBuffer_, groupCountX, groupCountY, groupCountZ);

  numDispatches_++;
}

void VulkanReplayHandle::recordComputeBarrier() {
  if (state_ != ReplayState::CAPTURING) {
    sd_printf("VulkanReplayHandle: recordComputeBarrier called outside of capture\n", 0);
    return;
  }

  // Full memory barrier for compute shader writes -> compute shader reads.
  // This is the compute equivalent of a global memory fence between dispatches.
  VkMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(cmdBuffer_,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // src stage
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,  // dst stage
                       0,          // dependency flags
                       1, &barrier,  // memory barriers
                       0, nullptr,   // buffer barriers
                       0, nullptr);  // image barriers
}

// ── State & Statistics ───────────────────────────────────────────────────────

ReplayState VulkanReplayHandle::getState() const {
  return state_;
}

ReplayStatistics VulkanReplayHandle::getStatistics() const {
  ReplayStatistics stats;
  stats.numOperations = numDispatches_;
  stats.numMemoryOps = 0;  // Tracked separately if buffer copies are recorded
  stats.estimatedMemory = workspaceSize_;
  stats.captureTimeMs = captureTimeMs_;
  stats.lastReplayTimeMs = lastReplayTimeMs_;
  stats.replayCount = replayCount_;
  return stats;
}

// ── Workspace Management ─────────────────────────────────────────────────────

bool VulkanReplayHandle::allocateWorkspace(size_t bytes, int /*deviceId*/,
                                            void* /*registryPtr*/, int /*segIdx*/) {
  if (workspaceBuffer_ != VK_NULL_HANDLE) return true;  // Already allocated

  if (!initialized_ && !initVulkan()) return false;

  // Create a device-local storage buffer for workspace
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bytes;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(device_, &bufferInfo, nullptr, &workspaceBuffer_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkCreateBuffer failed for workspace (result=%d)\n",
              static_cast<int>(result));
    return false;
  }

  // Query memory requirements
  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device_, workspaceBuffer_, &memReqs);

  // Allocate device-local memory
  uint32_t memType = findMemoryType(memReqs.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (memType == UINT32_MAX) {
    sd_printf("VulkanReplayHandle: no suitable device-local memory type for workspace\n", 0);
    vkDestroyBuffer(device_, workspaceBuffer_, nullptr);
    workspaceBuffer_ = VK_NULL_HANDLE;
    return false;
  }

  VkMemoryAllocateInfo memAllocInfo = {};
  memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAllocInfo.allocationSize = memReqs.size;
  memAllocInfo.memoryTypeIndex = memType;

  result = vkAllocateMemory(device_, &memAllocInfo, nullptr, &workspaceMemory_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkAllocateMemory failed for workspace (result=%d)\n",
              static_cast<int>(result));
    vkDestroyBuffer(device_, workspaceBuffer_, nullptr);
    workspaceBuffer_ = VK_NULL_HANDLE;
    return false;
  }

  result = vkBindBufferMemory(device_, workspaceBuffer_, workspaceMemory_, 0);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkBindBufferMemory failed for workspace (result=%d)\n",
              static_cast<int>(result));
    vkFreeMemory(device_, workspaceMemory_, nullptr);
    vkDestroyBuffer(device_, workspaceBuffer_, nullptr);
    workspaceBuffer_ = VK_NULL_HANDLE;
    workspaceMemory_ = VK_NULL_HANDLE;
    return false;
  }

  workspaceSize_ = bytes;
  // Expose to base class for getWorkspacePtr() / getWorkspaceBytes()
  // Note: device-local memory is not host-mappable, so captureWorkspacePtr_
  // stores the VkBuffer handle cast for identification purposes only.
  // Actual GPU access goes through descriptor set bindings.
  captureWorkspaceBytes_ = bytes;

  sd_printf("VulkanReplayHandle: allocated %zuMB workspace (device-local)\n",
            bytes / (1024 * 1024));
  return true;
}

void VulkanReplayHandle::releaseWorkspace(void* /*registryPtr*/, int /*segIdx*/) {
  if (device_ == VK_NULL_HANDLE) return;

  if (workspaceBuffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, workspaceBuffer_, nullptr);
    workspaceBuffer_ = VK_NULL_HANDLE;
  }
  if (workspaceMemory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, workspaceMemory_, nullptr);
    workspaceMemory_ = VK_NULL_HANDLE;
  }
  workspaceSize_ = 0;
  captureWorkspacePtr_ = nullptr;
  captureWorkspaceBytes_ = 0;
}

void VulkanReplayHandle::freeHostPointers() {
  if (device_ == VK_NULL_HANDLE) return;

  // Free host-visible staging buffers
  for (size_t i = 0; i < hostBuffers_.size(); ++i) {
    if (hostBuffers_[i] != VK_NULL_HANDLE) {
      vkDestroyBuffer(device_, hostBuffers_[i], nullptr);
    }
  }
  hostBuffers_.clear();

  for (size_t i = 0; i < hostMemoryAllocations_.size(); ++i) {
    if (hostMemoryAllocations_[i] != VK_NULL_HANDLE) {
      vkFreeMemory(device_, hostMemoryAllocations_[i], nullptr);
    }
  }
  hostMemoryAllocations_.clear();

  // Also free any raw host pointers tracked by the base class
  for (auto* ptr : capturedHostPtrs_) {
    if (ptr != nullptr) {
      free(ptr);
    }
  }
  capturedHostPtrs_.clear();
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

void VulkanReplayHandle::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    // Wait for all device work to complete before destroying objects
    vkDeviceWaitIdle(device_);

    // Destroy in reverse creation order
    if (fence_ != VK_NULL_HANDLE) {
      vkDestroyFence(device_, fence_, nullptr);
      fence_ = VK_NULL_HANDLE;
    }

    // Command buffer is freed implicitly when the pool is destroyed
    cmdBuffer_ = VK_NULL_HANDLE;

    if (cmdPool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, cmdPool_, nullptr);
      cmdPool_ = VK_NULL_HANDLE;
    }

    vkDestroyDevice(device_, nullptr);
    device_ = VK_NULL_HANDLE;
  }

  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE;
  }

  physicalDevice_ = VK_NULL_HANDLE;
  computeQueue_ = VK_NULL_HANDLE;
  initialized_ = false;
  state_ = ReplayState::EMPTY;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
