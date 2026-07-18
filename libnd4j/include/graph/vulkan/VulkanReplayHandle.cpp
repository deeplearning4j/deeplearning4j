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

#include <system/common.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanReplayHandle.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/DspDiagnostics.h>

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

  auto& devMgr = VulkanDeviceManager::getInstance();
  if (!devMgr.initialize()) {
    sd_printf("VulkanReplayHandle: VulkanDeviceManager::initialize() failed — no Vulkan devices available\n", 0);
    return false;
  }

  deviceContext_ = VulkanDeviceContext::getContext(deviceId_);
  if (deviceContext_ == nullptr || deviceContext_->isLost()) {
    sd_printf("VulkanReplayHandle: no usable VulkanDeviceContext for device %d\n", deviceId_);
    deviceContext_ = nullptr;
    return false;
  }

  // Borrow the exact logical device whose enabled capabilities are reported by
  // VulkanDeviceCaps. Replay state must never create a parallel feature-less
  // VkDevice and then infer support from the physical device.
  instance_ = devMgr.getInstance_();
  physicalDevice_ = deviceContext_->physicalDevice();
  device_ = deviceContext_->device();
  computeQueueFamily_ = deviceContext_->caps().computeQueueFamily;

  const VulkanDeviceInfo* devInfo = devMgr.getDeviceInfo(deviceId_);
  deviceName_ = devInfo ? devInfo->name : "unknown";
  apiVersion_ = deviceContext_->caps().apiVersion;

  DSP_DIAG(BACKEND,
           "VulkanReplayHandle: borrowing device %s (index %d), fp16=%s storage16=%s fp64=%s subgroup=%u",
           deviceName_.c_str(), deviceId_,
           deviceContext_->caps().fp16 ? "yes" : "no",
           deviceContext_->caps().storage16 ? "yes" : "no",
           deviceContext_->caps().fp64 ? "yes" : "no",
           deviceContext_->caps().subgroupSize);

  VkResult result = VK_SUCCESS;

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

  detectUMA();

  // Create the pipeline cache now that device_ and physicalDevice_ are valid.
  // The cache compiles MLIR modules to SPIR-V and caches the resulting
  // VkPipeline objects so that each op signature is only compiled once.
  pipelineCache_ = std::make_unique<VulkanPipelineCache>(
      device_, physicalDevice_, deviceContext_->caps(),
      deviceContext_->pipelineCache());
  DSP_DIAG(BACKEND, "VulkanReplayHandle: VulkanPipelineCache created");

  initialized_ = true;
  return true;
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

void VulkanReplayHandle::detectUMA() {
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);

  umaDetected_ = false;

  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    VkMemoryPropertyFlags flags = memProps.memoryTypes[i].propertyFlags;
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      umaDetected_ = true;
      break;
    }
  }

  // Canonical log line checked by the build-verification grep:
  //   "VulkanReplayHandle: UMA detected=true"  -- mobile/integrated GPU
  //   "VulkanReplayHandle: UMA detected=false" -- discrete dGPU (staging path)
  sd_printf("VulkanReplayHandle: UMA detected=%s\n", umaDetected_ ? "true" : "false");
  if (umaDetected_) {
    sd_printf("VulkanReplayHandle: host-visible device-local memory available - zero-copy workspace path enabled\n", 0);
  } else {
    sd_printf("VulkanReplayHandle: discrete GPU memory model - staging buffers required for workspace\n", 0);
  }
}

// ── Capture Lifecycle ────────────────────────────────────────────────────────

bool VulkanReplayHandle::beginCapture(void* /*stream*/) {
  // Lazy initialization on first capture
  if (!initialized_ && !initVulkan()) {
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Wait for any in-flight replay to complete before resetting the command buffer
  vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);

  // Reset the command buffer for fresh recording
  VkResult result = vkResetCommandBuffer(cmdBuffer_, 0);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkResetCommandBuffer failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERRORED;
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
    state_ = ReplayState::ERRORED;
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
    state_ = ReplayState::ERRORED;
    return false;
  }

  captureTimeMs_ = nowMs() - captureStartTimeMs_;
  state_ = ReplayState::CAPTURED;

  // Emit structured GRAPH_REPLAY diagnostic event with Vulkan capture stats.
  // The message format is intentionally parseable (vulkan_backend key=value pairs)
  // so generateJsonReport() can extract and surface them in the "vulkan" JSON block.
  DSP_DIAG(GRAPH_REPLAY,
      "vulkan_backend CAPTURE_DONE device=\"%s\" api_version=0x%08x"
      " dispatches=%d capture_ms=%.3f workspace_bytes=%zu uma=%d fp16=%d",
      deviceName_.c_str(), apiVersion_,
      numDispatches_, captureTimeMs_,
      workspaceSize_, umaDetected_ ? 1 : 0, isFp16Supported() ? 1 : 0);

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

  // Tier-2 kill-safety flush (ADR 0115): capture just created any new
  // pipelines, and mobile processes rarely exit cleanly — persist the driver
  // pipeline-cache blob now rather than only at context destroy. No-op when
  // the blob size is unchanged.
  if (deviceContext_ != nullptr) deviceContext_->savePipelineCacheBlob();

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
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Reset fence for this submission
  result = vkResetFences(device_, 1, &fence_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkResetFences failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Submit the recorded command buffer
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuffer_;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.signalSemaphoreCount = 0;

  result = deviceContext_ != nullptr
               ? deviceContext_->submitCompute(1, &submitInfo, fence_)
               : VK_ERROR_INITIALIZATION_FAILED;
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkQueueSubmit failed (result=%d)\n", static_cast<int>(result));
    state_ = ReplayState::ERRORED;
    return false;
  }

  // Wait for completion (synchronous replay, matching CUDA graph behavior)
  result = vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanReplayHandle: vkWaitForFences failed after submit (result=%d)\n",
              static_cast<int>(result));
    state_ = ReplayState::ERRORED;
    return false;
  }

  lastReplayTimeMs_ = nowMs() - replayStart;
  replayCount_++;
  VulkanPipelineCache::recordKernelLaunches(
      static_cast<uint64_t>(numDispatches_));

  // Emit structured GRAPH_REPLAY diagnostic event with Vulkan replay stats.
  DSP_DIAG(GRAPH_REPLAY,
      "vulkan_backend REPLAY_DONE device=\"%s\" replay_count=%d"
      " replay_ms=%.3f dispatches=%d",
      deviceName_.c_str(), replayCount_,
      lastReplayTimeMs_, numDispatches_);

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

bool VulkanReplayHandle::recordDispatch(const std::string& opName,
                                         const std::string& mlirModuleStr,
                                         VkDescriptorSet descriptorSet,
                                         uint32_t groupCountX,
                                         uint32_t groupCountY,
                                         uint32_t groupCountZ) {
  if (state_ != ReplayState::CAPTURING) {
    sd_printf("VulkanReplayHandle: recordDispatch(MLIR) called outside of capture for op '%s'\n",
              opName.c_str());
    return false;
  }

  if (!pipelineCache_) {
    sd_printf("VulkanReplayHandle: pipelineCache_ is null — was initVulkan() called?\n", 0);
    return false;
  }

  // Obtain (or compile+cache) the VkPipeline for this op.
  VkPipeline pipeline = pipelineCache_->getOrCompile(opName, mlirModuleStr, device_);
  if (pipeline == VK_NULL_HANDLE) {
    sd_printf("VulkanReplayHandle: pipeline compilation failed for op '%s'\n", opName.c_str());
    return false;
  }

  // Retrieve the layout that was created alongside the pipeline.
  // getPipelineLayout() is a cache-read-only lookup (O(log n)) and will never
  // recompile; it returns VK_NULL_HANDLE only if the key was never compiled,
  // which cannot happen here because getOrCompile() just succeeded above.
  VkPipelineLayout layout = pipelineCache_->getPipelineLayout(opName, mlirModuleStr);
  if (layout == VK_NULL_HANDLE) {
    // This should never happen: getOrCompile() succeeded, so the entry is
    // present.  Guard defensively anyway.
    sd_printf("VulkanReplayHandle: internal error — layout missing after successful "
              "getOrCompile() for op '%s'\n", opName.c_str());
    return false;
  }

  // Bind the compute pipeline, then bind the caller-supplied descriptor set
  // using the layout obtained from the cache.  vkCmdBindDescriptorSets
  // requires the same VkPipelineLayout that was used at pipeline creation,
  // which is exactly what getPipelineLayout() returns.
  vkCmdBindPipeline(cmdBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

  if (descriptorSet != VK_NULL_HANDLE) {
    vkCmdBindDescriptorSets(cmdBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                            0, 1, &descriptorSet, 0, nullptr);
  }

  vkCmdDispatch(cmdBuffer_, groupCountX, groupCountY, groupCountZ);
  numDispatches_++;
  return true;
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
  // Vulkan-specific fields (populated after initVulkan())
  stats.deviceName = deviceName_;
  stats.apiVersion = apiVersion_;
  stats.memoryBudgetBytes = workspaceSize_;  // allocated workspace size
  return stats;
}

// ── Workspace Management ─────────────────────────────────────────────────────

bool VulkanReplayHandle::allocateWorkspace(size_t bytes, int /*deviceId*/,
                                            void* /*registryPtr*/, int /*segIdx*/) {
  if (workspaceBuffer_ != VK_NULL_HANDLE) return true;  // Already allocated

  if (!initialized_ && !initVulkan()) return false;

  // Create a storage buffer for workspace
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

  uint32_t memType = UINT32_MAX;
  const char* strategyUsed = nullptr;

  // Try UMA path (HOST_VISIBLE | DEVICE_LOCAL) if available
  if (umaDetected_) {
    memType = findMemoryType(memReqs.memoryTypeBits,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType != UINT32_MAX) {
      strategyUsed = "UMA (host-visible device-local)";
    }
  }

  // Fall back to device-local only if UMA allocation failed or not available
  if (memType == UINT32_MAX) {
    memType = findMemoryType(memReqs.memoryTypeBits,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType != UINT32_MAX) {
      strategyUsed = "discrete GPU (device-local with staging)";
    }
  }

  if (memType == UINT32_MAX) {
    sd_printf("VulkanReplayHandle: no suitable memory type for workspace\n", 0);
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
  captureWorkspaceBytes_ = bytes;

  DSP_DIAG(MEMORY, "VulkanReplayHandle: allocated %zuMB workspace (%s)",
           bytes / (1024 * 1024), strategyUsed);
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

// ── Android Lifecycle Management ─────────────────────────────────────────────

void VulkanReplayHandle::suspend() {
  // On Android pause or device-lost events, invalidate command buffers and
  // release device-side state.  The Vulkan spec requires all outstanding
  // submissions to finish before destroying any device objects, so we wait
  // for idle first and then reset to EMPTY so the next beginCapture() will
  // reinitialize properly via initVulkan().
  if (!initialized_) return;

  // Drain any in-flight GPU work before tearing down command buffers.
  if (device_ != VK_NULL_HANDLE && fence_ != VK_NULL_HANDLE) {
    VkResult waitResult = waitForReplayIdle();
    if (waitResult != VK_SUCCESS) {
      sd_printf("VulkanReplayHandle: suspend fence wait failed "
                "(result=%d)\n",
                static_cast<int>(waitResult));
      state_ = ReplayState::ERRORED;
      return;
    }
  }

  // Invalidate the recorded command buffer — the recorded kernels reference
  // device addresses that may be invalid after a device-lost event.
  if (cmdBuffer_ != VK_NULL_HANDLE && cmdPool_ != VK_NULL_HANDLE &&
      device_ != VK_NULL_HANDLE) {
    vkResetCommandBuffer(cmdBuffer_, 0);
  }

  // Mark the handle as device-lost and return to EMPTY so callers know a
  // fresh capture is required after resume().
  deviceLost_ = true;
  state_ = ReplayState::EMPTY;

  DSP_DIAG(GRAPH_REPLAY, "VulkanReplayHandle: suspended (device=%d, deviceLost=true)", deviceId_);
}

void VulkanReplayHandle::resume() {
  // Called when the Android activity resumes after a pause/device-lost event.
  // We destroy and recreate all Vulkan objects so that the handle is ready for
  // a fresh beginCapture() -> endCapture() -> finalize() -> replay() cycle.
  if (!deviceLost_ && initialized_) {
    // Device was not actually lost — nothing to do.
    return;
  }

  // Tear down existing (potentially invalid) Vulkan state and reinitialize.
  cleanup();

  deviceLost_ = false;
  state_ = ReplayState::EMPTY;

  // Eagerly reinitialize so callers can immediately start a new capture.
  // On failure we leave the handle in ERRORED so callers can detect the
  // condition via getState() == ERRORED and isDeviceLost() == false.
  if (!initVulkan()) {
    sd_printf("VulkanReplayHandle: resume failed to reinitialize Vulkan (device=%d)\n", deviceId_);
    state_ = ReplayState::ERRORED;
    return;
  }

  DSP_DIAG(GRAPH_REPLAY, "VulkanReplayHandle: resumed successfully (device=%d)", deviceId_);
}

bool VulkanReplayHandle::isDeviceLost() const {
  return deviceLost_;
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

void VulkanReplayHandle::cleanup() {
  if (device_ != VK_NULL_HANDLE) {
    // The replay-local fence is the ownership proof for every object below.
    if (fence_ != VK_NULL_HANDLE &&
        waitForReplayIdle() != VK_SUCCESS) {
      state_ = ReplayState::ERRORED;
      return;
    }

    // The recorder owns descriptor pools and buffers created from this device.
    // Release it while the borrowed device is still available.
    recorder_.reset();

    // Destroy the compiled-pipeline cache while device_ is still valid.
    if (pipelineCache_) {
      pipelineCache_->clear();
      pipelineCache_.reset();
    }

    // Destroy only replay-local objects. The logical device and queue are
    // borrowed from VulkanDeviceContext and remain valid for other handles.
    if (fence_ != VK_NULL_HANDLE) {
      vkDestroyFence(device_, fence_, nullptr);
      fence_ = VK_NULL_HANDLE;
    }

    // Command buffer is freed implicitly when the pool is destroyed.
    cmdBuffer_ = VK_NULL_HANDLE;

    if (cmdPool_ != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device_, cmdPool_, nullptr);
      cmdPool_ = VK_NULL_HANDLE;
    }
  }

  instance_ = VK_NULL_HANDLE;
  physicalDevice_ = VK_NULL_HANDLE;
  device_ = VK_NULL_HANDLE;
  deviceContext_ = nullptr;
  initialized_ = false;
  state_ = ReplayState::EMPTY;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
