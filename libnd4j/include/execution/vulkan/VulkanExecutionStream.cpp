/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <execution/vulkan/VulkanExecutionStream.h>

#if defined(HAVE_VULKAN) && HAVE_VULKAN

#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/logger.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace sd {
namespace graph {

std::mutex VulkanExecutionStream::registryMutex_;
std::unordered_set<VulkanExecutionStream*> VulkanExecutionStream::allStreams_;
std::unordered_set<VulkanExecutionStream*> VulkanExecutionStream::activeStreams_;
std::vector<VulkanExecutionStream*> VulkanExecutionStream::ownedStreams_;
std::unordered_map<int, VulkanExecutionStream*> VulkanExecutionStream::defaultExecutionStreams_;
std::unordered_map<int, VulkanExecutionStream*> VulkanExecutionStream::defaultCopyStreams_;

std::mutex VulkanExecutionEvent::registryMutex_;
std::unordered_set<VulkanExecutionEvent*> VulkanExecutionEvent::events_;

namespace {

thread_local VulkanExecutionStream* currentExecutionStream = nullptr;

VkDeviceSize allocationBufferSize(VkDeviceSize logicalSize) {
  if (logicalSize == 0 ||
      logicalSize > std::numeric_limits<VkDeviceSize>::max() - 3u) {
    return 0;
  }
  return (logicalSize + 3u) & ~VkDeviceSize(3u);
}

void recordTransferBarriers(VkCommandBuffer commandBuffer, bool before) {
  VkMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  if (before) {
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
  } else {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
  }
}

#if !defined(_WIN32)
struct ExternalCopyResources {
  VkDevice sourceDevice = VK_NULL_HANDLE;
  VkDevice targetDevice = VK_NULL_HANDLE;
  VkSemaphore sourceSemaphore = VK_NULL_HANDLE;
  VkSemaphore targetSemaphore = VK_NULL_HANDLE;
  VkFence sourceFence = VK_NULL_HANDLE;
  bool sourceSubmitted = false;
  VkBuffer targetAliasBuffer = VK_NULL_HANDLE;
  VkDeviceMemory targetAliasMemory = VK_NULL_HANDLE;
  int memoryFd = -1;
  int semaphoreFd = -1;

  ~ExternalCopyResources() {
    // A completed target fence proves the imported semaphore was signalled.
    // On setup/submit failure, destroy source-side synchronization objects only
    // when their fence proves completion. A failed wait is not completion: keep
    // those Vulkan children alive for later device-idle/device-teardown cleanup
    // rather than destroying objects that may still be referenced by the queue.
    bool sourceCompletionProven = !sourceSubmitted;
    if (sourceSubmitted && sourceDevice != VK_NULL_HANDLE &&
        sourceFence != VK_NULL_HANDLE) {
      sourceCompletionProven =
          vkWaitForFences(sourceDevice, 1, &sourceFence, VK_TRUE,
                          std::numeric_limits<uint64_t>::max()) == VK_SUCCESS;
    }
    if (memoryFd >= 0) close(memoryFd);
    if (semaphoreFd >= 0) close(semaphoreFd);
    if (targetDevice != VK_NULL_HANDLE) {
      if (targetSemaphore != VK_NULL_HANDLE)
        vkDestroySemaphore(targetDevice, targetSemaphore, nullptr);
      if (targetAliasBuffer != VK_NULL_HANDLE)
        vkDestroyBuffer(targetDevice, targetAliasBuffer, nullptr);
      if (targetAliasMemory != VK_NULL_HANDLE)
        vkFreeMemory(targetDevice, targetAliasMemory, nullptr);
    }
    if (sourceCompletionProven && sourceDevice != VK_NULL_HANDLE) {
      if (sourceSemaphore != VK_NULL_HANDLE)
        vkDestroySemaphore(sourceDevice, sourceSemaphore, nullptr);
      if (sourceFence != VK_NULL_HANDLE)
        vkDestroyFence(sourceDevice, sourceFence, nullptr);
    }
  }
};
#endif

}  // namespace

VulkanExecutionStream::VulkanExecutionStream(int deviceId, bool backendOwned)
    : deviceId_(deviceId), backendOwned_(backendOwned) {}

VulkanExecutionStream::~VulkanExecutionStream() {
  bool destructionSafe = synchronize();
  if (!destructionSafe && context_ != nullptr && context_->isLost()) {
    // collectCompleted only marks the context lost for VK_ERROR_DEVICE_LOST.
    // In that state submitted work cannot complete normally, so immediate
    // discard is the only valid ownership transition.
    std::lock_guard<std::mutex> guard(mutex_);
    while (!pending_.empty()) {
      discardAndDestroy(pending_.front());
      pending_.pop_front();
    }
    destructionSafe = true;
  } else if (!destructionSafe && device_ != VK_NULL_HANDLE) {
    // This is teardown-only error recovery, not a transfer fallback. A
    // non-device-loss fence error leaves submissions owned by pending_. Prove
    // the whole logical device idle before completing and destroying them.
    const VkResult idleResult = vkDeviceWaitIdle(device_);
    if (idleResult == VK_SUCCESS) {
      std::lock_guard<std::mutex> guard(mutex_);
      while (!pending_.empty()) {
        completeAndDestroy(pending_.front());
        pending_.pop_front();
      }
      destructionSafe = true;
    } else if (idleResult == VK_ERROR_DEVICE_LOST) {
      if (context_ != nullptr) context_->markLost();
      std::lock_guard<std::mutex> guard(mutex_);
      while (!pending_.empty()) {
        discardAndDestroy(pending_.front());
        pending_.pop_front();
      }
      destructionSafe = true;
    }
  }

  if (destructionSafe && device_ != VK_NULL_HANDLE &&
      commandPool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, commandPool_, nullptr);
    commandPool_ = VK_NULL_HANDLE;
  }
}

bool VulkanExecutionStream::initialize() {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) {
    sd_printf("VulkanExecutionStream::initialize: device manager initialization failed device=%d\n",
              deviceId_);
    return false;
  }
  if (deviceId_ < 0 || deviceId_ >= manager.deviceCount()) {
    sd_printf("VulkanExecutionStream::initialize: invalid device=%d count=%d\n",
              deviceId_, manager.deviceCount());
    return false;
  }

  context_ = VulkanDeviceContext::getContext(deviceId_);
  if (context_ == nullptr) {
    sd_printf("VulkanExecutionStream::initialize: no device context device=%d\n",
              deviceId_);
    return false;
  }
  if (context_->isLost()) {
    sd_printf("VulkanExecutionStream::initialize: device is lost device=%d\n",
              deviceId_);
    return false;
  }

  device_ = context_->device();
  if (device_ == VK_NULL_HANDLE) {
    sd_printf("VulkanExecutionStream::initialize: null logical device device=%d\n",
              deviceId_);
    return false;
  }

  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                   VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  poolInfo.queueFamilyIndex = context_->caps().computeQueueFamily;
  const VkResult result =
      vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_);
  if (result != VK_SUCCESS) {
    sd_printf("VulkanExecutionStream::initialize: vkCreateCommandPool failed "
              "device=%d queueFamily=%u flags=0x%x result=%d\n",
              deviceId_, static_cast<unsigned>(poolInfo.queueFamilyIndex),
              static_cast<unsigned>(poolInfo.flags), static_cast<int>(result));
    commandPool_ = VK_NULL_HANDLE;
    return false;
  }
  active_ = true;
  return true;
}

VulkanExecutionStream* VulkanExecutionStream::create(int deviceId) {
  auto* result = new VulkanExecutionStream(deviceId, false);
  if (!result->initialize()) {
    delete result;
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(registryMutex_);
  allStreams_.insert(result);
  activeStreams_.insert(result);
  ownedStreams_.push_back(result);
  return result;
}

VulkanExecutionStream* VulkanExecutionStream::createBackendOwned(int deviceId) {
  auto* result = new VulkanExecutionStream(deviceId, true);
  if (!result->initialize()) {
    delete result;
    return nullptr;
  }
  std::lock_guard<std::mutex> guard(registryMutex_);
  allStreams_.insert(result);
  activeStreams_.insert(result);
  ownedStreams_.push_back(result);
  return result;
}

bool VulkanExecutionStream::destroy(VulkanExecutionStream* stream) {
  if (stream == nullptr) return false;
  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    if (activeStreams_.find(stream) == activeStreams_.end() || stream->backendOwned_) return false;
  }
  if (!stream->synchronize()) return false;

  // A recorded event remains valid after CUDA-style stream destruction: stream
  // destruction proves its recorded position complete, so detach such events.
  {
    std::lock_guard<std::mutex> eventRegistryGuard(VulkanExecutionEvent::registryMutex_);
    for (auto* event : VulkanExecutionEvent::events_) {
      std::lock_guard<std::mutex> eventGuard(event->mutex_);
      if (event->stream_ == stream) {
        event->stream_ = nullptr;
        event->sequence_ = 0;
        event->completed_ = true;
      }
    }
  }

  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    stream->active_ = false;
    activeStreams_.erase(stream);
    allStreams_.erase(stream);
    auto it = std::find(ownedStreams_.begin(), ownedStreams_.end(), stream);
    if (it == ownedStreams_.end()) return false;
    ownedStreams_.erase(it);
  }
  if (currentExecutionStream == stream) currentExecutionStream = nullptr;
  delete stream;
  return true;
}

VulkanExecutionStream* VulkanExecutionStream::lookup(void* opaque, bool includeDestroyed) {
  if (opaque == nullptr) return nullptr;
  auto* candidate = reinterpret_cast<VulkanExecutionStream*>(opaque);
  std::lock_guard<std::mutex> guard(registryMutex_);
  const auto& registry = includeDestroyed ? allStreams_ : activeStreams_;
  return registry.find(candidate) == registry.end() ? nullptr : candidate;
}

VulkanExecutionStream* VulkanExecutionStream::fromOpaque(void* opaque, bool allowDefault) {
  if (opaque == nullptr) {
    return allowDefault ? currentOrDefault() : nullptr;
  }
  return lookup(opaque, false);
}

VulkanExecutionStream* VulkanExecutionStream::currentOrDefault(int deviceId) {
  if (currentExecutionStream != nullptr && currentExecutionStream->isActive() &&
      (deviceId < 0 || currentExecutionStream->deviceId() == deviceId)) {
    return currentExecutionStream;
  }
  const int resolvedDevice =
      deviceId >= 0 ? deviceId : VulkanDeviceManager::currentDeviceId();
  return defaultExecution(resolvedDevice);
}

VulkanExecutionStream* VulkanExecutionStream::setCurrent(VulkanExecutionStream* stream) {
  if (stream != nullptr && !stream->isActive()) return currentExecutionStream;
  VulkanExecutionStream* previous = currentExecutionStream;
  currentExecutionStream = stream;
  return previous;
}

bool VulkanExecutionStream::isCrossDeviceCopySupported(
    int srcDeviceId, int dstDeviceId, std::string* reason) {
  auto fail = [&](const std::string& message) {
    if (reason != nullptr) *reason = message;
    return false;
  };
  if (srcDeviceId < 0 || dstDeviceId < 0) {
    return fail("invalid Vulkan device id");
  }
  auto* source = VulkanDeviceContext::getContext(srcDeviceId);
  auto* target = VulkanDeviceContext::getContext(dstDeviceId);
  if (source == nullptr || target == nullptr) {
    return fail("Vulkan logical device context is unavailable");
  }
  if (source->isLost() || target->isLost()) {
    return fail("Vulkan logical device is lost");
  }
  if (srcDeviceId == dstDeviceId) {
    if (reason != nullptr) reason->clear();
    return true;
  }
  if (source->physicalDevice() != target->physicalDevice()) {
    return fail(
        "Vulkan device IDs refer to different underlying physical devices; "
        "opaque-FD import is only valid for memory exported by the same "
        "underlying physical device, while direct peer memory requires one "
        "shared device-group VkDevice and heap-specific peer support");
  }
#if defined(_WIN32)
  return fail("same-physical-device opaque-FD Vulkan sharing is unavailable on Windows");
#else
  if (!source->hasExternalDeviceSharing()) {
    return fail("source Vulkan device lacks opaque-FD external memory/semaphore export");
  }
  if (!target->hasExternalDeviceSharing()) {
    return fail("destination Vulkan device lacks opaque-FD external memory/semaphore import");
  }
  if (reason != nullptr) reason->clear();
  return true;
#endif
}

VulkanExecutionStream* VulkanExecutionStream::defaultExecution(int deviceId) {
  std::lock_guard<std::mutex> guard(registryMutex_);
  auto it = defaultExecutionStreams_.find(deviceId);
  if (it != defaultExecutionStreams_.end()) return it->second;

  auto* created = new VulkanExecutionStream(deviceId, true);
  if (!created->initialize()) {
    delete created;
    return nullptr;
  }
  allStreams_.insert(created);
  activeStreams_.insert(created);
  ownedStreams_.push_back(created);
  defaultExecutionStreams_[deviceId] = created;
  return created;
}

VulkanExecutionStream* VulkanExecutionStream::defaultCopy(int deviceId) {
  std::lock_guard<std::mutex> guard(registryMutex_);
  auto it = defaultCopyStreams_.find(deviceId);
  if (it != defaultCopyStreams_.end()) return it->second;

  auto* created = new VulkanExecutionStream(deviceId, true);
  if (!created->initialize()) {
    delete created;
    return nullptr;
  }
  allStreams_.insert(created);
  activeStreams_.insert(created);
  ownedStreams_.push_back(created);
  defaultCopyStreams_[deviceId] = created;
  return created;
}

bool VulkanExecutionStream::synchronizeDevice(int deviceId) {
  std::vector<VulkanExecutionStream*> streams;
  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    streams.reserve(allStreams_.size());
    for (auto* stream : allStreams_) {
      if (stream->deviceId_ == deviceId) streams.push_back(stream);
    }
  }
  bool ok = true;
  for (auto* stream : streams) ok = stream->synchronize() && ok;
  return ok;
}

void VulkanExecutionStream::destroyAll() {
  std::vector<VulkanExecutionStream*> snapshot;
  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    snapshot.assign(allStreams_.begin(), allStreams_.end());
  }
  for (auto* stream : snapshot) stream->synchronize();

  {
    std::lock_guard<std::mutex> eventRegistryGuard(VulkanExecutionEvent::registryMutex_);
    for (auto* event : VulkanExecutionEvent::events_) {
      std::lock_guard<std::mutex> eventGuard(event->mutex_);
      event->stream_ = nullptr;
      event->sequence_ = 0;
      event->completed_ = true;
    }
  }

  std::vector<VulkanExecutionStream*> owned;
  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    for (auto* stream : allStreams_) stream->active_ = false;
    activeStreams_.clear();
    allStreams_.clear();
    defaultExecutionStreams_.clear();
    defaultCopyStreams_.clear();
    owned.swap(ownedStreams_);
  }
  currentExecutionStream = nullptr;
  for (auto* stream : owned) delete stream;
}

bool VulkanExecutionStream::isActive() const {
  std::lock_guard<std::mutex> guard(registryMutex_);
  return activeStreams_.find(const_cast<VulkanExecutionStream*>(this)) != activeStreams_.end();
}

uint64_t VulkanExecutionStream::lastSequence() const {
  std::lock_guard<std::mutex> guard(mutex_);
  return nextSequence_ - 1;
}

bool VulkanExecutionStream::createStaging(VkDeviceSize bytes, VkBufferUsageFlags usage,
                                          StagingAllocation& staging) {
  if (bytes == 0 || device_ == VK_NULL_HANDLE) return false;
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bytes;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(device_, &bufferInfo, nullptr, &staging.buffer) != VK_SUCCESS) return false;

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device_, staging.buffer, &requirements);
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(context_->physicalDevice(), &properties);

  int bestType = -1;
  int bestScore = -1;
  bool coherent = false;
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    if ((requirements.memoryTypeBits & (1u << i)) == 0) continue;
    const auto flags = properties.memoryTypes[i].propertyFlags;
    if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
    int score = 0;
    if ((flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) score += 4;
    if ((flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0) score += 2;
    if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) score += 1;
    if (score > bestScore) {
      bestScore = score;
      bestType = static_cast<int>(i);
      coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
  }
  if (bestType < 0) {
    vkDestroyBuffer(device_, staging.buffer, nullptr);
    staging.buffer = VK_NULL_HANDLE;
    return false;
  }

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = requirements.size;
  allocInfo.memoryTypeIndex = static_cast<uint32_t>(bestType);
  if (vkAllocateMemory(device_, &allocInfo, nullptr, &staging.memory) != VK_SUCCESS ||
      vkBindBufferMemory(device_, staging.buffer, staging.memory, 0) != VK_SUCCESS ||
      vkMapMemory(device_, staging.memory, 0, VK_WHOLE_SIZE, 0, &staging.mapped) != VK_SUCCESS) {
    if (staging.mapped != nullptr) vkUnmapMemory(device_, staging.memory);
    if (staging.memory != VK_NULL_HANDLE) vkFreeMemory(device_, staging.memory, nullptr);
    vkDestroyBuffer(device_, staging.buffer, nullptr);
    staging = {};
    return false;
  }
  staging.device = device_;
  staging.coherent = coherent;
  return true;
}

bool VulkanExecutionStream::flushStaging(const StagingAllocation& staging) {
  if (staging.coherent) return true;
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = staging.memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;
  return vkFlushMappedMemoryRanges(device_, 1, &range) == VK_SUCCESS;
}

bool VulkanExecutionStream::invalidateStaging(const StagingAllocation& staging) {
  if (staging.coherent) return true;
  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = staging.memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;
  return vkInvalidateMappedMemoryRanges(device_, 1, &range) == VK_SUCCESS;
}

void VulkanExecutionStream::destroyStaging(StagingAllocation& staging) {
  if (staging.device == VK_NULL_HANDLE) return;
  if (staging.mapped != nullptr) vkUnmapMemory(staging.device, staging.memory);
  if (staging.buffer != VK_NULL_HANDLE) vkDestroyBuffer(staging.device, staging.buffer, nullptr);
  if (staging.memory != VK_NULL_HANDLE) vkFreeMemory(staging.device, staging.memory, nullptr);
  staging = {};
}

VkCommandBuffer VulkanExecutionStream::beginOneShot() {
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool_;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    return VK_NULL_HANDLE;
  }
  return commandBuffer;
}

bool VulkanExecutionStream::endOneShot(VkCommandBuffer commandBuffer) {
  return commandBuffer != VK_NULL_HANDLE && vkEndCommandBuffer(commandBuffer) == VK_SUCCESS;
}

uint64_t VulkanExecutionStream::submit(
    VkCommandBuffer commandBuffer, bool ownsCommandBuffer,
    std::vector<StagingAllocation>&& staging, std::vector<HostCopy>&& hostCopies,
    std::vector<std::function<void()>>&& callbacks,
    VkSemaphore waitSemaphore, VkPipelineStageFlags waitStage,
    std::vector<std::function<void()>>&& cleanupCallbacks) {
  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  if (vkCreateFence(device_, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
    for (auto& allocation : staging) destroyStaging(allocation);
    if (ownsCommandBuffer && commandBuffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    }
    for (auto& cleanup : cleanupCallbacks) cleanup();
    return 0;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount =
      waitSemaphore == VK_NULL_HANDLE ? 0u : 1u;
  submitInfo.pWaitSemaphores =
      waitSemaphore == VK_NULL_HANDLE ? nullptr : &waitSemaphore;
  submitInfo.pWaitDstStageMask =
      waitSemaphore == VK_NULL_HANDLE ? nullptr : &waitStage;
  submitInfo.commandBufferCount = commandBuffer == VK_NULL_HANDLE ? 0u : 1u;
  submitInfo.pCommandBuffers = commandBuffer == VK_NULL_HANDLE ? nullptr : &commandBuffer;
  const VkResult result = context_->submitCompute(1, &submitInfo, fence);
  if (result != VK_SUCCESS) {
    vkDestroyFence(device_, fence, nullptr);
    for (auto& allocation : staging) destroyStaging(allocation);
    if (ownsCommandBuffer && commandBuffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    }
    for (auto& cleanup : cleanupCallbacks) cleanup();
    if (result == VK_ERROR_DEVICE_LOST) context_->markLost();
    return 0;
  }

  PendingSubmission pending;
  pending.sequence = nextSequence_++;
  pending.fence = fence;
  pending.commandBuffer = commandBuffer;
  pending.ownsCommandBuffer = ownsCommandBuffer;
  pending.staging = std::move(staging);
  pending.hostCopies = std::move(hostCopies);
  pending.completionCallbacks = std::move(callbacks);
  pending.cleanupCallbacks = std::move(cleanupCallbacks);
  const uint64_t sequence = pending.sequence;
  pending_.emplace_back(std::move(pending));
  return collectCompleted(false, 0) ? sequence : 0;
}

uint64_t VulkanExecutionStream::enqueueCommands(
    const std::function<bool(VkCommandBuffer)>& recorder,
    std::vector<std::function<void()>> cleanupCallbacks) {
  if (!isActive() || !recorder) {
    for (auto& cleanup : cleanupCallbacks) cleanup();
    return 0;
  }

  std::lock_guard<std::mutex> guard(mutex_);
  VkCommandBuffer commandBuffer = beginOneShot();
  if (commandBuffer == VK_NULL_HANDLE || !recorder(commandBuffer) ||
      !endOneShot(commandBuffer)) {
    if (commandBuffer != VK_NULL_HANDLE) {
      vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    }
    for (auto& cleanup : cleanupCallbacks) cleanup();
    return 0;
  }

  return submit(commandBuffer, true, {}, {}, {}, VK_NULL_HANDLE,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                std::move(cleanupCallbacks));
}

bool VulkanExecutionStream::enqueueCopy(void* dst, const void* src, VkDeviceSize bytes,
                                        int direction, VkDeviceSize dstOffset,
                                        VkDeviceSize srcOffset) {
  if (!isActive() || dst == nullptr || src == nullptr || direction < 0 || direction > 3) return false;
  if (bytes == 0) return true;

  std::lock_guard<std::mutex> guard(mutex_);
  // vkCmdCopyBuffer is byte-granular. Preserve the caller's exact range;
  // padding a request would overwrite adjacent logical array data.
  const VkDeviceSize commandBytes = bytes;

  auto& pool = VulkanMemoryPool::getInstance();
  VulkanAllocRecord dstRecord;
  VulkanAllocRecord srcRecord;
  std::vector<StagingAllocation> staging;
  std::vector<HostCopy> hostCopies;
  std::vector<std::function<void()>> callbacks;

  VkBuffer sourceBuffer = VK_NULL_HANDLE;
  VkBuffer destinationBuffer = VK_NULL_HANDLE;
  VkDeviceSize sourceOffset = srcOffset;
  VkDeviceSize destinationOffset = dstOffset;

  if (direction == 0) {
    VkCommandBuffer commandBuffer = beginOneShot();
    if (commandBuffer == VK_NULL_HANDLE) return false;
    if (!endOneShot(commandBuffer)) {
      vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
      return false;
    }
    callbacks.emplace_back([dst, src, bytes, dstOffset, srcOffset]() {
      std::memmove(static_cast<uint8_t*>(dst) + dstOffset,
                   static_cast<const uint8_t*>(src) + srcOffset,
                   static_cast<size_t>(bytes));
    });
    return submit(commandBuffer, true, {}, {}, std::move(callbacks)) != 0;
  }

  if (direction == 1) {
    if (!pool.queryRecord(dst, dstRecord) ||
        dstRecord.deviceId != deviceId_ ||
        dstOffset > dstRecord.logicalSize ||
        bytes > dstRecord.logicalSize - dstOffset) {
      return false;
    }
    StagingAllocation upload;
    if (!createStaging(commandBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, upload)) return false;
    std::memcpy(upload.mapped,
                static_cast<const uint8_t*>(src) + srcOffset,
                static_cast<size_t>(bytes));
    if (!flushStaging(upload)) {
      destroyStaging(upload);
      return false;
    }
    sourceBuffer = upload.buffer;
    destinationBuffer = dstRecord.buffer;
    sourceOffset = 0;
    staging.emplace_back(std::move(upload));
  } else if (direction == 2) {
    if (!pool.queryRecord(const_cast<void*>(src), srcRecord) ||
        srcRecord.deviceId != deviceId_ ||
        srcOffset > srcRecord.logicalSize ||
        bytes > srcRecord.logicalSize - srcOffset) {
      return false;
    }
    StagingAllocation download;
    if (!createStaging(commandBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, download)) return false;
    sourceBuffer = srcRecord.buffer;
    destinationBuffer = download.buffer;
    destinationOffset = 0;
    hostCopies.push_back({static_cast<uint8_t*>(dst) + dstOffset,
                          download.mapped, static_cast<size_t>(bytes)});
    staging.emplace_back(std::move(download));
  } else {
    if (!pool.queryRecord(dst, dstRecord) ||
        !pool.queryRecord(const_cast<void*>(src), srcRecord) ||
        dstRecord.deviceId != deviceId_ ||
        dstOffset > dstRecord.logicalSize ||
        bytes > dstRecord.logicalSize - dstOffset ||
        srcOffset > srcRecord.logicalSize ||
        bytes > srcRecord.logicalSize - srcOffset) {
      return false;
    }

    if (srcRecord.deviceId == deviceId_) {
      sourceBuffer = srcRecord.buffer;
      destinationBuffer = dstRecord.buffer;
    } else {
#if defined(_WIN32)
      return false;
#else
      std::string capabilityReason;
      if (!isCrossDeviceCopySupported(
              srcRecord.deviceId, dstRecord.deviceId, &capabilityReason) ||
          !srcRecord.externalShareable ||
          srcRecord.backingMemory == VK_NULL_HANDLE ||
          srcRecord.backingSize == 0 ||
          (context_->caps().externalMemoryDedicatedOnly &&
           !srcRecord.dedicated)) {
        return false;
      }

      auto* sourceContext =
          VulkanDeviceContext::getContext(srcRecord.deviceId);
      if (sourceContext == nullptr) return false;

      auto* resources = new ExternalCopyResources();
      auto failExternal = [&resources]() {
        delete resources;
        resources = nullptr;
        return false;
      };
      resources->sourceDevice = sourceContext->device();
      resources->targetDevice = device_;

      VkMemoryGetFdInfoKHR memoryFdInfo = {};
      memoryFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
      memoryFdInfo.memory = srcRecord.backingMemory;
      memoryFdInfo.handleType =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      if (sourceContext->getMemoryFdFn()(
              resources->sourceDevice, &memoryFdInfo,
              &resources->memoryFd) != VK_SUCCESS ||
          resources->memoryFd < 0) {
        return failExternal();
      }

      // Imported buffers must reproduce the exported buffer's creation
      // geometry, especially for dedicated external-memory allocations.
      const VkDeviceSize aliasSize = allocationBufferSize(srcRecord.logicalSize);
      if (aliasSize == 0) return failExternal();

      VkExternalMemoryBufferCreateInfo externalBufferInfo = {};
      externalBufferInfo.sType =
          VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
      externalBufferInfo.handleTypes =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

      VkBufferCreateInfo aliasInfo = {};
      aliasInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      aliasInfo.pNext = &externalBufferInfo;
      aliasInfo.size = aliasSize;
      aliasInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
      aliasInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      if (vkCreateBuffer(device_, &aliasInfo, nullptr,
                         &resources->targetAliasBuffer) != VK_SUCCESS) {
        return failExternal();
      }

      VkMemoryRequirements aliasRequirements = {};
      vkGetBufferMemoryRequirements(
          device_, resources->targetAliasBuffer, &aliasRequirements);
      if (aliasRequirements.size == 0 ||
          aliasRequirements.memoryTypeBits == 0 ||
          aliasRequirements.alignment == 0 ||
          (srcRecord.offsetInBlock % aliasRequirements.alignment) != 0 ||
          srcRecord.offsetInBlock + aliasRequirements.size <
              srcRecord.offsetInBlock ||
          srcRecord.offsetInBlock + aliasRequirements.size >
              srcRecord.backingSize) {
        return failExternal();
      }

      // OPAQUE_FD imports must come from the same underlying physical device.
      // For that handle type vkGetMemoryFdPropertiesKHR is forbidden, so retain
      // the exported allocation's memory type and validate it against the alias.
      VkPhysicalDeviceMemoryProperties targetMemoryProperties = {};
      vkGetPhysicalDeviceMemoryProperties(
          context_->physicalDevice(), &targetMemoryProperties);
      const uint32_t targetMemoryType = srcRecord.memTypeIdx;
      if (targetMemoryType >= targetMemoryProperties.memoryTypeCount ||
          (aliasRequirements.memoryTypeBits & (1u << targetMemoryType)) == 0 ||
          (targetMemoryProperties.memoryTypes[targetMemoryType].propertyFlags &
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0) {
        return failExternal();
      }

      VkMemoryDedicatedAllocateInfo dedicatedInfo = {};
      dedicatedInfo.sType =
          VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
      dedicatedInfo.buffer = resources->targetAliasBuffer;

      VkImportMemoryFdInfoKHR importMemoryInfo = {};
      importMemoryInfo.sType =
          VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
      importMemoryInfo.pNext =
          srcRecord.dedicated ? &dedicatedInfo : nullptr;
      importMemoryInfo.handleType =
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      importMemoryInfo.fd = resources->memoryFd;

      VkMemoryAllocateInfo importAllocateInfo = {};
      importAllocateInfo.sType =
          VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      importAllocateInfo.pNext = &importMemoryInfo;
      importAllocateInfo.allocationSize = srcRecord.backingSize;
      importAllocateInfo.memoryTypeIndex = targetMemoryType;
      if (vkAllocateMemory(
              device_, &importAllocateInfo, nullptr,
              &resources->targetAliasMemory) != VK_SUCCESS) {
        return failExternal();
      }
      // A successful Vulkan import consumes the file descriptor.
      resources->memoryFd = -1;

      if (vkBindBufferMemory(
              device_, resources->targetAliasBuffer,
              resources->targetAliasMemory,
              srcRecord.offsetInBlock) != VK_SUCCESS) {
        return failExternal();
      }

      VkExportSemaphoreCreateInfo exportSemaphoreInfo = {};
      exportSemaphoreInfo.sType =
          VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
      exportSemaphoreInfo.handleTypes =
          VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      VkSemaphoreCreateInfo sourceSemaphoreInfo = {};
      sourceSemaphoreInfo.sType =
          VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      sourceSemaphoreInfo.pNext = &exportSemaphoreInfo;
      if (vkCreateSemaphore(
              resources->sourceDevice, &sourceSemaphoreInfo, nullptr,
              &resources->sourceSemaphore) != VK_SUCCESS) {
        return failExternal();
      }

      VkFenceCreateInfo sourceFenceInfo = {};
      sourceFenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      if (vkCreateFence(
              resources->sourceDevice, &sourceFenceInfo, nullptr,
              &resources->sourceFence) != VK_SUCCESS) {
        return failExternal();
      }

      VkSubmitInfo sourceSignal = {};
      sourceSignal.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      sourceSignal.signalSemaphoreCount = 1;
      sourceSignal.pSignalSemaphores = &resources->sourceSemaphore;
      const VkResult sourceSubmitResult = sourceContext->submitCompute(
          1, &sourceSignal, resources->sourceFence);
      if (sourceSubmitResult != VK_SUCCESS) {
        if (sourceSubmitResult == VK_ERROR_DEVICE_LOST)
          sourceContext->markLost();
        return failExternal();
      }
      resources->sourceSubmitted = true;

      VkSemaphoreGetFdInfoKHR semaphoreFdInfo = {};
      semaphoreFdInfo.sType =
          VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
      semaphoreFdInfo.semaphore = resources->sourceSemaphore;
      semaphoreFdInfo.handleType =
          VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      if (sourceContext->getSemaphoreFdFn()(
              resources->sourceDevice, &semaphoreFdInfo,
              &resources->semaphoreFd) != VK_SUCCESS ||
          resources->semaphoreFd < 0) {
        return failExternal();
      }

      VkSemaphoreCreateInfo targetSemaphoreInfo = {};
      targetSemaphoreInfo.sType =
          VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      if (vkCreateSemaphore(
              device_, &targetSemaphoreInfo, nullptr,
              &resources->targetSemaphore) != VK_SUCCESS) {
        return failExternal();
      }

      VkImportSemaphoreFdInfoKHR importSemaphoreInfo = {};
      importSemaphoreInfo.sType =
          VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
      importSemaphoreInfo.semaphore = resources->targetSemaphore;
      importSemaphoreInfo.handleType =
          VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
      importSemaphoreInfo.fd = resources->semaphoreFd;
      if (context_->importSemaphoreFdFn()(
              device_, &importSemaphoreInfo) != VK_SUCCESS) {
        return failExternal();
      }
      // A successful Vulkan import consumes the file descriptor.
      resources->semaphoreFd = -1;

      VkCommandBuffer commandBuffer = beginOneShot();
      if (commandBuffer == VK_NULL_HANDLE) return failExternal();
      recordTransferBarriers(commandBuffer, true);
      VkBufferCopy region = {};
      region.srcOffset = srcOffset;
      region.dstOffset = dstOffset;
      region.size = commandBytes;
      vkCmdCopyBuffer(commandBuffer, resources->targetAliasBuffer,
                      dstRecord.buffer, 1, &region);
      recordTransferBarriers(commandBuffer, false);
      if (!endOneShot(commandBuffer)) {
        vkFreeCommandBuffers(
            device_, commandPool_, 1, &commandBuffer);
        return failExternal();
      }

      std::vector<std::function<void()>> externalCleanup;
      externalCleanup.emplace_back([resources]() { delete resources; });
      return submit(commandBuffer, true, {}, {}, {},
                    resources->targetSemaphore,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    std::move(externalCleanup)) != 0;
#endif
    }
  }

  VkCommandBuffer commandBuffer = beginOneShot();
  if (commandBuffer == VK_NULL_HANDLE) {
    for (auto& allocation : staging) destroyStaging(allocation);
    return false;
  }
  recordTransferBarriers(commandBuffer, true);
  VkBufferCopy region{};
  region.srcOffset = sourceOffset;
  region.dstOffset = destinationOffset;
  region.size = commandBytes;
  vkCmdCopyBuffer(commandBuffer, sourceBuffer, destinationBuffer, 1, &region);
  recordTransferBarriers(commandBuffer, false);
  if (!endOneShot(commandBuffer)) {
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    for (auto& allocation : staging) destroyStaging(allocation);
    return false;
  }
  return submit(commandBuffer, true, std::move(staging), std::move(hostCopies), {}) != 0;
}

bool VulkanExecutionStream::enqueueFill(void* dst, int value, VkDeviceSize bytes,
                                        VkDeviceSize dstOffset) {
  if (!isActive() || dst == nullptr) return false;
  if (bytes == 0) return true;
  auto& pool = VulkanMemoryPool::getInstance();
  VulkanAllocRecord record;
  if (!pool.queryRecord(dst, record) || record.deviceId != deviceId_ ||
      dstOffset > record.logicalSize ||
      bytes > record.logicalSize - dstOffset) {
    return false;
  }

  const bool wholeBuffer = dstOffset == 0 && bytes == record.logicalSize;
  const VkDeviceSize endOffset = dstOffset + bytes;
  const VkDeviceSize maxOffset =
      std::numeric_limits<VkDeviceSize>::max();
  const VkDeviceSize alignedCandidate =
      dstOffset > maxOffset - 3u
          ? endOffset
          : (dstOffset + 3u) & ~VkDeviceSize(3u);
  const VkDeviceSize alignedBegin = std::min(endOffset, alignedCandidate);
  const VkDeviceSize prefixBytes = wholeBuffer ? 0 : alignedBegin - dstOffset;
  const VkDeviceSize middleBytes =
      wholeBuffer ? 0 : (endOffset - alignedBegin) & ~VkDeviceSize(3u);
  const VkDeviceSize suffixBytes =
      wholeBuffer ? 0 : endOffset - alignedBegin - middleBytes;
  const VkDeviceSize edgeBytes = prefixBytes + suffixBytes;

  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<StagingAllocation> staging;
  VkBuffer edgeBuffer = VK_NULL_HANDLE;
  if (edgeBytes != 0) {
    StagingAllocation edges;
    if (!createStaging(edgeBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, edges)) {
      return false;
    }
    std::memset(edges.mapped, static_cast<unsigned char>(value),
                static_cast<size_t>(edgeBytes));
    if (!flushStaging(edges)) {
      destroyStaging(edges);
      return false;
    }
    edgeBuffer = edges.buffer;
    staging.emplace_back(std::move(edges));
  }

  VkCommandBuffer commandBuffer = beginOneShot();
  if (commandBuffer == VK_NULL_HANDLE) {
    for (auto& allocation : staging) destroyStaging(allocation);
    return false;
  }
  const uint32_t byteValue =
      static_cast<uint32_t>(static_cast<uint8_t>(value));
  const uint32_t pattern =
      byteValue | (byteValue << 8u) | (byteValue << 16u) |
      (byteValue << 24u);
  recordTransferBarriers(commandBuffer, true);
  if (wholeBuffer) {
    vkCmdFillBuffer(commandBuffer, record.buffer, 0, VK_WHOLE_SIZE, pattern);
  } else {
    if (prefixBytes != 0) {
      VkBufferCopy prefix = {};
      prefix.srcOffset = 0;
      prefix.dstOffset = dstOffset;
      prefix.size = prefixBytes;
      vkCmdCopyBuffer(commandBuffer, edgeBuffer, record.buffer, 1, &prefix);
    }
    if (middleBytes != 0) {
      vkCmdFillBuffer(commandBuffer, record.buffer, alignedBegin, middleBytes,
                      pattern);
    }
    if (suffixBytes != 0) {
      VkBufferCopy suffix = {};
      suffix.srcOffset = prefixBytes;
      suffix.dstOffset = alignedBegin + middleBytes;
      suffix.size = suffixBytes;
      vkCmdCopyBuffer(commandBuffer, edgeBuffer, record.buffer, 1, &suffix);
    }
  }
  recordTransferBarriers(commandBuffer, false);
  if (!endOneShot(commandBuffer)) {
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    for (auto& allocation : staging) destroyStaging(allocation);
    return false;
  }
  return submit(commandBuffer, true, std::move(staging), {}, {}) != 0;
}

uint64_t VulkanExecutionStream::submitExternal(VkCommandBuffer commandBuffer) {
  if (!isActive() || commandBuffer == VK_NULL_HANDLE) return 0;
  std::lock_guard<std::mutex> guard(mutex_);
  return submit(commandBuffer, false, {}, {}, {});
}

uint64_t VulkanExecutionStream::enqueueHostCallback(std::function<void()> callback) {
  if (!isActive() || !callback) return 0;
  std::lock_guard<std::mutex> guard(mutex_);
  VkCommandBuffer commandBuffer = beginOneShot();
  if (commandBuffer == VK_NULL_HANDLE) return 0;
  if (!endOneShot(commandBuffer)) {
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
    return 0;
  }
  std::vector<std::function<void()>> callbacks;
  callbacks.emplace_back(std::move(callback));
  return submit(commandBuffer, true, {}, {}, std::move(callbacks));
}

void VulkanExecutionStream::completeAndDestroy(PendingSubmission& pending) {
  for (const auto& allocation : pending.staging) invalidateStaging(allocation);
  for (const auto& copy : pending.hostCopies) {
    if (copy.dst != nullptr && copy.src != nullptr && copy.bytes > 0) {
      std::memcpy(copy.dst, copy.src, copy.bytes);
    }
  }
  for (auto& callback : pending.completionCallbacks) callback();
  for (auto& allocation : pending.staging) destroyStaging(allocation);
  if (pending.ownsCommandBuffer && pending.commandBuffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(device_, commandPool_, 1, &pending.commandBuffer);
  }
  if (pending.fence != VK_NULL_HANDLE) {
    vkDestroyFence(device_, pending.fence, nullptr);
  }
  for (auto& cleanup : pending.cleanupCallbacks) cleanup();
  completedSequence_ = std::max(completedSequence_, pending.sequence);
}

void VulkanExecutionStream::discardAndDestroy(PendingSubmission& pending) {
  // Only confirmed device loss reaches this path. Host copies and completion
  // callbacks must not run because device writes are unproven, but Vulkan
  // handles and explicit cleanup ownership still have to be retired.
  for (auto& allocation : pending.staging) destroyStaging(allocation);
  if (pending.ownsCommandBuffer && pending.commandBuffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(device_, commandPool_, 1, &pending.commandBuffer);
  }
  if (pending.fence != VK_NULL_HANDLE) {
    vkDestroyFence(device_, pending.fence, nullptr);
  }
  for (auto& cleanup : pending.cleanupCallbacks) cleanup();
}

bool VulkanExecutionStream::collectCompleted(bool wait, uint64_t throughSequence) {
  while (!pending_.empty()) {
    PendingSubmission& pending = pending_.front();
    const VkResult result =
        wait ? vkWaitForFences(device_, 1, &pending.fence, VK_TRUE,
                               std::numeric_limits<uint64_t>::max())
             : vkGetFenceStatus(device_, pending.fence);
    if (!wait && result == VK_NOT_READY) break;
    if (result == VK_ERROR_DEVICE_LOST) {
      context_->markLost();
      while (!pending_.empty()) {
        discardAndDestroy(pending_.front());
        pending_.pop_front();
      }
      return false;
    }
    if (result != VK_SUCCESS) {
      // OOM, validation, and unknown fence errors do not prove queue completion.
      // Keep every submission and its cleanup ownership intact so a later poll,
      // synchronization, or device-idle teardown can resolve it safely.
      return false;
    }
    completeAndDestroy(pending);
    pending_.pop_front();
    if (wait && throughSequence != 0 && completedSequence_ >= throughSequence) break;
  }
  return throughSequence == 0 || completedSequence_ >= throughSequence;
}

bool VulkanExecutionStream::waitThrough(uint64_t sequence) {
  if (sequence == 0) return true;
  std::lock_guard<std::mutex> guard(mutex_);
  if (completedSequence_ >= sequence) return true;
  if (sequence >= nextSequence_) return false;
  return collectCompleted(true, sequence);
}

bool VulkanExecutionStream::synchronize() {
  std::lock_guard<std::mutex> guard(mutex_);
  const uint64_t target = nextSequence_ - 1;
  if (target == 0 || completedSequence_ >= target) return true;
  return collectCompleted(true, target);
}

bool VulkanExecutionStream::waitEvent(const VulkanExecutionEvent& event) {
  std::lock_guard<std::mutex> eventGuard(event.mutex_);
  if (event.deviceId_ != deviceId_) return false;
  if (event.completed_) return true;
  if (event.stream_ == nullptr || event.sequence_ == 0) return false;
  // record() submits an explicit marker on the canonical per-device VkQueue.
  // Queue submission is externally synchronized by VulkanDeviceContext; every
  // submission made on this stream after waitEvent therefore follows that marker.
  return true;
}

bool VulkanExecutionStream::retireAllocation(void* ptr) {
  if (!isActive() || ptr == nullptr) return false;
  auto& pool = VulkanMemoryPool::getInstance();
  VulkanAllocRecord record;
  if (!pool.queryRecord(ptr, record) || record.deviceId != deviceId_) return false;

  // Always submit a marker before releasing the allocation. Every stream on a
  // logical device uses the canonical queue, so the marker is ordered after all
  // earlier device work even when this stream had no local pending submission.
  return enqueueHostCallback([ptr]() {
           VulkanMemoryPool::getInstance().freeImmediate(ptr);
         }) != 0;
}

VulkanExecutionEvent* VulkanExecutionEvent::create() {
  auto* event = new VulkanExecutionEvent();
  std::lock_guard<std::mutex> guard(registryMutex_);
  events_.insert(event);
  return event;
}

VulkanExecutionEvent* VulkanExecutionEvent::fromOpaque(void* opaque) {
  if (opaque == nullptr) return nullptr;
  auto* candidate = reinterpret_cast<VulkanExecutionEvent*>(opaque);
  std::lock_guard<std::mutex> guard(registryMutex_);
  return events_.find(candidate) == events_.end() ? nullptr : candidate;
}

bool VulkanExecutionEvent::destroy(VulkanExecutionEvent* event) {
  if (event == nullptr) return false;
  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    auto it = events_.find(event);
    if (it == events_.end()) return false;
    events_.erase(it);
  }
  // Events are logical queue positions, not Vulkan handles. Destroying one
  // drops only the observer; the owning stream retains its submitted work and
  // completion resources. Matching cudaEventDestroy, this is non-blocking.
  delete event;
  return true;
}

void VulkanExecutionEvent::destroyAll() {
  std::vector<VulkanExecutionEvent*> snapshot;
  {
    std::lock_guard<std::mutex> guard(registryMutex_);
    snapshot.assign(events_.begin(), events_.end());
    events_.clear();
  }
  for (auto* event : snapshot) {
    delete event;
  }
}

bool VulkanExecutionEvent::record(VulkanExecutionStream* stream) {
  if (stream == nullptr || !stream->isActive()) return false;
  const uint64_t marker = stream->enqueueHostCallback([]() {});
  if (marker == 0) return false;
  std::lock_guard<std::mutex> guard(mutex_);
  stream_ = stream;
  sequence_ = marker;
  deviceId_ = stream->deviceId();
  completed_ = false;
  return true;
}

bool VulkanExecutionEvent::synchronize() {
  VulkanExecutionStream* stream = nullptr;
  uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    stream = stream_;
    sequence = sequence_;
  }
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (completed_) return true;
  }
  return stream != nullptr && stream->waitThrough(sequence);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN
