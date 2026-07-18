/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <graph/ResourceBinderDeviceDispatch.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/DspDeviceDispatch.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/logger.h>

#include <cstdlib>

namespace sd {
namespace graph {
namespace vulkan {

void* ResourceBinder_getDspExecutionStream() {
  return dspGetExecutionStream();
}

void ResourceBinder_setDspExecutionStream(void* stream) {
  dspSetExecutionStream(stream);
}

void* ResourceBinder_createCompletionEvent() {
  return dspCreateEvent();
}

void ResourceBinder_destroyCompletionEvent(void* event) {
  dspDestroyEvent(event);
}

void ResourceBinder_recordEvent(void* event, void* stream) {
  dspEventRecord(event, stream);
}

void ResourceBinder_streamWaitEvent(void* stream, void* event) {
  dspStreamWaitEvent(stream, event);
}

void* ResourceBinder_createStream() {
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize()) return nullptr;
  return VulkanExecutionStream::create(VulkanDeviceManager::currentDeviceId());
}

void ResourceBinder_destroyStream(void* stream, int deviceId) {
  auto* resolved = VulkanExecutionStream::fromOpaque(stream, false);
  if (resolved == nullptr || resolved->deviceId() != deviceId ||
      !VulkanExecutionStream::destroy(resolved)) {
    THROW_EXCEPTION(
        "ResourceBinder_destroyStream: invalid Vulkan stream/device ownership");
  }
}

void* ResourceBinder_deviceAlloc(size_t bytes, int deviceId) {
  if (bytes == 0) return nullptr;
  auto& manager = VulkanDeviceManager::getInstance();
  if (!manager.initialize() || deviceId < 0 ||
      deviceId >= manager.deviceCount()) {
    THROW_EXCEPTION("ResourceBinder_deviceAlloc: invalid Vulkan device");
  }
  void* ptr = VulkanMemoryPool::getInstance().allocate(
      deviceId, static_cast<VkDeviceSize>(bytes));
  if (ptr == nullptr) {
    THROW_EXCEPTION(
        "ResourceBinder_deviceAlloc: Vulkan device-local allocation failed");
  }
  return ptr;
}

void ResourceBinder_deviceFree(void* ptr, int deviceId) {
  if (ptr == nullptr) return;
  auto& pool = VulkanMemoryPool::getInstance();
  if (pool.getDeviceId(ptr) != deviceId || !pool.freeSynchronized(ptr)) {
    THROW_EXCEPTION(
        "ResourceBinder_deviceFree: invalid Vulkan allocation/device ownership");
  }
}

void* ResourceBinder_pinnedAlloc(size_t bytes) {
  if (bytes == 0) return nullptr;
  void* ptr = std::malloc(bytes);
  if (ptr == nullptr) {
    THROW_EXCEPTION("ResourceBinder_pinnedAlloc: host staging allocation failed");
  }
  return ptr;
}

void ResourceBinder_pinnedFree(void* ptr) {
  std::free(ptr);
}

void ResourceBinder_memcpyD2HAsync(void* dst, const void* src, size_t bytes,
                                   void* stream) {
  if (bytes == 0) return;
  if (dst == nullptr || src == nullptr) {
    THROW_EXCEPTION("ResourceBinder_memcpyD2HAsync: null copy operand");
  }

  auto& pool = VulkanMemoryPool::getInstance();
  const int deviceId = pool.getDeviceId(const_cast<void*>(src));
  auto* resolved = VulkanExecutionStream::fromOpaque(stream, true);
  if (deviceId < 0 || resolved == nullptr ||
      resolved->deviceId() != deviceId ||
      !resolved->enqueueCopy(dst, src, static_cast<VkDeviceSize>(bytes), 2)) {
    THROW_EXCEPTION(
        "ResourceBinder_memcpyD2HAsync: Vulkan transfer submission failed");
  }
}

}  // namespace vulkan
}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
