/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if !defined(SD_VULKAN)
#error "array/vulkan/ExtraArguments.cpp is only valid for SD_VULKAN"
#endif

#if !defined(HAVE_VULKAN) || !HAVE_VULKAN
#error "SD_VULKAN requires HAVE_VULKAN=1"
#endif

#include <array/ExtraArguments_device.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <system/op_boilerplate.h>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace extra_args_detail {
namespace {

int allocationDevice(void* ptr) {
  const int deviceId = graph::VulkanMemoryPool::getInstance().getDeviceId(ptr);
  if (deviceId < 0) {
    THROW_EXCEPTION("ExtraArguments: pointer is not owned by VulkanMemoryPool");
  }
  return deviceId;
}

graph::VulkanExecutionStream* orderedStream(int deviceId) {
  auto* stream = graph::VulkanExecutionStream::currentOrDefault(deviceId);
  if (stream == nullptr || stream->deviceId() != deviceId) {
    THROW_EXCEPTION("ExtraArguments: no active Vulkan stream for allocation device");
  }
  return stream;
}

}  // namespace

void* extraArgsAllocDevice(size_t bytes) {
  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  auto* ptr = graph::VulkanMemoryPool::getInstance().allocate(
      deviceId, static_cast<VkDeviceSize>(bytes));
  if (ptr == nullptr) {
    THROW_EXCEPTION("ExtraArguments: VulkanMemoryPool::allocate failed");
  }
  return ptr;
}

void extraArgsFreeDevice(void* ptr) {
  if (ptr == nullptr) return;
  const int deviceId = allocationDevice(ptr);
  if (!orderedStream(deviceId)->retireAllocation(ptr)) {
    THROW_EXCEPTION("ExtraArguments: fence-ordered Vulkan release failed");
  }
}

void extraArgsCopyH2DDispatch(void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return;
  if (dst == nullptr || src == nullptr) {
    THROW_EXCEPTION("ExtraArguments: Vulkan H2D copy requires non-null pointers");
  }
  const int deviceId = allocationDevice(dst);
  if (!orderedStream(deviceId)->enqueueCopy(
          dst, src, static_cast<VkDeviceSize>(bytes), 1)) {
    THROW_EXCEPTION("ExtraArguments: stream-ordered Vulkan H2D copy failed");
  }
}

}  // namespace extra_args_detail
SD_BACKEND_ABI_NAMESPACE_END
