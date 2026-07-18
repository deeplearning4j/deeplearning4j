/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if !defined(SD_VULKAN)
#error "array/vulkan/VulkanPointerDeallocator.cpp is only valid for SD_VULKAN"
#endif

#if !defined(HAVE_VULKAN) || !HAVE_VULKAN
#error "SD_VULKAN requires HAVE_VULKAN=1"
#endif

#include <array/VulkanPointerDeallocator.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <system/op_boilerplate.h>

namespace sd {

void VulkanPointerDeallocator::release(void* ptr) {
  if (ptr == nullptr) return;

  auto& pool = graph::VulkanMemoryPool::getInstance();
  const int deviceId = pool.getDeviceId(ptr);
  if (deviceId < 0) {
    THROW_EXCEPTION(
        "VulkanPointerDeallocator: pointer is not owned by VulkanMemoryPool");
  }

  auto* stream = graph::VulkanExecutionStream::currentOrDefault(deviceId);
  if (stream == nullptr || stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "VulkanPointerDeallocator: no active stream for allocation device");
  }

  if (!stream->retireAllocation(ptr)) {
    THROW_EXCEPTION(
        "VulkanPointerDeallocator: fence-ordered allocation release failed");
  }
}

}  // namespace sd
