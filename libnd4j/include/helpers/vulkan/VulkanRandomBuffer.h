/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_RANDOM_BUFFER_H
#define LIBND4J_VULKAN_RANDOM_BUFFER_H

#if !defined(SD_VULKAN)
#error "VulkanRandomBuffer is only available in SD_VULKAN builds"
#endif

#include <helpers/helper_generator.h>

namespace sd {
namespace random {

/**
 * Vulkan ownership wrapper for the backend-neutral random algorithm state.
 *
 * The RandomBuffer base subobject is the only data copied to device metadata.
 * Vulkan allocation identity and device ownership remain host-only here.
 */
class VulkanRandomBuffer final : public RandomBuffer {
 private:
  void* _metadata = nullptr;
  int _deviceId = -1;

 public:
  VulkanRandomBuffer(sd::LongType seed, sd::LongType size,
                     uint64_t* hostBuffer, uint64_t* deviceBuffer,
                     int deviceId);
  ~VulkanRandomBuffer();

  VulkanRandomBuffer(const VulkanRandomBuffer&) = delete;
  VulkanRandomBuffer& operator=(const VulkanRandomBuffer&) = delete;
  VulkanRandomBuffer(VulkanRandomBuffer&&) = delete;
  VulkanRandomBuffer& operator=(VulkanRandomBuffer&&) = delete;

  sd::Pointer metadataPointer() const;
  int deviceId() const { return _deviceId; }
  void propagateToDevice(void* stream);
};

}  // namespace random
}  // namespace sd

#endif  // LIBND4J_VULKAN_RANDOM_BUFFER_H
