/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef SD_VULKAN_POINTER_DEALLOCATOR_H_
#define SD_VULKAN_POINTER_DEALLOCATOR_H_

#include <array/PointerDeallocator.h>
#include <system/common.h>

namespace sd {

class SD_LIB_EXPORT VulkanPointerDeallocator : public PointerDeallocator {
 public:
  VulkanPointerDeallocator() = default;
  ~VulkanPointerDeallocator() override = default;

  void release(void* ptr) override;
};

}  // namespace sd

#endif  // SD_VULKAN_POINTER_DEALLOCATOR_H_
