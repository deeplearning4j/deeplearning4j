/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if !defined(SD_VULKAN)
#error "helpers/vulkan/ShapeBufferPlatformHelper.cpp is only valid for SD_VULKAN"
#endif

#include <helpers/ShapeBufferPlatformHelper.h>
#include <helpers/vulkan/VulkanShapeBufferCreator.h>

#include <mutex>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace {

struct ShapeBufferInitializer {
  ShapeBufferInitializer() { ShapeBufferPlatformHelper::initialize(); }
};

ShapeBufferInitializer forceEarlyInitialization;

}  // namespace

void ShapeBufferPlatformHelper::initialize() {
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    ShapeBufferCreatorHelper::setCurrentCreator(
        &VulkanShapeBufferCreator::getInstance());
  });
}

SD_BACKEND_ABI_NAMESPACE_END
