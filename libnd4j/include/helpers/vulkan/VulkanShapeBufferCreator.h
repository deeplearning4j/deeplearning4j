/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef LIBND4J_VULKAN_SHAPE_BUFFER_CREATOR_H
#define LIBND4J_VULKAN_SHAPE_BUFFER_CREATOR_H

#include <helpers/ShapeBufferCreator.h>

namespace sd {

/**
 * Vulkan implementation of ShapeBufferCreator.
 *
 * Shape metadata has a host copy and an immutable Vulkan pool allocation, just
 * as CUDA shape metadata has primary and special storage.
 */
class VulkanShapeBufferCreator : public ShapeBufferCreator {
 public:
  ConstantShapeBuffer* create(const LongType* shapeInfo, int rank) override;

  static VulkanShapeBufferCreator& getInstance();

 private:
  VulkanShapeBufferCreator() = default;
};

}  // namespace sd

#endif  // LIBND4J_VULKAN_SHAPE_BUFFER_CREATOR_H
