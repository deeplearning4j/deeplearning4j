/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if !defined(SD_VULKAN)
#error "helpers/vulkan/TadCalculator.cpp is only valid for SD_VULKAN"
#endif

#if !defined(HAVE_VULKAN) || !HAVE_VULKAN
#error "SD_VULKAN requires HAVE_VULKAN=1"
#endif

#include <array/PrimaryPointerDeallocator.h>
#include <array/VulkanPointerDeallocator.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/TadCalculatorPlatform.h>
#include <helpers/vulkan/VulkanShapeBufferCreator.h>
#include <system/op_boilerplate.h>

#include <memory>

SD_BACKEND_ABI_NAMESPACE_BEGIN
namespace tad_platform {
namespace {

int allocationDevice(void* pointer) {
  const int deviceId =
      graph::VulkanMemoryPool::getInstance().getDeviceId(pointer);
  if (deviceId < 0) {
    THROW_EXCEPTION(
        "TadCalculator: offset allocation is not owned by VulkanMemoryPool");
  }
  return deviceId;
}

graph::VulkanExecutionStream* orderedStream(int deviceId) {
  auto* stream = graph::VulkanExecutionStream::currentOrDefault(deviceId);
  if (stream == nullptr || stream->deviceId() != deviceId) {
    THROW_EXCEPTION("TadCalculator: no active Vulkan stream");
  }
  return stream;
}

}  // namespace

TadShapeOwnership shapeOwnership() {
  return TadShapeOwnership::Owned;
}

ConstantShapeBuffer* createShape(const LongType* shapeInfo, LongType rank) {
  return VulkanShapeBufferCreator::getInstance().create(
      shapeInfo, static_cast<int>(rank));
}

ConstantOffsetsBuffer* createOffsets(
    std::unique_ptr<LongType[]> offsets, LongType count) {
  if (offsets == nullptr || count < 0) {
    THROW_EXCEPTION("TadCalculator: invalid Vulkan offsets");
  }

  const size_t allocationElements =
      static_cast<size_t>(count) + SD_SHAPE_ALLOC_PADDING;
  const VkDeviceSize allocationBytes =
      static_cast<VkDeviceSize>(allocationElements * sizeof(LongType));
  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  auto* stream = orderedStream(deviceId);

  auto primary = std::make_shared<PointerWrapper>(
      offsets.release(), std::make_shared<PrimaryPointerDeallocator>());
  void* specialPointer =
      graph::VulkanMemoryPool::getInstance().allocate(deviceId, allocationBytes);
  if (specialPointer == nullptr) {
    THROW_EXCEPTION("TadCalculator: Vulkan offset allocation failed");
  }

  if (!stream->enqueueCopy(
          specialPointer, primary->pointer(), allocationBytes, 1)) {
    if (!stream->retireAllocation(specialPointer)) {
      THROW_EXCEPTION(
          "TadCalculator: failed to retire Vulkan offset allocation");
    }
    THROW_EXCEPTION("TadCalculator: Vulkan offset H2D copy failed");
  }

  const uint64_t copySequence = stream->lastSequence();
  if (!stream->waitThrough(copySequence)) {
    if (!stream->retireAllocation(specialPointer)) {
      THROW_EXCEPTION(
          "TadCalculator: failed to retire Vulkan offset allocation");
    }
    THROW_EXCEPTION("TadCalculator: Vulkan offset H2D completion failed");
  }

  const int actualDevice = allocationDevice(specialPointer);
  if (actualDevice != deviceId) {
    if (!stream->retireAllocation(specialPointer)) {
      THROW_EXCEPTION(
          "TadCalculator: failed to retire misplaced Vulkan allocation");
    }
    THROW_EXCEPTION("TadCalculator: Vulkan offset allocated on wrong device");
  }

  auto special = std::make_shared<PointerWrapper>(
      specialPointer, std::make_shared<VulkanPointerDeallocator>());
  return new ConstantOffsetsBuffer(primary, special);
}

}  // namespace tad_platform
SD_BACKEND_ABI_NAMESPACE_END
