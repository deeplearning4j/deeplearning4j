/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#if !defined(SD_VULKAN)
#error "helpers/vulkan/VulkanShapeBufferCreator.cpp is only valid for SD_VULKAN"
#endif

#if !defined(HAVE_VULKAN) || !HAVE_VULKAN
#error "SD_VULKAN requires HAVE_VULKAN=1"
#endif

#include <array/PrimaryPointerDeallocator.h>
#include <array/VulkanPointerDeallocator.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <graph/vulkan/VulkanDeviceManager.h>
#include <graph/vulkan/VulkanMemoryPool.h>
#include <helpers/shape.h>
#include <helpers/vulkan/VulkanShapeBufferCreator.h>
#include <system/CanaryConstants.h>
#include <system/op_boilerplate.h>

#include <memory>
#include <mutex>
#include <string>

namespace sd {
namespace {

graph::VulkanExecutionStream* shapeStream(int deviceId) {
  auto* stream = graph::VulkanExecutionStream::currentOrDefault(deviceId);
  if (stream == nullptr || stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "VulkanShapeBufferCreator: no active stream for allocation device");
  }
  return stream;
}

void retireFailedShapeAllocation(graph::VulkanExecutionStream* stream,
                                 void* devicePointer) {
  if (devicePointer != nullptr &&
      (stream == nullptr || !stream->retireAllocation(devicePointer))) {
    THROW_EXCEPTION(
        "VulkanShapeBufferCreator: failed to retire device allocation");
  }
}

}  // namespace

ConstantShapeBuffer* VulkanShapeBufferCreator::create(
    const LongType* shapeInfo, int rank) {
  if (shapeInfo == nullptr) {
    THROW_EXCEPTION("VulkanShapeBufferCreator::create: shapeInfo is null");
  }
  if (rank < 0 || rank > SD_MAX_RANK) {
    const std::string message =
        "VulkanShapeBufferCreator::create: invalid rank: " +
        std::to_string(rank);
    THROW_EXCEPTION(message.c_str());
  }

  const LongType inputRank = shapeInfo[0];
  if (inputRank != rank) {
    const std::string message =
        "VulkanShapeBufferCreator::create: shapeInfo rank mismatch. Expected: " +
        std::to_string(rank) + ", found: " + std::to_string(inputRank);
    THROW_EXCEPTION(message.c_str());
  }

  const int shapeInfoLength = shape::shapeInfoLength(rank);
  const size_t allocationElements =
      static_cast<size_t>(shapeInfoLength) + SD_SHAPE_ALLOC_PADDING;
  const VkDeviceSize allocationBytes =
      static_cast<VkDeviceSize>(allocationElements * sizeof(LongType));

  auto* shapeCopy = new LongType[allocationElements]();
  for (int i = 0; i < shapeInfoLength; ++i) {
    shapeCopy[i] = shapeInfo[i];
  }
  for (int i = 0; i < 8 && shapeInfoLength + i < allocationElements; ++i) {
    shapeCopy[shapeInfoLength + i] =
        sd::CanaryConstants::SHAPE_BUFFER_CANARY;
  }

  if (shapeCopy[0] != rank) {
    const LongType copiedRank = shapeCopy[0];
    delete[] shapeCopy;
    const std::string message =
        "VulkanShapeBufferCreator::create: copy verification failed. Expected: " +
        std::to_string(rank) + ", copied: " + std::to_string(copiedRank);
    THROW_EXCEPTION(message.c_str());
  }

  const int deviceId = graph::VulkanDeviceManager::currentDeviceId();
  auto* stream = shapeStream(deviceId);
  auto& pool = graph::VulkanMemoryPool::getInstance();
  void* devicePointer = pool.allocate(deviceId, allocationBytes);
  if (devicePointer == nullptr) {
    delete[] shapeCopy;
    THROW_EXCEPTION(
        "VulkanShapeBufferCreator::create: Vulkan allocation failed");
  }
  if (!stream->enqueueCopy(devicePointer, shapeCopy, allocationBytes, 1)) {
    retireFailedShapeAllocation(stream, devicePointer);
    delete[] shapeCopy;
    THROW_EXCEPTION(
        "VulkanShapeBufferCreator::create: Vulkan H2D copy failed");
  }

  const uint64_t copySequence = stream->lastSequence();
  if (!stream->waitThrough(copySequence)) {
    retireFailedShapeAllocation(stream, devicePointer);
    delete[] shapeCopy;
    THROW_EXCEPTION(
        "VulkanShapeBufferCreator::create: Vulkan H2D completion failed");
  }

  auto hostDeallocator = std::make_shared<PrimaryPointerDeallocator>();
  auto* hostPointer = new PointerWrapper(shapeCopy, hostDeallocator);
  auto* specialPointer = new PointerWrapper(
      devicePointer, std::make_shared<VulkanPointerDeallocator>());
  return new ConstantShapeBuffer(hostPointer, specialPointer);
}

VulkanShapeBufferCreator& VulkanShapeBufferCreator::getInstance() {
  static VulkanShapeBufferCreator* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new VulkanShapeBufferCreator();
  });
  return *instance;
}

}  // namespace sd
