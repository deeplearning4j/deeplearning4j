/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN)

#if !defined(HAVE_VULKAN) || !HAVE_VULKAN
#error "SD_VULKAN requires HAVE_VULKAN=1"
#endif

#include <array/DataBuffer.h>
#include <array/NDArray.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/headers/transforms.h>
#include <system/op_boilerplate.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace sd {
namespace {

class ScopedVulkanDevice {
 public:
  explicit ScopedVulkanDevice(int deviceId)
      : previousDeviceId_(AffinityManager::currentDeviceId()),
        changed_(previousDeviceId_ != deviceId) {
    if (deviceId < 0) {
      THROW_EXCEPTION("ScopedVulkanDevice requires a physical Vulkan device");
    }
    if (changed_) {
      AffinityManager::setCurrentDevice(deviceId);
    }
  }

  ~ScopedVulkanDevice() noexcept {
    if (!changed_) {
      return;
    }
    try {
      AffinityManager::setCurrentDevice(previousDeviceId_);
    } catch (...) {
      // Destructors must not throw while unwinding. The exact-device operation
      // has already completed or failed; restoration is best-effort only here.
    }
  }

  ScopedVulkanDevice(const ScopedVulkanDevice&) = delete;
  ScopedVulkanDevice& operator=(const ScopedVulkanDevice&) = delete;

 private:
  int previousDeviceId_;
  bool changed_;
};

bool validateRepetitions(const std::vector<LongType>& repetitions) {
  bool identity = true;
  for (const LongType repetition : repetitions) {
    if (repetition < 1) {
      THROW_EXCEPTION("NDArray::tile: repetitions must be positive");
    }
    identity = identity && repetition == 1;
  }
  return identity;
}

std::vector<LongType> normalizeRepetitions(
    int inputRank, const std::vector<LongType>& repetitions) {
  const int normalizedRank =
      std::max(inputRank, static_cast<int>(repetitions.size()));
  std::vector<LongType> normalized(static_cast<size_t>(normalizedRank), 1);
  std::copy(repetitions.begin(), repetitions.end(),
            normalized.end() - repetitions.size());
  return normalized;
}

std::unique_ptr<NDArray> alignInputRank(NDArray& input, int targetRank) {
  if (input.rankOf() == targetRank) return nullptr;

  std::vector<LongType> alignedShape(static_cast<size_t>(targetRank), 1);
  const int offset = targetRank - input.rankOf();
  for (int axis = 0; axis < input.rankOf(); ++axis) {
    alignedShape[static_cast<size_t>(offset + axis)] = input.sizeAt(axis);
  }

  auto* view = input.reshape(input.ordering(), alignedShape, false);
  if (view == nullptr) {
    THROW_EXCEPTION(
        "NDArray::tile could not create the required no-copy rank-aligned "
        "Vulkan view");
  }
  return std::unique_ptr<NDArray>(view);
}

void executeTileDescriptor(NDArray& input, NDArray& target,
                           const std::vector<LongType>& repetitions) {
#if NOT_EXCLUDED(OP_tile)
  if (input.getDataBuffer() == nullptr || target.getDataBuffer() == nullptr) {
    THROW_EXCEPTION("NDArray::tile received an array without a data buffer");
  }

  const std::vector<LongType> normalizedRepetitions =
      normalizeRepetitions(input.rankOf(), repetitions);
  if (target.rankOf() != static_cast<int>(normalizedRepetitions.size())) {
    THROW_EXCEPTION(
        "NDArray::tile target rank does not match the normalized repetition "
        "rank");
  }

  auto alignedOwner = alignInputRank(input, target.rankOf());
  NDArray* descriptorInput =
      alignedOwner == nullptr ? &input : alignedOwner.get();

  for (int axis = 0; axis < target.rankOf(); ++axis) {
    if (target.sizeAt(axis) !=
        descriptorInput->sizeAt(axis) *
            normalizedRepetitions[static_cast<size_t>(axis)]) {
      THROW_EXCEPTION(
          "NDArray::tile target shape does not match the repetition contract");
    }
  }

  const int deviceId = descriptorInput->getDataBuffer()->deviceId();
  LaunchContext* launchContext = input.getContext();
  if (deviceId < 0 || launchContext == nullptr ||
      launchContext->getDeviceID() != deviceId) {
    THROW_EXCEPTION("NDArray::tile received an invalid Vulkan device context");
  }
  if (target.getDataBuffer()->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "NDArray::tile input and target must reside on one physical device");
  }

  ScopedVulkanDevice deviceScope(deviceId);

  const auto contextStream = graph::vulkanExecutionStream(launchContext);
  auto* stream =
      contextStream == nullptr
          ? graph::VulkanExecutionStream::defaultExecution(deviceId)
          : graph::VulkanExecutionStream::fromOpaque(contextStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        contextStream == nullptr
            ? "NDArray::tile could not resolve the exact-device default "
              "Vulkan execution stream"
            : "NDArray::tile received an invalid context-owned Vulkan "
              "execution stream");
  }

  std::vector<NDArray*> inputs{descriptorInput};
  std::vector<NDArray*> outputs{&target};
  NDArray::prepareSpecialUse(outputs, inputs);

  graph::Context opContext(0);
  opContext.setInputArrays(1, inputs.data(), false);
  opContext.setOutputArrays(1, outputs.data(), false);
  opContext.setIArguments(normalizedRepetitions);

  ops::tile descriptor;
  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *stream, &error);
  if (status != Status::OK) {
    if (error.empty()) {
      error = "NDArray::tile Vulkan descriptor execution failed";
    }
    THROW_EXCEPTION(error.c_str());
  }

  NDArray::registerSpecialUse(outputs, inputs);
#else
  (void)input;
  (void)target;
  (void)repetitions;
  THROW_EXCEPTION(
      "NDArray::tile requires the registered Vulkan tile descriptor");
#endif
}

}  // namespace

NDArray NDArray::tile(const std::vector<LongType>& repetitions) {
  const bool identity = validateRepetitions(repetitions);
  if (getDataBuffer() == nullptr || getContext() == nullptr) {
    THROW_EXCEPTION("NDArray::tile received an array without a Vulkan device context");
  }
  const int deviceId = getDataBuffer()->deviceId();
  if (deviceId < 0 || getContext()->getDeviceID() != deviceId) {
    THROW_EXCEPTION("NDArray::tile received an invalid Vulkan device context");
  }
  ScopedVulkanDevice deviceScope(deviceId);

  const int inputRank = rankOf();
  const int rankDifference =
      inputRank - static_cast<int>(repetitions.size());

  if (identity) {
    NDArray result(*this);
    if (rankDifference < 0) {
      std::vector<LongType> alignedShape = repetitions;
      std::copy(shapeInfo() + 1, shapeInfo() + 1 + inputRank,
                alignedShape.begin() + static_cast<size_t>(-rankDifference));
      result.reshapei(ordering(), alignedShape);
    }
    return result;
  }

  auto* newShapeInfo = ShapeUtils::evalTileShapeInfo(
      *this, repetitions, getContext()->getWorkspace());
  auto* newBuffer =
      new DataBuffer(shape::length(newShapeInfo) * sizeOfT(), dataType(),
                     getContext()->getWorkspace(), true);
  NDArray result(newBuffer, const_cast<LongType*>(newShapeInfo), getContext());

  executeTileDescriptor(*this, result, repetitions);
  return result;
}

void NDArray::tile(const std::vector<LongType>& repetitions, NDArray& target) {
  validateRepetitions(repetitions);

  auto* expectedShapeInfo = ShapeUtils::evalTileShapeInfo(
      *this, repetitions, getContext()->getWorkspace());
  if (!shape::equalsSoft(expectedShapeInfo, target.shapeInfo())) {
    THROW_EXCEPTION(
        "NDArray::tile target shape is not suitable for the requested "
        "repetitions");
  }

  executeTileDescriptor(*this, target, repetitions);
}

void NDArray::tile(NDArray& target) {
  if (rankOf() > target.rankOf()) {
    THROW_EXCEPTION(
        "NDArray::tile target rank must be greater than or equal to the input "
        "rank");
  }
  if (!ShapeUtils::areShapesBroadcastable(*this, target)) {
    THROW_EXCEPTION(
        "NDArray::tile target shape is not broadcast-compatible with the "
        "input");
  }

  std::vector<LongType> repetitions(
      static_cast<size_t>(target.rankOf()), 1);
  const int offset = target.rankOf() - rankOf();
  for (int axis = 0; axis < target.rankOf(); ++axis) {
    const LongType inputDimension =
        axis < offset ? 1 : sizeAt(axis - offset);
    const LongType targetDimension = target.sizeAt(axis);
    if (inputDimension < 1 || targetDimension < 1 ||
        targetDimension % inputDimension != 0) {
      THROW_EXCEPTION(
          "NDArray::tile target shape does not define integral repetitions");
    }
    repetitions[static_cast<size_t>(axis)] =
        targetDimension / inputDimension;
  }

  executeTileDescriptor(*this, target, repetitions);
}

}  // namespace sd

#endif  // SD_VULKAN
