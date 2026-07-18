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

#include <array/NDArray.h>
#include <execution/AffinityManager.h>
#include <execution/LaunchContext.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <helpers/ShapeUtils.h>
#include <legacy/NativeOpExecutioner.h>
#include <ops/declarable/headers/transforms.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>

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
      THROW_EXCEPTION("NDArray Vulkan operation requires a physical device");
    }
    if (changed_) AffinityManager::setCurrentDevice(deviceId);
  }

  ~ScopedVulkanDevice() noexcept {
    if (!changed_) return;
    try {
      AffinityManager::setCurrentDevice(previousDeviceId_);
    } catch (...) {
      // Device restoration must not throw while another exception is active.
    }
  }

  ScopedVulkanDevice(const ScopedVulkanDevice&) = delete;
  ScopedVulkanDevice& operator=(const ScopedVulkanDevice&) = delete;

 private:
  int previousDeviceId_;
  bool changed_;
};

int validateSameDevice(NDArray& input, NDArray& output,
                       const char* operation) {
  if (input.getDataBuffer() == nullptr || output.getDataBuffer() == nullptr ||
      input.getContext() == nullptr) {
    std::string message(operation);
    message += ": arrays must have Vulkan data buffers and a launch context";
    THROW_EXCEPTION(message.c_str());
  }

  const int deviceId = input.getDataBuffer()->deviceId();
  if (deviceId < 0 || input.getContext()->getDeviceID() != deviceId ||
      output.getDataBuffer()->deviceId() != deviceId) {
    std::string message(operation);
    message += ": arrays must reside on the launch context's physical device";
    THROW_EXCEPTION(message.c_str());
  }
  return deviceId;
}

graph::VulkanExecutionStream* resolveStream(LaunchContext* launchContext,
                                            int deviceId,
                                            const char* operation) {
  const auto contextStream = graph::vulkanExecutionStream(launchContext);
  auto* stream =
      contextStream == nullptr
          ? graph::VulkanExecutionStream::defaultExecution(deviceId)
          : graph::VulkanExecutionStream::fromOpaque(contextStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    std::string message(operation);
    message += ": could not resolve an active exact-device execution stream";
    THROW_EXCEPTION(message.c_str());
  }
  return stream;
}

void executeRepeat(NDArray& input, NDArray& target, int axis,
                   const std::vector<LongType>& repeats) {
#if NOT_EXCLUDED(OP_repeat)
  const int deviceId =
      validateSameDevice(input, target, "NDArray::repeat");
  ScopedVulkanDevice deviceScope(deviceId);

  std::vector<LongType> iArguments(repeats);
  iArguments.push_back(axis);

  std::vector<NDArray*> inputs{&input};
  std::vector<NDArray*> outputs{&target};
  NDArray::prepareSpecialUse(outputs, inputs);

  graph::Context opContext(0);
  opContext.setInputArrays(1, inputs.data(), false);
  opContext.setOutputArrays(1, outputs.data(), false);
  opContext.setIArguments(iArguments);

  ops::repeat descriptor;
  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext,
      *resolveStream(input.getContext(), deviceId, "NDArray::repeat"), &error);
  if (status != Status::OK) {
    if (error.empty()) error = "NDArray::repeat Vulkan execution failed";
    THROW_EXCEPTION(error.c_str());
  }

  NDArray::registerSpecialUse(outputs, inputs);
#else
  (void)input;
  (void)target;
  (void)axis;
  (void)repeats;
  THROW_EXCEPTION("NDArray::repeat requires the registered Vulkan repeat descriptor");
#endif
}

void deviceAssign(NDArray& source, NDArray& target) {
  NDArray::prepareSpecialUse({&target}, {&source});
  NativeOpExecutioner::execTransformAny(
      source.getContext(), transform::Assign, source.buffer(),
      source.shapeInfo(), source.specialBuffer(), source.specialShapeInfo(),
      target.buffer(), target.shapeInfo(), target.specialBuffer(),
      target.specialShapeInfo(), nullptr, false);
  NDArray::registerSpecialUse({&target}, {&source});
}

}  // namespace

NDArray NDArray::repeat(const int axis,
                        const std::vector<LongType>& repeats) {
  auto* nonConst = const_cast<NDArray*>(this);
  std::vector<LongType> outputShape =
      ShapeUtils::evalRepeatShape(axis, repeats, *nonConst);
  NDArray output('c', outputShape, dataType(), getContext());
  executeRepeat(*nonConst, output, axis, repeats);
  return output;
}

void NDArray::repeat(const int axis, const std::vector<LongType>& repeats,
                     NDArray& target) {
  auto* nonConst = const_cast<NDArray*>(this);
  std::vector<LongType> outputShape =
      ShapeUtils::evalRepeatShape(axis, repeats, *nonConst);
  if (!target.isSameShape(outputShape)) {
    THROW_EXCEPTION(
        "NDArray::repeat(const int axis, const std::vector<int>& repeats, "
        "NDArray& target) method: wrong shape of target array!");
  }

  validateSameDevice(*nonConst, target, "NDArray::repeat");
  if (dataType() == target.dataType()) {
    executeRepeat(*nonConst, target, axis, repeats);
    return;
  }

  NDArray sameTypeTarget('c', outputShape, dataType(), getContext());
  executeRepeat(*nonConst, sameTypeTarget, axis, repeats);
  deviceAssign(sameTypeTarget, target);
}

void NDArray::setIdentity() {
  if (isS()) {
    THROW_EXCEPTION(
        "NDArray::setIdentity: you can't use this method on String array!");
  }
  if (getDataBuffer() == nullptr || getContext() == nullptr) {
    THROW_EXCEPTION(
        "NDArray::setIdentity requires a Vulkan data buffer and launch context");
  }

  const int deviceId = getDataBuffer()->deviceId();
  if (deviceId < 0 || getContext()->getDeviceID() != deviceId) {
    THROW_EXCEPTION(
        "NDArray::setIdentity requires the launch context's physical device");
  }
  ScopedVulkanDevice deviceScope(deviceId);

  int zero = 0;
  assign(zero);
  NDArray diagonalView = diagonal('c');
  int one = 1;
  diagonalView.assign(one);
}

void NDArray::swapUnsafe(NDArray& other) {
  if (dataType() != other.dataType()) {
    THROW_EXCEPTION(
        "NDArray::swapUnsage method: both arrays must have the same data type");
  }
  if (specialBuffer() == nullptr || other.specialBuffer() == nullptr) {
    THROW_EXCEPTION(
        "NDArray::swapUnsafe method: input array should not be empty!");
  }
  if (lengthOf() != other.lengthOf()) {
    THROW_EXCEPTION(
        "NDArray::swapUnsafe method: input arrays should have the same length!");
  }

  const int deviceId =
      validateSameDevice(*this, other, "NDArray::swapUnsafe");
  ScopedVulkanDevice deviceScope(deviceId);

  std::vector<LongType> temporaryShape(shapeOf(), shapeOf() + rankOf());
  NDArray temporary(ordering(), temporaryShape, dataType(), getContext());
  deviceAssign(*this, temporary);
  deviceAssign(other, *this);
  deviceAssign(temporary, other);
  synchronize("NDArray::swapUnsafe");
}

}  // namespace sd

#endif  // SD_VULKAN
