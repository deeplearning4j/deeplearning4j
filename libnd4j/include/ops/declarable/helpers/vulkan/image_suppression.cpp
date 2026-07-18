/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/image_suppression.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {
namespace {

graph::VulkanExecutionStream* resolveStream(LaunchContext* context, int deviceId) {
  const auto opaque =
      context == nullptr ? nullptr : graph::vulkanExecutionStream(context);
  auto* stream = opaque == nullptr
                     ? graph::VulkanExecutionStream::defaultExecution(deviceId)
                     : graph::VulkanExecutionStream::fromOpaque(opaque, false);
  if (stream == nullptr || !stream->isActive() || stream->deviceId() != deviceId) {
    THROW_EXCEPTION(opaque == nullptr
                        ? "Vulkan NMS could not resolve the exact-device default execution stream"
                        : "Vulkan NMS received an invalid context-owned execution stream");
  }
  return stream;
}

template <typename Op>
LongType executeNms(Op& descriptor, LaunchContext* context, NDArray* boxes,
                    NDArray* scores, int maxSize, double overlapThreshold,
                    double scoreThreshold, NDArray* output) {
  if (output == nullptr) {
    THROW_EXCEPTION(
        "Vulkan NMS shape discovery requires a device-produced selected-count "
        "scalar API; VulkanEagerExecutor currently exposes tensor outputs only");
  }
  if (boxes == nullptr || scores == nullptr || boxes->getDataBuffer() == nullptr ||
      scores->getDataBuffer() == nullptr || output->getDataBuffer() == nullptr) {
    THROW_EXCEPTION("Vulkan NMS received an invalid tensor buffer");
  }

  const int deviceId = boxes->getDataBuffer()->deviceId();
  if (deviceId < 0 || scores->getDataBuffer()->deviceId() != deviceId ||
      output->getDataBuffer()->deviceId() != deviceId ||
      (context != nullptr && context->getDeviceID() != deviceId)) {
    THROW_EXCEPTION("Vulkan NMS tensors and launch context must use one device");
  }

  std::vector<NDArray*> inputs{boxes, scores};
  std::vector<NDArray*> outputs{output};
  NDArray::prepareSpecialUse(outputs, inputs);

  graph::Context opContext(0);
  opContext.setInputArrays(static_cast<int>(inputs.size()), inputs.data(), false);
  opContext.setOutputArrays(static_cast<int>(outputs.size()), outputs.data(), false);
  opContext.setIArguments(std::vector<LongType>{maxSize});
  opContext.setTArguments(
      std::vector<double>{overlapThreshold, scoreThreshold});

  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *resolveStream(context, deviceId), &error);
  if (status != Status::OK) {
    if (error.empty()) error = "Vulkan NMS descriptor execution failed";
    THROW_EXCEPTION(error.c_str());
  }

  NDArray::registerSpecialUse(outputs, inputs);
  return output->lengthOf();
}

}  // namespace

LongType nonMaxSuppressionV3(LaunchContext* context, NDArray* boxes,
                             NDArray* scores, int maxSize,
                             double overlapThreshold, double scoreThreshold,
                             NDArray* output) {
#if NOT_EXCLUDED(OP_non_max_suppression_v3)
  sd::ops::non_max_suppression_v3 descriptor;
  return executeNms(descriptor, context, boxes, scores, maxSize,
                    overlapThreshold, scoreThreshold, output);
#else
  THROW_EXCEPTION("Vulkan non_max_suppression_v3 is excluded from this build");
  return 0;
#endif
}

LongType nonMaxSuppressionGeneric(LaunchContext* context, NDArray* boxes,
                                  NDArray* scores, int maxSize,
                                  double overlapThreshold,
                                  double scoreThreshold, NDArray* output) {
#if NOT_EXCLUDED(OP_non_max_suppression_overlaps)
  sd::ops::non_max_suppression_overlaps descriptor;
  return executeNms(descriptor, context, boxes, scores, maxSize,
                    overlapThreshold, scoreThreshold, output);
#else
  THROW_EXCEPTION(
      "Vulkan non_max_suppression_overlaps is excluded from this build");
  return 0;
#endif
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
