/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>
#include <system/op_boilerplate.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN && \
    (NOT_EXCLUDED(OP_unique) || NOT_EXCLUDED(OP_unique_with_counts))

#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/unique.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

LongType uniqueCount(LaunchContext* context, NDArray* input) {
  THROW_EXCEPTION(
      "Vulkan unique shape inference requires the shared device scalar "
      "metadata API for stable first-occurrence counts");
}

Status uniqueFunctor(LaunchContext* context, NDArray* input, NDArray* values,
                     NDArray* indices, NDArray* counts) {
  std::vector<NDArray*> inputs{input};
  std::vector<NDArray*> outputs{values, indices};
  if (counts != nullptr) outputs.push_back(counts);
  NDArray::prepareSpecialUse(outputs, inputs);

  graph::Context opContext(0);
  opContext.setInputArrays(static_cast<int>(inputs.size()), inputs.data(), false);
  opContext.setOutputArrays(static_cast<int>(outputs.size()), outputs.data(),
                            false);

  const int deviceId = input->getDataBuffer()->deviceId();
  const auto contextStream =
      context == nullptr ? nullptr : graph::vulkanExecutionStream(context);
  auto* stream =
      contextStream == nullptr
          ? graph::VulkanExecutionStream::defaultExecution(deviceId)
          : graph::VulkanExecutionStream::fromOpaque(contextStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    return Status::VALIDATION;
  }

  std::string error;
  Status status;
#if NOT_EXCLUDED(OP_unique) && NOT_EXCLUDED(OP_unique_with_counts)
  if (counts == nullptr) {
    sd::ops::unique descriptor;
    status = graph::VulkanEagerExecutor::execute(
        descriptor.getOpHash(), opContext, *stream, &error);
  } else {
    sd::ops::unique_with_counts descriptor;
    status = graph::VulkanEagerExecutor::execute(
        descriptor.getOpHash(), opContext, *stream, &error);
  }
#elif NOT_EXCLUDED(OP_unique)
  sd::ops::unique descriptor;
  status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *stream, &error);
#else
  sd::ops::unique_with_counts descriptor;
  status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *stream, &error);
#endif

  if (status == Status::OK) NDArray::registerSpecialUse(outputs, inputs);
  return status;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
