/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>
#include <system/op_boilerplate.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN && NOT_EXCLUDED(OP_listdiff)

#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/listdiff.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

LongType listDiffCount(LaunchContext* context, NDArray* values, NDArray* keep) {
  THROW_EXCEPTION(
      "Vulkan listdiff shape inference requires the shared device scalar "
      "metadata API for stable-compaction counts");
}

Status listDiffFunctor(LaunchContext* context, NDArray* values, NDArray* keep,
                       NDArray* output1, NDArray* output2) {
  std::vector<NDArray*> inputs{values, keep};
  std::vector<NDArray*> outputs{output1, output2};
  NDArray::prepareSpecialUse(outputs, inputs);

  graph::Context opContext(0);
  opContext.setInputArrays(static_cast<int>(inputs.size()), inputs.data(), false);
  opContext.setOutputArrays(static_cast<int>(outputs.size()), outputs.data(),
                            false);

  const int deviceId = values->getDataBuffer()->deviceId();
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

  sd::ops::listdiff descriptor;
  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *stream, &error);
  if (status == Status::OK) NDArray::registerSpecialUse(outputs, inputs);
  return status;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
