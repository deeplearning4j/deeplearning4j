/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>
#include <system/op_boilerplate.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN && NOT_EXCLUDED(OP_choose)

#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <ops/declarable/headers/boolean.h>
#include <ops/declarable/helpers/choose.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {
namespace {

void executeChoose(LaunchContext* launchContext,
                   const std::vector<NDArray*>& inputs, int mode,
                   const std::vector<double>& tArgs, NDArray* result,
                   NDArray* numResults) {
  if (result == nullptr) {
    THROW_EXCEPTION(
        "Vulkan choose shape inference requires the shared device scalar "
        "metadata API for stable-compaction counts");
  }

  std::vector<NDArray*> outputs{result, numResults};
  NDArray::prepareSpecialUse(outputs, inputs);

  graph::Context opContext(0);
  opContext.setInputArrays(static_cast<int>(inputs.size()),
                           const_cast<NDArray**>(inputs.data()), false);
  opContext.setOutputArrays(static_cast<int>(outputs.size()), outputs.data(),
                            false);
  opContext.setIArguments(std::vector<LongType>{mode});
  if (!tArgs.empty()) opContext.setTArguments(tArgs);

  const int deviceId = inputs.front()->getDataBuffer()->deviceId();
  const auto contextStream =
      launchContext == nullptr ? nullptr
                               : graph::vulkanExecutionStream(launchContext);
  auto* stream =
      contextStream == nullptr
          ? graph::VulkanExecutionStream::defaultExecution(deviceId)
          : graph::VulkanExecutionStream::fromOpaque(contextStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    THROW_EXCEPTION("Vulkan choose received an invalid execution stream");
  }

  sd::ops::choose descriptor;
  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *stream, &error);
  if (status != Status::OK) {
    if (error.empty()) error = "Vulkan choose descriptor execution failed";
    THROW_EXCEPTION(error.c_str());
  }

  NDArray::registerSpecialUse(outputs, inputs);
}

}  // namespace

void chooseFunctorArray(LaunchContext* context, NDArray* arg, NDArray* comp,
                        int mode, NDArray* result, NDArray* numResults) {
  executeChoose(context, std::vector<NDArray*>{arg, comp}, mode, {}, result,
                numResults);
}

void chooseFunctorScalar(LaunchContext* context, NDArray* arg, double scalar,
                         int mode, NDArray* result, NDArray* numResults) {
  executeChoose(context, std::vector<NDArray*>{arg}, mode,
                std::vector<double>{scalar}, result, numResults);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
