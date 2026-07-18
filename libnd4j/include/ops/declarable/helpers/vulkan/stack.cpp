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
#include <ops/declarable/helpers/stack.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {
namespace {

template <typename Op>
void executeDataMovement(Op& op, LaunchContext* launchContext,
                         const std::vector<NDArray*>& inputs,
                         const std::vector<NDArray*>& outputs, int dimension) {
  if (inputs.empty() || outputs.empty()) return;

  for (auto* array : inputs) {
    if (array == nullptr || array->getDataBuffer() == nullptr) {
      THROW_EXCEPTION(
          "Vulkan data-movement helper received an invalid input buffer");
    }
  }
  for (auto* array : outputs) {
    if (array == nullptr || array->getDataBuffer() == nullptr) {
      THROW_EXCEPTION(
          "Vulkan data-movement helper received an invalid output buffer");
    }
  }

  NDArray::prepareSpecialUse(outputs, inputs);

  const int deviceId = inputs.front()->getDataBuffer()->deviceId();
  if (deviceId < 0 ||
      (launchContext != nullptr && launchContext->getDeviceID() != deviceId)) {
    THROW_EXCEPTION(
        "Vulkan data-movement helper received an invalid device context");
  }

  for (auto* array : inputs) {
    if (array->getDataBuffer()->deviceId() != deviceId) {
      THROW_EXCEPTION(
          "Vulkan data-movement inputs must reside on one device");
    }
  }
  for (auto* array : outputs) {
    if (array->getDataBuffer()->deviceId() != deviceId) {
      THROW_EXCEPTION(
          "Vulkan data-movement outputs must reside on the input device");
    }
  }

  const auto contextStream =
      launchContext == nullptr
          ? nullptr
          : graph::vulkanExecutionStream(launchContext);
  auto* stream =
      contextStream == nullptr
          ? graph::VulkanExecutionStream::defaultExecution(deviceId)
          : graph::VulkanExecutionStream::fromOpaque(contextStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        contextStream == nullptr
            ? "Vulkan data-movement helper could not resolve the exact-device "
              "default execution stream"
            : "Vulkan data-movement helper received an invalid context-owned "
              "execution stream");
  }

  graph::Context opContext(0);
  opContext.setInputArrays(static_cast<int>(inputs.size()),
                           const_cast<NDArray**>(inputs.data()), false);
  opContext.setOutputArrays(static_cast<int>(outputs.size()),
                            const_cast<NDArray**>(outputs.data()), false);
  opContext.setIArguments(std::vector<LongType>{dimension});

  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      op.getOpHash(), opContext, *stream, &error);
  if (status != Status::OK) {
    if (error.empty()) error = "Vulkan data-movement execution failed";
    THROW_EXCEPTION(error.c_str());
  }
}

}  // namespace

#if NOT_EXCLUDED(OP_stack)
void stack(LaunchContext* context, const std::vector<NDArray*>& inArrs,
           NDArray& output, const int dim) {
  sd::ops::stack op;
  executeDataMovement(op, context, inArrs, std::vector<NDArray*>{&output}, dim);
}
#endif

#if NOT_EXCLUDED(OP_unstack)
void unstack(LaunchContext* context, NDArray& input,
             const std::vector<NDArray*>& outArrs, const int dim) {
  sd::ops::unstack op;
  executeDataMovement(op, context, std::vector<NDArray*>{&input}, outArrs, dim);
}
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
