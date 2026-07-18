/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if defined(SD_VULKAN) && defined(HAVE_VULKAN) && HAVE_VULKAN

#include <array/NDArray.h>
#include <execution/vulkan/VulkanExecutionStream.h>
#include <execution/vulkan/VulkanLaunchContext.h>
#include <graph/Context.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/fusedElementwiseChain.h>

#include <string>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

#if NOT_EXCLUDED(OP_fused_elementwise_chain)
void fusedElementwiseChain(NDArray* input, NDArray* output,
                           const FusedElemOp* ops, int numOps,
                           NDArray** secondaryInputs, const double* clipMin,
                           const double* clipMax, LaunchContext* context) {
  if (input == nullptr || output == nullptr || ops == nullptr || numOps <= 0) {
    return;
  }
  if (numOps > 8) {
    THROW_EXCEPTION(
        "fusedElementwiseChain: chain length exceeds the fused-kernel maximum "
        "of 8 ops; the caller must split the chain into multiple fused calls.");
  }

  std::vector<NDArray*> inputs;
  inputs.reserve(static_cast<size_t>(numOps) + 1);
  inputs.push_back(input);

  std::vector<LongType> opCodes;
  opCodes.reserve(static_cast<size_t>(numOps));

  bool hasClip = false;
  for (int index = 0; index < numOps; ++index) {
    opCodes.push_back(static_cast<LongType>(ops[index]));
    hasClip = hasClip || ops[index] == FUSED_CLIP;

    if (!isBinaryFusedOp(ops[index])) continue;
    if (secondaryInputs == nullptr || secondaryInputs[index] == nullptr) {
      THROW_EXCEPTION(
          "Vulkan fused element-wise execution requires a secondary input "
          "for every binary chain step");
    }
    inputs.push_back(secondaryInputs[index]);
  }

  std::vector<NDArray*> outputs{output};
  for (auto* array : inputs) {
    if (array == nullptr || array->getDataBuffer() == nullptr) {
      THROW_EXCEPTION(
          "Vulkan fused element-wise execution received an invalid input "
          "buffer");
    }
  }
  for (auto* array : outputs) {
    if (array == nullptr || array->getDataBuffer() == nullptr) {
      THROW_EXCEPTION(
          "Vulkan fused element-wise execution received an invalid output "
          "buffer");
    }
  }

  NDArray::prepareSpecialUse(outputs, inputs);

  const int deviceId = input->getDataBuffer()->deviceId();
  if (deviceId < 0 ||
      (context != nullptr && context->getDeviceID() != deviceId)) {
    THROW_EXCEPTION(
        "Vulkan fused element-wise execution received an invalid device "
        "context");
  }

  for (auto* array : inputs) {
    if (array->getDataBuffer()->deviceId() != deviceId) {
      THROW_EXCEPTION(
          "Vulkan fused element-wise inputs must reside on one device");
    }
  }
  if (output->getDataBuffer()->deviceId() != deviceId) {
    THROW_EXCEPTION(
        "Vulkan fused element-wise output must reside on the input device");
  }

  const auto contextStream =
      context == nullptr ? nullptr : graph::vulkanExecutionStream(context);
  auto* stream =
      contextStream == nullptr
          ? graph::VulkanExecutionStream::defaultExecution(deviceId)
          : graph::VulkanExecutionStream::fromOpaque(contextStream, false);
  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    THROW_EXCEPTION(
        contextStream == nullptr
            ? "Vulkan fused element-wise execution could not resolve the "
              "exact-device default execution stream"
            : "Vulkan fused element-wise execution received an invalid "
              "context-owned execution stream");
  }

  graph::Context opContext(0);
  opContext.setInputArrays(static_cast<int>(inputs.size()), inputs.data(),
                           false);
  opContext.setOutputArrays(static_cast<int>(outputs.size()), outputs.data(),
                            false);
  opContext.setIArguments(opCodes);
  if (hasClip) {
    opContext.setTArguments(std::vector<double>{
        clipMin == nullptr ? 0.0 : *clipMin,
        clipMax == nullptr ? 0.0 : *clipMax});
  }

  sd::ops::fused_elementwise_chain descriptor;
  std::string error;
  const Status status = graph::VulkanEagerExecutor::execute(
      descriptor.getOpHash(), opContext, *stream, &error);
  if (status != Status::OK) {
    if (error.empty()) {
      error = "Vulkan fused element-wise descriptor execution failed";
    }
    THROW_EXCEPTION(error.c_str());
  }
}
#endif

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
