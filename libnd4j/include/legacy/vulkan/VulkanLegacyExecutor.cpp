/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
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
#include <helpers/shape.h>
#include <legacy/vulkan/VulkanLegacyExecutor.h>

#include <memory>
#include <sstream>
#include <utility>

namespace sd {
namespace graph {
namespace {

void setError(std::string* target, const std::string& message) {
  if (target != nullptr) *target = message;
}

VulkanExecutionStream* resolveStream(sd::LaunchContext* launchContext,
                                     std::string* errorMessage) {
  if (launchContext == nullptr) {
    setError(errorMessage, "Vulkan legacy execution has no launch context");
    return nullptr;
  }

  const int deviceId = launchContext->getDeviceID();
  void* contextOwned = vulkanExecutionStream(launchContext);
  VulkanExecutionStream* stream = nullptr;
  if (contextOwned != nullptr) {
    stream = VulkanExecutionStream::fromOpaque(contextOwned, false);
  } else {
    stream = VulkanExecutionStream::defaultExecution(deviceId);
  }

  if (stream == nullptr || !stream->isActive() ||
      stream->deviceId() != deviceId) {
    setError(errorMessage,
             "Vulkan legacy execution could not resolve the exact-device stream");
    return nullptr;
  }
  return stream;
}

bool validateTensor(const VulkanLegacyTensor& tensor,
                    const char* role,
                    std::string* errorMessage) {
  if (tensor.hostShapeInfo == nullptr) {
    setError(errorMessage,
             std::string("Vulkan legacy ") + role +
                 " tensor has no host shape information");
    return false;
  }

  const sd::LongType length = shape::length(tensor.hostShapeInfo);
  if (length > 0 && tensor.deviceData == nullptr) {
    setError(errorMessage,
             std::string("Vulkan legacy ") + role +
                 " tensor has no Vulkan device allocation");
    return false;
  }
  return true;
}

std::unique_ptr<NDArray> wrapTensor(const VulkanLegacyTensor& tensor,
                                    sd::LaunchContext* launchContext) {
  return std::make_unique<NDArray>(
      tensor.hostData, tensor.deviceData,
      const_cast<sd::LongType*>(tensor.hostShapeInfo), launchContext,
      /*isBuffAlloc=*/false, /*isBuffDAlloc=*/false, /*offset=*/0);
}

template <typename Execute>
Status executeInvocation(sd::LaunchContext* launchContext,
                         const VulkanInvocationArguments& invocation,
                         std::string* errorMessage, Execute&& execute) {
#if !defined(HAVE_MLIR) || !HAVE_MLIR
  setError(errorMessage, "Vulkan execution requires MLIR support");
  return Status::VALIDATION;
#else
  VulkanExecutionStream* stream = resolveStream(launchContext, errorMessage);
  if (stream == nullptr) return Status::KERNEL_FAILURE;

  std::vector<std::unique_ptr<NDArray>> ownedInputs;
  std::vector<std::unique_ptr<NDArray>> ownedOutputs;
  ownedInputs.reserve(invocation.inputs.size());
  ownedOutputs.reserve(invocation.outputs.size());

  Context context(1);
  for (size_t index = 0; index < invocation.inputs.size(); ++index) {
    if (!validateTensor(invocation.inputs[index], "input", errorMessage)) {
      return Status::VALIDATION;
    }
    ownedInputs.emplace_back(
        wrapTensor(invocation.inputs[index], launchContext));
    context.setInputArray(static_cast<int>(index), ownedInputs.back().get(),
                          /*removable=*/false);
  }
  for (size_t index = 0; index < invocation.outputs.size(); ++index) {
    if (!validateTensor(invocation.outputs[index], "output", errorMessage)) {
      return Status::VALIDATION;
    }
    ownedOutputs.emplace_back(
        wrapTensor(invocation.outputs[index], launchContext));
    context.setOutputArray(static_cast<int>(index), ownedOutputs.back().get(),
                           /*removable=*/false);
  }

  context.setIArguments(invocation.integerArguments);
  context.setTArguments(invocation.floatingArguments);
  context.setBArguments(invocation.booleanArguments);
  return execute(context, *stream);
#endif
}

Status executeDescriptorHash(
    sd::LaunchContext* launchContext, sd::LongType descriptorHash,
    const VulkanInvocationArguments& invocation, std::string* errorMessage) {
  return executeInvocation(
      launchContext, invocation, errorMessage,
      [&](Context& context, VulkanExecutionStream& stream) {
        return VulkanEagerExecutor::execute(descriptorHash, context, stream,
                                            errorMessage);
      });
}

}  // namespace

Status executeVulkanLegacy(sd::LaunchContext* launchContext,
                           const VulkanLegacyInvocation& invocation,
                           std::string* errorMessage) {
  if (VulkanLegacyOpCatalog::lookup(invocation.family,
                                    invocation.opNum) == nullptr) {
    setError(errorMessage,
             "Vulkan legacy execution received a non-canonical typed identity");
    return Status::VALIDATION;
  }

  return executeInvocation(
      launchContext, invocation, errorMessage,
      [&](Context& context, VulkanExecutionStream& stream) {
        return VulkanEagerExecutor::execute(
            invocation.family, invocation.opNum, context, stream,
            reinterpret_cast<RandomGenerator*>(invocation.randomState),
            errorMessage);
      });
}

Status executeVulkanDescriptor(sd::LaunchContext* launchContext,
                               const VulkanDescriptorInvocation& invocation,
                               std::string* errorMessage) {
  return executeDescriptorHash(launchContext, invocation.descriptorHash,
                               invocation, errorMessage);
}

void requireVulkanLegacyExecution(sd::LaunchContext* launchContext,
                                  const VulkanLegacyInvocation& invocation) {
  std::string errorMessage;
  const Status status =
      executeVulkanLegacy(launchContext, invocation, &errorMessage);
  if (status == Status::OK) return;

  std::ostringstream message;
  message << "Vulkan legacy execution failed"
          << " (family=" << static_cast<int>(invocation.family)
          << ", opNum=" << invocation.opNum << ')';
  if (!errorMessage.empty()) message << ": " << errorMessage;
  THROW_EXCEPTION(message.str().c_str());
}

void requireVulkanDescriptorExecution(
    sd::LaunchContext* launchContext,
    const VulkanDescriptorInvocation& invocation) {
  std::string errorMessage;
  const Status status =
      executeVulkanDescriptor(launchContext, invocation, &errorMessage);
  if (status == Status::OK) return;

  std::ostringstream message;
  message << "Vulkan descriptor execution failed"
          << " (hash=" << invocation.descriptorHash << ')';
  if (!errorMessage.empty()) message << ": " << errorMessage;
  THROW_EXCEPTION(message.str().c_str());
}

}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
