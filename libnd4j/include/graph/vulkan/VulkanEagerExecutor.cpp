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
#include <graph/NativeDynamicShapePlan.h>
#include <graph/vulkan/VulkanDeviceContext.h>
#include <graph/vulkan/VulkanEagerExecutor.h>
#include <graph/vulkan/VulkanKernelEmitterCatalog.h>
#include <graph/vulkan/VulkanReplayHandle.h>
#include <graph/vulkan/VulkanSegmentRecorder.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace sd {
namespace graph {
namespace {

template <typename T>
T* copyArguments(const std::vector<T>& source) {
  if (source.empty()) return nullptr;
  auto* result = new T[source.size()];
  std::copy(source.begin(), source.end(), result);
  return result;
}

bool* copyBoolArguments(const std::vector<bool>& source) {
  if (source.empty()) return nullptr;
  auto* result = new bool[source.size()];
  for (size_t index = 0; index < source.size(); ++index) {
    result[index] = source[index];
  }
  return result;
}

void setError(std::string* target, const std::string& message) {
  if (target != nullptr) *target = message;
}

bool outputsAreResolvedAliases(const std::vector<NDArray*>& inputs,
                               const std::vector<NDArray*>& outputs) {
  if (outputs.empty()) return false;

  for (auto* output : outputs) {
    if (output == nullptr || !output->hasValidShapeInfo() ||
        output->dataBuffer() == nullptr) {
      return false;
    }

    bool aliasesInput = false;
    for (auto* input : inputs) {
      if (input != nullptr && input->dataBuffer() == output->dataBuffer()) {
        aliasesInput = true;
        break;
      }
    }
    if (!aliasesInput) return false;
  }

  return true;
}

void copyContextArguments(Context& context, NativeSlot& slot) {
  const auto* iArgs = context.getIArguments();
  const auto* tArgs = context.getTArguments();
  const auto* bArgs = context.getBArguments();
  const auto* dArgs = context.getDArguments();
  const auto* sArgs = context.getSArguments();

  slot.args.numIArgs = iArgs == nullptr ? 0 : static_cast<int>(iArgs->size());
  slot.args.iArgs = iArgs == nullptr ? nullptr : copyArguments(*iArgs);
  slot.args.numTArgs = tArgs == nullptr ? 0 : static_cast<int>(tArgs->size());
  slot.args.tArgs = tArgs == nullptr ? nullptr : copyArguments(*tArgs);
  slot.args.numBArgs = bArgs == nullptr ? 0 : static_cast<int>(bArgs->size());
  slot.args.bArgs = bArgs == nullptr ? nullptr : copyBoolArguments(*bArgs);
  slot.args.numDArgs = dArgs == nullptr ? 0 : static_cast<int>(dArgs->size());
  slot.args.dArgs = dArgs == nullptr ? nullptr : copyArguments(*dArgs);
  slot.args.numSArgs = sArgs == nullptr ? 0 : static_cast<int>(sArgs->size());
  slot.args.sArgs = sArgs == nullptr ? nullptr : copyArguments(*sArgs);
}

Status recordAndExecuteSlot(NativeSlot& slot, Context& context,
                            VulkanExecutionStream& stream,
                            VulkanDeviceContext& deviceContext,
                            RandomGenerator* randomState,
                            std::string* errorMessage) {
  auto& inputs = context.fastpath_in();
  auto& outputs = context.fastpath_out();
  NDArray** inputData = inputs.data();
  NDArray** outputData = outputs.data();

  if (!VulkanSegmentRecorder::opIsRecordable(
          slot, inputData, slot.wiring.numInputs, outputData,
          slot.wiring.numOutputs, deviceContext.caps())) {
    setError(errorMessage,
             "Vulkan eager metadata is not recordable for these operands");
    return Status::VALIDATION;
  }

  VulkanExecutionStreamGuard streamGuard(&stream);
  VulkanReplayHandle handle(stream.deviceId());
  if (!handle.beginCapture(&stream)) {
    setError(errorMessage, "Vulkan eager command capture could not begin");
    return Status::KERNEL_FAILURE;
  }

  auto recorder = std::make_unique<VulkanSegmentRecorder>(&handle);
  if (!recorder->recordOp(slot, inputData, slot.wiring.numInputs, outputData,
                          slot.wiring.numOutputs, randomState) ||
      handle.getNumDispatches() <= 0 || !handle.endCapture(&stream) ||
      !handle.finalize() || !recorder->prepareReplayInputs(&stream)) {
    setError(errorMessage,
             "Vulkan eager recording or finalization failed");
    return Status::KERNEL_FAILURE;
  }

  handle.setRecorder(std::move(recorder));
  if (!handle.replay(&stream)) {
    setError(errorMessage, "Vulkan eager command submission failed");
    return Status::KERNEL_FAILURE;
  }

  handle.getRecorder()->markReplayOutputs();
  return Status::OK;
}

}  // namespace

Status VulkanEagerExecutor::execute(LongType descriptorHash, Context& context,
                                    std::string* errorMessage) {
  auto* launchContext = context.launchContext();
  if (launchContext == nullptr) {
    setError(errorMessage, "Vulkan eager execution has no launch context");
    return Status::KERNEL_FAILURE;
  }

  const int deviceId = launchContext->getDeviceID();
  void* contextOwnedStream = vulkanExecutionStream(launchContext);
  VulkanExecutionStream* stream = nullptr;
  if (contextOwnedStream != nullptr) {
    stream = VulkanExecutionStream::fromOpaque(contextOwnedStream, false);
    if (stream == nullptr || !stream->isActive() ||
        stream->deviceId() != deviceId) {
      setError(errorMessage,
               "Vulkan eager execution received an invalid context-owned stream");
      return Status::KERNEL_FAILURE;
    }
  } else {
    stream = VulkanExecutionStream::defaultExecution(deviceId);
    if (stream == nullptr || !stream->isActive() ||
        stream->deviceId() != deviceId) {
      setError(errorMessage,
               "Vulkan eager execution could not resolve the exact-device default stream");
      return Status::KERNEL_FAILURE;
    }
  }

  return execute(descriptorHash, context, *stream, errorMessage);
}

Status VulkanEagerExecutor::execute(LongType descriptorHash, Context& context,
                                    VulkanExecutionStream& stream,
                                    std::string* errorMessage) {
  if (!stream.isActive()) {
    setError(errorMessage, "Vulkan eager execution received an inactive stream");
    return Status::KERNEL_FAILURE;
  }

  auto* deviceContext = VulkanDeviceContext::getContext(stream.deviceId());
  if (deviceContext == nullptr || deviceContext->isLost()) {
    setError(errorMessage, "Vulkan eager execution has no usable device context");
    return Status::KERNEL_FAILURE;
  }

  auto* op = ops::OpRegistrator::getInstance().getOperation(descriptorHash);
  if (op == nullptr || op->getOpDescriptor() == nullptr) {
    setError(errorMessage,
             "Vulkan eager execution could not resolve the descriptor hash");
    return Status::VALIDATION;
  }

  auto& inputs = context.fastpath_in();
  auto& outputs = context.fastpath_out();

  // View and identity operations are metadata-only when the framework has
  // installed outputs that share their source buffers. Dispatching a writing
  // kernel against those aliases would corrupt unread input elements. CUDA and
  // the generic op bodies treat this state as complete; Vulkan follows the same
  // descriptor-trait contract, and replay records only downstream device work.
  if (op->getOpDescriptor()->hasAnyTrait(
          sd::ops::OP_TRAIT_VIEW_PRODUCING | sd::ops::OP_TRAIT_IDENTITY) &&
      outputsAreResolvedAliases(inputs, outputs)) {
    return Status::OK;
  }

  if (findVulkanKernelEmitter(descriptorHash) == nullptr) {
    setError(errorMessage,
             "Vulkan eager execution does not support this descriptor hash");
    return Status::VALIDATION;
  }

  NativeSlot slot;
  slot.ident.opHash = descriptorHash;
  slot.ident.op = op;
  if (op->getOpName() != nullptr) slot.ident.opName = *op->getOpName();
  slot.opTraits_ = op->getOpDescriptor()->getTraits();
  slot.targetDeviceId = stream.deviceId();
  slot.wiring.numInputs = static_cast<int>(inputs.size());
  slot.wiring.numOutputs = static_cast<int>(outputs.size());
  slot.flags.structuralIArgCount =
      op->getOpDescriptor()->getNumberOfStructuralIArgs();

  copyContextArguments(context, slot);
  return recordAndExecuteSlot(slot, context, stream, *deviceContext,
                              nullptr, errorMessage);
}

Status VulkanEagerExecutor::execute(VulkanLegacyOpFamily family, int opNum,
                                    Context& context,
                                    VulkanExecutionStream& stream,
                                    RandomGenerator* randomState,
                                    std::string* errorMessage) {
  if (!stream.isActive()) {
    setError(errorMessage, "Vulkan eager execution received an inactive stream");
    return Status::KERNEL_FAILURE;
  }

  auto* deviceContext = VulkanDeviceContext::getContext(stream.deviceId());
  if (deviceContext == nullptr || deviceContext->isLost()) {
    setError(errorMessage, "Vulkan eager execution has no usable device context");
    return Status::KERNEL_FAILURE;
  }

  const auto* opInfo = VulkanLegacyOpCatalog::lookup(family, opNum);
  const auto* emitter = findVulkanLegacyKernelEmitter(family, opNum);
  const auto legacyTypeCode = vulkanLegacyTypeCode(family);
  if (opInfo == nullptr || emitter == nullptr || !legacyTypeCode.has_value()) {
    setError(errorMessage,
             "Vulkan eager execution does not support this typed legacy identity");
    return Status::VALIDATION;
  }

  auto& inputs = context.fastpath_in();
  auto& outputs = context.fastpath_out();

  NativeSlot slot;
  slot.ident.opHash = 0;
  slot.ident.op = nullptr;
  slot.ident.opName = opInfo->diagnosticName();
  slot.legacy.legacyOpType = *legacyTypeCode;
  slot.legacy.legacyOpNum = opNum;
  slot.opTraits_ = emitter->traits;
  slot.targetDeviceId = stream.deviceId();
  slot.wiring.numInputs = static_cast<int>(inputs.size());
  slot.wiring.numOutputs = static_cast<int>(outputs.size());
  slot.flags.structuralIArgCount = -1;
  copyContextArguments(context, slot);

  return recordAndExecuteSlot(slot, context, stream, *deviceContext,
                              randomState, errorMessage);
}

}  // namespace graph
}  // namespace sd

#endif  // SD_VULKAN && HAVE_VULKAN
