/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifdef SD_TPU

#include <graph/tpu/TpuEagerExecutor.h>

#include <graph/Context.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/tpu/PjrtClientManager.h>
#include <graph/tpu/StableHloGraphLowering.h>
#include <graph/tpu/TpuReplayHandle.h>
#include <ops/declarable/OpRegistrator.h>

#include <algorithm>
#include <string>
#include <vector>

namespace sd {
namespace graph {
namespace {

template <typename T>
T* copyArguments(const std::vector<T>* source) {
  if (source == nullptr || source->empty()) return nullptr;
  auto* result = new T[source->size()];
  std::copy(source->begin(), source->end(), result);
  return result;
}

bool* copyBoolArguments(const std::vector<bool>* source) {
  if (source == nullptr || source->empty()) return nullptr;
  auto* result = new bool[source->size()];
  for (size_t i = 0; i < source->size(); ++i) result[i] = (*source)[i];
  return result;
}

void setError(std::string* target, const std::string& message) {
  if (target != nullptr) *target = message;
}

void copyContextArguments(Context& context, NativeSlot& slot) {
  const auto* iArgs = context.getIArguments();
  const auto* tArgs = context.getTArguments();
  const auto* bArgs = context.getBArguments();
  const auto* dArgs = context.getDArguments();
  const auto* sArgs = context.getSArguments();
  slot.args.numIArgs = iArgs == nullptr ? 0 : static_cast<int>(iArgs->size());
  slot.args.iArgs = copyArguments(iArgs);
  slot.args.numTArgs = tArgs == nullptr ? 0 : static_cast<int>(tArgs->size());
  slot.args.tArgs = copyArguments(tArgs);
  slot.args.numBArgs = bArgs == nullptr ? 0 : static_cast<int>(bArgs->size());
  slot.args.bArgs = copyBoolArguments(bArgs);
  slot.args.numDArgs = dArgs == nullptr ? 0 : static_cast<int>(dArgs->size());
  slot.args.dArgs = copyArguments(dArgs);
  slot.args.numSArgs = sArgs == nullptr ? 0 : static_cast<int>(sArgs->size());
  slot.args.sArgs = copyArguments(sArgs);
}

bool outputsAreResolvedAliases(const std::vector<NDArray*>& inputs,
                               const std::vector<NDArray*>& outputs) {
  if (outputs.empty()) return false;
  for (auto* output : outputs) {
    if (output == nullptr || output->dataBuffer() == nullptr) return false;
    bool aliases = false;
    for (auto* input : inputs) {
      if (input != nullptr && input->dataBuffer() == output->dataBuffer()) {
        aliases = true;
        break;
      }
    }
    if (!aliases) return false;
  }
  return true;
}

}  // namespace

Status TpuEagerExecutor::execute(LongType descriptorHash, Context& context,
                                 std::string* errorMessage) {
  auto* op = ops::OpRegistrator::getInstance().getOperation(descriptorHash);
  if (op == nullptr || op->getOpDescriptor() == nullptr) {
    setError(errorMessage, "TPU eager execution could not resolve descriptor hash");
    return Status::VALIDATION;
  }

  auto& inputs = context.fastpath_in();
  auto& outputs = context.fastpath_out();
  if (op->getOpDescriptor()->hasAnyTrait(
          sd::ops::OP_TRAIT_VIEW_PRODUCING | sd::ops::OP_TRAIT_IDENTITY) &&
      outputsAreResolvedAliases(inputs, outputs)) {
    return Status::OK;
  }

  NativeSlot slot;
  slot.ident.opHash = descriptorHash;
  slot.ident.op = op;
  if (op->getOpName() != nullptr) slot.ident.opName = *op->getOpName();
  slot.opTraits_ = op->getOpDescriptor()->getTraits64();
  slot.targetDeviceId = PjrtClientManager::getInstance().getCurrentDevice();
  slot.flags.structuralIArgCount =
      op->getOpDescriptor()->getNumberOfStructuralIArgs();
  slot.wiring.numInputs = static_cast<int>(inputs.size());
  slot.wiring.numOutputs = static_cast<int>(outputs.size());
  slot.wiring.inputSourceIndices =
      slot.wiring.numInputs == 0 ? nullptr : new int[slot.wiring.numInputs];
  slot.wiring.outputSlotIndices =
      slot.wiring.numOutputs == 0 ? nullptr : new int[slot.wiring.numOutputs];
  for (int i = 0; i < slot.wiring.numInputs; ++i)
    slot.wiring.inputSourceIndices[i] = -(i + 1);
  for (int i = 0; i < slot.wiring.numOutputs; ++i)
    slot.wiring.outputSlotIndices[i] = i;
  copyContextArguments(context, slot);

  std::vector<int> requestedOutputs(static_cast<size_t>(slot.wiring.numOutputs));
  for (int i = 0; i < slot.wiring.numOutputs; ++i) requestedOutputs[i] = i;
  StableHloLoweringResult lowered = StableHloGraphLowering::lower(
      &slot, 0, 0, inputs.empty() ? nullptr : inputs.data(),
      static_cast<int>(inputs.size()), outputs.empty() ? nullptr : outputs.data(),
      static_cast<int>(outputs.size()), 1,
      requestedOutputs.empty() ? nullptr : requestedOutputs.data(),
      static_cast<int>(requestedOutputs.size()));
  if (!lowered.success) {
    setError(errorMessage, "TPU eager StableHLO lowering failed: " + lowered.error);
    return Status::VALIDATION;
  }

  TpuReplayHandle handle(slot.targetDeviceId);
  handle.setProgram(lowered.program, lowered.format,
                    lowered.boundary.inputSourceIndices,
                    lowered.boundary.outputSlotIndices, 1);
  if (!handle.beginCapture(nullptr) || !handle.endCapture(nullptr) ||
      !handle.finalize()) {
    setError(errorMessage, "TPU eager PJRT compilation failed: " +
                               PjrtClientManager::getInstance().getLastError());
    return Status::KERNEL_FAILURE;
  }

  std::vector<NDArray*> boundInputs;
  for (int source : lowered.boundary.inputSourceIndices) {
    const int inputIndex = -(source + 1);
    if (inputIndex < 0 || inputIndex >= static_cast<int>(inputs.size())) {
      setError(errorMessage, "TPU eager input boundary is invalid");
      return Status::VALIDATION;
    }
    boundInputs.push_back(inputs[static_cast<size_t>(inputIndex)]);
  }
  std::vector<NDArray*> boundOutputs;
  for (int outputIndex : lowered.boundary.outputSlotIndices) {
    if (outputIndex < 0 || outputIndex >= static_cast<int>(outputs.size())) {
      setError(errorMessage, "TPU eager output boundary is invalid");
      return Status::VALIDATION;
    }
    boundOutputs.push_back(outputs[static_cast<size_t>(outputIndex)]);
  }
  handle.bindArrays(boundInputs.empty() ? nullptr : boundInputs.data(),
                    static_cast<int>(boundInputs.size()),
                    boundOutputs.empty() ? nullptr : boundOutputs.data(),
                    static_cast<int>(boundOutputs.size()));
  if (!handle.replay(nullptr)) {
    setError(errorMessage, "TPU eager PJRT execution failed: " +
                               PjrtClientManager::getInstance().getLastError());
    return Status::KERNEL_FAILURE;
  }
  return Status::OK;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
