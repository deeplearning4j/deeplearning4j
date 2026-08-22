/* ******************************************************************************
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifdef SD_TPU

#include <graph/tpu/StableHloGraphLowering.h>

#include <graph/DspDiagnostics.h>
#include <graph/kernelspec/KernelSpec.h>
#include <graph/tpu/StableHloKernelExprEmitter.h>
#include <helpers/shape.h>
#include <ops/declarable/CustomOperations.h>
#include <system/op_boilerplate.h>

#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace sd {
namespace graph {
namespace {

enum class StructuralRecipe : uint8_t { MATMUL, RESHAPE };

using StructuralCatalog = std::unordered_map<LongType, StructuralRecipe>;

template <typename Op>
void registerStructural(StructuralCatalog& catalog, StructuralRecipe recipe) {
  static Op op;
  op.initializeDescriptor();
  catalog.emplace(op.getOpDescriptor()->getHash(), recipe);
}

const StructuralCatalog& structuralCatalog() {
  static StructuralCatalog catalog;
  static std::once_flag once;
  std::call_once(once, []() {
#if NOT_EXCLUDED(OP_matmul)
    registerStructural<sd::ops::matmul>(catalog, StructuralRecipe::MATMUL);
#endif
#if NOT_EXCLUDED(OP_reshape)
    registerStructural<sd::ops::reshape>(catalog, StructuralRecipe::RESHAPE);
#endif
#if NOT_EXCLUDED(OP_reshape_no_copy)
    registerStructural<sd::ops::reshape_no_copy>(catalog, StructuralRecipe::RESHAPE);
#endif
  });
  return catalog;
}

bool hasAllTraits(const NativeSlot& slot, uint64_t traits) {
  return (slot.opTraits() & traits) == traits;
}

uint64_t categoryTrait(kernelspec::KernelCategory category) {
  using kernelspec::KernelCategory;
  switch (category) {
    case KernelCategory::UNARY_ELEMENTWISE:
      return sd::ops::OP_TRAIT_UNARY_ELEMENTWISE;
    case KernelCategory::BINARY_ELEMENTWISE:
      return sd::ops::OP_TRAIT_BINARY_ELEMENTWISE;
    case KernelCategory::TERNARY_ELEMENTWISE:
      return sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE;
    case KernelCategory::COMPARISON:
      return sd::ops::OP_TRAIT_COMPARISON;
    case KernelCategory::LOGICAL:
      return sd::ops::OP_TRAIT_LOGICAL;
    case KernelCategory::REDUCTION:
      return sd::ops::OP_TRAIT_REDUCTION;
    case KernelCategory::IDENTITY:
      return sd::ops::OP_TRAIT_IDENTITY;
  }
  return 0;
}

uint32_t dtypeMask(DataType dataType) {
  using namespace kernelspec;
  switch (dataType) {
    case FLOAT32: return KDT_F32;
    case HALF: return KDT_F16;
    case BFLOAT16: return KDT_BF16;
    case DOUBLE: return KDT_F64;
    case INT32: return KDT_I32;
    case INT64: return KDT_I64;
    case INT8: return KDT_I8;
    case BOOL: return KDT_BOOL;
    default: return 0;
  }
}

std::string elementType(DataType dataType) {
  switch (dataType) {
    case BOOL: return "i1";
    case INT8: return "i8";
    case INT16: return "i16";
    case INT32: return "i32";
    case INT64: return "i64";
    case HALF: return "f16";
    case FLOAT32: return "f32";
    case DOUBLE: return "f64";
    case BFLOAT16: return "bf16";
    default: return "";
  }
}

std::string tensorType(NDArray* array, bool booleanElements = false) {
  if (array == nullptr) return "";
  const std::string element = booleanElements ? "i1" : elementType(array->dataType());
  if (element.empty()) return "";
  std::ostringstream result;
  result << "tensor<";
  for (int i = 0; i < array->rankOf(); ++i) result << array->sizeAt(i) << "x";
  result << element << ">";
  return result.str();
}

NDArray* arrayForSource(int sourceIndex,
                        NDArray** externalInputs, int numExternalInputs,
                        NDArray** outputSlots, int totalOutputSlots) {
  if (sourceIndex < 0) {
    const int externalIndex = -(sourceIndex + 1);
    return externalInputs != nullptr && externalIndex >= 0 &&
                   externalIndex < numExternalInputs
               ? externalInputs[externalIndex]
               : nullptr;
  }
  return outputSlots != nullptr && sourceIndex >= 0 &&
                 sourceIndex < totalOutputSlots
             ? outputSlots[sourceIndex]
             : nullptr;
}

bool sameShape(NDArray* first, NDArray* second) {
  if (first == nullptr || second == nullptr || first->rankOf() != second->rankOf()) {
    return false;
  }
  for (int i = 0; i < first->rankOf(); ++i)
    if (first->sizeAt(i) != second->sizeAt(i)) return false;
  return true;
}

bool broadcastDimensions(NDArray* input, NDArray* output,
                         std::vector<int64_t>& dimensions) {
  dimensions.clear();
  if (input == nullptr || output == nullptr ||
      input->dataType() != output->dataType() ||
      input->rankOf() > output->rankOf()) {
    return false;
  }
  const int offset = output->rankOf() - input->rankOf();
  for (int i = 0; i < input->rankOf(); ++i) {
    if (input->sizeAt(i) != 1 && input->sizeAt(i) != output->sizeAt(offset + i)) {
      return false;
    }
    dimensions.push_back(offset + i);
  }
  return true;
}

std::string join(const std::vector<std::string>& values) {
  std::ostringstream result;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) result << ", ";
    result << values[i];
  }
  return result.str();
}

bool validateStructuralForm(const NativeSlot& slot, StructuralRecipe recipe,
                            std::string* reason) {
  auto reject = [&](const char* message) {
    if (reason != nullptr) *reason = message;
    return false;
  };
  if (slot.wiring.numOutputs != 1) return reject("structural recipe requires one output");
  if (recipe == StructuralRecipe::MATMUL) {
    if (!hasAllTraits(slot, sd::ops::OP_TRAIT_MATMUL) || slot.wiring.numInputs != 2)
      return reject("matmul recipe/traits/arity mismatch");
    const bool transX = slot.args.numIArgs > 0 && slot.args.iArgs[0] != 0;
    const bool transY = slot.args.numIArgs > 1 && slot.args.iArgs[1] != 0;
    const bool transZ = slot.args.numIArgs > 2 && slot.args.iArgs[2] != 0;
    const double alpha = slot.args.numTArgs > 0 ? slot.args.tArgs[0] : 1.0;
    const double beta = slot.args.numTArgs > 1 ? slot.args.tArgs[1] : 0.0;
    if (transX || transY || transZ || alpha != 1.0 || beta != 0.0)
      return reject("matmul transpose/alpha/beta form has no StableHLO recipe");
    return true;
  }
  if (!slot.isViewCapableOp() || slot.wiring.numInputs != 1)
    return reject("reshape recipe requires one view-producing input");
  return true;
}

const kernelspec::KernelSpec* expressionSpec(const NativeSlot& slot) {
  kernelspec::registerBuiltinKernelSpecs();
  return kernelspec::KernelSpecRegistry::getInstance().find(slot.ident.opHash);
}

}  // namespace

bool StableHloGraphLowering::canLowerSlot(const NativeSlot& slot,
                                          std::string* reason) {
  auto reject = [&](const std::string& message) {
    if (reason != nullptr) *reason = message;
    return false;
  };
  if (!slot.isCompilerBackendEligible())
    return reject("slot is not compiler-backend eligible");
  if (slot.hasOpTrait(sd::ops::OP_TRAIT_STATEFUL))
    return reject("stateful op requires an explicit token/state ABI");
  if (slot.hasDynamicOutputSize())
    return reject("dynamic output extent is unsupported");

  auto structural = structuralCatalog().find(slot.ident.opHash);
  if (structural != structuralCatalog().end())
    return validateStructuralForm(slot, structural->second, reason);

  const auto* spec = expressionSpec(slot);
  if (spec == nullptr) return reject("canonical descriptor has no shared KernelSpec");
  if (!hasAllTraits(slot, spec->traits) ||
      !hasAllTraits(slot, categoryTrait(spec->category))) {
    return reject("KernelSpec family does not match op-local NativeSlot traits");
  }
  if (!slot.isFullyWriting()) return reject("expression op is not fully writing");
  if (spec->category == kernelspec::KernelCategory::REDUCTION)
    return reject("StableHLO reduction target recipe is not implemented");
  if (!spec->hasBody || slot.wiring.numInputs != spec->numInputs ||
      slot.wiring.numOutputs != 1) {
    return reject("KernelSpec invocation arity/result mismatch");
  }
  return true;
}

StableHloLoweringResult StableHloGraphLowering::lower(
    NativeSlot* slots, int start, int end,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    int totalSlots, int* requestedOutputSlotIndices,
    int numRequestedOutputs) {
  StableHloLoweringResult result;
  auto fail = [&](int slot, const std::string& message) {
    result.failedSlot = slot;
    result.error = message;
    DSP_DIAG(COMPILE, "StableHloGraphLowering: slot=%d %s", slot, message.c_str());
    return result;
  };
  if (slots == nullptr || start < 0 || end < start)
    return fail(-1, "invalid inclusive slot range");

  for (int slot = start; slot <= end; ++slot) {
    std::string reason;
    if (!canLowerSlot(slots[slot], &reason)) return fail(slot, reason);
  }

  result.boundary = computeFunctionalGraphBoundary(
      slots, start, end, totalSlots, requestedOutputSlotIndices,
      numRequestedOutputs);
  if (result.boundary.outputSlotIndices.empty())
    return fail(-1, "segment has no externally visible outputs");

  std::unordered_map<int, std::string> values;
  std::vector<std::string> parameterTypes;
  for (size_t i = 0; i < result.boundary.inputSourceIndices.size(); ++i) {
    const int source = result.boundary.inputSourceIndices[i];
    NDArray* array = arrayForSource(source, externalInputs, numExternalInputs,
                                    outputSlots, totalOutputSlots);
    const std::string type = tensorType(array);
    if (array == nullptr || type.empty())
      return fail(-1, "boundary input is not a supported materialized tensor");
    values[source] = "%arg" + std::to_string(i);
    parameterTypes.push_back(type);
  }

  std::vector<std::string> outputTypes;
  for (int output : result.boundary.outputSlotIndices) {
    NDArray* array = arrayForSource(output, externalInputs, numExternalInputs,
                                    outputSlots, totalOutputSlots);
    if (array == nullptr || array->ordering() != 'c' ||
        !shape::strideDescendingCAscendingF(array->shapeInfo())) {
      return fail(-1, "boundary output must be a dense C-order tensor");
    }
    const std::string type = tensorType(array);
    if (type.empty()) return fail(-1, "boundary output dtype is unsupported");
    outputTypes.push_back(type);
  }

  std::ostringstream body;
  int nextValueId = 0;
  for (int slotIndex = start; slotIndex <= end; ++slotIndex) {
    NativeSlot& slot = slots[slotIndex];
    const int outputIndex = slot.wiring.outputSlotIndices[0];
    NDArray* output = arrayForSource(outputIndex, externalInputs, numExternalInputs,
                                     outputSlots, totalOutputSlots);
    if (output == nullptr) return fail(slotIndex, "slot output is not materialized");
    const std::string outputType = tensorType(output);

    std::vector<NDArray*> inputArrays;
    std::vector<std::string> inputValues;
    std::vector<std::string> inputTypes;
    for (int input = 0; input < slot.wiring.numInputs; ++input) {
      const int source = slot.wiring.inputSourceIndices[input];
      auto value = values.find(source);
      NDArray* array = arrayForSource(source, externalInputs, numExternalInputs,
                                      outputSlots, totalOutputSlots);
      if (value == values.end() || array == nullptr)
        return fail(slotIndex, "slot has an unresolved SSA input");
      inputArrays.push_back(array);
      inputValues.push_back(value->second);
      inputTypes.push_back(tensorType(array));
    }

    std::string emitted;
    auto structural = structuralCatalog().find(slot.ident.opHash);
    if (structural != structuralCatalog().end()) {
      emitted = "%v" + std::to_string(nextValueId++);
      if (structural->second == StructuralRecipe::RESHAPE) {
        if (inputArrays[0]->dataType() != output->dataType() ||
            inputArrays[0]->lengthOf() != output->lengthOf() ||
            inputArrays[0]->ordering() != 'c' || output->ordering() != 'c' ||
            !shape::strideDescendingCAscendingF(inputArrays[0]->shapeInfo()) ||
            !shape::strideDescendingCAscendingF(output->shapeInfo())) {
          return fail(slotIndex, "reshape requires dense C-order equal-length tensors");
        }
        body << "    " << emitted << " = stablehlo.reshape " << inputValues[0]
             << " : (" << inputTypes[0] << ") -> " << outputType << "\n";
      } else {
        if (inputArrays[0]->rankOf() != 2 || inputArrays[1]->rankOf() != 2 ||
            output->rankOf() != 2 ||
            inputArrays[0]->dataType() != inputArrays[1]->dataType() ||
            inputArrays[0]->dataType() != output->dataType() ||
            inputArrays[0]->sizeAt(1) != inputArrays[1]->sizeAt(0) ||
            output->sizeAt(0) != inputArrays[0]->sizeAt(0) ||
            output->sizeAt(1) != inputArrays[1]->sizeAt(1)) {
          return fail(slotIndex, "rank-2 matmul tensor contract mismatch");
        }
        body << "    " << emitted << " = stablehlo.dot_general "
             << inputValues[0] << ", " << inputValues[1]
             << ", contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : ("
             << inputTypes[0] << ", " << inputTypes[1] << ") -> "
             << outputType << "\n";
      }
    } else {
      const auto* spec = expressionSpec(slot);
      if (spec == nullptr || (spec->dtypes & dtypeMask(output->dataType())) == 0)
        return fail(slotIndex, "KernelSpec does not support the output dtype");

      std::vector<std::string> expressionInputs;
      for (size_t input = 0; input < inputArrays.size(); ++input) {
        if (inputArrays[input]->dataType() != output->dataType())
          return fail(slotIndex, "KernelSpec inputs must share the output dtype");
        if (sameShape(inputArrays[input], output)) {
          expressionInputs.push_back(inputValues[input]);
          continue;
        }
        std::vector<int64_t> dimensions;
        if (!broadcastDimensions(inputArrays[input], output, dimensions))
          return fail(slotIndex, "KernelSpec input is not broadcast-compatible");
        const std::string broadcast = "%v" + std::to_string(nextValueId++);
        body << "    " << broadcast << " = stablehlo.broadcast_in_dim "
             << inputValues[input] << ", dims = [";
        for (size_t d = 0; d < dimensions.size(); ++d) {
          if (d > 0) body << ", ";
          body << dimensions[d];
        }
        body << "] : (" << inputTypes[input] << ") -> " << outputType << "\n";
        expressionInputs.push_back(broadcast);
      }

      std::vector<double> scalarValues;
      scalarValues.reserve(spec->scalars.size());
      for (const auto& scalar : spec->scalars) {
        scalarValues.push_back(scalar.tArgIndex < slot.args.numTArgs
                                   ? slot.args.tArgs[scalar.tArgIndex]
                                   : scalar.defaultValue);
      }
      const auto expression = StableHloKernelExprEmitter::emit(
          spec->body, expressionInputs, scalarValues, outputType,
          tensorType(output, true), nextValueId, body);
      if (!expression.success) return fail(slotIndex, expression.error);
      if (expression.booleanValue != (output->dataType() == BOOL))
        return fail(slotIndex, "KernelExpr result dtype does not match output");
      emitted = expression.value;
    }

    values[outputIndex] = emitted;
  }

  std::vector<std::string> returnValues;
  for (int output : result.boundary.outputSlotIndices) {
    auto value = values.find(output);
    if (value == values.end()) return fail(-1, "boundary output has no SSA value");
    returnValues.push_back(value->second);
  }

  std::ostringstream module;
  module << "module @dsp_segment_" << start << "_" << end << " {\n";
  module << "  func.func public @main(";
  for (size_t i = 0; i < parameterTypes.size(); ++i) {
    if (i > 0) module << ", ";
    module << "%arg" << i << ": " << parameterTypes[i];
  }
  module << ") -> ";
  module << (outputTypes.size() == 1 ? outputTypes[0]
                                     : "(" + join(outputTypes) + ")");
  module << " {\n" << body.str() << "    return " << join(returnValues)
         << " : " << join(outputTypes) << "\n  }\n}\n";

  result.success = true;
  result.program = module.str();
  return result;
}

}  // namespace graph
}  // namespace sd

#endif  // SD_TPU
