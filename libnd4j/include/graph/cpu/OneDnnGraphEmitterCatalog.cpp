/* ******************************************************************************
 *
 * Copyright (c) 2026 Eclipse Foundation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <graph/cpu/OneDnnGraphEmitterCatalog.h>

#if HAVE_ONEDNN

#include <ops/declarable/CustomOperations.h>
#include <system/op_boilerplate.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace sd {
namespace graph {
namespace {

bool isSupportedStridedLayout(const NDArray* array) {
  if (array == nullptr || array->isEmpty()) return false;
  std::vector<std::pair<LongType, LongType>> dimensions;
  dimensions.reserve(static_cast<size_t>(array->rankOf()));
  for (int dimension = 0; dimension < array->rankOf(); ++dimension) {
    const LongType size = array->sizeAt(dimension);
    if (size <= 1) continue;
    const LongType stride = array->strideAt(dimension);
    if (stride <= 0) return false;
    dimensions.emplace_back(stride, size);
  }
  std::sort(dimensions.begin(), dimensions.end());
  LongType requiredSpan = 1;
  for (const auto& dimension : dimensions) {
    if (dimension.first < requiredSpan ||
        dimension.first > std::numeric_limits<LongType>::max() /
                              dimension.second) {
      return false;
    }
    requiredSpan = dimension.first * dimension.second;
  }
  return true;
}

bool hasOnlyStructuralArguments(const NativeSlot& slot) {
  return slot.args.numIArgs == 0 && slot.args.numTArgs == 0 &&
         slot.args.numBArgs == 0 && slot.args.numDArgs == 0 &&
         slot.args.numSArgs == 0;
}

bool sameShape(const NDArray* left, const NDArray* right) {
  return left != nullptr && right != nullptr &&
         const_cast<NDArray*>(left)->isSameShape(const_cast<NDArray*>(right));
}

bool validateF32Dense(const std::vector<NDArray*>& arrays,
                      std::string& rejectionReason) {
  for (size_t index = 0; index < arrays.size(); ++index) {
    const NDArray* array = arrays[index];
    if (array == nullptr) {
      rejectionReason = "required array " + std::to_string(index) + " is null";
      return false;
    }
    if (array->dataType() != DataType::FLOAT32) {
      rejectionReason = "only exact f32 lowering is registered";
      return false;
    }
    if (!isSupportedStridedLayout(array)) {
      rejectionReason = "layout has negative or overlapping strides";
      return false;
    }
  }
  return true;
}

bool lowerMatMul(const OneDnnLoweringContext& context,
                 OneDnnLoweredOp& lowered,
                 std::string& rejectionReason) {
  if (context.inputs.size() != 2 || context.outputs.size() != 1 ||
      !validateF32Dense(context.inputs, rejectionReason) ||
      !validateF32Dense(context.outputs, rejectionReason)) {
    if (rejectionReason.empty()) rejectionReason = "matmul requires two inputs and one output";
    return false;
  }
  if (context.inputs[0]->rankOf() == 0 || context.inputs[1]->rankOf() == 0) {
    rejectionReason = "matmul scalar inputs are not valid";
    return false;
  }
  if (context.slot.args.numIArgs > 3 || context.slot.args.numTArgs > 2 ||
      context.slot.args.numBArgs != 0 || context.slot.args.numDArgs != 0 ||
      context.slot.args.numSArgs != 0) {
    rejectionReason = "matmul argument schema is not supported";
    return false;
  }
  const bool transposeA = context.slot.args.numIArgs > 0 && context.slot.args.iArgs[0] != 0;
  const bool transposeB = context.slot.args.numIArgs > 1 && context.slot.args.iArgs[1] != 0;
  const bool transposeOutput = context.slot.args.numIArgs > 2 && context.slot.args.iArgs[2] != 0;
  const double alpha = context.slot.args.numTArgs > 0 ? context.slot.args.tArgs[0] : 1.0;
  const double beta = context.slot.args.numTArgs > 1 ? context.slot.args.tArgs[1] : 0.0;
  if (transposeOutput || alpha != 1.0 || beta != 0.0) {
    rejectionReason = "oneDNN MatMul cannot preserve transZ/alpha/beta semantics";
    return false;
  }
  lowered.operation.set_attr<bool>(dg::op::attr::transpose_a, transposeA);
  lowered.operation.set_attr<bool>(dg::op::attr::transpose_b, transposeB);
  lowered.frameworkInputOrder = {0, 1};
  return true;
}

bool lowerBinaryF32(const OneDnnLoweringContext& context,
                    OneDnnLoweredOp& lowered,
                    std::string& rejectionReason) {
  if (context.inputs.size() != 2 || context.outputs.size() != 1 ||
      !hasOnlyStructuralArguments(context.slot)) {
    rejectionReason = "binary operation requires two inputs, one output, and no arguments";
    return false;
  }
  if (!validateF32Dense(context.inputs, rejectionReason) ||
      !validateF32Dense(context.outputs, rejectionReason)) {
    return false;
  }
  lowered.frameworkInputOrder = {0, 1};
  return true;
}

bool lowerUnaryF32(const OneDnnLoweringContext& context,
                   OneDnnLoweredOp& lowered,
                   std::string& rejectionReason) {
  if (context.inputs.size() != 1 || context.outputs.size() != 1 ||
      !hasOnlyStructuralArguments(context.slot)) {
    rejectionReason = "unary operation requires one input, one output, and no arguments";
    return false;
  }
  if (!validateF32Dense(context.inputs, rejectionReason) ||
      !validateF32Dense(context.outputs, rejectionReason) ||
      !sameShape(context.inputs[0], context.outputs[0])) {
    if (rejectionReason.empty()) rejectionReason = "unary input/output shapes differ";
    return false;
  }
  lowered.frameworkInputOrder = {0};
  return true;
}

bool lowerRelu(const OneDnnLoweringContext& context,
               OneDnnLoweredOp& lowered,
               std::string& rejectionReason) {
  if (context.slot.args.numIArgs != 0 || context.slot.args.numBArgs != 0 ||
      context.slot.args.numDArgs != 0 || context.slot.args.numSArgs != 0 ||
      context.slot.args.numTArgs > 1 ||
      (context.slot.args.numTArgs == 1 && context.slot.args.tArgs[0] != 0.0)) {
    rejectionReason = "oneDNN ReLU is exact only for zero threshold";
    return false;
  }
  if (context.inputs.size() != 1 || context.outputs.size() != 1 ||
      !validateF32Dense(context.inputs, rejectionReason) ||
      !validateF32Dense(context.outputs, rejectionReason) ||
      !sameShape(context.inputs[0], context.outputs[0])) {
    if (rejectionReason.empty()) rejectionReason = "relu input/output contract mismatch";
    return false;
  }
  lowered.frameworkInputOrder = {0};
  return true;
}

bool lowerBiasAdd(const OneDnnLoweringContext& context,
                  OneDnnLoweredOp& lowered,
                  std::string& rejectionReason) {
  if (context.inputs.size() != 2 || context.outputs.size() != 1 ||
      context.slot.args.numIArgs != 0 || context.slot.args.numTArgs != 0 ||
      context.slot.args.numBArgs > 1 || context.slot.args.numDArgs != 0 ||
      context.slot.args.numSArgs != 0 ||
      !validateF32Dense(context.inputs, rejectionReason) ||
      !validateF32Dense(context.outputs, rejectionReason)) {
    if (rejectionReason.empty()) rejectionReason = "biasadd argument or tensor contract mismatch";
    return false;
  }
  const NDArray* input = context.inputs[0];
  const NDArray* bias = context.inputs[1];
  const bool nchw = context.slot.args.numBArgs == 1 && context.slot.args.bArgs[0];
  const int channelDimension = nchw ? 1 : input->rankOf() - 1;
  if (input->rankOf() < 2 || bias->rankOf() != 1 ||
      bias->sizeAt(0) != input->sizeAt(channelDimension) ||
      !sameShape(input, context.outputs[0])) {
    rejectionReason = "biasadd channel or output shape mismatch";
    return false;
  }
  lowered.operation.set_attr<std::string>(dg::op::attr::data_format,
                                          nchw ? "NCX" : "NXC");
  lowered.frameworkInputOrder = {0, 1};
  return true;
}

bool lowerSelect(const OneDnnLoweringContext& context,
                 OneDnnLoweredOp& lowered,
                 std::string& rejectionReason) {
  if (context.inputs.size() != 3 || context.outputs.size() != 1 ||
      !hasOnlyStructuralArguments(context.slot)) {
    rejectionReason = "select requires three inputs, one output, and no arguments";
    return false;
  }
  const NDArray* condition = context.inputs[0];
  const NDArray* left = context.inputs[1];
  const NDArray* right = context.inputs[2];
  const NDArray* output = context.outputs[0];
  if (condition == nullptr || condition->dataType() != DataType::BOOL ||
      !isSupportedStridedLayout(condition) ||
      !validateF32Dense({context.inputs[1], context.inputs[2],
                                                context.outputs[0]}, rejectionReason) ||
      !sameShape(left, right) || !sameShape(left, output) ||
      !(condition->isScalar() || sameShape(condition, left))) {
    if (rejectionReason.empty()) {
      rejectionReason = "select only supports scalar or elementwise boolean conditions";
    }
    return false;
  }
  lowered.operation.set_attr<std::string>(dg::op::attr::auto_broadcast, "numpy");
  lowered.frameworkInputOrder = {0, 1, 2};
  return true;
}

bool lowerGreaterEqual(const OneDnnLoweringContext& context,
                       OneDnnLoweredOp& lowered,
                       std::string& rejectionReason) {
  if (context.inputs.size() != 2 || context.outputs.size() != 1 ||
      !hasOnlyStructuralArguments(context.slot) ||
      !validateF32Dense(context.inputs, rejectionReason)) {
    if (rejectionReason.empty()) rejectionReason = "greater_equal input contract mismatch";
    return false;
  }
  NDArray* output = context.outputs[0];
  if (output == nullptr || output->dataType() != DataType::BOOL ||
      !isSupportedStridedLayout(output)) {
    rejectionReason = "greater_equal requires a valid strided boolean output";
    return false;
  }
  lowered.frameworkInputOrder = {0, 1};
  return true;
}

struct CatalogData {
  std::vector<OneDnnGraphEmitterInfo> entries;
  std::unordered_map<LongType, size_t> indices;
};

template <typename Op>
void registerEmitter(CatalogData& catalog, dg::op::kind kind, bool anchor,
                     OneDnnLowerer lower) {
  static Op op;
  op.initializeDescriptor();
  auto* descriptor = op.getOpDescriptor();
  OneDnnGraphEmitterInfo info{descriptor->getHash(), descriptor->getTraits64(),
                              kind, anchor, lower};
  const size_t index = catalog.entries.size();
  if (!catalog.indices.emplace(info.descriptorHash, index).second) {
    throw std::logic_error("duplicate canonical descriptor in oneDNN emitter catalog");
  }
  catalog.entries.push_back(info);
}

CatalogData& catalogData() {
  static std::once_flag once;
  static CatalogData* data = nullptr;
  std::call_once(once, [] {
    data = new CatalogData();
#if NOT_EXCLUDED(OP_matmul)
    registerEmitter<sd::ops::matmul>(*data, dg::op::kind::MatMul, true, lowerMatMul);
#endif
#if NOT_EXCLUDED(OP_add)
    registerEmitter<sd::ops::add>(*data, dg::op::kind::Add, false, lowerBinaryF32);
#endif
#if NOT_EXCLUDED(OP_subtract)
    registerEmitter<sd::ops::subtract>(*data, dg::op::kind::Subtract, false, lowerBinaryF32);
#endif
#if NOT_EXCLUDED(OP_multiply)
    registerEmitter<sd::ops::multiply>(*data, dg::op::kind::Multiply, false, lowerBinaryF32);
#endif
#if NOT_EXCLUDED(OP_squaredsubtract)
    registerEmitter<sd::ops::squaredsubtract>(*data, dg::op::kind::SquaredDifference,
                                              false, lowerBinaryF32);
#endif
#if NOT_EXCLUDED(OP_relu)
    registerEmitter<sd::ops::relu>(*data, dg::op::kind::ReLU, false, lowerRelu);
#endif
#if NOT_EXCLUDED(OP_square)
    registerEmitter<sd::ops::square>(*data, dg::op::kind::Square, false, lowerUnaryF32);
#endif
#if NOT_EXCLUDED(OP_biasadd)
    registerEmitter<sd::ops::biasadd>(*data, dg::op::kind::BiasAdd, false, lowerBiasAdd);
#endif
#if NOT_EXCLUDED(OP_select)
    registerEmitter<sd::ops::select>(*data, dg::op::kind::Select, false, lowerSelect);
#endif
#if NOT_EXCLUDED(OP_greater_equal)
    registerEmitter<sd::ops::greater_equal>(*data, dg::op::kind::GreaterEqual,
                                            false, lowerGreaterEqual);
#endif
  });
  return *data;
}

}  // namespace

const OneDnnGraphEmitterInfo* findOneDnnGraphEmitter(const NativeSlot& slot) {
  auto& catalog = catalogData();
  auto found = catalog.indices.find(slot.ident.opHash);
  return found == catalog.indices.end() ? nullptr : &catalog.entries[found->second];
}

const std::vector<OneDnnGraphEmitterInfo>& getOneDnnGraphEmitterCatalog() {
  return catalogData().entries;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ONEDNN
