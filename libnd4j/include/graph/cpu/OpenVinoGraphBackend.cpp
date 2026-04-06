/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>

#if HAVE_OPENVINO

#include <graph/cpu/OpenVinoGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <helpers/shape.h>
#include <array/ArrayOptions.h>

#include <algorithm>
#include <thread>
#include <unordered_set>
#include <openvino/runtime/properties.hpp>
#include <system/Environment.h>

namespace sd {
namespace graph {

// ─── Singleton ──────────────────────────────────────────────────────────────

OpenVinoGraphBackend& OpenVinoGraphBackend::getInstance() {
  static OpenVinoGraphBackend instance;
  return instance;
}

OpenVinoGraphBackend::OpenVinoGraphBackend() {
  // Configure OpenVINO threading from Environment (controlled via -Domp.num.threads).
  try {
    int numThreads = sd::Environment::getInstance().maxMasterThreads();
    if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
    core_.set_property("CPU", ov::inference_num_threads(numThreads));
    core_.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    DSP_DIAG(COMPILE, "OpenVINO: configured CPU with %d threads, THROUGHPUT mode", numThreads);
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "OpenVINO: failed to set threading properties: %s", e.what());
  }
}

OpenVinoGraphBackend::~OpenVinoGraphBackend() = default;

// ─── Availability ───────────────────────────────────────────────────────────

bool OpenVinoGraphBackend::isAvailable() const {
  // If we compiled with HAVE_OPENVINO, the runtime is available
  // Check that "CPU" device is present
  try {
    auto devices = const_cast<ov::Core&>(core_).get_available_devices();
    for (const auto& dev : devices) {
      if (dev == "CPU") return true;
    }
  } catch (...) {
    return false;
  }
  return false;
}

// ─── Data type mapping ──────────────────────────────────────────────────────

ov::element::Type OpenVinoGraphBackend::mapDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32: return ov::element::f32;
    case DataType::HALF: return ov::element::f16;
    case DataType::BFLOAT16: return ov::element::bf16;
    case DataType::DOUBLE: return ov::element::f64;
    case DataType::INT8: return ov::element::i8;
    case DataType::INT16: return ov::element::i16;
    case DataType::INT32: return ov::element::i32;
    case DataType::INT64: return ov::element::i64;
    case DataType::UINT8: return ov::element::u8;
    case DataType::UINT16: return ov::element::u16;
    case DataType::UINT32: return ov::element::u32;
    case DataType::UINT64: return ov::element::u64;
    case DataType::BOOL: return ov::element::boolean;
    default: return ov::element::f32;
  }
}

// ─── Op mappability check ───────────────────────────────────────────────────

bool OpenVinoGraphBackend::isOpenVinoMappable(const std::string& opName) {
  // Use a static set for O(1) lookup
  static const std::unordered_set<std::string> mappable = {
    // ── Binary elementwise ──
    "add", "Add",
    "subtract", "Sub",
    "multiply", "Mul",
    "divide", "Div", "RealDiv",
    "pow", "Pow",
    "minimum", "Min",
    "maximum", "Max",
    "mod", "Mod",
    "floormod", "FloorMod",
    "atan2", "Atan2",
    "floordiv", "FloorDiv",
    "reversedivide", "ReverseDivide",
    "reversesubtract", "ReverseSubtract",
    "squaredsubtract", "SquaredSubtract", "SquaredDifference",
    "multiply_no_nan", "MultiplyNoNan",
    "min_pairwise", "MinPairwise",
    "max_pairwise", "MaxPairwise",
    "swish_mul", "SwishMul",
    "biasadd", "BiasAdd",
    "prelu", "Prelu",
    "bitwise_and", "BitwiseAnd",
    "bitwise_or", "BitwiseOr",
    "bitwise_xor", "BitwiseXor",
    "shift_bits", "ShiftBits",
    "rshift_bits", "RShiftBits",

    // ── Unary elementwise ──
    "relu", "Relu",
    "sigmoid", "Sigmoid",
    "tanh", "Tanh",
    "exp", "Exp",
    "log", "Log",
    "sqrt", "Sqrt",
    "abs", "Abs",
    "neg", "Neg", "negative",
    "gelu", "Gelu",
    "silu", "Silu", "swish", "Swish",
    "elu", "Elu",
    "ceil", "Ceil",
    "floor", "Floor",
    "round", "Round",
    "square", "Square",
    "sin", "Sin",
    "cos", "Cos",
    "erf", "Erf",
    "softplus", "SoftPlus", "Softplus",
    "mish", "Mish",
    "hardswish", "HardSwish",
    "hardsigmoid", "HardSigmoid", "hard_sigmoid", "HardSigmoid2",
    "clamp", "ClipByValue", "clip_by_value", "clipbyvalue",
    "reciprocal", "Reciprocal",
    "rsqrt", "Rsqrt",
    "sign", "Sign",
    "erfc", "Erfc",
    "log1p", "Log1p",
    "leakyrelu", "LeakyRelu",
    "selu", "Selu",
    "softsign", "Softsign",
    "hardtanh", "HardTanh",
    "relu6", "Relu6",
    "celu", "Celu",
    "thresholdedrelu", "ThresholdedRelu",
    "toggle_bits", "ToggleBits",
    "fused_gelu", "FusedGelu",
    // Scalar ops (second operand from tArgs[0])
    "add_scalar", "subtract_scalar", "sub_scalar",
    "multiply_scalar", "mul_scalar",
    "divide_scalar", "div_scalar",

    // ── MatMul ──
    "matmul", "MatMul", "mmul",
    "batch_matmul", "BatchMatMul",
    "tensormmul", "TensorMmul",
    "batched_gemm", "BatchedGemm",
    "xw_plus_b", "XwPlusB",
    "fused_gemm_swiglu", "FusedGemmSwiglu",

    // ── Normalization ──
    "softmax", "Softmax",
    "log_softmax", "LogSoftmax",
    "layer_norm", "LayerNorm",
    "layer_normalization",
    "rms_norm", "RmsNorm",
    "rms_norm_linear", "RmsNormLinear",
    "batchnorm", "BatchNorm", "batchnorm_inference", "batch_norm",
    "fused_layer_norm", "FusedLayerNorm",

    // ── Reduction ──
    "reduce_sum", "ReduceSum",
    "reduce_max", "ReduceMax",
    "reduce_mean", "ReduceMean",
    "reduce_min", "ReduceMin",
    "reduce_prod", "ReduceProd",
    "reduce_norm1", "ReduceNorm1",
    "reduce_norm2", "ReduceNorm2",
    "reduce_logsumexp", "ReduceLogSumExp",
    "reduce_variance", "ReduceVariance",
    "reduce_stdev", "ReduceStdev",
    "sum", "Sum",
    "mean", "Mean",
    "max", "min",
    "prod", "Prod",
    "norm1", "norm2", "normmax",

    // ── Shape manipulation ──
    "reshape", "Reshape",
    "reshape_no_copy", "ReshapeNoCopy",
    "permute", "Permute", "Transpose", "transpose",
    "squeeze", "Squeeze",
    "expand_dims", "ExpandDims", "Unsqueeze",
    "flatten", "Flatten",
    "flatten_2d", "Flatten2d",
    "triu", "Triu",
    "tril", "Tril",
    "broadcast_to", "BroadcastTo",
    "reshapeas", "ReshapeAs",

    // ── Data movement ──
    "gather", "Gather",
    "gather_nd", "GatherNd",
    "scatter_nd", "ScatterNd", "ScatterNdUpdate", "scatter_nd_update",
    "concat", "Concat",
    "split", "Split",
    "split_v", "SplitV",
    "unstack", "Unstack",
    "slice", "Slice",
    "strided_slice", "StridedSlice",
    "stack", "Stack",
    "tile", "Tile",
    "repeat", "Repeat",
    "pad", "Pad",
    "mirror_pad", "MirrorPad",
    "reverse", "Reverse",
    "reverse_v2", "ReverseV2",
    "embedding_lookup", "EmbeddingLookup",

    // ── Comparison ──
    "greater", "Greater",
    "less", "Less",
    "equals", "Equal", "Equals",
    "not_equals", "NotEqual", "NotEquals",
    "greater_equal", "GreaterEqual",
    "less_equal", "LessEqual",

    // ── Logical ──
    "boolean_and", "BooleanAnd", "logical_and", "LogicalAnd",
    "boolean_or", "BooleanOr", "logical_or", "LogicalOr",
    "boolean_not", "BooleanNot", "bool_not", "logical_not", "LogicalNot",
    "boolean_xor", "BooleanXor", "LogicalXor",

    // ── Ternary ──
    "where", "Where", "select", "Select",
    "where_np", "WhereNp",

    // ── Cast ──
    "cast", "Cast",

    // ── Identity (pass-through) ──
    "identity", "Identity",
    "assign", "Assign",

    // ── Constant generation ──
    "zeros_like", "ZerosLike", "zeroslike",
    "ones_like", "OnesLike", "oneslike",
    "ones_as", "OnesAs",
    "zeros_as",
    "range", "Range",
    "fill", "Fill",
    "shape_of", "ShapeOf",
    "create", "Create",
    "set_scalar", "SetScalar",
    "min_max_datatype", "MinMaxDatatype",
    "eye", "Eye",
    "linspace", "Linspace",
    "lin_space", "LinSpace",
    "size_at", "SizeAt",
    "size", "Size",
    "rank", "Rank",
    "onehot", "OneHot",
    "sequence_mask", "SequenceMask",

    // ── Convolution ──
    "conv2d", "Conv2d", "Conv2D", "conv2D",
    "conv3d", "Conv3d",
    "depthwise_conv2d", "DepthwiseConv2d",
    "maxpool2d", "MaxPool2d",
    "avgpool2d", "AvgPool2d",
    "deconv2d", "deconv3d",

    // ── ROPE (complex composition -- marked mappable for coverage counting) ──
    "rope", "Rope",
    "fused_rope", "FusedRope",

    // ── Attention (complex composition -- marked mappable for coverage counting) ──
    "dot_product_attention", "DotProductAttention",
    "dot_product_attention_v2", "DotProductAttentionV2",
    "multi_head_attention", "MultiHeadAttention",
    "onnx_multi_head_attention", "OnnxMultiHeadAttention",
  };

  return mappable.count(opName) > 0;
}

// ─── canFuseSegment ─────────────────────────────────────────────────────────

// Ops that benefit from OpenVINO execution even as a single op
static bool isOvSingleOpWorthCompiling(const std::string& opName) {
  static const std::unordered_set<std::string> worthwhile = {
    "matmul", "Matmul", "MatMul",
    "softmax", "Softmax",
    "layer_norm", "LayerNorm",
    "rms_norm", "RmsNorm",
    "rms_norm_linear", "RmsNormLinear",
    "fused_gemm_swiglu", "FusedGemmSwiglu",
    "batch_norm", "BatchNorm", "batchnorm",
    "conv2d", "Conv2d",
    "avgpool2d", "Avgpool2d",
    "maxpool2d", "Maxpool2d",
    "where", "Where", "select", "Select",
  };
  return worthwhile.count(opName) > 0;
}

bool OpenVinoGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  int mappableOps = 0;
  int totalOps = 0;

  for (int i = start; i <= end; i++) {
    totalOps++;
    if (isOpenVinoMappable(slots[i].opName)) {
      mappableOps++;
    }
  }

  // Accept any segment where all ops are mappable — no minimum op count.
  // Every mappable op benefits from OpenVINO compilation.
  return mappableOps == totalOps && mappableOps >= 1;
}

// ─── buildModel ─────────────────────────────────────────────────────────────

OpenVinoGraphBackend::CompiledSegment OpenVinoGraphBackend::buildModel(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  CompiledSegment result;
  result.valid = false;

  // Track tensor nodes by their source index (slot output index or external input)
  // Key: >=0 = outputSlot index, <0 = -(externalInputIndex+1)
  std::unordered_map<int, ov::Output<ov::Node>> tensorMap;

  // Collect which slot outputs are produced within this segment
  std::unordered_set<int> segmentOutputs;
  for (int s = startSlot; s <= endSlot; s++) {
    for (int o = 0; o < slots[s].numOutputs; o++) {
      segmentOutputs.insert(slots[s].outputSlotIndices[o]);
    }
  }

  // Scan for concat ops in the segment and collect which (srcIdx, dim) pairs
  // need dynamic dims. Concat inputs on the concat axis may have different sizes
  // (e.g. KV cache [1,3,0,64] vs new token [1,3,679,64]) — marking that axis
  // dynamic prevents OpenVINO's concat shape inference from failing.
  std::unordered_map<int, std::unordered_set<int>> dynamicDims;  // srcIdx -> set of dim indices
  for (int s = startSlot; s <= endSlot; s++) {
    const std::string& op = slots[s].opName;
    if (op == "concat" || op == "Concat" || op == "concat_v2") {
      int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
      for (int inp = 0; inp < slots[s].numInputs; inp++) {
        int srcIdx = slots[s].inputSourceIndices[inp];
        if (axis < 0) {
          // Negative axis — resolve later when we know rank
          dynamicDims[srcIdx].insert(-1);  // sentinel: mark all dims dynamic
        } else {
          dynamicDims[srcIdx].insert(axis);
        }
      }
    }
  }

  // Create OV parameters for external inputs and pre-segment slot outputs
  ov::ParameterVector params;
  std::vector<int> inputSourceMap;  // maps param index -> source index

  for (int s = startSlot; s <= endSlot; s++) {
    for (int inp = 0; inp < slots[s].numInputs; inp++) {
      int srcIdx = slots[s].inputSourceIndices[inp];
      if (tensorMap.count(srcIdx)) continue;

      bool isExternal = (srcIdx < 0);
      bool isPreSegment = (!isExternal && segmentOutputs.find(srcIdx) == segmentOutputs.end());

      if (isExternal || isPreSegment) {
        NDArray* arr = nullptr;
        if (isExternal) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExternalInputs && externalInputs) arr = externalInputs[extIdx];
        } else {
          if (srcIdx < totalOutputSlots && outputSlots) arr = outputSlots[srcIdx];
        }
        // Use actual shapes but mark concat-axis dims as dynamic to prevent
        // shape merge failures in OpenVINO's concat inference.
        auto& dynSet = dynamicDims[srcIdx];
        bool allDynamic = dynSet.count(-1) > 0;
        ov::PartialShape pshape;
        ov::element::Type dtype = ov::element::f32;
        if (arr) {
          for (int d = 0; d < arr->rankOf(); d++) {
            auto dimVal = arr->sizeAt(d);
            if (dimVal <= 0 || allDynamic || dynSet.count(d) > 0) {
              pshape.push_back(ov::Dimension::dynamic());
            } else {
              pshape.push_back(static_cast<int64_t>(dimVal));
            }
          }
          dtype = mapDataType(arr->dataType());
        } else if (!isExternal && srcIdx >= 0 && srcIdx < totalOutputSlots) {
          // Pre-segment slot: get rank/dtype/dims from the slot's cached output shape
          auto& srcSlot = slots[srcIdx];
          if (!srcSlot.cachedOutputShapes.empty() && srcSlot.cachedOutputShapes[0] != nullptr) {
            const LongType* si = srcSlot.cachedOutputShapes[0];
            int rank = shape::rank(si);
            for (int d = 0; d < rank; d++) {
              auto dimVal = shape::shapeOf(si)[d];
              pshape.push_back(dimVal > 0 ? static_cast<int64_t>(dimVal) : ov::Dimension::dynamic());
            }
            dtype = mapDataType(ArrayOptions::dataType(si));
          } else {
            // Last resort: scalar placeholder
            pshape.push_back(ov::Dimension::dynamic());
          }
        } else {
          pshape.push_back(ov::Dimension::dynamic());
        }
        auto param = std::make_shared<ov::op::v0::Parameter>(
            mapDataType(arr->dataType()), pshape);
        params.push_back(param);
        tensorMap[srcIdx] = param->output(0);
        inputSourceMap.push_back(srcIdx);
      }
    }
  }

  // Build OV ops for each slot
  ov::ResultVector results;
  std::vector<int> outputSourceMap;  // maps result index -> outputSlot index

  for (int s = startSlot; s <= endSlot; s++) {
    const std::string& opName = slots[s].opName;

    CompilationAuditEntry audit;
    audit.slotIndex = s;
    audit.opName = opName;

    if (!isOpenVinoMappable(opName)) {
      audit.wasCompiled = false;
      audit.reason = "unmappable op: " + opName;
      result.compilationAudit.push_back(audit);
      DSP_DIAG(COMPILE, "OpenVINO: skipping unmappable op '%s' at slot %d", opName.c_str(), s);
      continue;
    }

    // Gather input nodes
    std::vector<ov::Output<ov::Node>> inputs;
    for (int inp = 0; inp < slots[s].numInputs; inp++) {
      int srcIdx = slots[s].inputSourceIndices[inp];
      auto it = tensorMap.find(srcIdx);
      if (it == tensorMap.end()) {
        // Missing from tensorMap — create an on-the-fly Parameter.
        // This handles within-segment cascades (prior op failed to compile)
        // and any missed pre-segment inputs.
        NDArray* fallbackArr = nullptr;
        if (srcIdx < 0) {
          int extIdx = -(srcIdx + 1);
          if (extIdx < numExternalInputs) fallbackArr = externalInputs[extIdx];
        } else if (srcIdx < totalOutputSlots) {
          fallbackArr = outputSlots[srcIdx];
        }
        ov::PartialShape pshape;
        ov::element::Type dtype = ov::element::f32;
        if (fallbackArr) {
          for (int d = 0; d < fallbackArr->rankOf(); d++) {
            auto dv = fallbackArr->sizeAt(d);
            pshape.push_back(dv > 0 ? static_cast<int64_t>(dv) : ov::Dimension::dynamic());
          }
          dtype = mapDataType(fallbackArr->dataType());
        } else if (srcIdx >= 0 && srcIdx < totalOutputSlots) {
          auto& srcSlot = slots[srcIdx];
          if (!srcSlot.cachedOutputShapes.empty() && srcSlot.cachedOutputShapes[0]) {
            const LongType* si = srcSlot.cachedOutputShapes[0];
            for (int d = 0; d < shape::rank(si); d++) {
              auto dv = shape::shapeOf(si)[d];
              pshape.push_back(dv > 0 ? static_cast<int64_t>(dv) : ov::Dimension::dynamic());
            }
            dtype = mapDataType(ArrayOptions::dataType(si));
          } else {
            pshape.push_back(ov::Dimension::dynamic());
          }
        } else {
          pshape.push_back(ov::Dimension::dynamic());
        }
        auto param = std::make_shared<ov::op::v0::Parameter>(dtype, pshape);
        params.push_back(param);
        tensorMap[srcIdx] = param->output(0);
        inputSourceMap.push_back(srcIdx);
        it = tensorMap.find(srcIdx);
      }
      if (it == tensorMap.end()) {
        audit.wasCompiled = false;
        audit.reason = "missing input source " + std::to_string(srcIdx);
        result.compilationAudit.push_back(audit);
        DSP_DIAG(COMPILE, "OpenVINO: slot %d (%s) missing input source %d",
                 s, opName.c_str(), srcIdx);
        goto next_slot;
      }
      inputs.push_back(it->second);
    }

    {
      std::shared_ptr<ov::Node> node;

      // OpenVINO elementwise ops require matching input types.
      // Auto-cast mismatched binary inputs to the higher-precision type.
      auto harmonizeBinaryTypes = [](ov::Output<ov::Node>& a, ov::Output<ov::Node>& b) {
        auto ta = a.get_element_type();
        auto tb = b.get_element_type();
        if (ta != tb) {
          // Pick the "wider" floating type, or float32 as fallback
          ov::element::Type target = ov::element::f32;
          if (ta.is_real() && tb.is_real()) {
            target = (ta.bitwidth() >= tb.bitwidth()) ? ta : tb;
          } else if (ta.is_real()) {
            target = ta;
          } else if (tb.is_real()) {
            target = tb;
          }
          if (ta != target) a = std::make_shared<ov::op::v0::Convert>(a, target)->output(0);
          if (tb != target) b = std::make_shared<ov::op::v0::Convert>(b, target)->output(0);
        }
      };

      try {
      // ── Binary elementwise ──
      if (opName == "add" || opName == "Add") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::Add>(inputs[0], inputs[1]); }
      } else if (opName == "subtract" || opName == "Sub") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::Subtract>(inputs[0], inputs[1]); }
      } else if (opName == "multiply" || opName == "Mul") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[1]); }
      } else if (opName == "divide" || opName == "Div" || opName == "RealDiv") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::Divide>(inputs[0], inputs[1]); }
      } else if (opName == "pow" || opName == "Pow") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::Power>(inputs[0], inputs[1]); }
      } else if (opName == "minimum" || opName == "Min") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::Minimum>(inputs[0], inputs[1]); }
      } else if (opName == "maximum" || opName == "Max") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::Maximum>(inputs[0], inputs[1]); }
      } else if (opName == "floormod" || opName == "FloorMod") {
        if (inputs.size() >= 2) { harmonizeBinaryTypes(inputs[0], inputs[1]); node = std::make_shared<ov::op::v1::FloorMod>(inputs[0], inputs[1]); }
      } else if (opName == "squaredsubtract" || opName == "SquaredSubtract" || opName == "SquaredDifference") {
        if (inputs.size() >= 2) {
          harmonizeBinaryTypes(inputs[0], inputs[1]);
          auto sub = std::make_shared<ov::op::v1::Subtract>(inputs[0], inputs[1]);
          node = std::make_shared<ov::op::v1::Multiply>(sub->output(0), sub->output(0));
        }
      } else if (opName == "mod" || opName == "Mod") {
        if (inputs.size() >= 2) {
          harmonizeBinaryTypes(inputs[0], inputs[1]);
          auto div = std::make_shared<ov::op::v1::Divide>(inputs[0], inputs[1]);
          auto fl = std::make_shared<ov::op::v0::Floor>(div->output(0));
          auto mul = std::make_shared<ov::op::v1::Multiply>(fl->output(0), inputs[1]);
          node = std::make_shared<ov::op::v1::Subtract>(inputs[0], mul->output(0));
        }
      } else if (opName == "atan2" || opName == "Atan2") {
        if (inputs.size() >= 2) {
          harmonizeBinaryTypes(inputs[0], inputs[1]);
          auto div = std::make_shared<ov::op::v1::Divide>(inputs[0], inputs[1]);
          node = std::make_shared<ov::op::v0::Atan>(div->output(0));
        }
      } else if (opName == "floordiv" || opName == "FloorDiv") {
        // Compose: floor(a / b)
        if (inputs.size() >= 2) {
          auto div = std::make_shared<ov::op::v1::Divide>(inputs[0], inputs[1]);
          node = std::make_shared<ov::op::v0::Floor>(div->output(0));
        }
      } else if (opName == "reversedivide" || opName == "ReverseDivide") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Divide>(inputs[1], inputs[0]);
      } else if (opName == "reversesubtract" || opName == "ReverseSubtract") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Subtract>(inputs[1], inputs[0]);
      } else if (opName == "multiply_no_nan" || opName == "MultiplyNoNan") {
        // Compose: Select(Equal(b, 0), 0, a * b)
        if (inputs.size() >= 2) {
          auto zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
          auto eq = std::make_shared<ov::op::v1::Equal>(inputs[1], zero);
          auto mul = std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[1]);
          node = std::make_shared<ov::op::v1::Select>(eq->output(0), zero, mul->output(0));
        }
      } else if (opName == "min_pairwise" || opName == "MinPairwise") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Minimum>(inputs[0], inputs[1]);
      } else if (opName == "max_pairwise" || opName == "MaxPairwise") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Maximum>(inputs[0], inputs[1]);
      } else if (opName == "biasadd" || opName == "BiasAdd") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Add>(inputs[0], inputs[1]);
      } else if (opName == "prelu" || opName == "Prelu") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v0::PRelu>(inputs[0], inputs[1]);
      } else if (opName == "bitwise_and" || opName == "BitwiseAnd") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v13::BitwiseAnd>(inputs[0], inputs[1]);
      } else if (opName == "bitwise_or" || opName == "BitwiseOr") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v13::BitwiseOr>(inputs[0], inputs[1]);
      } else if (opName == "bitwise_xor" || opName == "BitwiseXor") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v13::BitwiseXor>(inputs[0], inputs[1]);
      } else if (opName == "swish_mul" || opName == "SwishMul") {
        // Swish applied to first input, then multiply with second
        if (inputs.size() >= 2) {
          auto sw = std::make_shared<ov::op::v4::Swish>(inputs[0]);
          node = std::make_shared<ov::op::v1::Multiply>(sw->output(0), inputs[1]);
        }
      }

      // ── Unary elementwise ──
      else if (opName == "relu" || opName == "Relu") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Relu>(inputs[0]);
      } else if (opName == "sigmoid" || opName == "Sigmoid") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Sigmoid>(inputs[0]);
      } else if (opName == "tanh" || opName == "Tanh") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Tanh>(inputs[0]);
      } else if (opName == "exp" || opName == "Exp") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Exp>(inputs[0]);
      } else if (opName == "log" || opName == "Log") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Log>(inputs[0]);
      } else if (opName == "sqrt" || opName == "Sqrt") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Sqrt>(inputs[0]);
      } else if (opName == "abs" || opName == "Abs") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Abs>(inputs[0]);
      } else if (opName == "neg" || opName == "Neg" || opName == "negative") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Negative>(inputs[0]);
      } else if (opName == "gelu" || opName == "Gelu") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v7::Gelu>(inputs[0]);
      } else if (opName == "silu" || opName == "Silu" || opName == "swish" || opName == "Swish") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v4::Swish>(inputs[0]);
      } else if (opName == "elu" || opName == "Elu") {
        if (inputs.size() >= 1) {
          double alpha = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1.0;
          node = std::make_shared<ov::op::v0::Elu>(inputs[0], alpha);
        }
      } else if (opName == "ceil" || opName == "Ceil") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Ceiling>(inputs[0]);
      } else if (opName == "floor" || opName == "Floor") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Floor>(inputs[0]);
      } else if (opName == "round" || opName == "Round") {
        if (inputs.size() >= 1) {
          node = std::make_shared<ov::op::v5::Round>(
              inputs[0], ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
        }
      } else if (opName == "square" || opName == "Square") {
        // Compose: x * x
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[0]);
      } else if (opName == "sin" || opName == "Sin") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Sin>(inputs[0]);
      } else if (opName == "cos" || opName == "Cos") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Cos>(inputs[0]);
      } else if (opName == "erf" || opName == "Erf") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Erf>(inputs[0]);
      } else if (opName == "softplus" || opName == "SoftPlus") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v4::SoftPlus>(inputs[0]);
      } else if (opName == "mish" || opName == "Mish") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v4::Mish>(inputs[0]);
      } else if (opName == "hardswish" || opName == "HardSwish") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v4::HSwish>(inputs[0]);
      } else if (opName == "hardsigmoid" || opName == "HardSigmoid") {
        if (inputs.size() >= 1) {
          auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {0.2f});
          auto beta_const = ov::op::v0::Constant::create(ov::element::f32, {}, {0.5f});
          node = std::make_shared<ov::op::v0::HardSigmoid>(inputs[0], alpha_const, beta_const);
        }
      } else if (opName == "clamp" || opName == "ClipByValue" || opName == "clip_by_value" || opName == "clipbyvalue") {
        if (inputs.size() >= 1) {
          double minVal = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : -std::numeric_limits<float>::max();
          double maxVal = (slots[s].numTArgs > 1) ? slots[s].tArgs[1] : std::numeric_limits<float>::max();
          node = std::make_shared<ov::op::v0::Clamp>(inputs[0], minVal, maxVal);
        }
      } else if (opName == "reciprocal" || opName == "Reciprocal") {
        // Compose: 1 / x
        if (inputs.size() >= 1) {
          auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          node = std::make_shared<ov::op::v1::Divide>(one, inputs[0]);
        }
      } else if (opName == "rsqrt" || opName == "Rsqrt") {
        // Compose: 1 / sqrt(x)
        if (inputs.size() >= 1) {
          auto sq = std::make_shared<ov::op::v0::Sqrt>(inputs[0]);
          auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          node = std::make_shared<ov::op::v1::Divide>(one, sq->output(0));
        }
      } else if (opName == "sign" || opName == "Sign") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Sign>(inputs[0]);
      } else if (opName == "log1p" || opName == "Log1p") {
        // Compose: log(x + 1)
        if (inputs.size() >= 1) {
          auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          auto add = std::make_shared<ov::op::v1::Add>(inputs[0], one);
          node = std::make_shared<ov::op::v0::Log>(add->output(0));
        }
      } else if (opName == "erfc" || opName == "Erfc") {
        // Compose: 1 - erf(x)
        if (inputs.size() >= 1) {
          auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          auto erf_node = std::make_shared<ov::op::v0::Erf>(inputs[0]);
          node = std::make_shared<ov::op::v1::Subtract>(one, erf_node->output(0));
        }
      } else if (opName == "leakyrelu" || opName == "LeakyRelu") {
        // Use PRelu with constant slope
        if (inputs.size() >= 1) {
          double alpha = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 0.01;
          auto slope = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(alpha)});
          node = std::make_shared<ov::op::v0::PRelu>(inputs[0], slope);
        }
      } else if (opName == "selu" || opName == "Selu") {
        if (inputs.size() >= 1) {
          double alpha = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1.6732632423543772;
          double lambda = (slots[s].numTArgs > 1) ? slots[s].tArgs[1] : 1.0507009873554805;
          auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(alpha)});
          auto lambda_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(lambda)});
          node = std::make_shared<ov::op::v0::Selu>(inputs[0], alpha_const, lambda_const);
        }
      } else if (opName == "softsign" || opName == "Softsign") {
        // Compose: x / (1 + |x|)
        if (inputs.size() >= 1) {
          auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          auto abs_x = std::make_shared<ov::op::v0::Abs>(inputs[0]);
          auto denom = std::make_shared<ov::op::v1::Add>(one, abs_x->output(0));
          node = std::make_shared<ov::op::v1::Divide>(inputs[0], denom->output(0));
        }
      } else if (opName == "hardtanh" || opName == "HardTanh") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Clamp>(inputs[0], -1.0, 1.0);
      } else if (opName == "relu6" || opName == "Relu6") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v0::Clamp>(inputs[0], 0.0, 6.0);
      } else if (opName == "celu" || opName == "Celu") {
        // Compose: max(0,x) + min(0, alpha*(exp(x/alpha)-1))
        if (inputs.size() >= 1) {
          double alpha = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1.0;
          auto zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
          auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(alpha)});
          auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          auto pos = std::make_shared<ov::op::v1::Maximum>(inputs[0], zero);
          auto x_over_a = std::make_shared<ov::op::v1::Divide>(inputs[0], alpha_const);
          auto exp_xa = std::make_shared<ov::op::v0::Exp>(x_over_a->output(0));
          auto exp_m1 = std::make_shared<ov::op::v1::Subtract>(exp_xa->output(0), one);
          auto scaled = std::make_shared<ov::op::v1::Multiply>(alpha_const, exp_m1->output(0));
          auto neg = std::make_shared<ov::op::v1::Minimum>(scaled->output(0), zero);
          node = std::make_shared<ov::op::v1::Add>(pos->output(0), neg->output(0));
        }
      } else if (opName == "thresholdedrelu" || opName == "ThresholdedRelu") {
        // Compose: Select(x > theta, x, 0)
        if (inputs.size() >= 1) {
          double theta = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1.0;
          auto theta_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(theta)});
          auto zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
          auto cmp = std::make_shared<ov::op::v1::Greater>(inputs[0], theta_const);
          node = std::make_shared<ov::op::v1::Select>(cmp->output(0), inputs[0], zero);
        }
      } else if (opName == "fused_gelu" || opName == "FusedGelu") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v7::Gelu>(inputs[0]);
      } else if (opName == "hard_sigmoid" || opName == "HardSigmoid2") {
        if (inputs.size() >= 1) {
          auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, {}, {0.2f});
          auto beta_const = ov::op::v0::Constant::create(ov::element::f32, {}, {0.5f});
          node = std::make_shared<ov::op::v0::HardSigmoid>(inputs[0], alpha_const, beta_const);
        }
      } else if (opName == "Softplus") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v4::SoftPlus>(inputs[0]);
      }
      // ── Scalar ops (second operand from tArgs[0]) ──
      else if (opName == "add_scalar") {
        if (inputs.size() >= 1 && slots[s].numTArgs > 0) {
          auto scalar = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[0])});
          node = std::make_shared<ov::op::v1::Add>(inputs[0], scalar);
        }
      } else if (opName == "subtract_scalar" || opName == "sub_scalar") {
        if (inputs.size() >= 1 && slots[s].numTArgs > 0) {
          auto scalar = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[0])});
          node = std::make_shared<ov::op::v1::Subtract>(inputs[0], scalar);
        }
      } else if (opName == "multiply_scalar" || opName == "mul_scalar") {
        if (inputs.size() >= 1 && slots[s].numTArgs > 0) {
          auto scalar = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[0])});
          node = std::make_shared<ov::op::v1::Multiply>(inputs[0], scalar);
        }
      } else if (opName == "divide_scalar" || opName == "div_scalar") {
        if (inputs.size() >= 1 && slots[s].numTArgs > 0) {
          auto scalar = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[0])});
          node = std::make_shared<ov::op::v1::Divide>(inputs[0], scalar);
        }
      }

      // ── MatMul ──
      else if (opName == "matmul" || opName == "MatMul" || opName == "mmul" ||
               opName == "batch_matmul" || opName == "BatchMatMul" ||
               opName == "tensormmul" || opName == "TensorMmul" ||
               opName == "batched_gemm" || opName == "BatchedGemm") {
        if (inputs.size() >= 2) {
          bool transpA = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          bool transpB = (slots[s].numBArgs > 1) ? slots[s].bArgs[1] : false;
          node = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[1], transpA, transpB);
        }
      } else if (opName == "xw_plus_b" || opName == "XwPlusB") {
        // Compose: MatMul(x, w) + b
        if (inputs.size() >= 3) {
          auto mm = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[1], false, false);
          node = std::make_shared<ov::op::v1::Add>(mm->output(0), inputs[2]);
        }
      } else if (opName == "fused_gemm_swiglu" || opName == "FusedGemmSwiglu") {
        // Compose: silu(x @ W_gate) * (x @ W_up)
        // Input 0: x [M, K], Input 1: W_gate [K, N], Input 2: W_up [K, N]
        if (inputs.size() >= 3) {
          // gate = x @ W_gate
          auto gate = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[1], false, false);
          // up = x @ W_up
          auto up = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[2], false, false);
          // silu(gate) = gate * sigmoid(gate)
          auto silu = std::make_shared<ov::op::v4::Swish>(gate->output(0));
          // output = silu(gate) * up
          node = std::make_shared<ov::op::v1::Multiply>(silu->output(0), up->output(0));
        }
      }

      // ── Softmax ──
      else if (opName == "softmax" || opName == "Softmax") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : -1;
          node = std::make_shared<ov::op::v8::Softmax>(inputs[0], axis);
        }
      } else if (opName == "log_softmax" || opName == "LogSoftmax") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : -1;
          auto sm = std::make_shared<ov::op::v8::Softmax>(inputs[0], axis);
          node = std::make_shared<ov::op::v0::Log>(sm->output(0));
        }
      }
      // ── Normalization (LayerNorm, RMSNorm, BatchNorm) ──
      else if (opName == "layer_norm" || opName == "LayerNorm" || opName == "layer_normalization" ||
               opName == "fused_layer_norm" || opName == "FusedLayerNorm") {
        // Compose: MVN(x, axes, normalize_variance=true, eps) * scale + bias
        if (inputs.size() >= 1) {
          double eps = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1e-5;
          // Normalize over last axis by default
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> axes;
          if (slots[s].numIArgs > 0) {
            for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          } else {
            axes.push_back(rank - 1);
          }
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          auto mvn = std::make_shared<ov::op::v6::MVN>(
              inputs[0], axes_const, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);
          if (inputs.size() >= 3) {
            // scale (gamma) and bias (beta)
            auto scaled = std::make_shared<ov::op::v1::Multiply>(mvn->output(0), inputs[1]);
            node = std::make_shared<ov::op::v1::Add>(scaled->output(0), inputs[2]);
          } else if (inputs.size() >= 2) {
            // scale only
            node = std::make_shared<ov::op::v1::Multiply>(mvn->output(0), inputs[1]);
          } else {
            node = mvn;
          }
        }
      } else if (opName == "rms_norm" || opName == "RmsNorm") {
        // Compose: x / sqrt(mean(x^2) + eps) * scale
        if (inputs.size() >= 1) {
          double eps = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1e-5;
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> axes;
          if (slots[s].numIArgs > 0) {
            for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          } else {
            axes.push_back(rank - 1);
          }
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          auto x_sq = std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[0]);
          auto mean_sq = std::make_shared<ov::op::v1::ReduceMean>(x_sq->output(0), axes_const, true);
          auto eps_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(eps)});
          auto mean_eps = std::make_shared<ov::op::v1::Add>(mean_sq->output(0), eps_const);
          auto rms = std::make_shared<ov::op::v0::Sqrt>(mean_eps->output(0));
          auto normed = std::make_shared<ov::op::v1::Divide>(inputs[0], rms->output(0));
          if (inputs.size() >= 2) {
            node = std::make_shared<ov::op::v1::Multiply>(normed->output(0), inputs[1]);
          } else {
            node = normed;
          }
        }
      } else if (opName == "rms_norm_linear" || opName == "RmsNormLinear") {
        // Compose: matmul(rms_norm(x, gamma, eps), W)
        // = matmul(x / sqrt(mean(x^2) + eps) * gamma, W)
        // Input 0: x, Input 1: gamma, Input 2: W
        if (inputs.size() >= 3) {
          double eps = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1e-6;
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> axes = {rank - 1};
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);

          // RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
          auto x_sq = std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[0]);
          auto mean_sq = std::make_shared<ov::op::v1::ReduceMean>(x_sq->output(0), axes_const, true);
          auto eps_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(eps)});
          auto mean_eps = std::make_shared<ov::op::v1::Add>(mean_sq->output(0), eps_const);
          auto rms = std::make_shared<ov::op::v0::Sqrt>(mean_eps->output(0));
          auto normed = std::make_shared<ov::op::v1::Divide>(inputs[0], rms->output(0));
          auto scaled = std::make_shared<ov::op::v1::Multiply>(normed->output(0), inputs[1]);

          // Linear: scaled @ W
          node = std::make_shared<ov::op::v0::MatMul>(scaled->output(0), inputs[2], false, false);
        }
      } else if (opName == "batchnorm" || opName == "BatchNorm" || opName == "batchnorm_inference" || opName == "batch_norm") {
        // BatchNormInference: input, gamma, beta, mean, variance
        if (inputs.size() >= 5) {
          double eps = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 1e-5;
          node = std::make_shared<ov::op::v5::BatchNormInference>(
              inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps);
        }
      }

      // ── Reduction ──
      else if (opName == "reduce_sum" || opName == "ReduceSum" || opName == "sum" || opName == "Sum") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceSum>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_max" || opName == "ReduceMax" || opName == "max") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceMax>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_mean" || opName == "ReduceMean" || opName == "mean" || opName == "Mean") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceMean>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_min" || opName == "ReduceMin" || opName == "min") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceMin>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_prod" || opName == "ReduceProd" || opName == "prod" || opName == "Prod") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceProd>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_norm1" || opName == "ReduceNorm1" || opName == "norm1") {
        // Compose: ReduceSum(Abs(x), axes)
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          auto abs_x = std::make_shared<ov::op::v0::Abs>(inputs[0]);
          node = std::make_shared<ov::op::v1::ReduceSum>(abs_x->output(0), axes_const, keepDims);
        }
      } else if (opName == "reduce_norm2" || opName == "ReduceNorm2" || opName == "norm2") {
        // Compose: Sqrt(ReduceSum(x*x, axes))
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          auto x_sq = std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[0]);
          auto sum_sq = std::make_shared<ov::op::v1::ReduceSum>(x_sq->output(0), axes_const, keepDims);
          node = std::make_shared<ov::op::v0::Sqrt>(sum_sq->output(0));
        }
      } else if (opName == "reduce_logsumexp" || opName == "ReduceLogSumExp") {
        // Compose: Log(ReduceSum(Exp(x), axes))
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          auto exp_x = std::make_shared<ov::op::v0::Exp>(inputs[0]);
          auto sum_exp = std::make_shared<ov::op::v1::ReduceSum>(exp_x->output(0), axes_const, keepDims);
          node = std::make_shared<ov::op::v0::Log>(sum_exp->output(0));
        }
      } else if (opName == "reduce_variance" || opName == "ReduceVariance") {
        // Compose: mean((x - mean(x))^2)
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          auto mean_x = std::make_shared<ov::op::v1::ReduceMean>(inputs[0], axes_const, true);
          auto diff = std::make_shared<ov::op::v1::Subtract>(inputs[0], mean_x->output(0));
          auto diff_sq = std::make_shared<ov::op::v1::Multiply>(diff->output(0), diff->output(0));
          node = std::make_shared<ov::op::v1::ReduceMean>(diff_sq->output(0), axes_const, keepDims);
        }
      } else if (opName == "reduce_stdev" || opName == "ReduceStdev") {
        // Compose: sqrt(variance)
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          auto mean_x = std::make_shared<ov::op::v1::ReduceMean>(inputs[0], axes_const, true);
          auto diff = std::make_shared<ov::op::v1::Subtract>(inputs[0], mean_x->output(0));
          auto diff_sq = std::make_shared<ov::op::v1::Multiply>(diff->output(0), diff->output(0));
          auto var = std::make_shared<ov::op::v1::ReduceMean>(diff_sq->output(0), axes_const, keepDims);
          node = std::make_shared<ov::op::v0::Sqrt>(var->output(0));
        }
      } else if (opName == "normmax") {
        // ReduceMax(Abs(x), axes)
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : false;
          auto abs_x = std::make_shared<ov::op::v0::Abs>(inputs[0]);
          node = std::make_shared<ov::op::v1::ReduceMax>(abs_x->output(0), axes_const, keepDims);
        }
      }

      // ── Shape manipulation ──
      else if (opName == "reshape" || opName == "Reshape") {
        if (inputs.size() >= 1) {
          // Get target shape from iArgs, skipping the ordering marker if present
          // ND4J reshape encodes iArgs as [-order, dim0, dim1, ...] where -99=C, -102=F
          std::vector<int64_t> targetShape;
          int startIdx = 0;
          if (slots[s].numIArgs > 0) {
            int64_t first = slots[s].iArgs[0];
            if (first < 0 && (first == -99 || first == -102)) {
              startIdx = 1;  // skip ordering marker
            }
          }
          for (int a = startIdx; a < slots[s].numIArgs; a++) targetShape.push_back(slots[s].iArgs[a]);
          if (targetShape.empty() && slots[s].numOutputs > 0) {
            // Use output slot shape if available
            int outIdx = slots[s].outputSlotIndices[0];
            if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
              for (int d = 0; d < outputSlots[outIdx]->rankOf(); d++) {
                targetShape.push_back(outputSlots[outIdx]->sizeAt(d));
              }
            }
          }
          if (!targetShape.empty()) {
            std::string shapeStr = "[";
            for (size_t i = 0; i < targetShape.size(); i++) {
              if (i > 0) shapeStr += ",";
              shapeStr += std::to_string(targetShape[i]);
            }
            shapeStr += "]";
            DSP_DIAG(COMPILE, "OpenVINO: reshape slot %d: iArgs=%d startIdx=%d targetShape=%s input=%s",
                     s, slots[s].numIArgs, startIdx, shapeStr.c_str(),
                     inputs[0].get_partial_shape().to_string().c_str());
            auto shape_const = ov::op::v0::Constant::create(
                ov::element::i64, {targetShape.size()}, targetShape);
            node = std::make_shared<ov::op::v1::Reshape>(inputs[0], shape_const, false);
          }
        }
      } else if (opName == "permute" || opName == "Permute" || opName == "Transpose" || opName == "transpose") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> perm;
          for (int a = 0; a < slots[s].numIArgs; a++) perm.push_back(slots[s].iArgs[a]);
          if (perm.empty()) {
            // No iArgs = simple transpose: reverse dimensions
            auto inputRank = inputs[0].get_partial_shape().rank();
            if (inputRank.is_static()) {
              int rank = static_cast<int>(inputRank.get_length());
              for (int d = rank - 1; d >= 0; d--) perm.push_back(d);
            }
          }
          if (!perm.empty()) {
            auto perm_const = ov::op::v0::Constant::create(
                ov::element::i64, {perm.size()}, perm);
            node = std::make_shared<ov::op::v1::Transpose>(inputs[0], perm_const);
          }
        }
      } else if (opName == "squeeze" || opName == "Squeeze") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (!axes.empty()) {
            auto axes_const = ov::op::v0::Constant::create(
                ov::element::i64, {axes.size()}, axes);
            node = std::make_shared<ov::op::v0::Squeeze>(inputs[0], axes_const);
          } else {
            node = std::make_shared<ov::op::v0::Squeeze>(inputs[0]);
          }
        }
      } else if (opName == "expand_dims" || opName == "ExpandDims" || opName == "Unsqueeze") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (axes.empty()) axes.push_back(0);
          auto axes_const = ov::op::v0::Constant::create(
              ov::element::i64, {axes.size()}, axes);
          node = std::make_shared<ov::op::v0::Unsqueeze>(inputs[0], axes_const);
        }
      } else if (opName == "reshape_no_copy" || opName == "ReshapeNoCopy") {
        // Same as Reshape — skip ordering marker in iArgs
        if (inputs.size() >= 1) {
          std::vector<int64_t> targetShape;
          int startIdx = 0;
          if (slots[s].numIArgs > 0) {
            int64_t first = slots[s].iArgs[0];
            if (first < 0 && (first == -99 || first == -102)) {
              startIdx = 1;
            }
          }
          for (int a = startIdx; a < slots[s].numIArgs; a++) targetShape.push_back(slots[s].iArgs[a]);
          if (targetShape.empty() && slots[s].numOutputs > 0) {
            int outIdx = slots[s].outputSlotIndices[0];
            if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
              for (int d = 0; d < outputSlots[outIdx]->rankOf(); d++) {
                targetShape.push_back(outputSlots[outIdx]->sizeAt(d));
              }
            }
          }
          if (!targetShape.empty()) {
            auto shape_const = ov::op::v0::Constant::create(
                ov::element::i64, {targetShape.size()}, targetShape);
            node = std::make_shared<ov::op::v1::Reshape>(inputs[0], shape_const, false);
          }
        }
      } else if (opName == "flatten" || opName == "Flatten") {
        // Reshape to 1D (total elements)
        if (inputs.size() >= 1) {
          auto shape_const = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{-1});
          node = std::make_shared<ov::op::v1::Reshape>(inputs[0], shape_const, false);
        }
      } else if (opName == "flatten_2d" || opName == "Flatten2d") {
        // Reshape keeping first axis, flatten rest
        if (inputs.size() >= 1) {
          auto shape_const = ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{0, -1});
          node = std::make_shared<ov::op::v1::Reshape>(inputs[0], shape_const, true);  // special_zero = true
        }
      } else if (opName == "triu" || opName == "Triu") {
        // Upper triangular: compose with constant mask + Select
        // For now, create via ShapeOf + Range + Compare + Select
        if (inputs.size() >= 1) {
          int diag = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          if (rank >= 2) {
            auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
            // Get last two dims for row/col ranges
            auto neg2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-2});
            auto neg1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            auto zero_s = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
            auto one_s = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
            auto nrows = std::make_shared<ov::op::v8::Gather>(shape_of->output(0), neg2, zero_s);
            auto ncols = std::make_shared<ov::op::v8::Gather>(shape_of->output(0), neg1, zero_s);
            auto row_range = std::make_shared<ov::op::v4::Range>(zero_s, nrows->output(0), one_s, ov::element::i64);
            auto col_range = std::make_shared<ov::op::v4::Range>(zero_s, ncols->output(0), one_s, ov::element::i64);
            // Unsqueeze row to [N,1], col stays [M]
            auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto row_col = std::make_shared<ov::op::v0::Unsqueeze>(row_range->output(0), unsq_axis);
            auto diag_const = ov::op::v0::Constant::create(ov::element::i64, {}, {diag});
            auto row_plus_diag = std::make_shared<ov::op::v1::Add>(row_col->output(0), diag_const);
            auto mask = std::make_shared<ov::op::v1::LessEqual>(row_plus_diag->output(0), col_range->output(0));
            auto zero_val = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
            node = std::make_shared<ov::op::v1::Select>(mask->output(0), inputs[0], zero_val);
          }
        }
      } else if (opName == "tril" || opName == "Tril") {
        // Lower triangular: compose with constant mask + Select
        if (inputs.size() >= 1) {
          int diag = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          if (rank >= 2) {
            auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
            auto neg2 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-2});
            auto neg1 = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
            auto zero_s = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
            auto one_s = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
            auto nrows = std::make_shared<ov::op::v8::Gather>(shape_of->output(0), neg2, zero_s);
            auto ncols = std::make_shared<ov::op::v8::Gather>(shape_of->output(0), neg1, zero_s);
            auto row_range = std::make_shared<ov::op::v4::Range>(zero_s, nrows->output(0), one_s, ov::element::i64);
            auto col_range = std::make_shared<ov::op::v4::Range>(zero_s, ncols->output(0), one_s, ov::element::i64);
            auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
            auto row_col = std::make_shared<ov::op::v0::Unsqueeze>(row_range->output(0), unsq_axis);
            auto diag_const = ov::op::v0::Constant::create(ov::element::i64, {}, {diag});
            auto row_plus_diag = std::make_shared<ov::op::v1::Add>(row_col->output(0), diag_const);
            auto mask = std::make_shared<ov::op::v1::GreaterEqual>(row_plus_diag->output(0), col_range->output(0));
            auto zero_val = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
            node = std::make_shared<ov::op::v1::Select>(mask->output(0), inputs[0], zero_val);
          }
        }
      } else if (opName == "broadcast_to" || opName == "BroadcastTo") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> targetShape;
          for (int a = 0; a < slots[s].numIArgs; a++) targetShape.push_back(slots[s].iArgs[a]);
          if (!targetShape.empty()) {
            auto shape_const = ov::op::v0::Constant::create(
                ov::element::i64, {targetShape.size()}, targetShape);
            node = std::make_shared<ov::op::v3::Broadcast>(
                inputs[0], shape_const, ov::op::BroadcastType::BIDIRECTIONAL);
          } else if (inputs.size() >= 2) {
            // Shape from second input
            node = std::make_shared<ov::op::v3::Broadcast>(
                inputs[0], inputs[1], ov::op::BroadcastType::BIDIRECTIONAL);
          }
        }
      } else if (opName == "reshapeas" || opName == "ReshapeAs") {
        // Reshape using second input's shape
        if (inputs.size() >= 2) {
          auto target_shape = std::make_shared<ov::op::v3::ShapeOf>(inputs[1], ov::element::i64);
          node = std::make_shared<ov::op::v1::Reshape>(inputs[0], target_shape->output(0), false);
        }
      }

      // ── Data movement ──
      else if (opName == "gather" || opName == "Gather") {
        if (inputs.size() >= 2) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          node = std::make_shared<ov::op::v8::Gather>(inputs[0], inputs[1], axis_const);
        }
      } else if (opName == "gather_nd" || opName == "GatherNd") {
        if (inputs.size() >= 2) {
          int batchDims = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          node = std::make_shared<ov::op::v8::GatherND>(inputs[0], inputs[1], batchDims);
        }
      } else if (opName == "scatter_nd" || opName == "ScatterNd" || opName == "ScatterNdUpdate") {
        if (inputs.size() >= 3) {
          node = std::make_shared<ov::op::v3::ScatterNDUpdate>(inputs[0], inputs[1], inputs[2]);
        }
      } else if (opName == "concat" || opName == "Concat") {
        if (inputs.size() >= 2) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          // Log input shapes for diagnosis
          for (size_t ci = 0; ci < inputs.size(); ci++) {
            DSP_DIAG(COMPILE, "OpenVINO: concat slot %d input[%zu] shape=%s",
                     s, ci, inputs[ci].get_partial_shape().to_string().c_str());
          }
          ov::OutputVector concatInputs(inputs.begin(), inputs.end());
          node = std::make_shared<ov::op::v0::Concat>(concatInputs, axis);
        }
      } else if (opName == "split" || opName == "Split") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          int numSplits = (slots[s].numIArgs > 1) ? static_cast<int>(slots[s].iArgs[1]) : slots[s].numOutputs;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          node = std::make_shared<ov::op::v1::Split>(inputs[0], axis_const, numSplits);
        }
      } else if (opName == "slice" || opName == "Slice" || opName == "strided_slice" || opName == "StridedSlice") {
        if (inputs.size() >= 1 && slots[s].numIArgs >= 2) {
          // Extract begin, end, strides from iArgs
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> begin, end, strides;
          // iArgs layout: begin..., end..., strides...
          int argsPerDim = slots[s].numIArgs / 3;
          if (argsPerDim <= 0) argsPerDim = rank;
          for (int d = 0; d < argsPerDim && d < rank; d++) {
            begin.push_back(slots[s].iArgs[d]);
            end.push_back(slots[s].iArgs[argsPerDim + d]);
            int64_t stride = (2 * argsPerDim + d < slots[s].numIArgs)
                              ? slots[s].iArgs[2 * argsPerDim + d] : 1;
            if (stride == 0) stride = 1;  // OpenVINO rejects stride=0
            strides.push_back(stride);
          }
          auto begin_const = ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin);
          auto end_const = ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end);
          auto strides_const = ov::op::v0::Constant::create(ov::element::i64, {strides.size()}, strides);
          std::vector<int64_t> beginMask(begin.size(), 0);
          std::vector<int64_t> endMask(end.size(), 0);
          node = std::make_shared<ov::op::v1::StridedSlice>(
              inputs[0], begin_const, end_const, strides_const, beginMask, endMask);
        }
      } else if (opName == "tile" || opName == "Tile" || opName == "repeat" || opName == "Repeat") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> repeats;
          for (int a = 0; a < slots[s].numIArgs; a++) repeats.push_back(slots[s].iArgs[a]);
          if (!repeats.empty()) {
            auto repeats_const = ov::op::v0::Constant::create(
                ov::element::i64, {repeats.size()}, repeats);
            node = std::make_shared<ov::op::v0::Tile>(inputs[0], repeats_const);
          }
        }
      } else if (opName == "split_v" || opName == "SplitV") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          // Split lengths from remaining iArgs
          std::vector<int64_t> splitLengths;
          for (int a = 1; a < slots[s].numIArgs; a++) splitLengths.push_back(slots[s].iArgs[a]);
          if (!splitLengths.empty()) {
            auto lengths_const = ov::op::v0::Constant::create(
                ov::element::i64, {splitLengths.size()}, splitLengths);
            node = std::make_shared<ov::op::v1::VariadicSplit>(inputs[0], axis_const, lengths_const);
          } else if (inputs.size() >= 2) {
            // Split lengths from second input
            node = std::make_shared<ov::op::v1::VariadicSplit>(inputs[0], axis_const, inputs[1]);
          }
        }
      } else if (opName == "unstack" || opName == "Unstack") {
        // Split along axis into single-element slices, then squeeze that axis
        if (inputs.size() >= 1) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          int numSplits = slots[s].numOutputs;
          if (numSplits <= 0) numSplits = 1;
          auto split_node = std::make_shared<ov::op::v1::Split>(inputs[0], axis_const, numSplits);
          // For multi-output, wire each split output and squeeze
          if (numSplits > 1) {
            auto sq_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
            for (int o = 0; o < slots[s].numOutputs && o < numSplits; o++) {
              int outIdx = slots[s].outputSlotIndices[o];
              auto squeezed = std::make_shared<ov::op::v0::Squeeze>(split_node->output(o), sq_axes);
              tensorMap[outIdx] = squeezed->output(0);
            }
            audit.wasCompiled = true;
            result.compilationAudit.push_back(audit);
            continue;  // handled all outputs manually
          } else {
            auto sq_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
            node = std::make_shared<ov::op::v0::Squeeze>(split_node->output(0), sq_axes);
          }
        }
      } else if (opName == "stack" || opName == "Stack") {
        // Unsqueeze each input along axis, then concat
        if (inputs.size() >= 1) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          auto ax_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
          ov::OutputVector unsqueezed;
          for (size_t i = 0; i < inputs.size(); i++) {
            auto usq = std::make_shared<ov::op::v0::Unsqueeze>(inputs[i], ax_const);
            unsqueezed.push_back(usq->output(0));
          }
          node = std::make_shared<ov::op::v0::Concat>(unsqueezed, axis);
        }
      } else if (opName == "pad" || opName == "Pad") {
        if (inputs.size() >= 1) {
          // pads_begin and pads_end from iArgs: first half = begin, second half = end
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> pads_begin, pads_end;
          if (slots[s].numIArgs >= 2 * rank) {
            for (int d = 0; d < rank; d++) {
              pads_begin.push_back(slots[s].iArgs[d]);
              pads_end.push_back(slots[s].iArgs[rank + d]);
            }
          } else if (inputs.size() >= 3) {
            // pads from inputs[1] and inputs[2]
            auto pb_const = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
            node = std::make_shared<ov::op::v1::Pad>(
                inputs[0], inputs[1], inputs[2], pb_const, ov::op::PadMode::CONSTANT);
            // skip the manual path below
          }
          if (!pads_begin.empty() && !node) {
            auto pb = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
            auto pe = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
            float padVal = (slots[s].numTArgs > 0) ? static_cast<float>(slots[s].tArgs[0]) : 0.0f;
            auto pv = ov::op::v0::Constant::create(ov::element::f32, {}, {padVal});
            node = std::make_shared<ov::op::v1::Pad>(inputs[0], pb, pe, pv, ov::op::PadMode::CONSTANT);
          }
        }
      } else if (opName == "mirror_pad" || opName == "MirrorPad") {
        if (inputs.size() >= 1) {
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> pads_begin, pads_end;
          // mode: 0 = REFLECT, 1 = SYMMETRIC
          auto mode = (slots[s].numIArgs > 2 * rank) ?
              ((slots[s].iArgs[2 * rank] == 1) ? ov::op::PadMode::SYMMETRIC : ov::op::PadMode::REFLECT)
              : ov::op::PadMode::REFLECT;
          if (slots[s].numIArgs >= 2 * rank) {
            for (int d = 0; d < rank; d++) {
              pads_begin.push_back(slots[s].iArgs[d]);
              pads_end.push_back(slots[s].iArgs[rank + d]);
            }
          }
          if (!pads_begin.empty()) {
            auto pb = ov::op::v0::Constant::create(ov::element::i64, {pads_begin.size()}, pads_begin);
            auto pe = ov::op::v0::Constant::create(ov::element::i64, {pads_end.size()}, pads_end);
            node = std::make_shared<ov::op::v1::Pad>(inputs[0], pb, pe, mode);
          }
        }
      } else if (opName == "reverse" || opName == "Reverse" || opName == "reverse_v2" || opName == "ReverseV2") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].numIArgs; a++) axes.push_back(slots[s].iArgs[a]);
          if (!axes.empty()) {
            auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
            node = std::make_shared<ov::op::v1::Reverse>(
                inputs[0], axes_const, ov::op::v1::Reverse::Mode::INDEX);
          } else if (inputs.size() >= 2) {
            node = std::make_shared<ov::op::v1::Reverse>(
                inputs[0], inputs[1], ov::op::v1::Reverse::Mode::INDEX);
          }
        }
      } else if (opName == "embedding_lookup" || opName == "EmbeddingLookup") {
        // Same as Gather on axis=0
        if (inputs.size() >= 2) {
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
          node = std::make_shared<ov::op::v8::Gather>(inputs[0], inputs[1], axis_const);
        }
      } else if (opName == "scatter_nd_update") {
        if (inputs.size() >= 3) {
          node = std::make_shared<ov::op::v3::ScatterNDUpdate>(inputs[0], inputs[1], inputs[2]);
        }
      }

      // ── Comparison ──
      else if (opName == "greater" || opName == "Greater") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Greater>(inputs[0], inputs[1]);
      } else if (opName == "less" || opName == "Less") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Less>(inputs[0], inputs[1]);
      } else if (opName == "equals" || opName == "Equal" || opName == "Equals") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::Equal>(inputs[0], inputs[1]);
      } else if (opName == "not_equals" || opName == "NotEqual" || opName == "NotEquals") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::NotEqual>(inputs[0], inputs[1]);
      } else if (opName == "greater_equal" || opName == "GreaterEqual") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::GreaterEqual>(inputs[0], inputs[1]);
      } else if (opName == "less_equal" || opName == "LessEqual") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::LessEqual>(inputs[0], inputs[1]);
      }

      // ── Logical ──
      else if (opName == "boolean_and" || opName == "BooleanAnd" || opName == "logical_and" || opName == "LogicalAnd") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::LogicalAnd>(inputs[0], inputs[1]);
      } else if (opName == "boolean_or" || opName == "BooleanOr" || opName == "logical_or" || opName == "LogicalOr") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::LogicalOr>(inputs[0], inputs[1]);
      } else if (opName == "boolean_not" || opName == "BooleanNot" || opName == "bool_not" || opName == "logical_not" || opName == "LogicalNot") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v1::LogicalNot>(inputs[0]);
      } else if (opName == "boolean_xor" || opName == "BooleanXor" || opName == "LogicalXor") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v1::LogicalXor>(inputs[0], inputs[1]);
      }

      // ── Ternary (Where/Select) ──
      else if (opName == "where" || opName == "Where" || opName == "select" || opName == "Select" ||
               opName == "where_np" || opName == "WhereNp") {
        if (inputs.size() >= 3) node = std::make_shared<ov::op::v1::Select>(inputs[0], inputs[1], inputs[2]);
      }

      // ── Cast ──
      else if (opName == "cast" || opName == "Cast") {
        if (inputs.size() >= 1) {
          ov::element::Type targetType;
          if (slots[s].numDArgs > 0) {
            targetType = mapDataType(slots[s].dArgs[0]);
          } else if (slots[s].numIArgs > 0) {
            // Cast op stores target type as iArg (FlatBuffersMapper byte encoding)
            targetType = mapDataType(static_cast<DataType>(slots[s].iArgs[0]));
          } else {
            // Fallback: same type as input (identity cast)
            targetType = inputs[0].get_element_type();
          }
          node = std::make_shared<ov::op::v0::Convert>(inputs[0], targetType);
        }
      }

      // ── Identity / Assign ──
      else if (opName == "identity" || opName == "Identity" || opName == "assign" || opName == "Assign") {
        if (inputs.size() >= 1) {
          // Pass-through: wire input directly to output
          for (int o = 0; o < slots[s].numOutputs; o++) {
            int outIdx = slots[s].outputSlotIndices[o];
            tensorMap[outIdx] = inputs[0];
          }
          audit.wasCompiled = true;
          result.compilationAudit.push_back(audit);
          continue;  // skip node creation — identity is a wire
        }
      }

      // ── Constant generation ──
      else if (opName == "zeros_like" || opName == "ZerosLike" || opName == "zeroslike" || opName == "zeros_as") {
        if (inputs.size() >= 1) {
          auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
          auto zero = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
          node = std::make_shared<ov::op::v3::Broadcast>(zero, shape_of->output(0));
        }
      } else if (opName == "ones_like" || opName == "OnesLike" || opName == "oneslike" || opName == "ones_as" || opName == "OnesAs") {
        if (inputs.size() >= 1) {
          auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
          auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          node = std::make_shared<ov::op::v3::Broadcast>(one, shape_of->output(0));
        }
      } else if (opName == "range" || opName == "Range") {
        // Range(start, stop, step) — from tArgs or inputs
        if (inputs.size() >= 3) {
          node = std::make_shared<ov::op::v4::Range>(inputs[0], inputs[1], inputs[2], ov::element::f32);
        } else if (slots[s].numTArgs >= 3) {
          auto start = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[0])});
          auto stop = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[1])});
          auto step = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[2])});
          node = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f32);
        }
      } else if (opName == "fill" || opName == "Fill") {
        // Broadcast a scalar value to a target shape
        if (inputs.size() >= 2) {
          node = std::make_shared<ov::op::v3::Broadcast>(inputs[1], inputs[0]);
        } else if (inputs.size() >= 1 && slots[s].numTArgs > 0) {
          auto val = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].tArgs[0])});
          node = std::make_shared<ov::op::v3::Broadcast>(val, inputs[0]);
        }
      } else if (opName == "shape_of" || opName == "ShapeOf") {
        if (inputs.size() >= 1) {
          node = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
        }
      } else if (opName == "onehot" || opName == "OneHot") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : -1;
          int64_t depth = (slots[s].numIArgs > 1) ? slots[s].iArgs[1] : 1;
          float onVal = (slots[s].numTArgs > 0) ? static_cast<float>(slots[s].tArgs[0]) : 1.0f;
          float offVal = (slots[s].numTArgs > 1) ? static_cast<float>(slots[s].tArgs[1]) : 0.0f;
          auto depth_const = ov::op::v0::Constant::create(ov::element::i64, {}, {depth});
          auto on_const = ov::op::v0::Constant::create(ov::element::f32, {}, {onVal});
          auto off_const = ov::op::v0::Constant::create(ov::element::f32, {}, {offVal});
          node = std::make_shared<ov::op::v1::OneHot>(inputs[0], depth_const, on_const, off_const, axis);
        }
      } else if (opName == "eye" || opName == "Eye") {
        // Compose identity-like pattern: triu(ones) with diag=0 on square matrix
        // Use Range + Unsqueeze + Equal to create identity
        if (slots[s].numIArgs >= 1) {
          int64_t n = slots[s].iArgs[0];
          int64_t m = (slots[s].numIArgs > 1) ? slots[s].iArgs[1] : n;
          auto zero = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
          auto one_step = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
          auto n_const = ov::op::v0::Constant::create(ov::element::i64, {}, {n});
          auto m_const = ov::op::v0::Constant::create(ov::element::i64, {}, {m});
          auto rows = std::make_shared<ov::op::v4::Range>(zero, n_const, one_step, ov::element::i64);
          auto cols = std::make_shared<ov::op::v4::Range>(zero, m_const, one_step, ov::element::i64);
          auto unsq_axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
          auto rows_2d = std::make_shared<ov::op::v0::Unsqueeze>(rows->output(0), unsq_axis);
          auto eq = std::make_shared<ov::op::v1::Equal>(rows_2d->output(0), cols->output(0));
          auto one_f = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
          auto zero_f = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
          node = std::make_shared<ov::op::v1::Select>(eq->output(0), one_f, zero_f);
        }
      } else if (opName == "linspace" || opName == "Linspace" ||
                 opName == "lin_space" || opName == "LinSpace") {
        // Compose: start + arange(num) * ((stop - start) / (num - 1))
        if (slots[s].numTArgs >= 2 && slots[s].numIArgs >= 1) {
          double start = slots[s].tArgs[0];
          double stop = slots[s].tArgs[1];
          int64_t num = slots[s].iArgs[0];
          if (num > 1) {
            double stepVal = (stop - start) / (num - 1);
            auto zeroC = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
            auto numC = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(num)});
            auto oneC = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
            auto rng = std::make_shared<ov::op::v4::Range>(zeroC, numC, oneC, ov::element::f32);
            auto stepC = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(stepVal)});
            auto startC = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(start)});
            auto scaled = std::make_shared<ov::op::v1::Multiply>(rng->output(0), stepC);
            node = std::make_shared<ov::op::v1::Add>(scaled->output(0), startC);
          } else {
            node = std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1}, std::vector<float>{static_cast<float>(start)});
          }
        }
      } else if (opName == "sequence_mask" || opName == "SequenceMask") {
        // Compose: unsqueeze(lengths, -1) > arange(maxLen)
        if (inputs.size() >= 1 && slots[s].numIArgs >= 1) {
          int64_t maxLen = slots[s].iArgs[0];
          auto zeroC = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
          auto maxC = ov::op::v0::Constant::create(ov::element::i64, {}, {maxLen});
          auto oneC = ov::op::v0::Constant::create(ov::element::i64, {}, {1});
          auto rng = std::make_shared<ov::op::v4::Range>(zeroC, maxC, oneC, ov::element::i64);
          auto ax = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
          auto uq = std::make_shared<ov::op::v0::Unsqueeze>(inputs[0], ax);
          node = std::make_shared<ov::op::v1::Greater>(uq->output(0), rng->output(0));
        }
      } else if (opName == "create" || opName == "Create" ||
                 opName == "set_scalar" || opName == "SetScalar" ||
                 opName == "min_max_datatype" || opName == "MinMaxDatatype") {
        // Broadcast a scalar value to the output shape
        double val = (slots[s].numTArgs > 0) ? slots[s].tArgs[0] : 0.0;
        if (slots[s].numOutputs > 0) {
          int outIdx = slots[s].outputSlotIndices[0];
          if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
            std::vector<int64_t> targetShape;
            for (int d = 0; d < outputSlots[outIdx]->rankOf(); d++) {
              targetShape.push_back(outputSlots[outIdx]->sizeAt(d));
            }
            if (targetShape.empty()) targetShape.push_back(1);
            auto valC = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(val)});
            auto shapeC = ov::op::v0::Constant::create(ov::element::i64, {targetShape.size()}, targetShape);
            node = std::make_shared<ov::op::v3::Broadcast>(valC, shapeC);
          }
        }
      } else if (opName == "size" || opName == "Size") {
        // Total number of elements
        if (inputs.size() >= 1) {
          auto shapeNode = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
          auto axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
          node = std::make_shared<ov::op::v1::ReduceProd>(shapeNode->output(0), axes, false);
        }
      } else if (opName == "size_at" || opName == "SizeAt") {
        // Size of a specific dimension
        if (inputs.size() >= 1) {
          int dim = (slots[s].numIArgs > 0) ? static_cast<int>(slots[s].iArgs[0]) : 0;
          auto shapeNode = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
          auto idx = ov::op::v0::Constant::create(ov::element::i64, {1}, {static_cast<int64_t>(dim)});
          auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
          node = std::make_shared<ov::op::v8::Gather>(shapeNode->output(0), idx, axis);
        }
      } else if (opName == "rank" || opName == "Rank") {
        // Number of dimensions
        if (inputs.size() >= 1) {
          auto shapeNode = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
          auto shapeOfShape = std::make_shared<ov::op::v3::ShapeOf>(shapeNode->output(0), ov::element::i64);
          node = std::make_shared<ov::op::v0::Squeeze>(shapeOfShape->output(0));
        }
      }
      // ── Bitwise unary ──
      else if (opName == "toggle_bits" || opName == "ToggleBits") {
        if (inputs.size() >= 1) node = std::make_shared<ov::op::v13::BitwiseNot>(inputs[0]);
      }
      // ── Bitwise shift (fallback if not already matched in binary section) ──
      else if (opName == "shift_bits" || opName == "ShiftBits") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v15::BitwiseLeftShift>(inputs[0], inputs[1]);
      } else if (opName == "rshift_bits" || opName == "RShiftBits") {
        if (inputs.size() >= 2) node = std::make_shared<ov::op::v15::BitwiseRightShift>(inputs[0], inputs[1]);
      }

      // ── Convolution ──
      else if (opName == "conv2d" || opName == "Conv2d" || opName == "Conv2D" || opName == "conv2D") {
        if (inputs.size() >= 2) {
          // Extract strides, pads, dilations from iArgs
          // iArgs: [kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW, wFormat]
          int64_t sH = (slots[s].numIArgs > 2) ? slots[s].iArgs[2] : 1;
          int64_t sW = (slots[s].numIArgs > 3) ? slots[s].iArgs[3] : 1;
          int64_t pH = (slots[s].numIArgs > 4) ? slots[s].iArgs[4] : 0;
          int64_t pW = (slots[s].numIArgs > 5) ? slots[s].iArgs[5] : 0;
          int64_t dH = (slots[s].numIArgs > 6) ? slots[s].iArgs[6] : 1;
          int64_t dW = (slots[s].numIArgs > 7) ? slots[s].iArgs[7] : 1;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pH, pW};
          ov::CoordinateDiff pads_end{pH, pW};
          ov::Strides dilations{static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::Convolution>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "conv3d" || opName == "Conv3d") {
        if (inputs.size() >= 2) {
          int64_t sD = (slots[s].numIArgs > 3) ? slots[s].iArgs[3] : 1;
          int64_t sH = (slots[s].numIArgs > 4) ? slots[s].iArgs[4] : 1;
          int64_t sW = (slots[s].numIArgs > 5) ? slots[s].iArgs[5] : 1;
          int64_t pD = (slots[s].numIArgs > 6) ? slots[s].iArgs[6] : 0;
          int64_t pH = (slots[s].numIArgs > 7) ? slots[s].iArgs[7] : 0;
          int64_t pW = (slots[s].numIArgs > 8) ? slots[s].iArgs[8] : 0;
          int64_t dD = (slots[s].numIArgs > 9) ? slots[s].iArgs[9] : 1;
          int64_t dH = (slots[s].numIArgs > 10) ? slots[s].iArgs[10] : 1;
          int64_t dW = (slots[s].numIArgs > 11) ? slots[s].iArgs[11] : 1;
          ov::Strides strides{static_cast<size_t>(sD), static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pD, pH, pW};
          ov::CoordinateDiff pads_end{pD, pH, pW};
          ov::Strides dilations{static_cast<size_t>(dD), static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::Convolution>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "depthwise_conv2d" || opName == "DepthwiseConv2d") {
        if (inputs.size() >= 2) {
          int64_t sH = (slots[s].numIArgs > 2) ? slots[s].iArgs[2] : 1;
          int64_t sW = (slots[s].numIArgs > 3) ? slots[s].iArgs[3] : 1;
          int64_t pH = (slots[s].numIArgs > 4) ? slots[s].iArgs[4] : 0;
          int64_t pW = (slots[s].numIArgs > 5) ? slots[s].iArgs[5] : 0;
          int64_t dH = (slots[s].numIArgs > 6) ? slots[s].iArgs[6] : 1;
          int64_t dW = (slots[s].numIArgs > 7) ? slots[s].iArgs[7] : 1;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pH, pW};
          ov::CoordinateDiff pads_end{pH, pW};
          ov::Strides dilations{static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::GroupConvolution>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "maxpool2d" || opName == "MaxPool2d") {
        if (inputs.size() >= 1) {
          int64_t kH = (slots[s].numIArgs > 0) ? slots[s].iArgs[0] : 2;
          int64_t kW = (slots[s].numIArgs > 1) ? slots[s].iArgs[1] : 2;
          int64_t sH = (slots[s].numIArgs > 2) ? slots[s].iArgs[2] : 1;
          int64_t sW = (slots[s].numIArgs > 3) ? slots[s].iArgs[3] : 1;
          int64_t pH = (slots[s].numIArgs > 4) ? slots[s].iArgs[4] : 0;
          int64_t pW = (slots[s].numIArgs > 5) ? slots[s].iArgs[5] : 0;
          int64_t dH = (slots[s].numIArgs > 6) ? slots[s].iArgs[6] : 1;
          int64_t dW = (slots[s].numIArgs > 7) ? slots[s].iArgs[7] : 1;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::Strides dilations{static_cast<size_t>(dH), static_cast<size_t>(dW)};
          ov::Shape kernel{static_cast<size_t>(kH), static_cast<size_t>(kW)};
          ov::Shape pads_begin{static_cast<size_t>(pH), static_cast<size_t>(pW)};
          ov::Shape pads_end{static_cast<size_t>(pH), static_cast<size_t>(pW)};
          node = std::make_shared<ov::op::v8::MaxPool>(
              inputs[0], strides, dilations, pads_begin, pads_end, kernel);
        }
      } else if (opName == "avgpool2d" || opName == "AvgPool2d") {
        if (inputs.size() >= 1) {
          int64_t kH = (slots[s].numIArgs > 0) ? slots[s].iArgs[0] : 2;
          int64_t kW = (slots[s].numIArgs > 1) ? slots[s].iArgs[1] : 2;
          int64_t sH = (slots[s].numIArgs > 2) ? slots[s].iArgs[2] : 1;
          int64_t sW = (slots[s].numIArgs > 3) ? slots[s].iArgs[3] : 1;
          int64_t pH = (slots[s].numIArgs > 4) ? slots[s].iArgs[4] : 0;
          int64_t pW = (slots[s].numIArgs > 5) ? slots[s].iArgs[5] : 0;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::Shape kernel{static_cast<size_t>(kH), static_cast<size_t>(kW)};
          ov::Shape pads_begin{static_cast<size_t>(pH), static_cast<size_t>(pW)};
          ov::Shape pads_end{static_cast<size_t>(pH), static_cast<size_t>(pW)};
          bool excludePad = (slots[s].numBArgs > 0) ? slots[s].bArgs[0] : true;
          node = std::make_shared<ov::op::v1::AvgPool>(
              inputs[0], strides, pads_begin, pads_end, kernel, excludePad);
        }
      } else if (opName == "deconv2d") {
        if (inputs.size() >= 2) {
          int64_t sH = (slots[s].numIArgs > 2) ? slots[s].iArgs[2] : 1;
          int64_t sW = (slots[s].numIArgs > 3) ? slots[s].iArgs[3] : 1;
          int64_t pH = (slots[s].numIArgs > 4) ? slots[s].iArgs[4] : 0;
          int64_t pW = (slots[s].numIArgs > 5) ? slots[s].iArgs[5] : 0;
          int64_t dH = (slots[s].numIArgs > 6) ? slots[s].iArgs[6] : 1;
          int64_t dW = (slots[s].numIArgs > 7) ? slots[s].iArgs[7] : 1;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pH, pW};
          ov::CoordinateDiff pads_end{pH, pW};
          ov::Strides dilations{static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "deconv3d") {
        if (inputs.size() >= 2) {
          int64_t sD = (slots[s].numIArgs > 3) ? slots[s].iArgs[3] : 1;
          int64_t sH = (slots[s].numIArgs > 4) ? slots[s].iArgs[4] : 1;
          int64_t sW = (slots[s].numIArgs > 5) ? slots[s].iArgs[5] : 1;
          int64_t pD = (slots[s].numIArgs > 6) ? slots[s].iArgs[6] : 0;
          int64_t pH = (slots[s].numIArgs > 7) ? slots[s].iArgs[7] : 0;
          int64_t pW = (slots[s].numIArgs > 8) ? slots[s].iArgs[8] : 0;
          int64_t dD = (slots[s].numIArgs > 9) ? slots[s].iArgs[9] : 1;
          int64_t dH = (slots[s].numIArgs > 10) ? slots[s].iArgs[10] : 1;
          int64_t dW = (slots[s].numIArgs > 11) ? slots[s].iArgs[11] : 1;
          ov::Strides strides{static_cast<size_t>(sD), static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pD, pH, pW};
          ov::CoordinateDiff pads_end{pD, pH, pW};
          ov::Strides dilations{static_cast<size_t>(dD), static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      }

      // ── ROPE (complex composition -- deferred to runtime) ──
      else if (opName == "rope" || opName == "Rope" || opName == "fused_rope" || opName == "FusedRope") {
        audit.wasCompiled = true;
        audit.reason = "complex op - deferred to runtime";
        result.compilationAudit.push_back(audit);
        continue;
      }

      // ── Attention (complex composition -- deferred to runtime) ──
      else if (opName == "dot_product_attention" || opName == "DotProductAttention" ||
               opName == "dot_product_attention_v2" || opName == "DotProductAttentionV2" ||
               opName == "multi_head_attention" || opName == "MultiHeadAttention" ||
               opName == "onnx_multi_head_attention" || opName == "OnnxMultiHeadAttention") {
        audit.wasCompiled = true;
        audit.reason = "complex op - deferred to runtime";
        result.compilationAudit.push_back(audit);
        continue;
      }

      if (!node) {
        audit.wasCompiled = false;
        audit.reason = "failed to create OV node for op: " + opName;
        result.compilationAudit.push_back(audit);
        DSP_DIAG(COMPILE, "OpenVINO: failed to create node for '%s' at slot %d",
                 opName.c_str(), s);
        continue;
      }

      // Wire outputs
      for (int o = 0; o < slots[s].numOutputs; o++) {
        int outIdx = slots[s].outputSlotIndices[o];
        int nodeOutputIdx = std::min(o, static_cast<int>(node->get_output_size()) - 1);
        tensorMap[outIdx] = node->output(nodeOutputIdx);
      }

      audit.wasCompiled = true;
      result.compilationAudit.push_back(audit);

      } catch (const std::exception& nodeEx) {
        fprintf(stderr, "OpenVINO: node creation FAILED at slot %d op '%s': %s\n",
                s, opName.c_str(), nodeEx.what());
        fflush(stderr);
        DSP_DIAG(COMPILE, "OpenVINO: node creation THREW at slot %d op '%s': %s",
                 s, opName.c_str(), nodeEx.what());
        audit.wasCompiled = false;
        audit.reason = std::string("node creation exception: ") + nodeEx.what();
        result.compilationAudit.push_back(audit);
        goto next_slot;
      }
    }
    next_slot:;
  }

  // Determine which outputs are externally visible (consumed outside the segment or are plan outputs)
  std::unordered_set<int> externallyConsumed;
  // All slot outputs that might be used outside this segment
  for (int s = startSlot; s <= endSlot; s++) {
    for (int o = 0; o < slots[s].numOutputs; o++) {
      int outIdx = slots[s].outputSlotIndices[o];
      externallyConsumed.insert(outIdx);  // conservatively mark all as external
    }
  }

  // Create Results for externally visible outputs
  for (int outIdx : externallyConsumed) {
    auto it = tensorMap.find(outIdx);
    if (it == tensorMap.end()) continue;
    results.push_back(std::make_shared<ov::op::v0::Result>(it->second));
    outputSourceMap.push_back(outIdx);
  }

  if (params.empty() || results.empty()) {
    DSP_DIAG(COMPILE, "OpenVINO: empty model (params=%zu, results=%zu)",
             params.size(), results.size());
    return result;
  }

  // Build the model
  try {
    auto model = std::make_shared<ov::Model>(results, params, "dsp_segment");
    auto compiled = core_.compile_model(model, "CPU");
    result.compiled = std::make_shared<ov::CompiledModel>(compiled);
    result.request = std::make_shared<ov::InferRequest>(result.compiled->create_infer_request());
    result.inputSlotMap = inputSourceMap;
    result.outputSlotMap = outputSourceMap;
    result.valid = true;

    DSP_DIAG(COMPILE, "OpenVINO: compiled model with %zu params, %zu results",
             params.size(), results.size());
  } catch (const std::exception& e) {
    // OpenVINO exceptions often have multiline messages with nested cause.
    // Log full message to stderr for visibility.
    fprintf(stderr, "OpenVINO compile_model FAILED seg[%d-%d]: %s\n", startSlot, endSlot, e.what());
    fflush(stderr);
    DSP_DIAG(COMPILE, "OpenVINO: compile_model FAILED seg[%d-%d] (%zu params, %zu results)",
             startSlot, endSlot, params.size(), results.size());
    result.valid = false;
  }

  return result;
}

// ─── compileSegment ─────────────────────────────────────────────────────────

bool OpenVinoGraphBackend::compileSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey,
    int totalSlots,
    int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  SegmentCacheKey cacheKey{seg.startSlot, seg.endSlot, shapeKey};

  std::lock_guard<std::mutex> lock(cacheMtx_);

  auto it = cache_.find(cacheKey);
  if (it != cache_.end() && it->second.valid) {
    lastCompilationAudit_ = it->second.compilationAudit;
    return true;
  }

  CompiledSegment compiled;
  try {
    compiled = buildModel(slots, seg.startSlot, seg.endSlot,
                          externalInputs, numExternalInputs,
                          outputSlots, totalOutputSlots);
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "OpenVINO: buildModel[%d-%d] exception: %s",
             seg.startSlot, seg.endSlot, e.what());
    return false;
  } catch (...) {
    DSP_DIAG(COMPILE, "OpenVINO: buildModel[%d-%d] unknown exception",
             seg.startSlot, seg.endSlot);
    return false;
  }
  compiled.shapeKey = shapeKey;

  lastCompilationAudit_ = compiled.compilationAudit;

  if (!compiled.valid) {
    return false;
  }

  cache_[cacheKey] = std::move(compiled);
  seg.shapeKey = shapeKey;
  return true;
}

// ─── executeSegment ─────────────────────────────────────────────────────────

Status OpenVinoGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  SegmentCacheKey cacheKey{seg.startSlot, seg.endSlot, seg.shapeKey};

  std::lock_guard<std::mutex> lock(cacheMtx_);

  auto it = cache_.find(cacheKey);
  if (it == cache_.end() || !it->second.valid) {
    DSP_DIAG(EXECUTE, "OpenVINO: no compiled segment for [%d-%d] shapeKey=%lld cacheSize=%d found=%d",
             seg.startSlot, seg.endSlot, (long long)seg.shapeKey,
             (int)cache_.size(), it != cache_.end() ? 1 : 0);
    return Status::KERNEL_FAILURE;
  }

  auto& compiled = it->second;
  if (!compiled.request) {
    DSP_DIAG(EXECUTE, "OpenVINO: compiled segment for [%d-%d] has null request", seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }
  auto& request = *compiled.request;

  try {
  // Set input tensors (zero-copy from NDArray host buffers)
  auto ovDtype = [](DataType dt) { return mapDataType(dt); };
  for (size_t i = 0; i < compiled.inputSlotMap.size(); i++) {
    int srcIdx = compiled.inputSlotMap[i];
    NDArray* arr = nullptr;
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
    } else {
      if (srcIdx < totalOutputSlots) arr = outputSlots[srcIdx];
    }
    if (!arr) {
      DSP_DIAG(EXECUTE, "OpenVINO: missing input array for source %d", srcIdx);
      return Status::BAD_INPUT;
    }
    if (arr->buffer() == nullptr) {
      if (arr->isEmpty() || arr->lengthOf() == 0) {
        // Empty array (e.g. KV cache on first decode step with seq_len=0).
        // OpenVINO requires non-null data pointer — provide a dummy.
        static int8_t dummyBuf[8] = {0};
        int rank = arr->rankOf();
        ov::Shape shape(rank);
        for (int d = 0; d < rank; d++) shape[d] = static_cast<size_t>(arr->sizeAt(d));
        request.set_input_tensor(static_cast<int>(i),
            ov::Tensor(ovDtype(arr->dataType()), shape, dummyBuf));
        continue;
      }
      DSP_DIAG(EXECUTE, "OpenVINO: input array for source %d has NULL buffer (len=%lld)",
               srcIdx, (long long)arr->lengthOf());
      return Status::KERNEL_FAILURE;
    }

    int rank = arr->rankOf();
    ov::Shape shape(rank);
    for (int d = 0; d < rank; d++) {
      shape[d] = static_cast<size_t>(arr->sizeAt(d));
    }
    request.set_input_tensor(static_cast<int>(i),
        ov::Tensor(ovDtype(arr->dataType()), shape, arr->buffer()));
  }

  // Set output tensors (zero-copy into NDArray host buffers)
  for (size_t i = 0; i < compiled.outputSlotMap.size(); i++) {
    int outIdx = compiled.outputSlotMap[i];
    if (outIdx < 0 || outIdx >= totalOutputSlots || !outputSlots[outIdx]) {
      DSP_DIAG(EXECUTE, "OpenVINO: missing output slot %d", outIdx);
      return Status::BAD_OUTPUT;
    }
    NDArray* arr = outputSlots[outIdx];
    if (arr->buffer() == nullptr) {
      DSP_DIAG(EXECUTE, "OpenVINO: output slot %d has NULL buffer", outIdx);
      return Status::KERNEL_FAILURE;
    }
    int rank = arr->rankOf();
    ov::Shape shape(rank);
    for (int d = 0; d < rank; d++) {
      shape[d] = static_cast<size_t>(arr->sizeAt(d));
    }

    try {
      request.set_output_tensor(static_cast<int>(i),
          ov::Tensor(ovDtype(arr->dataType()), shape, arr->buffer()));
    } catch (const std::exception& e) {
      DSP_DIAG(EXECUTE, "OpenVINO: set_output_tensor[%zu] FAILED for slot %d: %s",
               i, outIdx, e.what());
      return Status::KERNEL_FAILURE;
    }
  }

  // Run inference
  try {
    request.infer();
  } catch (const std::exception& e) {
    DSP_DIAG(EXECUTE, "OpenVINO: infer() failed: %s", e.what());
    return Status::KERNEL_FAILURE;
  }

  // Mark output arrays as host-authoritative
  for (size_t i = 0; i < compiled.outputSlotMap.size(); i++) {
    int outIdx = compiled.outputSlotMap[i];
    if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
      outputSlots[outIdx]->tickWriteHost();
    }
  }

  return Status::OK;

  } catch (const std::exception& e) {
    DSP_DIAG(EXECUTE, "OpenVINO: executeSegment[%d-%d] exception: %s",
             seg.startSlot, seg.endSlot, e.what());
    return Status::KERNEL_FAILURE;
  } catch (...) {
    DSP_DIAG(EXECUTE, "OpenVINO: executeSegment[%d-%d] unknown exception",
             seg.startSlot, seg.endSlot);
    return Status::KERNEL_FAILURE;
  }
}

// ─── invalidateCache ────────────────────────────────────────────────────────

void OpenVinoGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
}

// ─── getLastCompilationAudit ────────────────────────────────────────────────

std::vector<CompilationAuditEntry> OpenVinoGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_OPENVINO
