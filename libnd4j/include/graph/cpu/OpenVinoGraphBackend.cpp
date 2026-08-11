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
#include <ops/declarable/helpers/causal_conv1d.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <openvino/runtime/properties.hpp>
#include <system/Environment.h>
#include <system/common.h>

#if defined(__F16C__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if HAVE_ONEDNN
#include <dnnl.h>
#endif

namespace sd {
namespace graph {

// ─── ISA detection ─────────────────────────────────────────────────────────
// Query OneDNN once at startup to determine if the CPU has native FP16 support
// (AVX512-FP16 / AMX-FP16). If not, the OpenVINO backend must promote f16
// parameters to f32 so ALL ops compute in f32. Without this, CPUs like AMD
// Ryzen (no AVX512-FP16) silently produce zeros for f16 matmul/compute.
static bool cpuHasNativeFp16() {
#if HAVE_ONEDNN
  dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
  return (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
#else
  return false;  // conservative: assume no FP16 if we can't check
#endif
}

static const bool kCpuHasNativeFp16 = cpuHasNativeFp16();

// Thread-local native executor for mixed-segment interleaving
thread_local OpenVinoGraphBackend::NativeSlotExecutor OpenVinoGraphBackend::nativeExecutor_ = nullptr;

void OpenVinoGraphBackend::setNativeSlotExecutor(NativeSlotExecutor executor) {
  nativeExecutor_ = std::move(executor);
}

void OpenVinoGraphBackend::clearNativeSlotExecutor() {
  nativeExecutor_ = nullptr;
}

// ─── Singleton ──────────────────────────────────────────────────────────────

OpenVinoGraphBackend& OpenVinoGraphBackend::getInstance() {
  static OpenVinoGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new OpenVinoGraphBackend();
  });
  return *instance;
}

OpenVinoGraphBackend::OpenVinoGraphBackend() {
  // Configure OpenVINO for low-latency single-request inference (autoregressive decode).
  try {
    int numThreads = sd::Environment::getInstance().maxMasterThreads();
    if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
    core_.set_property("CPU", ov::inference_num_threads(numThreads));

    // LATENCY mode = single stream, all threads for intra-op parallelism.
    // THROUGHPUT mode replicates model buffers per stream → OOM on large models.
    // For autoregressive decode (single request), LATENCY with many threads
    // gives maximum parallelism within each matmul/attention op.
    core_.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));

    // Single outstanding infer request — no queuing overhead for sequential decode.
    core_.set_property("CPU", ov::hint::num_requests(1));

    // Disable hyper-threading: use physical cores only.
    // HT siblings share execution units and memory bandwidth — using both
    // halves effective per-thread bandwidth for memory-bound matmul/attention.
    core_.set_property("CPU", ov::hint::enable_hyper_threading(false));

    // Pin threads to physical cores to eliminate OS scheduler jitter between
    // decode steps. On Linux this is the default with LATENCY hint, but
    // setting explicitly ensures it's active regardless of platform.
    core_.set_property("CPU", ov::hint::enable_cpu_pinning(true));

    // On hybrid CPUs (Intel Alder Lake+), prefer P-cores only.
    // E-cores have lower IPC and clock speed — mixing them in slows down
    // the critical path for latency-sensitive decode.
    core_.set_property("CPU", ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::PCORE_ONLY));

    // Enable OpenVINO disk cache for compiled models.
    // Content-based: identical model topologies share cached artifacts across
    // process restarts, eliminating recompilation overhead on warm starts.
    const char* home = std::getenv("HOME");
    if (home) {
      std::string cacheDir = std::string(home) + "/.nd4j/openvino_cache";
      core_.set_property(ov::cache_dir(cacheDir));
      DSP_DIAG(COMPILE, "OpenVINO: configured CPU with %d threads, LATENCY mode, "
               "HT=off, pinning=on, P-core-only, disk cache=%s",
               numThreads, cacheDir.c_str());
    } else {
      DSP_DIAG(COMPILE, "OpenVINO: configured CPU with %d threads, LATENCY mode, "
               "HT=off, pinning=on, P-core-only (no disk cache — HOME not set)",
               numThreads);
    }
    DSP_DIAG(COMPILE, "OpenVINO: CPU native FP16 ISA: %s (f16 params %s to f32)",
             kCpuHasNativeFp16 ? "YES" : "NO",
             kCpuHasNativeFp16 ? "kept as f16" : "promoted");
  } catch (const std::exception& e) {
    DSP_DIAG(COMPILE, "OpenVINO: failed to set properties: %s", e.what());
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

bool OpenVinoGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_OPENVINO ||
         request.executionMode == GraphExecutionMode::GEM_AUTO ||
         request.executionMode == GraphExecutionMode::GEM_PORTABLE_REPLAY;
}

int OpenVinoGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_OPENVINO ? 1000 : 600;
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
    "causal_conv1d", "CausalConv1d",
    "maxpool2d", "MaxPool2d",
    "avgpool2d", "AvgPool2d",
    "deconv2d", "deconv3d",

    // ── SSM recurrence (complex stateful ops -- deferred to runtime) ──
    "gated_delta_rule", "GatedDeltaRule",
    "mamba2_ssm", "Mamba2SSM", "Mamba2Ssm",

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

// ─── Native-deferred ops ────────────────────────────────────────────────────
// These ops are in the mappable list (for canFuseSegment coverage), but are
// too complex or stateful to decompose into OV opset. They run natively via
// the NativeSlotExecutor callback during mixed-segment execution.

bool OpenVinoGraphBackend::isNativeDeferredOp(const std::string& opName) {
  // Only genuinely sequential/stateful ops that CAN'T be decomposed into a
  // static OV graph. Everything else should be properly implemented.
  static const std::unordered_set<std::string> deferred = {
    // Sequential SSM recurrence: state[t] depends on state[t-1]. No OV scan op.
    "gated_delta_rule", "GatedDeltaRule",
    "mamba2_ssm", "Mamba2SSM", "Mamba2Ssm",
    "rope", "Rope", "fused_rope", "FusedRope",
    "dot_product_attention", "DotProductAttention",
    "dot_product_attention_v2", "DotProductAttentionV2",
    "multi_head_attention", "MultiHeadAttention",
    "onnx_multi_head_attention", "OnnxMultiHeadAttention",
  };
  return deferred.count(opName) > 0;
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
    if (isOpenVinoMappable(slots[i].ident.opName)) {
      mappableOps++;
    } else {
      DSP_DIAG(BACKEND, "OpenVinoGraphBackend::canFuseSegment: unmappable op '%s' at slot %d",
               slots[i].ident.opName.c_str(), i);
    }
  }

  // Accept if all ops are either OV-mappable or native-deferred.
  // Need at least 1 OV-compilable op (non-deferred) to justify the segment.
  int ovCompilableOps = 0;
  for (int i = start; i <= end; i++) {
    if (isOpenVinoMappable(slots[i].ident.opName) && !isNativeDeferredOp(slots[i].ident.opName)) {
      ovCompilableOps++;
    }
  }
  bool canFuse = mappableOps == totalOps && ovCompilableOps >= 1;
  DSP_DIAG(SEGMENT, "OpenVinoGraphBackend::canFuseSegment [%d-%d]: mappable=%d/%d ovCompilable=%d canFuse=%s",
           start, end, mappableOps, totalOps, ovCompilableOps, canFuse ? "true" : "false");
  return canFuse;
}

// ─── computeIslandTopologyHash ──────────────────────────────────────────────
// Produces a slot-index-agnostic hash from the op sequence + input shapes.
// Segments from different transformer layers with the same topology will
// hash identically, letting them share one CompiledModel.

uint64_t OpenVinoGraphBackend::computeIslandTopologyHash(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a offset basis
  auto mix = [&](uint64_t v) {
    h ^= v;
    h *= 0x100000001b3ULL;  // FNV-1a prime
  };

  int numSlots = endSlot - startSlot + 1;
  mix(static_cast<uint64_t>(numSlots));

  for (int s = startSlot; s <= endSlot; s++) {
    // Hash op name (not slot index — so identical layers match)
    const std::string& opName = slots[s].ident.opName;
    for (char c : opName) mix(static_cast<uint64_t>(c));
    mix(0xFF);  // separator

    // Hash number of inputs and outputs
    mix(static_cast<uint64_t>(slots[s].wiring.numInputs));
    mix(static_cast<uint64_t>(slots[s].wiring.numOutputs));

    // Hash input shapes (rank + dims) — slot-index-agnostic
    for (int i = 0; i < slots[s].wiring.numInputs; i++) {
      int srcIdx = slots[s].wiring.inputSourceIndices[i];
      NDArray* arr = nullptr;
      if (srcIdx < 0) {
        int extIdx = -(srcIdx + 1);
        if (extIdx < numExternalInputs) arr = externalInputs[extIdx];
      } else if (srcIdx < totalOutputSlots) {
        arr = outputSlots[srcIdx];
      }
      if (arr) {
        mix(static_cast<uint64_t>(arr->rankOf()));
        for (int d = 0; d < arr->rankOf(); d++) {
          mix(static_cast<uint64_t>(arr->sizeAt(d)));
        }
        mix(static_cast<uint64_t>(arr->dataType()));
      }
    }

    // Hash integer args (affect op behavior e.g. strided_slice masks)
    for (int a = 0; a < slots[s].args.numIArgs; a++) {
      mix(static_cast<uint64_t>(slots[s].args.iArgs[a]));
    }
  }

  return h;
}

// ─── buildIsland ────────────────────────────────────────────────────────────
// Builds an OV model from a contiguous range of OV-compilable slots.

OpenVinoGraphBackend::OvIsland OpenVinoGraphBackend::buildIsland(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    int totalSlots,
    const int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  OvIsland result;
  result.startSlot = startSlot;
  result.endSlot = endSlot;

  // Track tensor nodes by their source index (slot output index or external input)
  // Key: >=0 = outputSlot index, <0 = -(externalInputIndex+1)
  std::unordered_map<int, ov::Output<ov::Node>> tensorMap;

  // Collect which slot outputs are produced within this segment
  std::unordered_set<int> segmentOutputs;
  for (int s = startSlot; s <= endSlot; s++) {
    for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
      segmentOutputs.insert(slots[s].wiring.outputSlotIndices[o]);
    }
  }

  // Scan for concat ops in the segment and collect which (srcIdx, dim) pairs
  // need dynamic dims. Concat inputs on the concat axis may have different sizes
  // (e.g. KV cache [1,3,0,64] vs new token [1,3,679,64]) — marking that axis
  // dynamic prevents OpenVINO's concat shape inference from failing.
  std::unordered_map<int, std::unordered_set<int>> dynamicDims;  // srcIdx -> set of dim indices
  for (int s = startSlot; s <= endSlot; s++) {
    const std::string& op = slots[s].ident.opName;
    if (op == "concat" || op == "Concat" || op == "concat_v2") {
      int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
      for (int inp = 0; inp < slots[s].wiring.numInputs; inp++) {
        int srcIdx = slots[s].wiring.inputSourceIndices[inp];
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
    for (int inp = 0; inp < slots[s].wiring.numInputs; inp++) {
      int srcIdx = slots[s].wiring.inputSourceIndices[inp];
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
        // Use actual shapes for static dims (best performance), with these dims
        // marked dynamic:
        //  1. Concat axes (collected in dynamicDims above)
        //  2. Any dim where compile-time value is 0 (empty placeholder) — these
        //     are typically uninitialized buffers that get populated at runtime
        //  3. If ANY dim of an input is 0, mark ALL dims of that input dynamic
        //     because the whole tensor is "uninitialized" and may grow.
        auto& dynSet = dynamicDims[srcIdx];
        bool allDynamic = dynSet.count(-1) > 0;
        ov::PartialShape pshape;
        ov::element::Type dtype = ov::element::f32;
        int rank = 0;
        const LongType* shapeSrc = nullptr;
        if (arr) {
          rank = arr->rankOf();
          dtype = mapDataType(arr->dataType());
          // Detect empty tensor — any dim == 0
          for (int d = 0; d < rank; d++) {
            if (arr->sizeAt(d) == 0) { allDynamic = true; break; }
          }
        } else if (!isExternal && srcIdx >= 0 && srcIdx < totalOutputSlots) {
          auto& srcSlot = slots[srcIdx];
          if (!srcSlot.shapeCache.cachedOutputShapes.empty() && srcSlot.shapeCache.cachedOutputShapes[0] != nullptr) {
            shapeSrc = srcSlot.shapeCache.cachedOutputShapes[0];
            rank = shape::rank(shapeSrc);
            dtype = mapDataType(ArrayOptions::dataType(shapeSrc));
            for (int d = 0; d < rank; d++) {
              if (shape::shapeOf(shapeSrc)[d] == 0) { allDynamic = true; break; }
            }
          }
        }
        for (int d = 0; d < rank; d++) {
          int64_t dimVal = arr ? arr->sizeAt(d) : (shapeSrc ? shape::shapeOf(shapeSrc)[d] : -1);
          if (dimVal <= 0 || allDynamic || dynSet.count(d) > 0) {
            pshape.push_back(ov::Dimension::dynamic());
          } else {
            pshape.push_back(dimVal);
          }
        }
        // rank == 0 → empty pshape (true scalar)
        // Use the dtype we already computed (may be from cachedOutputShapes when arr is null)
        //
        // ISA-based FP16 promotion: if the CPU lacks native FP16 support
        // (AVX512-FP16 / AMX-FP16), promote f16 parameters to f32 so the
        // entire graph computes in f32. The output boundary (Result nodes)
        // already handles converting back to the expected NDArray type.
        auto paramDtype = dtype;
        if (!kCpuHasNativeFp16 && dtype == ov::element::f16) {
          paramDtype = ov::element::f32;
        }
        auto param = std::make_shared<ov::op::v0::Parameter>(paramDtype, pshape);
        params.push_back(param);
        tensorMap[srcIdx] = param->output(0);
        inputSourceMap.push_back(srcIdx);
        DSP_DIAG(COMPILE, "OpenVINO: buildIsland[%d-%d] param srcIdx=%d shape=%s dtype=%s%s allDynamic=%d hasArr=%d",
                 startSlot, endSlot, srcIdx, pshape.to_string().c_str(),
                 paramDtype.get_type_name().c_str(),
                 (paramDtype != dtype) ? " (promoted from f16 — no native FP16 ISA)" : "",
                 allDynamic, (arr != nullptr));
      }
    }
  }

  // Build OV ops for each slot
  ov::ResultVector results;
  std::vector<int> outputSourceMap;  // maps result index -> outputSlot index

  for (int s = startSlot; s <= endSlot; s++) {
    const std::string& opName = slots[s].ident.opName;

    if (!isOpenVinoMappable(opName)) {
      DSP_DIAG(COMPILE, "OpenVINO: skipping unmappable op '%s' at slot %d", opName.c_str(), s);
      continue;
    }

    // Native-deferred ops: skip silently — compileSegment handled them as NativeRange
    if (isNativeDeferredOp(opName)) {
      continue;
    }

    // Gather input nodes. If any input is missing from tensorMap, that means
    // an upstream slot failed to compile. We do NOT create on-the-fly Parameters
    // for missing inputs — that just creates ghost Parameters that have no source
    // at runtime and cascade into broken models. Fail the segment cleanly so DSP
    // can pick another backend.
    std::vector<ov::Output<ov::Node>> inputs;
    {
      std::string inputInfo;
      for (int inp = 0; inp < slots[s].wiring.numInputs; inp++) {
        int srcIdx = slots[s].wiring.inputSourceIndices[inp];
        if (!inputInfo.empty()) inputInfo += ", ";
        inputInfo += "src" + std::to_string(srcIdx);
        auto it2 = tensorMap.find(srcIdx);
        if (it2 != tensorMap.end()) {
          inputInfo += "=" + it2->second.get_partial_shape().to_string();
        } else {
          inputInfo += "=MISSING";
        }
      }
      DSP_DIAG(COMPILE, "OpenVINO: buildIsland slot %d op '%s' inputs=[%s] numOut=%d outSlots=[%s]",
               s, opName.c_str(), inputInfo.c_str(), slots[s].wiring.numOutputs,
               [&]() -> std::string {
                 std::string r;
                 for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
                   if (!r.empty()) r += ",";
                   r += std::to_string(slots[s].wiring.outputSlotIndices[o]);
                 }
                 return r;
               }().c_str());
    }
    for (int inp = 0; inp < slots[s].wiring.numInputs; inp++) {
      int srcIdx = slots[s].wiring.inputSourceIndices[inp];
      auto it = tensorMap.find(srcIdx);
      if (it == tensorMap.end()) {
        std::string msg = "OpenVINO: slot " + std::to_string(s) + " op '" + opName +
                          "' missing input source " + std::to_string(srcIdx) +
                          " (upstream slot must have failed to compile)";
        DSP_DIAG(COMPILE, "%s", msg.c_str());
        THROW_EXCEPTION(msg.c_str());
      }
      inputs.push_back(it->second);
    }

    {
      std::shared_ptr<ov::Node> node;

      // OpenVINO elementwise ops require matching input types.
      // Auto-cast mismatched binary inputs to the higher-precision type.
      // Harmonize binary input types: cast input[1] to match input[0]'s type.
      // input[0] is the "activation" (data flowing through the graph) and determines
      // the output type. Upcasting to f32 would poison all downstream ops and force
      // expensive conversions at every output boundary.
      auto harmonizeBinaryTypes = [](ov::Output<ov::Node>& a, ov::Output<ov::Node>& b) {
        auto ta = a.get_element_type();
        auto tb = b.get_element_type();
        if (ta != tb) {
          // Cast b to match a (activation-dominant: output type follows input[0])
          b = std::make_shared<ov::op::v0::Convert>(b, ta)->output(0);
        }
      };

      // No per-op try/catch — failures propagate up to compileSegment which
      // returns false, and DSP picks another backend at the segment level.
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
          double alpha = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1.0;
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
          double minVal = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : -std::numeric_limits<float>::max();
          double maxVal = (slots[s].args.numTArgs > 1) ? slots[s].args.tArgs[1] : std::numeric_limits<float>::max();
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
          double alpha = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 0.01;
          auto slope = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(alpha)});
          node = std::make_shared<ov::op::v0::PRelu>(inputs[0], slope);
        }
      } else if (opName == "selu" || opName == "Selu") {
        if (inputs.size() >= 1) {
          double alpha = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1.6732632423543772;
          double lambda = (slots[s].args.numTArgs > 1) ? slots[s].args.tArgs[1] : 1.0507009873554805;
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
          double alpha = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1.0;
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
          double theta = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1.0;
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
      // Create the scalar constant in the SAME dtype as the input to avoid
      // mixed-type validation failures (e.g. f16 input + f32 constant).
      else if (opName == "add_scalar") {
        if (inputs.size() >= 1 && slots[s].args.numTArgs > 0) {
          auto inType = inputs[0].get_element_type();
          auto scalar = ov::op::v0::Constant::create(inType, {}, {static_cast<float>(slots[s].args.tArgs[0])});
          node = std::make_shared<ov::op::v1::Add>(inputs[0], scalar);
        }
      } else if (opName == "subtract_scalar" || opName == "sub_scalar") {
        if (inputs.size() >= 1 && slots[s].args.numTArgs > 0) {
          auto inType = inputs[0].get_element_type();
          auto scalar = ov::op::v0::Constant::create(inType, {}, {static_cast<float>(slots[s].args.tArgs[0])});
          node = std::make_shared<ov::op::v1::Subtract>(inputs[0], scalar);
        }
      } else if (opName == "multiply_scalar" || opName == "mul_scalar") {
        if (inputs.size() >= 1 && slots[s].args.numTArgs > 0) {
          auto inType = inputs[0].get_element_type();
          auto scalar = ov::op::v0::Constant::create(inType, {}, {static_cast<float>(slots[s].args.tArgs[0])});
          node = std::make_shared<ov::op::v1::Multiply>(inputs[0], scalar);
        }
      } else if (opName == "divide_scalar" || opName == "div_scalar") {
        if (inputs.size() >= 1 && slots[s].args.numTArgs > 0) {
          auto inType = inputs[0].get_element_type();
          auto scalar = ov::op::v0::Constant::create(inType, {}, {static_cast<float>(slots[s].args.tArgs[0])});
          node = std::make_shared<ov::op::v1::Divide>(inputs[0], scalar);
        }
      }

      // ── MatMul ──
      // For MatMul: cast input[1] (weight) to match input[0] (activation) type.
      // This ensures the output type equals the activation type — no downstream
      // conversion needed.  harmonizeBinaryTypes upcasts to the WIDER type which
      // would force f32 outputs when only the weight is f32, requiring expensive
      // post-matmul conversion back to f16.
      else if (opName == "matmul" || opName == "MatMul" || opName == "mmul" ||
               opName == "batch_matmul" || opName == "BatchMatMul" ||
               opName == "tensormmul" || opName == "TensorMmul" ||
               opName == "batched_gemm" || opName == "BatchedGemm") {
        if (inputs.size() >= 2) {
          auto actType = inputs[0].get_element_type();
          if (inputs[1].get_element_type() != actType) {
            inputs[1] = std::make_shared<ov::op::v0::Convert>(inputs[1], actType)->output(0);
          }
          bool transpA = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          bool transpB = (slots[s].args.numBArgs > 1) ? slots[s].args.bArgs[1] : false;
          node = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[1], transpA, transpB);
        }
      } else if (opName == "xw_plus_b" || opName == "XwPlusB") {
        // Compose: MatMul(x, w) + b
        if (inputs.size() >= 3) {
          auto actType = inputs[0].get_element_type();
          if (inputs[1].get_element_type() != actType) {
            inputs[1] = std::make_shared<ov::op::v0::Convert>(inputs[1], actType)->output(0);
          }
          auto mm = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[1], false, false);
          node = std::make_shared<ov::op::v1::Add>(mm->output(0), inputs[2]);
        }
      } else if (opName == "fused_gemm_swiglu" || opName == "FusedGemmSwiglu") {
        // Compose: silu(x @ W_gate) * (x @ W_up)
        // Input 0: x [M, K], Input 1: W_gate [K, N], Input 2: W_up [K, N]
        if (inputs.size() >= 3) {
          auto actType = inputs[0].get_element_type();
          if (inputs[1].get_element_type() != actType) {
            inputs[1] = std::make_shared<ov::op::v0::Convert>(inputs[1], actType)->output(0);
          }
          auto gate = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[1], false, false);
          if (inputs[2].get_element_type() != actType) {
            inputs[2] = std::make_shared<ov::op::v0::Convert>(inputs[2], actType)->output(0);
          }
          auto up = std::make_shared<ov::op::v0::MatMul>(inputs[0], inputs[2], false, false);
          auto silu = std::make_shared<ov::op::v4::Swish>(gate->output(0));
          node = std::make_shared<ov::op::v1::Multiply>(silu->output(0), up->output(0));
        }
      }

      // ── Softmax ──
      else if (opName == "softmax" || opName == "Softmax") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : -1;
          node = std::make_shared<ov::op::v8::Softmax>(inputs[0], axis);
        }
      } else if (opName == "log_softmax" || opName == "LogSoftmax") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : -1;
          auto sm = std::make_shared<ov::op::v8::Softmax>(inputs[0], axis);
          node = std::make_shared<ov::op::v0::Log>(sm->output(0));
        }
      }
      // ── Normalization (LayerNorm, RMSNorm, BatchNorm) ──
      else if (opName == "layer_norm" || opName == "LayerNorm" || opName == "layer_normalization" ||
               opName == "fused_layer_norm" || opName == "FusedLayerNorm") {
        // Compose: MVN(x, axes, normalize_variance=true, eps) * scale + bias
        if (inputs.size() >= 1) {
          double eps = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1e-5;
          // Normalize over last axis by default
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> axes;
          if (slots[s].args.numIArgs > 0) {
            for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
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
        // Compute in FP32 to avoid HALF overflow on sum-of-squares.
        if (inputs.size() >= 1) {
          double eps = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1e-5;
          auto origType = inputs[0].get_element_type();

          // Promote to FP32
          auto xF32 = inputs[0];
          if (origType != ov::element::f32)
            xF32 = std::make_shared<ov::op::v0::Convert>(xF32, ov::element::f32)->output(0);

          int rank = static_cast<int>(xF32.get_partial_shape().rank().get_length());
          std::vector<int64_t> axes;
          if (slots[s].args.numIArgs > 0) {
            for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          } else {
            axes.push_back(rank - 1);
          }
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          auto x_sq = std::make_shared<ov::op::v1::Multiply>(xF32, xF32);
          auto mean_sq = std::make_shared<ov::op::v1::ReduceMean>(x_sq->output(0), axes_const, true);
          auto eps_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(eps)});
          auto mean_eps = std::make_shared<ov::op::v1::Add>(mean_sq->output(0), eps_const);
          auto rms = std::make_shared<ov::op::v0::Sqrt>(mean_eps->output(0));
          auto normed = std::make_shared<ov::op::v1::Divide>(xF32, rms->output(0));

          std::shared_ptr<ov::Node> result;
          if (inputs.size() >= 2) {
            auto gammaF32 = inputs[1];
            if (gammaF32.get_element_type() != ov::element::f32)
              gammaF32 = std::make_shared<ov::op::v0::Convert>(gammaF32, ov::element::f32)->output(0);
            result = std::make_shared<ov::op::v1::Multiply>(normed->output(0), gammaF32);
          } else {
            result = normed;
          }

          // Cast back to original dtype
          if (origType != ov::element::f32) {
            node = std::make_shared<ov::op::v0::Convert>(result->output(0), origType);
          } else {
            node = result;
          }
        }
      } else if (opName == "rms_norm_linear" || opName == "RmsNormLinear") {
        // Compose: matmul(rms_norm(x, gamma, eps), W)
        // All computation in FP32 to avoid HALF overflow (sum-of-squares for
        // 1024-dim exceeds 65504) and HALF matmul producing zeros on CPUs
        // without AMX-FP16.
        // Input 0: x, Input 1: gamma, Input 2: W
        if (inputs.size() >= 3) {
          double eps = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1e-6;

          // Promote inputs to FP32
          auto xF32 = inputs[0];
          if (xF32.get_element_type() != ov::element::f32)
            xF32 = std::make_shared<ov::op::v0::Convert>(xF32, ov::element::f32)->output(0);
          auto gammaF32 = inputs[1];
          if (gammaF32.get_element_type() != ov::element::f32)
            gammaF32 = std::make_shared<ov::op::v0::Convert>(gammaF32, ov::element::f32)->output(0);
          auto wF32 = inputs[2];
          if (wF32.get_element_type() != ov::element::f32)
            wF32 = std::make_shared<ov::op::v0::Convert>(wF32, ov::element::f32)->output(0);

          int rank = static_cast<int>(xF32.get_partial_shape().rank().get_length());
          std::vector<int64_t> axes = {rank - 1};
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);

          // RMSNorm in FP32: x / sqrt(mean(x^2) + eps) * gamma
          auto x_sq = std::make_shared<ov::op::v1::Multiply>(xF32, xF32);
          auto mean_sq = std::make_shared<ov::op::v1::ReduceMean>(x_sq->output(0), axes_const, true);
          auto eps_const = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(eps)});
          auto mean_eps = std::make_shared<ov::op::v1::Add>(mean_sq->output(0), eps_const);
          auto rms = std::make_shared<ov::op::v0::Sqrt>(mean_eps->output(0));
          auto normed = std::make_shared<ov::op::v1::Divide>(xF32, rms->output(0));
          auto scaled = std::make_shared<ov::op::v1::Multiply>(normed->output(0), gammaF32);

          // Linear in FP32: scaled @ W
          auto matmulOut = std::make_shared<ov::op::v0::MatMul>(scaled->output(0), wF32, false, false);

          // Cast back to input dtype if needed
          if (inputs[0].get_element_type() != ov::element::f32) {
            node = std::make_shared<ov::op::v0::Convert>(matmulOut->output(0), inputs[0].get_element_type());
          } else {
            node = matmulOut;
          }
        }
      } else if (opName == "batchnorm" || opName == "BatchNorm" || opName == "batchnorm_inference" || opName == "batch_norm") {
        // BatchNormInference: input, gamma, beta, mean, variance
        if (inputs.size() >= 5) {
          double eps = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 1e-5;
          node = std::make_shared<ov::op::v5::BatchNormInference>(
              inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps);
        }
      }

      // ── Reduction ──
      else if (opName == "reduce_sum" || opName == "ReduceSum" || opName == "sum" || opName == "Sum") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceSum>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_max" || opName == "ReduceMax" || opName == "max") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceMax>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_mean" || opName == "ReduceMean" || opName == "mean" || opName == "Mean") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceMean>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_min" || opName == "ReduceMin" || opName == "min") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceMin>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_prod" || opName == "ReduceProd" || opName == "prod" || opName == "Prod") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          node = std::make_shared<ov::op::v1::ReduceProd>(inputs[0], axes_const, keepDims);
        }
      } else if (opName == "reduce_norm1" || opName == "ReduceNorm1" || opName == "norm1") {
        // Compose: ReduceSum(Abs(x), axes)
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          auto abs_x = std::make_shared<ov::op::v0::Abs>(inputs[0]);
          node = std::make_shared<ov::op::v1::ReduceSum>(abs_x->output(0), axes_const, keepDims);
        }
      } else if (opName == "reduce_norm2" || opName == "ReduceNorm2" || opName == "norm2") {
        // Compose: Sqrt(ReduceSum(x*x, axes))
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          auto x_sq = std::make_shared<ov::op::v1::Multiply>(inputs[0], inputs[0]);
          auto sum_sq = std::make_shared<ov::op::v1::ReduceSum>(x_sq->output(0), axes_const, keepDims);
          node = std::make_shared<ov::op::v0::Sqrt>(sum_sq->output(0));
        }
      } else if (opName == "reduce_logsumexp" || opName == "ReduceLogSumExp") {
        // Compose: Log(ReduceSum(Exp(x), axes))
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          auto exp_x = std::make_shared<ov::op::v0::Exp>(inputs[0]);
          auto sum_exp = std::make_shared<ov::op::v1::ReduceSum>(exp_x->output(0), axes_const, keepDims);
          node = std::make_shared<ov::op::v0::Log>(sum_exp->output(0));
        }
      } else if (opName == "reduce_variance" || opName == "ReduceVariance") {
        // Compose: mean((x - mean(x))^2)
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
          auto mean_x = std::make_shared<ov::op::v1::ReduceMean>(inputs[0], axes_const, true);
          auto diff = std::make_shared<ov::op::v1::Subtract>(inputs[0], mean_x->output(0));
          auto diff_sq = std::make_shared<ov::op::v1::Multiply>(diff->output(0), diff->output(0));
          node = std::make_shared<ov::op::v1::ReduceMean>(diff_sq->output(0), axes_const, keepDims);
        }
      } else if (opName == "reduce_stdev" || opName == "ReduceStdev") {
        // Compose: sqrt(variance)
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
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
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
          if (axes.empty()) axes.push_back(-1);
          auto axes_const = ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes);
          bool keepDims = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : false;
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
          if (slots[s].args.numIArgs > 0) {
            int64_t first = slots[s].args.iArgs[0];
            if (first < 0 && (first == -99 || first == -102)) {
              startIdx = 1;  // skip ordering marker
            }
          }
          for (int a = startIdx; a < slots[s].args.numIArgs; a++) targetShape.push_back(slots[s].args.iArgs[a]);
          if (targetShape.empty() && slots[s].wiring.numOutputs > 0) {
            // Use output slot shape if available
            int outIdx = slots[s].wiring.outputSlotIndices[0];
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
                     s, slots[s].args.numIArgs, startIdx, shapeStr.c_str(),
                     inputs[0].get_partial_shape().to_string().c_str());
            auto shape_const = ov::op::v0::Constant::create(
                ov::element::i64, {targetShape.size()}, targetShape);
            // allow_special_zero=true: required for dynamic-shape models where
            // input dims may be unknown at graph construction time. Without this,
            // OV's reshape validation fails when it can't verify element counts.
            node = std::make_shared<ov::op::v1::Reshape>(inputs[0], shape_const, true);
          }
        }
      } else if (opName == "permute" || opName == "Permute" || opName == "Transpose" || opName == "transpose") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> perm;
          for (int a = 0; a < slots[s].args.numIArgs; a++) perm.push_back(slots[s].args.iArgs[a]);
          // nd4j permute op takes the permutation from EITHER iArgs OR input[1] (a tensor).
          // If iArgs is empty and we have a second input, use it as the order tensor.
          // OpenVINO Transpose accepts the order as a runtime Output<Node>.
          if (perm.empty() && inputs.size() >= 2) {
            node = std::make_shared<ov::op::v1::Transpose>(inputs[0], inputs[1]);
          } else {
            if (perm.empty()) {
              // No iArgs and no order tensor = simple transpose: reverse dimensions
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
        }
      } else if (opName == "squeeze" || opName == "Squeeze") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> axes;
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
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
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
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
          if (slots[s].args.numIArgs > 0) {
            int64_t first = slots[s].args.iArgs[0];
            if (first < 0 && (first == -99 || first == -102)) {
              startIdx = 1;
            }
          }
          for (int a = startIdx; a < slots[s].args.numIArgs; a++) targetShape.push_back(slots[s].args.iArgs[a]);
          if (targetShape.empty() && slots[s].wiring.numOutputs > 0) {
            int outIdx = slots[s].wiring.outputSlotIndices[0];
            if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
              for (int d = 0; d < outputSlots[outIdx]->rankOf(); d++) {
                targetShape.push_back(outputSlots[outIdx]->sizeAt(d));
              }
            }
          }
          if (!targetShape.empty()) {
            auto shape_const = ov::op::v0::Constant::create(
                ov::element::i64, {targetShape.size()}, targetShape);
            node = std::make_shared<ov::op::v1::Reshape>(inputs[0], shape_const, true);
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
          int diag = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
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
          int diag = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
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
          for (int a = 0; a < slots[s].args.numIArgs; a++) targetShape.push_back(slots[s].args.iArgs[a]);
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
          node = std::make_shared<ov::op::v1::Reshape>(inputs[0], target_shape->output(0), true);
        }
      }

      // ── Data movement ──
      else if (opName == "gather" || opName == "Gather") {
        if (inputs.size() >= 2) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          node = std::make_shared<ov::op::v8::Gather>(inputs[0], inputs[1], axis_const);
        }
      } else if (opName == "gather_nd" || opName == "GatherNd") {
        if (inputs.size() >= 2) {
          int batchDims = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
          node = std::make_shared<ov::op::v8::GatherND>(inputs[0], inputs[1], batchDims);
        }
      } else if (opName == "scatter_nd" || opName == "ScatterNd" || opName == "ScatterNdUpdate") {
        if (inputs.size() >= 3) {
          node = std::make_shared<ov::op::v3::ScatterNDUpdate>(inputs[0], inputs[1], inputs[2]);
        }
      } else if (opName == "concat" || opName == "Concat") {
        if (inputs.size() >= 2) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
          // Log input shapes for diagnosis
          for (size_t ci = 0; ci < inputs.size(); ci++) {
            DSP_DIAG(COMPILE, "OpenVINO: concat slot %d input[%zu] shape=%s",
                     s, ci, inputs[ci].get_partial_shape().to_string().c_str());
          }
          // Normalize all inputs to the same rank by prepending size-1 dimensions.
          // OpenVINO Concat requires all inputs to have the same rank; if some
          // inputs are scalars (rank 0) or lower rank, reshape them up front.
          int maxRankConcat = 0;
          for (size_t ci = 0; ci < inputs.size(); ci++) {
            auto ps = inputs[ci].get_partial_shape();
            if (ps.rank().is_static()) {
              int r = static_cast<int>(ps.rank().get_length());
              if (r > maxRankConcat) maxRankConcat = r;
            }
          }
          ov::OutputVector concatInputs;
          for (size_t ci = 0; ci < inputs.size(); ci++) {
            auto ps = inputs[ci].get_partial_shape();
            int r = ps.rank().is_static() ? static_cast<int>(ps.rank().get_length()) : maxRankConcat;
            if (r < maxRankConcat) {
              // Prepend (maxRankConcat - r) dimensions of size 1 via successive Unsqueeze(axis=0)
              ov::Output<ov::Node> promoted = inputs[ci];
              for (int d = 0; d < (maxRankConcat - r); d++) {
                auto leading_ax = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
                auto usq = std::make_shared<ov::op::v0::Unsqueeze>(promoted, leading_ax);
                promoted = usq->output(0);
              }
              concatInputs.push_back(promoted);
            } else {
              concatInputs.push_back(inputs[ci]);
            }
          }
          node = std::make_shared<ov::op::v0::Concat>(concatInputs, axis);
        }
      } else if (opName == "split" || opName == "Split") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
          int numSplits = (slots[s].args.numIArgs > 1) ? static_cast<int>(slots[s].args.iArgs[1]) : slots[s].wiring.numOutputs;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          node = std::make_shared<ov::op::v1::Split>(inputs[0], axis_const, numSplits);
        }
      } else if (opName == "slice" || opName == "Slice" || opName == "strided_slice" || opName == "StridedSlice") {
        if (inputs.size() >= 1 && slots[s].args.numIArgs >= 2) {
          // nd4j strided_slice iArgs layout:
          //   [beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask,
          //    begin_0..begin_N, end_0..end_N, strides_0..strides_N]
          // First 5 ints are mask flags, then begin/end/strides follow.
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());

          // Parse masks from first 5 iArgs (if available)
          // nd4j strided_slice iArgs order: [beginMask, ellipsisMask, endMask, newAxisMask, shrinkAxisMask]
          // (see strided_slice.cpp: INT_ARG(0)=begin_mask, INT_ARG(1)=ellipsis_mask, INT_ARG(2)=end_mask)
          int64_t beginMaskBits = (slots[s].args.numIArgs > 0) ? slots[s].args.iArgs[0] : 0;
          int64_t ellipsisMaskBits = (slots[s].args.numIArgs > 1) ? slots[s].args.iArgs[1] : 0;
          int64_t endMaskBits = (slots[s].args.numIArgs > 2) ? slots[s].args.iArgs[2] : 0;
          int64_t newAxisMaskBits = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 0;
          int64_t shrinkAxisMaskBits = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 0;

          int dataStart = 5;
          int remainingArgs = slots[s].args.numIArgs - dataStart;
          int argsPerDim = remainingArgs / 3;
          if (argsPerDim <= 0) argsPerDim = rank;

          std::vector<int64_t> begin, end, strides;
          for (int d = 0; d < argsPerDim && d < rank; d++) {
            begin.push_back(slots[s].args.iArgs[dataStart + d]);
            end.push_back(slots[s].args.iArgs[dataStart + argsPerDim + d]);
            int64_t stride = (dataStart + 2 * argsPerDim + d < slots[s].args.numIArgs)
                              ? slots[s].args.iArgs[dataStart + 2 * argsPerDim + d] : 1;
            if (stride == 0) stride = 1;  // OpenVINO rejects stride=0
            strides.push_back(stride);
          }

          // Convert bitmask to per-dim vectors for OV
          std::vector<int64_t> beginMask(begin.size(), 0);
          std::vector<int64_t> endMask(end.size(), 0);
          std::vector<int64_t> newAxisMask(begin.size(), 0);
          std::vector<int64_t> shrinkAxisMask(begin.size(), 0);
          std::vector<int64_t> ellipsisMask(begin.size(), 0);
          for (size_t d = 0; d < begin.size(); d++) {
            if (beginMaskBits & (1LL << d)) beginMask[d] = 1;
            if (endMaskBits & (1LL << d)) endMask[d] = 1;
            if (newAxisMaskBits & (1LL << d)) newAxisMask[d] = 1;
            if (shrinkAxisMaskBits & (1LL << d)) shrinkAxisMask[d] = 1;
            if (ellipsisMaskBits & (1LL << d)) ellipsisMask[d] = 1;
          }

          {
            std::string rawArgs = "[";
            for (int ia = 0; ia < slots[s].args.numIArgs; ia++) {
              if (ia > 0) rawArgs += ",";
              rawArgs += std::to_string(slots[s].args.iArgs[ia]);
            }
            rawArgs += "]";
            std::string beginStr = "[", endStr = "[", stridesStr = "[";
            for (size_t i = 0; i < begin.size(); i++) {
              if (i > 0) { beginStr += ","; endStr += ","; stridesStr += ","; }
              beginStr += std::to_string(begin[i]);
              endStr += std::to_string(end[i]);
              stridesStr += std::to_string(strides[i]);
            }
            beginStr += "]"; endStr += "]"; stridesStr += "]";
            DSP_DIAG(COMPILE, "OpenVINO: strided_slice slot %d: input=%s begin=%s end=%s strides=%s "
                     "beginMask=%lld endMask=%lld shrinkMask=%lld newAxisMask=%lld rawIArgs=%s",
                     s, inputs[0].get_partial_shape().to_string().c_str(),
                     beginStr.c_str(), endStr.c_str(), stridesStr.c_str(),
                     (long long)beginMaskBits, (long long)endMaskBits,
                     (long long)shrinkAxisMaskBits, (long long)newAxisMaskBits,
                     rawArgs.c_str());
          }

          auto begin_const = ov::op::v0::Constant::create(ov::element::i64, {begin.size()}, begin);
          auto end_const = ov::op::v0::Constant::create(ov::element::i64, {end.size()}, end);
          auto strides_const = ov::op::v0::Constant::create(ov::element::i64, {strides.size()}, strides);
          // OV requires ellipsis_mask to have at most 1 set bit. If no ellipsis bits are
          // set, pass an empty vector (not a vector of zeros — OV checks vector size for
          // the "at most 1 ellipsis" constraint, which converts the AxisSet from the mask).
          // Same for new_axis and shrink — pass empty when unused to avoid confusing OV's
          // mask-to-AxisSet conversion.
          std::vector<int64_t> newAxisMaskFinal = (newAxisMaskBits != 0) ? newAxisMask : std::vector<int64_t>{};
          std::vector<int64_t> shrinkAxisMaskFinal = (shrinkAxisMaskBits != 0) ? shrinkAxisMask : std::vector<int64_t>{};
          std::vector<int64_t> ellipsisMaskFinal = (ellipsisMaskBits != 0) ? ellipsisMask : std::vector<int64_t>{};
          node = std::make_shared<ov::op::v1::StridedSlice>(
              inputs[0], begin_const, end_const, strides_const,
              beginMask, endMask, newAxisMaskFinal, shrinkAxisMaskFinal, ellipsisMaskFinal);
        }
      } else if (opName == "tile" || opName == "Tile" || opName == "repeat" || opName == "Repeat") {
        if (inputs.size() >= 1) {
          std::vector<int64_t> repeats;
          for (int a = 0; a < slots[s].args.numIArgs; a++) repeats.push_back(slots[s].args.iArgs[a]);
          if (!repeats.empty()) {
            auto repeats_const = ov::op::v0::Constant::create(
                ov::element::i64, {repeats.size()}, repeats);
            node = std::make_shared<ov::op::v0::Tile>(inputs[0], repeats_const);
          }
        }
      } else if (opName == "split_v" || opName == "SplitV") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          // Split lengths from remaining iArgs
          std::vector<int64_t> splitLengths;
          for (int a = 1; a < slots[s].args.numIArgs; a++) splitLengths.push_back(slots[s].args.iArgs[a]);
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
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
          auto axis_const = ov::op::v0::Constant::create(ov::element::i64, {}, {axis});
          int numSplits = slots[s].wiring.numOutputs;
          if (numSplits <= 0) numSplits = 1;
          auto split_node = std::make_shared<ov::op::v1::Split>(inputs[0], axis_const, numSplits);
          // For multi-output, wire each split output and squeeze
          if (numSplits > 1) {
            auto sq_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
            for (int o = 0; o < slots[s].wiring.numOutputs && o < numSplits; o++) {
              int outIdx = slots[s].wiring.outputSlotIndices[o];
              auto squeezed = std::make_shared<ov::op::v0::Squeeze>(split_node->output(o), sq_axes);
              tensorMap[outIdx] = squeezed->output(0);
            }
            continue;  // handled all outputs manually
          } else {
            auto sq_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
            node = std::make_shared<ov::op::v0::Squeeze>(split_node->output(0), sq_axes);
          }
        }
      } else if (opName == "stack" || opName == "Stack") {
        // Unsqueeze each input along axis, then concat.
        // All inputs must have the same rank before unsqueeze so that after
        // unsqueeze they all reach rank N+1.  When inputs have mixed ranks
        // (e.g. scalars mixed with rank-1 tensors) we first promote every
        // input to the common (maximum) rank by prepending leading size-1
        // dimensions, then apply the unsqueeze uniformly.
        if (inputs.size() >= 1) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;

          // 1. Find the maximum rank across all inputs.
          int maxRankStack = 0;
          for (size_t i = 0; i < inputs.size(); i++) {
            auto ps = inputs[i].get_partial_shape();
            if (ps.rank().is_static()) {
              int r = static_cast<int>(ps.rank().get_length());
              if (r > maxRankStack) maxRankStack = r;
            }
          }

          // 2. Promote lower-rank inputs to maxRankStack by prepending size-1 dims.
          ov::OutputVector promoted;
          for (size_t i = 0; i < inputs.size(); i++) {
            auto ps = inputs[i].get_partial_shape();
            int r = ps.rank().is_static() ? static_cast<int>(ps.rank().get_length()) : maxRankStack;
            ov::Output<ov::Node> cur = inputs[i];
            for (int d = 0; d < (maxRankStack - r); d++) {
              auto leading_ax = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
              auto usq = std::make_shared<ov::op::v0::Unsqueeze>(cur, leading_ax);
              cur = usq->output(0);
            }
            promoted.push_back(cur);
          }

          // 3. Unsqueeze all (now same-rank) inputs along the stack axis.
          auto ax_const = ov::op::v0::Constant::create(ov::element::i64, {1}, {axis});
          ov::OutputVector unsqueezed;
          for (size_t i = 0; i < promoted.size(); i++) {
            auto usq = std::make_shared<ov::op::v0::Unsqueeze>(promoted[i], ax_const);
            unsqueezed.push_back(usq->output(0));
          }

          // 4. Concatenate along the same axis.
          node = std::make_shared<ov::op::v0::Concat>(unsqueezed, axis);
        }
      } else if (opName == "pad" || opName == "Pad") {
        if (inputs.size() >= 1) {
          // pads_begin and pads_end from iArgs: first half = begin, second half = end
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> pads_begin, pads_end;
          if (slots[s].args.numIArgs >= 2 * rank) {
            for (int d = 0; d < rank; d++) {
              pads_begin.push_back(slots[s].args.iArgs[d]);
              pads_end.push_back(slots[s].args.iArgs[rank + d]);
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
            float padVal = (slots[s].args.numTArgs > 0) ? static_cast<float>(slots[s].args.tArgs[0]) : 0.0f;
            auto pv = ov::op::v0::Constant::create(ov::element::f32, {}, {padVal});
            node = std::make_shared<ov::op::v1::Pad>(inputs[0], pb, pe, pv, ov::op::PadMode::CONSTANT);
          }
        }
      } else if (opName == "mirror_pad" || opName == "MirrorPad") {
        if (inputs.size() >= 1) {
          int rank = static_cast<int>(inputs[0].get_partial_shape().rank().get_length());
          std::vector<int64_t> pads_begin, pads_end;
          // mode: 0 = REFLECT, 1 = SYMMETRIC
          auto mode = (slots[s].args.numIArgs > 2 * rank) ?
              ((slots[s].args.iArgs[2 * rank] == 1) ? ov::op::PadMode::SYMMETRIC : ov::op::PadMode::REFLECT)
              : ov::op::PadMode::REFLECT;
          if (slots[s].args.numIArgs >= 2 * rank) {
            for (int d = 0; d < rank; d++) {
              pads_begin.push_back(slots[s].args.iArgs[d]);
              pads_end.push_back(slots[s].args.iArgs[rank + d]);
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
          for (int a = 0; a < slots[s].args.numIArgs; a++) axes.push_back(slots[s].args.iArgs[a]);
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
          if (slots[s].args.numDArgs > 0) {
            targetType = mapDataType(slots[s].args.dArgs[0]);
          } else if (slots[s].args.numIArgs > 0) {
            // Cast op stores target type as iArg (FlatBuffersMapper byte encoding)
            targetType = mapDataType(static_cast<DataType>(slots[s].args.iArgs[0]));
          } else {
            // Fallback: same type as input (identity cast)
            targetType = inputs[0].get_element_type();
          }
          // ISA-based FP16 promotion: when CPU lacks native FP16, redirect
          // casts to f16 → f32 so the entire graph stays in f32.
          if (!kCpuHasNativeFp16 && targetType == ov::element::f16) {
            targetType = ov::element::f32;
          }
          node = std::make_shared<ov::op::v0::Convert>(inputs[0], targetType);
        }
      }

      // ── Identity / Assign ──
      else if (opName == "identity" || opName == "Identity" || opName == "assign" || opName == "Assign") {
        if (inputs.size() >= 1) {
          // Pass-through: wire input directly to output
          for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
            int outIdx = slots[s].wiring.outputSlotIndices[o];
            tensorMap[outIdx] = inputs[0];
          }
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
        // Range(start, stop, step) — from tArgs or inputs.
        // OpenVINO Range requires rank-0 (scalar) inputs. Squeeze any rank-1 size-1 inputs.
        // Output type follows input type (don't hardcode f32 — inputs may be i64).
        // Squeeze without axes removes all size-1 dims; for an already-scalar it's a no-op.
        auto toScalar = [](const ov::Output<ov::Node>& in) -> ov::Output<ov::Node> {
          auto pshape = in.get_partial_shape();
          if (pshape.rank().is_static() && pshape.rank().get_length() == 0) return in;
          return std::make_shared<ov::op::v0::Squeeze>(in)->output(0);
        };
        if (inputs.size() >= 3) {
          auto start = toScalar(inputs[0]);
          auto stop = toScalar(inputs[1]);
          auto step = toScalar(inputs[2]);
          // Use input element type as output type
          auto outType = start.get_element_type();
          node = std::make_shared<ov::op::v4::Range>(start, stop, step, outType);
        } else if (slots[s].args.numTArgs >= 3) {
          auto start = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].args.tArgs[0])});
          auto stop = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].args.tArgs[1])});
          auto step = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].args.tArgs[2])});
          node = std::make_shared<ov::op::v4::Range>(start, stop, step, ov::element::f32);
        }
      } else if (opName == "fill" || opName == "Fill") {
        // Broadcast a scalar value to a target shape
        if (inputs.size() >= 2) {
          node = std::make_shared<ov::op::v3::Broadcast>(inputs[1], inputs[0]);
        } else if (inputs.size() >= 1 && slots[s].args.numTArgs > 0) {
          auto val = ov::op::v0::Constant::create(ov::element::f32, {}, {static_cast<float>(slots[s].args.tArgs[0])});
          node = std::make_shared<ov::op::v3::Broadcast>(val, inputs[0]);
        }
      } else if (opName == "shape_of" || opName == "ShapeOf") {
        if (inputs.size() >= 1) {
          node = std::make_shared<ov::op::v3::ShapeOf>(inputs[0], ov::element::i64);
        }
      } else if (opName == "onehot" || opName == "OneHot") {
        if (inputs.size() >= 1) {
          int axis = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : -1;
          int64_t depth = (slots[s].args.numIArgs > 1) ? slots[s].args.iArgs[1] : 1;
          float onVal = (slots[s].args.numTArgs > 0) ? static_cast<float>(slots[s].args.tArgs[0]) : 1.0f;
          float offVal = (slots[s].args.numTArgs > 1) ? static_cast<float>(slots[s].args.tArgs[1]) : 0.0f;
          auto depth_const = ov::op::v0::Constant::create(ov::element::i64, {}, {depth});
          auto on_const = ov::op::v0::Constant::create(ov::element::f32, {}, {onVal});
          auto off_const = ov::op::v0::Constant::create(ov::element::f32, {}, {offVal});
          node = std::make_shared<ov::op::v1::OneHot>(inputs[0], depth_const, on_const, off_const, axis);
        }
      } else if (opName == "eye" || opName == "Eye") {
        // Compose identity-like pattern: triu(ones) with diag=0 on square matrix
        // Use Range + Unsqueeze + Equal to create identity
        if (slots[s].args.numIArgs >= 1) {
          int64_t n = slots[s].args.iArgs[0];
          int64_t m = (slots[s].args.numIArgs > 1) ? slots[s].args.iArgs[1] : n;
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
        if (slots[s].args.numTArgs >= 2 && slots[s].args.numIArgs >= 1) {
          double start = slots[s].args.tArgs[0];
          double stop = slots[s].args.tArgs[1];
          int64_t num = slots[s].args.iArgs[0];
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
        if (inputs.size() >= 1 && slots[s].args.numIArgs >= 1) {
          int64_t maxLen = slots[s].args.iArgs[0];
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
        double val = (slots[s].args.numTArgs > 0) ? slots[s].args.tArgs[0] : 0.0;
        if (slots[s].wiring.numOutputs > 0) {
          int outIdx = slots[s].wiring.outputSlotIndices[0];
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
          int dim = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
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
          int64_t sH = (slots[s].args.numIArgs > 2) ? slots[s].args.iArgs[2] : 1;
          int64_t sW = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 1;
          int64_t pH = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 0;
          int64_t pW = (slots[s].args.numIArgs > 5) ? slots[s].args.iArgs[5] : 0;
          int64_t dH = (slots[s].args.numIArgs > 6) ? slots[s].args.iArgs[6] : 1;
          int64_t dW = (slots[s].args.numIArgs > 7) ? slots[s].args.iArgs[7] : 1;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pH, pW};
          ov::CoordinateDiff pads_end{pH, pW};
          ov::Strides dilations{static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::Convolution>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "conv3d" || opName == "Conv3d") {
        if (inputs.size() >= 2) {
          int64_t sD = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 1;
          int64_t sH = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 1;
          int64_t sW = (slots[s].args.numIArgs > 5) ? slots[s].args.iArgs[5] : 1;
          int64_t pD = (slots[s].args.numIArgs > 6) ? slots[s].args.iArgs[6] : 0;
          int64_t pH = (slots[s].args.numIArgs > 7) ? slots[s].args.iArgs[7] : 0;
          int64_t pW = (slots[s].args.numIArgs > 8) ? slots[s].args.iArgs[8] : 0;
          int64_t dD = (slots[s].args.numIArgs > 9) ? slots[s].args.iArgs[9] : 1;
          int64_t dH = (slots[s].args.numIArgs > 10) ? slots[s].args.iArgs[10] : 1;
          int64_t dW = (slots[s].args.numIArgs > 11) ? slots[s].args.iArgs[11] : 1;
          ov::Strides strides{static_cast<size_t>(sD), static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pD, pH, pW};
          ov::CoordinateDiff pads_end{pD, pH, pW};
          ov::Strides dilations{static_cast<size_t>(dD), static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::Convolution>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "depthwise_conv2d" || opName == "DepthwiseConv2d") {
        if (inputs.size() >= 2) {
          int64_t sH = (slots[s].args.numIArgs > 2) ? slots[s].args.iArgs[2] : 1;
          int64_t sW = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 1;
          int64_t pH = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 0;
          int64_t pW = (slots[s].args.numIArgs > 5) ? slots[s].args.iArgs[5] : 0;
          int64_t dH = (slots[s].args.numIArgs > 6) ? slots[s].args.iArgs[6] : 1;
          int64_t dW = (slots[s].args.numIArgs > 7) ? slots[s].args.iArgs[7] : 1;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pH, pW};
          ov::CoordinateDiff pads_end{pH, pW};
          ov::Strides dilations{static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::GroupConvolution>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "maxpool2d" || opName == "MaxPool2d") {
        if (inputs.size() >= 1) {
          int64_t kH = (slots[s].args.numIArgs > 0) ? slots[s].args.iArgs[0] : 2;
          int64_t kW = (slots[s].args.numIArgs > 1) ? slots[s].args.iArgs[1] : 2;
          int64_t sH = (slots[s].args.numIArgs > 2) ? slots[s].args.iArgs[2] : 1;
          int64_t sW = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 1;
          int64_t pH = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 0;
          int64_t pW = (slots[s].args.numIArgs > 5) ? slots[s].args.iArgs[5] : 0;
          int64_t dH = (slots[s].args.numIArgs > 6) ? slots[s].args.iArgs[6] : 1;
          int64_t dW = (slots[s].args.numIArgs > 7) ? slots[s].args.iArgs[7] : 1;
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
          int64_t kH = (slots[s].args.numIArgs > 0) ? slots[s].args.iArgs[0] : 2;
          int64_t kW = (slots[s].args.numIArgs > 1) ? slots[s].args.iArgs[1] : 2;
          int64_t sH = (slots[s].args.numIArgs > 2) ? slots[s].args.iArgs[2] : 1;
          int64_t sW = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 1;
          int64_t pH = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 0;
          int64_t pW = (slots[s].args.numIArgs > 5) ? slots[s].args.iArgs[5] : 0;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::Shape kernel{static_cast<size_t>(kH), static_cast<size_t>(kW)};
          ov::Shape pads_begin{static_cast<size_t>(pH), static_cast<size_t>(pW)};
          ov::Shape pads_end{static_cast<size_t>(pH), static_cast<size_t>(pW)};
          bool excludePad = (slots[s].args.numBArgs > 0) ? slots[s].args.bArgs[0] : true;
          node = std::make_shared<ov::op::v1::AvgPool>(
              inputs[0], strides, pads_begin, pads_end, kernel, excludePad);
        }
      } else if (opName == "deconv2d") {
        if (inputs.size() >= 2) {
          int64_t sH = (slots[s].args.numIArgs > 2) ? slots[s].args.iArgs[2] : 1;
          int64_t sW = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 1;
          int64_t pH = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 0;
          int64_t pW = (slots[s].args.numIArgs > 5) ? slots[s].args.iArgs[5] : 0;
          int64_t dH = (slots[s].args.numIArgs > 6) ? slots[s].args.iArgs[6] : 1;
          int64_t dW = (slots[s].args.numIArgs > 7) ? slots[s].args.iArgs[7] : 1;
          ov::Strides strides{static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pH, pW};
          ov::CoordinateDiff pads_end{pH, pW};
          ov::Strides dilations{static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      } else if (opName == "deconv3d") {
        if (inputs.size() >= 2) {
          int64_t sD = (slots[s].args.numIArgs > 3) ? slots[s].args.iArgs[3] : 1;
          int64_t sH = (slots[s].args.numIArgs > 4) ? slots[s].args.iArgs[4] : 1;
          int64_t sW = (slots[s].args.numIArgs > 5) ? slots[s].args.iArgs[5] : 1;
          int64_t pD = (slots[s].args.numIArgs > 6) ? slots[s].args.iArgs[6] : 0;
          int64_t pH = (slots[s].args.numIArgs > 7) ? slots[s].args.iArgs[7] : 0;
          int64_t pW = (slots[s].args.numIArgs > 8) ? slots[s].args.iArgs[8] : 0;
          int64_t dD = (slots[s].args.numIArgs > 9) ? slots[s].args.iArgs[9] : 1;
          int64_t dH = (slots[s].args.numIArgs > 10) ? slots[s].args.iArgs[10] : 1;
          int64_t dW = (slots[s].args.numIArgs > 11) ? slots[s].args.iArgs[11] : 1;
          ov::Strides strides{static_cast<size_t>(sD), static_cast<size_t>(sH), static_cast<size_t>(sW)};
          ov::CoordinateDiff pads_begin{pD, pH, pW};
          ov::CoordinateDiff pads_end{pD, pH, pW};
          ov::Strides dilations{static_cast<size_t>(dD), static_cast<size_t>(dH), static_cast<size_t>(dW)};
          node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
              inputs[0], inputs[1], strides, pads_begin, pads_end, dilations);
        }
      }

      // ── Causal Conv1D (depthwise causal 1D convolution with state) ──
      // Decomposes into: transpose → pad → GroupConv1D → bias → activation → transpose
      // Plus a slice for state_out. Has 2 outputs so wires tensorMap directly.
      else if (opName == "causal_conv1d" || opName == "CausalConv1d") {
        // inputs[0] = x [B,L,D], inputs[1] = weight [D,K] or [K,D]
        // optional inputs[2] = bias [D] or state_in [B,D,K-1]
        // optional inputs[3] = state_in [B,D,K-1]
        if (inputs.size() >= 2) {
          int activation = (slots[s].args.numIArgs > 0) ? static_cast<int>(slots[s].args.iArgs[0]) : 0;
          int wFormat = (slots[s].args.numIArgs > 1) ? static_cast<int>(slots[s].args.iArgs[1]) : 0;

          auto x_input = inputs[0];  // [B,L,D]
          auto w_input = inputs[1];  // [D,K] or [K,D]
          auto dtype = x_input.get_element_type();

          // Harmonize weight type to match input
          if (w_input.get_element_type() != dtype) {
            w_input = std::make_shared<ov::op::v0::Convert>(w_input, dtype)->output(0);
          }

          // Transpose x from [B,L,D] → [B,D,L] for GroupConv
          auto perm_blk = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3},
              std::vector<int64_t>{0, 2, 1});
          auto x_bdl = std::make_shared<ov::op::v1::Transpose>(x_input, perm_blk)->output(0);

          // Get K from weight shape. For wFormat=0: weight is [D,K], K is dim 1.
          // For wFormat=1: weight is [K,D], K is dim 0.
          // We need weight in [D,K] order for the reshape below.
          auto w_dk = w_input;
          if (wFormat == 1) {
            // Transpose [K,D] → [D,K]
            auto w_perm = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{2},
                std::vector<int64_t>{1, 0});
            w_dk = std::make_shared<ov::op::v1::Transpose>(w_input, w_perm)->output(0);
          }

          // Reshape weight [D,K] → [D,1,1,K] for OV GroupConvolution
          // GroupConv expects: [GROUPS, C_OUT/GROUPS, C_IN/GROUPS, K] = [D, 1, 1, K]
          auto w_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4},
              std::vector<int64_t>{0, 1, 1, 0});
          // Use ShapeOf + Gather to get D and K dynamically
          auto w_shape_node = std::make_shared<ov::op::v3::ShapeOf>(w_dk);
          auto idx_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
          auto idx_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
          auto axis_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
          auto D_val = std::make_shared<ov::op::v8::Gather>(w_shape_node, idx_0, axis_0)->output(0);
          auto K_val = std::make_shared<ov::op::v8::Gather>(w_shape_node, idx_1, axis_0)->output(0);
          auto one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
          auto target_shape = std::make_shared<ov::op::v0::Concat>(
              ov::OutputVector{D_val, one, one, K_val}, 0)->output(0);
          auto w_grouped = std::make_shared<ov::op::v1::Reshape>(w_dk, target_shape, true)->output(0);

          // Pad x_bdl [B,D,L] with K-1 zeros on the left (causal padding)
          // Pad format: pads_begin [0,0,K-1], pads_end [0,0,0]
          auto zero_i = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
          auto km1 = std::make_shared<ov::op::v1::Subtract>(K_val, one)->output(0);
          auto pads_begin = std::make_shared<ov::op::v0::Concat>(
              ov::OutputVector{zero_i, zero_i, km1}, 0)->output(0);
          auto pads_end = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3},
              std::vector<int64_t>{0, 0, 0});
          auto pad_val = std::make_shared<ov::op::v0::Constant>(dtype, ov::Shape{}, std::vector<float>{0.0f});

          // Resolve the backend-neutral op contract first, then lower the
          // resulting semantic roles into OpenVINO nodes. NNAPI and future graph
          // backends can reuse the same resolver without sharing OV tensor types.
          sd::ops::helpers::CausalConv1dInputRoles inputRoles;
          std::string resolutionFailure;
          const bool inputsResolved = sd::ops::helpers::resolveCausalConv1dInputRoles(
              static_cast<int>(inputs.size()),
              [&](int inputIndex) {
                const auto rank = inputs[inputIndex].get_partial_shape().rank();
                return rank.is_static() ? static_cast<int>(rank.get_length()) : -1;
              },
              inputRoles, &resolutionFailure);
          if (!inputsResolved) {
            throw std::runtime_error("OpenVINO causal_conv1d contract is not resolvable: " +
                                     resolutionFailure);
          }

          const bool hasBias = inputRoles.bias >= 0;
          const bool hasStateIn = inputRoles.stateIn >= 0;
          const bool hasActualLen = inputRoles.actualLen >= 0;
          ov::Output<ov::Node> biasNode;
          ov::Output<ov::Node> stateInNode;
          ov::Output<ov::Node> actualLenNode;
          if (hasBias) biasNode = inputs[inputRoles.bias];
          if (hasStateIn) stateInNode = inputs[inputRoles.stateIn];
          if (hasActualLen) actualLenNode = inputs[inputRoles.actualLen];

          ov::Output<ov::Node> x_padded;
          if (hasStateIn) {
            // Concat state_in [B,D,K-1] with x_bdl [B,D,L] along axis 2
            if (stateInNode.get_element_type() != dtype) {
              stateInNode = std::make_shared<ov::op::v0::Convert>(stateInNode, dtype)->output(0);
            }
            x_padded = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector{stateInNode, x_bdl}, 2)->output(0);
          } else {
            x_padded = std::make_shared<ov::op::v12::Pad>(
                x_bdl, pads_begin, pads_end, pad_val, ov::op::PadMode::CONSTANT)->output(0);
          }

          // GroupConvolution: groups=D, 1D conv along last dim
          // Input: [B,D,L+K-1], Weights: [D,1,1,K]
          // Output: [B,D,L]
          ov::Strides conv_strides{1};
          ov::CoordinateDiff conv_pads_begin{0};
          ov::CoordinateDiff conv_pads_end{0};
          ov::Strides conv_dilations{1};
          auto conv = std::make_shared<ov::op::v1::GroupConvolution>(
              x_padded, w_grouped, conv_strides, conv_pads_begin, conv_pads_end, conv_dilations);
          ov::Output<ov::Node> convOut = conv->output(0);  // [B,D,L]

          if (hasBias) {
            if (biasNode.get_element_type() != dtype) {
              biasNode = std::make_shared<ov::op::v0::Convert>(biasNode, dtype)->output(0);
            }
            // Reshape bias [D] → [1,D,1] for broadcasting with [B,D,L]
            auto bias_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3},
                std::vector<int64_t>{1, -1, 1});
            auto bias_reshaped = std::make_shared<ov::op::v1::Reshape>(biasNode, bias_shape, false)->output(0);
            convOut = std::make_shared<ov::op::v1::Add>(convOut, bias_reshaped)->output(0);
          }

          // Optional SiLU activation (activation == 1): y = x * sigmoid(x)
          if (activation == 1) {
            auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(convOut)->output(0);
            convOut = std::make_shared<ov::op::v1::Multiply>(convOut, sigmoid)->output(0);
          }

          // Transpose convOut [B,D,L] → [B,L,D] for output 0
          auto output0 = std::make_shared<ov::op::v1::Transpose>(convOut, perm_blk)->output(0);

          // State output: last K-1 timesteps of the PADDED sequence → [B,D,K-1]
          // x_padded is [B,D,L+K-1] (concat(stateIn,x) or zero-padded x).
          // The native kernel builds state_out by reaching back into stateIn
          // when L < K. Slicing from x_padded (not x_bdl) replicates this:
          //   decode L=1, K=4: x_padded=[stateIn[0..2], x[0]], last 3 = [stateIn[1], stateIn[2], x[0]]
          //   prefill L=7, K=4: x_padded=[pad/state, x[0..6]], last 3 = [x[4], x[5], x[6]]
          auto xp_shape_node = std::make_shared<ov::op::v3::ShapeOf>(x_padded);
          auto P_val = std::make_shared<ov::op::v8::Gather>(xp_shape_node,
              std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
              axis_0)->output(0);

          ov::Output<ov::Node> start_state;
          if (hasActualLen) {
            // Native causal_conv1d clamps actualLen to [0,L] and uses it only
            // to choose the state_out window. x_padded starts with K-1 history
            // values, so [actualLen, actualLen + K-1) is exactly the final
            // history after consuming actualLen real input timesteps.
            if (actualLenNode.get_element_type() != ov::element::i64) {
              actualLenNode =
                  std::make_shared<ov::op::v0::Convert>(actualLenNode, ov::element::i64)->output(0);
            }
            auto scalar_as_vector_shape = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
            auto actual_len_vector = std::make_shared<ov::op::v1::Reshape>(
                actualLenNode, scalar_as_vector_shape, false)->output(0);
            auto x_shape_node = std::make_shared<ov::op::v3::ShapeOf>(x_bdl);
            auto L_val = std::make_shared<ov::op::v8::Gather>(x_shape_node,
                std::make_shared<ov::op::v0::Constant>(
                    ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
                axis_0)->output(0);
            auto non_negative_len =
                std::make_shared<ov::op::v1::Maximum>(actual_len_vector, zero_i)->output(0);
            start_state =
                std::make_shared<ov::op::v1::Minimum>(non_negative_len, L_val)->output(0);
          } else {
            start_state = std::make_shared<ov::op::v1::Subtract>(P_val, km1)->output(0);
          }
          auto end_state = std::make_shared<ov::op::v1::Add>(start_state, km1)->output(0);

          // Pad start/stop/step to rank-3 [0,0,start] / [0,0,end] / [1,1,1]
          auto ss_begin = std::make_shared<ov::op::v0::Concat>(
              ov::OutputVector{zero_i, zero_i, start_state}, 0)->output(0);
          auto ss_end = std::make_shared<ov::op::v0::Concat>(
              ov::OutputVector{zero_i, zero_i, end_state}, 0)->output(0);
          auto ss_step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3},
              std::vector<int64_t>{1, 1, 1});
          // begin_mask/end_mask: mask dims 0,1 (batch, channel) so they pass through fully
          std::vector<int64_t> mask_01 = {1, 1, 0};
          auto output1 = std::make_shared<ov::op::v1::StridedSlice>(
              x_padded, ss_begin, ss_end, ss_step, mask_01, mask_01)->output(0);

          // Wire both outputs into tensorMap directly
          if (slots[s].wiring.numOutputs >= 1) {
            tensorMap[slots[s].wiring.outputSlotIndices[0]] = output0;
          }
          if (slots[s].wiring.numOutputs >= 2) {
            tensorMap[slots[s].wiring.outputSlotIndices[1]] = output1;
          }

          DSP_DIAG(COMPILE, "OpenVINO: causal_conv1d slot %d: activation=%d wFormat=%d hasBias=%d hasState=%d "
                   "hasActualLen=%d x_input=%s w_dk=%s x_bdl=%s convOut=%s output0=%s output1=%s",
                   s, activation, wFormat, hasBias, hasStateIn, hasActualLen,
                   x_input.get_partial_shape().to_string().c_str(),
                   w_dk.get_partial_shape().to_string().c_str(),
                   x_bdl.get_partial_shape().to_string().c_str(),
                   convOut.get_partial_shape().to_string().c_str(),
                   output0.get_partial_shape().to_string().c_str(),
                   output1.get_partial_shape().to_string().c_str());
          continue;  // skip standard wiring — already done
        }
      }

      // ── ROPE / Attention / SSM: already filtered by isNativeDeferredOp above ──
      // (These branches are unreachable but kept as defensive guards)

      if (!node) {
        DSP_DIAG(COMPILE, "OpenVINO: failed to create node for '%s' at slot %d",
                 opName.c_str(), s);
        continue;
      }

      // Wire outputs
      for (int o = 0; o < slots[s].wiring.numOutputs; o++) {
        int outIdx = slots[s].wiring.outputSlotIndices[o];
        int nodeOutputIdx = std::min(o, static_cast<int>(node->get_output_size()) - 1);
        tensorMap[outIdx] = node->output(nodeOutputIdx);
      }
    }
  }

  // Materialize only values that cross this island boundary, are requested
  // graph outputs, or are terminal. Keeping island-internal values in OpenVINO
  // avoids writes into view/constant storage and reduces result binding overhead.
  const auto externallyConsumed = computeExternallyVisibleOutputSlots(
      slots, startSlot, endSlot, totalSlots,
      requestedOutputSlotIndices, numRequestedOutputs);
  std::unordered_set<int> emittedOutputs;

  // Create Results in producer order. Deterministic ordering is required because
  // topology-equivalent islands share CompiledModels across transformer layers.
  // If the graph's compute type diverged from the expected output type (e.g. f32
  // intermediates from RMSNorm but output NDArray is f16), insert a Convert node
  // before the Result. This is a graph-level cast — OpenVINO fuses it into the
  // last kernel so there's no runtime temporary buffer.
  for (int s = startSlot; s <= endSlot; ++s) {
    for (int o = 0; o < slots[s].wiring.numOutputs; ++o) {
      const int outIdx = slots[s].wiring.outputSlotIndices[o];
      if (externallyConsumed.count(outIdx) == 0 ||
          !emittedOutputs.insert(outIdx).second) {
        continue;
      }
      auto it = tensorMap.find(outIdx);
      if (it == tensorMap.end()) continue;

      auto graphOutput = it->second;
      auto graphType = graphOutput.get_element_type();

      // Determine expected output type from the NDArray that warmup populated
      NDArray* outArr = (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots)
                        ? outputSlots[outIdx] : nullptr;
      if (outArr != nullptr) {
        auto expectedType = mapDataType(outArr->dataType());
        if (graphType != expectedType) {
          // When the CPU lacks native FP16 ISA (kCpuHasNativeFp16==false), the entire
          // island computes in f32 (ISA promotion). Inserting a Convert(f32→f16) here
          // would re-introduce f16 at island output boundaries: transformer intermediate
          // values (matmul products, attention scores) easily exceed f16 max (65504),
          // causing inf→NaN cascades through 24+ layers until logits are all NaN.
          //
          // Instead: leave the Result as f32. The runtime output binding (executeSegment)
          // detects the model-f32/NDArray-f16 mismatch and performs a saturating copy
          // (clamping to [-65504, 65504]) to fill the f16 NDArray buffer safely.
          bool skipConvert = (!kCpuHasNativeFp16
                              && graphType == ov::element::f32
                              && expectedType == ov::element::f16);
          if (skipConvert) {
            DSP_DIAG(COMPILE, "OpenVINO: output slot %d: skipping Convert(f32→f16) — no native FP16 ISA, "
                     "runtime will saturate-copy f32→f16", outIdx);
          } else {
            DSP_DIAG(COMPILE, "OpenVINO: output slot %d type mismatch: graph=%s expected=%s, inserting Convert",
                     outIdx, graphType.get_type_name().c_str(), expectedType.get_type_name().c_str());
            graphOutput = std::make_shared<ov::op::v0::Convert>(graphOutput, expectedType)->output(0);
          }
        }
      }

      results.push_back(std::make_shared<ov::op::v0::Result>(graphOutput));
      outputSourceMap.push_back(outIdx);
    }
  }

  if (params.empty() || results.empty()) {
    DSP_DIAG(COMPILE, "OpenVINO: empty model (params=%zu, results=%zu)",
             params.size(), results.size());
    return result;
  }

  // Build the model — check topology cache first to share CompiledModels
  // across segments with identical op structure (e.g. transformer layers).
  try {
    uint64_t topoHash = computeIslandTopologyHash(
        slots, startSlot, endSlot, externalInputs, numExternalInputs,
        outputSlots, totalOutputSlots);

    std::shared_ptr<ov::CompiledModel> sharedCompiled;
    {
      std::lock_guard<std::mutex> lock(modelCacheMtx_);
      auto it = modelCache_.find(topoHash);
      if (it != modelCache_.end()) {
        sharedCompiled = it->second;
        DSP_DIAG(COMPILE, "OpenVINO: island [%d-%d] topology HIT (hash=0x%llx) — reusing compiled model",
                 startSlot, endSlot, (unsigned long long)topoHash);
      }
    }

    if (!sharedCompiled) {
      auto model = std::make_shared<ov::Model>(results, params, "dsp_segment");
      auto compiledModel = core_.compile_model(model, "CPU");
      sharedCompiled = std::make_shared<ov::CompiledModel>(compiledModel);
      {
        std::lock_guard<std::mutex> lock(modelCacheMtx_);
        modelCache_[topoHash] = sharedCompiled;
      }
      DSP_DIAG(COMPILE, "OpenVINO: compiled island [%d-%d] with %zu params, %zu results (hash=0x%llx)",
               startSlot, endSlot, params.size(), results.size(), (unsigned long long)topoHash);
    }

    result.compiled = sharedCompiled;
    result.request = std::make_shared<ov::InferRequest>(result.compiled->create_infer_request());
    result.inputSlotMap = inputSourceMap;
    result.outputSlotMap = outputSourceMap;
  } catch (const std::exception& e) {
    // OpenVINO exceptions often have multiline messages with nested cause.
    // Log full message to stderr for visibility.
    fprintf(stderr, "OpenVINO compile_model FAILED island[%d-%d]: %s\n", startSlot, endSlot, e.what());
    fflush(stderr);
    DSP_DIAG(COMPILE, "OpenVINO: compile_model FAILED island[%d-%d] (%zu params, %zu results)",
             startSlot, endSlot, params.size(), results.size());
    // Leave result.compiled / result.request null — caller checks for null to detect failure
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

  SegmentCacheKey cacheKey{seg.def.startSlot, seg.def.endSlot, shapeKey};

  std::lock_guard<std::mutex> lock(cacheMtx_);

  auto it = cache_.find(cacheKey);
  if (it != cache_.end() && it->second->second.valid) {
    // Promote to MRU
    cacheLru_.splice(cacheLru_.begin(), cacheLru_, it->second);
    DSP_DIAG(JIT, "OpenVinoGraphBackend::compileSegment [%d-%d]: cache HIT (shapeKey=0x%llx)",
             seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);
    lastCompilationAudit_ = it->second->second.compilationAudit;
    return true;
  }

  // Remove existing invalid entry if present (prevents orphaned LRU nodes)
  if (it != cache_.end()) {
    cacheLru_.erase(it->second);
    cache_.erase(it);
  }

  DSP_DIAG(COMPILE, "OpenVinoGraphBackend::compileSegment [%d-%d]: cache MISS, building islands (shapeKey=0x%llx)",
           seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);

  CompiledSegment compiled;
  compiled.shapeKey = shapeKey;

  const int segStart = seg.def.startSlot;
  const int segEnd   = seg.def.endSlot;

  // Scan the segment and identify contiguous OV-compilable ranges vs native ranges.
  // A slot is "native" if it's in the deferred set; otherwise it's OV-compilable.
  int s = segStart;
  while (s <= segEnd) {
    const std::string& opName = slots[s].ident.opName;

    if (isNativeDeferredOp(opName)) {
      // Start a native range — extend until a non-deferred op
      int nativeStart = s;
      while (s <= segEnd && isNativeDeferredOp(slots[s].ident.opName)) {
        // Build audit entry for native slot
        CompilationAuditEntry audit;
        audit.slotIndex = s;
        audit.opName = slots[s].ident.opName;
        audit.wasCompiled = false;
        audit.isNativeHandled = true;
        audit.reason = "native-deferred op";
        compiled.compilationAudit.push_back(audit);
        s++;
      }
      int nativeEnd = s - 1;
      int nativeIdx = static_cast<int>(compiled.nativeRanges.size());
      compiled.nativeRanges.push_back({nativeStart, nativeEnd});
      compiled.executionSchedule.push_back({true, nativeIdx});
      DSP_DIAG(COMPILE, "OpenVINO: native range [%d-%d]", nativeStart, nativeEnd);
    } else {
      // Start an OV island — extend until a native-deferred op
      int islandStart = s;
      while (s <= segEnd && !isNativeDeferredOp(slots[s].ident.opName)) {
        s++;
      }
      int islandEnd = s - 1;

      OvIsland island;
      try {
        island = buildIsland(slots, islandStart, islandEnd,
                             externalInputs, numExternalInputs,
                             outputSlots, totalOutputSlots,
                             totalSlots, requestedOutputSlotIndices,
                             numRequestedOutputs);
      } catch (const std::exception& e) {
        DSP_DIAG(COMPILE, "OpenVINO: buildIsland[%d-%d] exception: %s",
                 islandStart, islandEnd, e.what());
        return false;
      } catch (...) {
        DSP_DIAG(COMPILE, "OpenVINO: buildIsland[%d-%d] unknown exception",
                 islandStart, islandEnd);
        return false;
      }

      bool islandOk = (island.compiled != nullptr && island.request != nullptr);

      // Build audit entries for slots in this island
      for (int sl = islandStart; sl <= islandEnd; sl++) {
        CompilationAuditEntry audit;
        audit.slotIndex = sl;
        audit.opName = slots[sl].ident.opName;
        audit.wasCompiled = islandOk;
        if (!islandOk) audit.reason = "island compilation failed";
        compiled.compilationAudit.push_back(audit);
      }

      if (!islandOk) {
        DSP_DIAG(COMPILE, "OpenVINO: island [%d-%d] failed to compile", islandStart, islandEnd);
        return false;
      }

      int islandIdx = static_cast<int>(compiled.ovIslands.size());
      compiled.ovIslands.push_back(std::move(island));
      compiled.executionSchedule.push_back({false, islandIdx});
      DSP_DIAG(COMPILE, "OpenVINO: OV island [%d-%d] compiled (island %d)",
               islandStart, islandEnd, islandIdx);
    }
  }

  // Segment is valid if at least one OV island compiled successfully
  compiled.valid = !compiled.ovIslands.empty();

  lastCompilationAudit_ = compiled.compilationAudit;

  if (!compiled.valid) {
    DSP_DIAG(COMPILE, "OpenVinoGraphBackend::compileSegment [%d-%d]: no OV islands compiled",
             segStart, segEnd);
    return false;
  }

  DSP_DIAG(COMPILE, "OpenVinoGraphBackend::compileSegment [%d-%d]: SUCCESS islands=%d nativeRanges=%d",
           segStart, segEnd, (int)compiled.ovIslands.size(), (int)compiled.nativeRanges.size());

  // Insert at MRU position and evict if over limit
  cacheLru_.emplace_front(cacheKey, std::move(compiled));
  cache_[cacheKey] = cacheLru_.begin();
  evictCacheIfOverLimitLocked();

  seg.def.shapeKeyState.markCompiled(shapeKey);
  return true;
}

// ─── executeSegment ─────────────────────────────────────────────────────────

Status OpenVinoGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  SegmentCacheKey cacheKey{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey};

  // Short-lived lock for cache lookup only — released before inference.
  CompiledSegment* compiledPtr = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(cacheKey);
    if (it != cache_.end() && it->second->second.valid) {
      compiledPtr = &it->second->second;
    }
  }

  if (!compiledPtr) {
    // Cache miss — recompile transparently.
    DSP_DIAG(EXECUTE, "OpenVINO: cache miss for seg[%d-%d] shapeKey=%lld cacheSize=%d — recompiling",
             seg.def.startSlot, seg.def.endSlot, (long long)seg.def.shapeKeyState.compiledShapeKey,
             (int)cache_.size());
    bool recompiled = compileSegment(seg, slots, externalInputs, numExternalInputs,
                                      outputSlots, totalOutputSlots,
                                      seg.def.shapeKeyState.compiledShapeKey, 0);
    if (!recompiled) {
      DSP_DIAG(EXECUTE, "OpenVINO: recompile FAILED for seg[%d-%d]",
               seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(cacheKey);
    if (it == cache_.end() || !it->second->second.valid) {
      DSP_DIAG(EXECUTE, "OpenVINO: cache still empty after recompile for seg[%d-%d]",
               seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiledPtr = &it->second->second;
    DSP_DIAG(EXECUTE, "OpenVINO: recompile OK for seg[%d-%d] — resuming execution",
             seg.def.startSlot, seg.def.endSlot);
  }

  auto& compiled = *compiledPtr;

  // Lambda to execute a single OV island — captures mapDataType via class scope
  auto runIsland = [&](OvIsland& island) -> Status {
    auto ovDtype = [](DataType dt) { return mapDataType(dt); };
    auto& request = *island.request;
    auto& compiledModel = *island.compiled;

    // Initialize caches on first execution
    if (!island.promotionInitialized) {
      island.cachedPromotedInputs.resize(island.inputSlotMap.size());
      island.cachedPromotedOutputs.resize(island.outputSlotMap.size());
      island.promotionInitialized = true;
    }
    if (!island.steadyStateCachesInitialized) {
      island.cachedInputShapes.resize(island.inputSlotMap.size());
      island.cachedOutputShapes.resize(island.outputSlotMap.size());
      island.cachedNeedsSaturatingCopy.resize(island.outputSlotMap.size(), false);
      island.steadyStateCachesInitialized = true;
    }

    // Set input tensors (zero-copy from NDArray host buffers)
    for (size_t i = 0; i < island.inputSlotMap.size(); i++) {
      int srcIdx = island.inputSlotMap[i];
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
          // Empty array — provide a dummy non-null pointer for OV
          static int8_t dummyBuf[8] = {0};
          auto& shape = island.cachedInputShapes[i];
          int rank = arr->rankOf();
          shape.resize(rank);
          for (int d = 0; d < rank; d++) shape[d] = static_cast<size_t>(arr->sizeAt(d));
          request.set_input_tensor(static_cast<int>(i),
              ov::Tensor(ovDtype(arr->dataType()), shape, dummyBuf));
          continue;
        }
        DSP_DIAG(EXECUTE, "OpenVINO: input array for source %d has NULL buffer (len=%lld)",
                 srcIdx, (long long)arr->lengthOf());
        return Status::KERNEL_FAILURE;
      }
      auto& shape = island.cachedInputShapes[i];
      int rank = arr->rankOf();
      shape.resize(rank);
      for (int d = 0; d < rank; d++) shape[d] = static_cast<size_t>(arr->sizeAt(d));

      auto modelInputType = compiledModel.input(static_cast<int>(i)).get_element_type();
      auto arrType = ovDtype(arr->dataType());
      if (modelInputType != arrType) {
        // ISA-promotion path: model expects f32, NDArray holds f16.
        // Use cached promoted tensor to avoid per-infer() malloc.
        auto& cached = island.cachedPromotedInputs[i];
        size_t numElems = 1;
        for (auto d : shape) numElems *= d;

        // Skip conversion only for constant external inputs (srcIdx < 0 = weight).
        // Internal output slots (srcIdx >= 0) change every step — their buffer
        // pointer stays the same but data is rewritten by upstream ops. Caching
        // the promoted tensor for those causes stale f32 data on subsequent infer()
        // calls, producing garbage LLM output.
        bool isConstantInput = (srcIdx < 0);
        if (isConstantInput && cached.lastSourceBuffer == arr->buffer() && cached.lastNumElems == numElems) {
          request.set_input_tensor(static_cast<int>(i), cached.tensor);
        } else {
          // Allocate or reuse promoted tensor
          if (cached.lastNumElems != numElems) {
            cached.tensor = ov::Tensor(modelInputType, shape);
            cached.lastNumElems = numElems;
          }
          // Convert f16→f32
          if (arrType == ov::element::f16 && modelInputType == ov::element::f32) {
            auto* src = static_cast<const ov::float16*>(arr->buffer());
            auto* dst = cached.tensor.data<float>();
#if defined(__F16C__) || defined(__AVX2__)
            size_t e = 0;
            for (; e + 8 <= numElems; e += 8) {
              __m128i h8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + e));
              __m256 f8 = _mm256_cvtph_ps(h8);
              _mm256_storeu_ps(dst + e, f8);
            }
            for (; e < numElems; e++) dst[e] = static_cast<float>(src[e]);
#else
            for (size_t e = 0; e < numElems; e++) dst[e] = static_cast<float>(src[e]);
#endif
          }
          cached.lastSourceBuffer = arr->buffer();
          request.set_input_tensor(static_cast<int>(i), cached.tensor);
        }
      } else {
        request.set_input_tensor(static_cast<int>(i),
            ov::Tensor(arrType, shape, arr->buffer()));
      }
    }

    // Set output tensors.
    // Normal path: zero-copy — wrap the NDArray host buffer directly in an OV tensor.
    // ISA-promotion path: when the CPU lacks native FP16 and the NDArray is f16 but
    //   the model output is f32 (because buildIsland skipped the Convert(f32→f16)
    //   node), we must NOT wrap the f16 buffer as f32 — that would corrupt memory.
    //   Instead, let OV allocate its own f32 buffer; after infer() we do a saturating
    //   f32→f16 copy into the NDArray buffer (clamping to [-65504, 65504] to prevent
    //   inf/NaN from transformer intermediate magnitudes exceeding f16 max).
    //
    // Shape selection: use the model output shape when it fits in the NDArray buffer
    // (handles rank differences like [4,1] vs [4]). If the model shape requires MORE
    // elements than the NDArray buffer holds, fall back to the NDArray's own shape
    // to prevent buffer overruns (can happen during frozen execution when the NDArray
    // was allocated for a different shape than the compiled model expects).
    //
    // needsSaturatingCopy[i] == true  ⟹  output i is ISA-promoted f32→f16 path.
    auto& needsSaturatingCopy = island.cachedNeedsSaturatingCopy;
    std::fill(needsSaturatingCopy.begin(), needsSaturatingCopy.end(), false);

    for (size_t i = 0; i < island.outputSlotMap.size(); i++) {
      int outIdx = island.outputSlotMap[i];
      if (outIdx < 0 || outIdx >= totalOutputSlots || !outputSlots[outIdx]) {
        DSP_DIAG(EXECUTE, "OpenVINO: missing output slot %d", outIdx);
        return Status::BAD_OUTPUT;
      }
      NDArray* arr = outputSlots[outIdx];
      if (arr->buffer() == nullptr) {
        DSP_DIAG(EXECUTE, "OpenVINO: output slot %d has NULL buffer", outIdx);
        return Status::KERNEL_FAILURE;
      }

      auto modelOutType = compiledModel.output(static_cast<int>(i)).get_element_type();
      auto arrType = ovDtype(arr->dataType());

      auto& shape = island.cachedOutputShapes[i];
      auto modelOutputShape = compiledModel.output(static_cast<int>(i)).get_partial_shape();
      if (modelOutputShape.is_static()) {
        auto modelShape = modelOutputShape.get_shape();
        // Compute total elements the model shape requires
        size_t modelElements = 1;
        for (auto d : modelShape) modelElements *= d;
        size_t arrElements = static_cast<size_t>(arr->lengthOf());
        // Also check against actual buffer capacity for views
        size_t bufBytes = arr->getDataBuffer()->getLenInBytes();
        size_t elemSize = DataTypeUtils::sizeOf(arr->dataType());
        size_t arrOffset = static_cast<size_t>(arr->offset());
        size_t bufCapacityElements = elemSize > 0 ? (bufBytes / elemSize) - arrOffset : 0;
        if (modelElements <= arrElements && modelElements <= bufCapacityElements) {
          shape = modelShape;
        } else {
          // Model shape exceeds buffer — use NDArray shape to prevent overrun
          DSP_DIAG(EXECUTE, "OpenVINO: output slot %d model shape exceeds buffer "
                   "(modelElems=%zu arrElems=%zu bufCapElems=%zu) — using NDArray shape",
                   outIdx, modelElements, arrElements, bufCapacityElements);
          int rank = arr->rankOf();
          shape.resize(rank);
          for (int d = 0; d < rank; d++) shape[d] = static_cast<size_t>(arr->sizeAt(d));
        }
      } else {
        int rank = arr->rankOf();
        shape.resize(rank);
        for (int d = 0; d < rank; d++) shape[d] = static_cast<size_t>(arr->sizeAt(d));
      }

      try {
        // ISA-promotion saturating-copy path: model outputs f32 but NDArray is f16.
        // Reuse cached promoted tensor to avoid per-infer() malloc.
        if (!kCpuHasNativeFp16
            && modelOutType == ov::element::f32
            && arrType == ov::element::f16) {
          needsSaturatingCopy[i] = true;
          auto& cached = island.cachedPromotedOutputs[i];
          size_t numElems = 1;
          for (auto d : shape) numElems *= d;
          if (cached.lastNumElems != numElems) {
            cached.tensor = ov::Tensor(ov::element::f32, shape);
            cached.lastNumElems = numElems;
          }
          request.set_output_tensor(static_cast<int>(i), cached.tensor);
        } else {
          request.set_output_tensor(static_cast<int>(i),
              ov::Tensor(arrType, shape, arr->buffer()));
        }
      } catch (const std::exception& e) {
        DSP_DIAG(EXECUTE, "OpenVINO: set_output_tensor[%zu] FAILED for slot %d: %s "
                 "(modelShape=%s arrLen=%lld)",
                 i, outIdx, e.what(),
                 modelOutputShape.to_string().c_str(),
                 (long long)arr->lengthOf());
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

    // Post-inference output validation: OV may infer a dynamic output shape that differs
    // from the PartialShape used at set_output_tensor time. Validate that the actual
    // shape reported by each output tensor after infer() fits within the bound NDArray
    // buffer. This catches overruns BEFORE corrupted data can propagate.
    for (size_t i = 0; i < island.outputSlotMap.size(); i++) {
      auto outTensor = request.get_output_tensor(static_cast<int>(i));
      auto actualShape = outTensor.get_shape();
      size_t actualElements = 1;
      for (auto d : actualShape) actualElements *= d;
      int outIdx = island.outputSlotMap[i];
      NDArray* arr = outputSlots[outIdx];
      size_t bufBytes = arr->getDataBuffer()->getLenInBytes();
      size_t elemSize = DataTypeUtils::sizeOf(arr->dataType());
      size_t bufCapacity = elemSize > 0 ? bufBytes / elemSize : 0;
      if (actualElements > bufCapacity) {
        DSP_DIAG(EXECUTE, "OpenVINO: POST-INFER output[%zu] slot %d actual shape exceeds buffer! "
                 "actualElems=%zu bufCap=%zu",
                 i, outIdx, actualElements, bufCapacity);
        return Status::KERNEL_FAILURE;
      }

      // ISA-promotion saturating copy: f32 OV tensor → f16 NDArray buffer.
      // Clamp each value to [-65504, 65504] (f16 finite range) and replace NaN
      // with 0.0f before narrowing. This prevents inf/NaN from transformer
      // intermediate magnitudes (matmul products >> 65504) overflowing f16.
      if (needsSaturatingCopy[i]) {
        auto* src = outTensor.data<float>();
        auto* dst = static_cast<ov::float16*>(arr->buffer());
#if defined(__F16C__) || defined(__AVX2__)
        // Vectorized f32→f16 saturating copy (8 elements per cycle).
        // Clamp to [-65504, 65504] and replace NaN with 0 before narrowing.
        const __m256 vMax = _mm256_set1_ps(65504.0f);
        const __m256 vMin = _mm256_set1_ps(-65504.0f);
        const __m256 vZero = _mm256_setzero_ps();
        size_t e = 0;
        for (; e + 8 <= actualElements; e += 8) {
          __m256 v = _mm256_loadu_ps(src + e);
          // Replace NaN with 0: NaN != NaN, so cmpord mask is 0 for NaN lanes
          __m256 nanMask = _mm256_cmp_ps(v, v, _CMP_ORD_Q);
          v = _mm256_and_ps(v, nanMask);  // NaN lanes become 0
          // Clamp to f16 finite range
          v = _mm256_min_ps(v, vMax);
          v = _mm256_max_ps(v, vMin);
          // Convert f32→f16 (round to nearest)
          __m128i h8 = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + e), h8);
        }
        // Scalar tail
        for (; e < actualElements; e++) {
          float val = src[e];
          if (std::isnan(val)) val = 0.0f;
          else if (val > 65504.0f) val = 65504.0f;
          else if (val < -65504.0f) val = -65504.0f;
          dst[e] = ov::float16(val);
        }
#else
        for (size_t e = 0; e < actualElements; e++) {
          float val = src[e];
          if (std::isnan(val)) val = 0.0f;
          else if (val > 65504.0f) val = 65504.0f;
          else if (val < -65504.0f) val = -65504.0f;
          dst[e] = ov::float16(val);
        }
#endif
        DSP_DIAG(EXECUTE, "OpenVINO: output slot %d saturating f32→f16 copy done (%zu elems)",
                 outIdx, actualElements);
      }
    }

    // Mark output arrays as host-authoritative
    for (size_t i = 0; i < island.outputSlotMap.size(); i++) {
      int outIdx = island.outputSlotMap[i];
      if (outIdx >= 0 && outIdx < totalOutputSlots && outputSlots[outIdx]) {
        outputSlots[outIdx]->tickWriteHost();
      }
    }
    return Status::OK;
  };

  try {
    // Walk the execution schedule in order: OV islands and native ranges interleaved
    for (const auto& step : compiled.executionSchedule) {
      if (!step.isNative) {
        // OV island step
        auto& island = compiled.ovIslands[step.index];
        Status st = runIsland(island);
        if (st != Status::OK) {
          DSP_DIAG(EXECUTE, "OpenVINO: island %d execution failed", step.index);
          return st;
        }
      } else {
        // Native range step
        auto& range = compiled.nativeRanges[step.index];
        if (!nativeExecutor_) {
          DSP_DIAG(EXECUTE, "OpenVINO: native range [%d-%d] requires NativeSlotExecutor but none is set",
                   range.startSlot, range.endSlot);
          return Status::KERNEL_FAILURE;
        }
        Status st = nativeExecutor_(range.startSlot, range.endSlot);
        if (st != Status::OK) {
          DSP_DIAG(EXECUTE, "OpenVINO: native range [%d-%d] execution failed",
                   range.startSlot, range.endSlot);
          return st;
        }
      }
    }

    return Status::OK;

  } catch (const std::exception& e) {
    DSP_DIAG(EXECUTE, "OpenVINO: executeSegment[%d-%d] exception: %s",
             seg.def.startSlot, seg.def.endSlot, e.what());
    return Status::KERNEL_FAILURE;
  } catch (...) {
    DSP_DIAG(EXECUTE, "OpenVINO: executeSegment[%d-%d] unknown exception",
             seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }
}

// ─── invalidateCache ────────────────────────────────────────────────────────

void OpenVinoGraphBackend::invalidateCache() {
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    cache_.clear();
    cacheLru_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(modelCacheMtx_);
    modelCache_.clear();
  }
}

void OpenVinoGraphBackend::evictCacheIfOverLimitLocked() {
  if (kMaxCacheEntries <= 0) return;  // 0 = unlimited — no eviction
  while (static_cast<int>(cacheLru_.size()) > kMaxCacheEntries) {
    // Evict from LRU end (oldest)
    auto& victim = cacheLru_.back();
    DSP_DIAG(COMPILE, "OpenVINO: evict LRU cache entry seg[%d-%d] shapeKey=0x%llx (cacheSize=%d > max=%d)",
             victim.first.startSlot, victim.first.endSlot,
             (long long)victim.first.shapeKey,
             (int)cacheLru_.size(), kMaxCacheEntries);
    cache_.erase(victim.first);
    cacheLru_.pop_back();
  }

  // Sweep modelCache_ for CompiledModels that are only held by the topology cache
  // (use_count==1 means no segment cache entry references them anymore).
  // Each CompiledModel holds ~50-200 MB of internal OpenVINO working buffers.
  {
    std::lock_guard<std::mutex> lock(modelCacheMtx_);
    for (auto it = modelCache_.begin(); it != modelCache_.end(); ) {
      if (it->second.use_count() == 1) {
        DSP_DIAG(COMPILE, "OpenVINO: evict unreferenced CompiledModel (topoHash=0x%llx)",
                 (unsigned long long)it->first);
        it = modelCache_.erase(it);
      } else {
        ++it;
      }
    }
  }
}

// ─── getLastCompilationAudit ────────────────────────────────────────────────

std::vector<CompilationAuditEntry> OpenVinoGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_OPENVINO
