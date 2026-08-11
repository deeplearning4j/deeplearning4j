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

#if HAVE_ONEDNN

#include <graph/cpu/OneDnnGraphBackend.h>
#include <graph/DspDiagnostics.h>
#include <helpers/shape.h>
#include <ops/declarable/OpDescriptor.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/platform/mkldnn/OnednnVersionProvider.h>
#include <system/Environment.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <mutex>
#include <thread>

namespace sd {
namespace graph {

// ─── Thread-local stream ────────────────────────────────────────────────────

static thread_local std::unique_ptr<dnnl::stream> tls_onednn_stream;

dnnl::stream& OneDnnGraphBackend::getThreadStream() {
  if (!tls_onednn_stream) {
    tls_onednn_stream = std::make_unique<dnnl::stream>(engine_);
  }
  return *tls_onednn_stream;
}

// ─── Singleton ──────────────────────────────────────────────────────────────

OneDnnGraphBackend& OneDnnGraphBackend::getInstance() {
  static OneDnnGraphBackend* instance = nullptr;
  static std::once_flag initFlag;
  std::call_once(initFlag, []() {
    instance = new OneDnnGraphBackend();
  });
  return *instance;
}

OneDnnGraphBackend::OneDnnGraphBackend()
    : engine_(dnnl::engine::kind::cpu, 0) {
  // Sync OMP thread count with Environment (controlled via -Domp.num.threads).
  // OneDNN with DNNL_CPU_RUNTIME=OMP uses omp_get_max_threads() at execution time.
  // KMP_BLOCKTIME/KMP_AFFINITY/GOMP_SPINCOUNT are already configured globally
  // by CoreConfig::initFromEnvironment() — no need to set them again here.
  int numThreads = sd::Environment::getInstance().maxMasterThreads();
  if (numThreads <= 0) numThreads = std::thread::hardware_concurrency();
  omp_set_num_threads(numThreads);

  DSP_DIAG(COMPILE, "OneDNN: configured %d OMP threads (blocktime/affinity set by CoreConfig)",
           numThreads);
}

OneDnnGraphBackend::~OneDnnGraphBackend() = default;

// ─── Availability ───────────────────────────────────────────────────────────

bool OneDnnGraphBackend::isAvailable() const {
  // -fno-threadsafe-statics: use std::call_once for thread-safe initialization.
  static std::once_flag selfTestFlag;
  std::call_once(selfTestFlag, []() {
    try {
      dg::graph selfTest(dnnl::engine::kind::cpu);
      auto st_in0 = dg::logical_tensor(90000, dg::logical_tensor::data_type::f32,
                                         {1, 7, 1024}, dg::logical_tensor::layout_type::strided);
      auto st_in1 = dg::logical_tensor(90001, dg::logical_tensor::data_type::f32,
                                         {1024, 2048}, dg::logical_tensor::layout_type::strided);
      auto st_out = dg::logical_tensor(90002, dg::logical_tensor::data_type::f32,
                                         {1, 7, 2048}, dg::logical_tensor::layout_type::strided);
      dg::op st_mm(90003, dg::op::kind::MatMul, "selftest_matmul");
      st_mm.set_attr<bool>(dg::op::attr::transpose_a, false);
      st_mm.set_attr<bool>(dg::op::attr::transpose_b, false);
      st_mm.add_inputs({st_in0, st_in1});
      st_mm.add_outputs({st_out});
      selfTest.add_op(st_mm);
      selfTest.finalize();
      auto stParts = selfTest.get_partitions();
      DSP_DIAG(COMPILE, "OneDNN SELF-TEST: %d partitions, first supported=%d",
               static_cast<int>(stParts.size()),
               stParts.empty() ? -1 : (stParts[0].is_supported() ? 1 : 0));
    } catch (const std::exception& e) {
      DSP_DIAG(COMPILE, "OneDNN SELF-TEST: EXCEPTION: %s", e.what());
    }
  });
  return sd::ops::platforms::onednn::OnednnVersionProvider::hasGraphApi();
}

bool OneDnnGraphBackend::isResolvable(
    const GraphBackendRequest& request) const {
  return request.executionMode == GraphExecutionMode::GEM_AUTO ||
         request.executionMode == GraphExecutionMode::GEM_PORTABLE_REPLAY;
}

int OneDnnGraphBackend::resolutionPriority(
    const GraphBackendRequest& request) const {
  (void)request;
  return 500;
}

// ─── Op kind mapping ────────────────────────────────────────────────────────
//
// STRICT mapping: only ops that oneDNN Graph API can genuinely compile and
// execute are included here. oneDNN's partitioner rejects ops it can't optimize
// individually or fuse, so mapping an op that oneDNN can't handle wastes
// compilation time and forces the entire segment to fall back.
//
// Excluded categories:
//   - Shape-only ops (reshape, permute, identity, squeeze, expand_dims):
//     oneDNN rejects these individually. They're zero-compute memory reorders
//     that are better handled natively.
//   - Scalar ops (add_scalar, mul_scalar, neg): Need synthetic scalar tensors
//     that our graph builder doesn't create. Better handled natively.
//   - Custom/composite ops (gated_delta_rule, fused_rope, causal_conv):
//     Not in oneDNN's vocabulary regardless of trait classification.
//   - Ops requiring special tensor layouts (concat with dynamic axis):
//     Can cause partition rejection.

dg::op::kind OneDnnGraphBackend::mapOpKind(const std::string& opName) {
  // ── Compute-heavy anchor ops (individually supported by oneDNN) ──────────
  if (opName == "matmul" || opName == "MatMul" || opName == "mmul") return dg::op::kind::MatMul;
  if (opName == "batch_matmul" || opName == "BatchMatMul") return dg::op::kind::MatMul;
  // Convolution: requires strides, pads_begin, pads_end, dilations attributes
  // that need complex arg-to-attribute mapping. Route to native execution.
  // if (opName == "conv2d" || opName == "Conv2D") return dg::op::kind::Convolution;

  // ── Normalization (individually supported) ───────────────────────────────
  if (opName == "softmax" || opName == "Softmax") return dg::op::kind::SoftMax;
  if (opName == "log_softmax" || opName == "LogSoftmax") return dg::op::kind::LogSoftmax;
  if (opName == "layer_norm" || opName == "LayerNorm") return dg::op::kind::LayerNorm;
  if (opName == "batchnorm" || opName == "BatchNorm") return dg::op::kind::BatchNormInference;
  if (opName == "group_norm" || opName == "GroupNorm") return dg::op::kind::GroupNorm;

  // ── Reduction (individually supported) ───────────────────────────────────
  if (opName == "reduce_sum" || opName == "ReduceSum") return dg::op::kind::ReduceSum;
  if (opName == "reduce_mean" || opName == "ReduceMean") return dg::op::kind::ReduceMean;
  if (opName == "reduce_min" || opName == "ReduceMin") return dg::op::kind::ReduceMin;
  if (opName == "reduce_max" || opName == "ReduceMax") return dg::op::kind::ReduceMax;
  if (opName == "reduce_prod" || opName == "ReduceProd") return dg::op::kind::ReduceProd;

  // ── Element-wise binary (fusible as post-ops into matmul/conv) ──────────
  if (opName == "add" || opName == "Add") return dg::op::kind::Add;
  if (opName == "subtract" || opName == "Sub") return dg::op::kind::Subtract;
  if (opName == "multiply" || opName == "Mul") return dg::op::kind::Multiply;
  if (opName == "divide" || opName == "Div" || opName == "RealDiv") return dg::op::kind::Divide;
  if (opName == "minimum" || opName == "Min") return dg::op::kind::Minimum;
  if (opName == "maximum" || opName == "Max") return dg::op::kind::Maximum;

  // ── Element-wise unary / activations (fusible as post-ops) ──────────────
  if (opName == "relu" || opName == "Relu") return dg::op::kind::ReLU;
  if (opName == "sigmoid" || opName == "Sigmoid") return dg::op::kind::Sigmoid;
  if (opName == "tanh" || opName == "Tanh") return dg::op::kind::Tanh;
  if (opName == "gelu" || opName == "Gelu") return dg::op::kind::GELU;
  if (opName == "elu" || opName == "Elu") return dg::op::kind::Elu;
  if (opName == "exp" || opName == "Exp") return dg::op::kind::Exp;
  if (opName == "log" || opName == "Log") return dg::op::kind::Log;
  if (opName == "abs" || opName == "Abs") return dg::op::kind::Abs;
  if (opName == "sqrt" || opName == "Sqrt") return dg::op::kind::Sqrt;
  if (opName == "square" || opName == "Square") return dg::op::kind::Square;
  if (opName == "pow" || opName == "Pow") return dg::op::kind::Pow;
  if (opName == "clamp" || opName == "ClipByValue" || opName == "clip_by_value") return dg::op::kind::Clamp;
  if (opName == "hardswish" || opName == "HardSwish") return dg::op::kind::HardSwish;
  if (opName == "hardsigmoid" || opName == "HardSigmoid") return dg::op::kind::HardSigmoid;
  if (opName == "mish" || opName == "Mish") return dg::op::kind::Mish;
  if (opName == "round" || opName == "Round") return dg::op::kind::Round;
  if (opName == "reciprocal" || opName == "Reciprocal") return dg::op::kind::Reciprocal;
  if (opName == "softplus" || opName == "SoftPlus" || opName == "Softplus") return dg::op::kind::SoftPlus;
  if (opName == "prelu" || opName == "PReLU" || opName == "Prelu") return dg::op::kind::PReLU;
  if (opName == "lrelu" || opName == "leakyrelu" || opName == "LeakyReLU" || opName == "LeakyRelu") return dg::op::kind::LeakyReLU;

  // ── Pooling ──────────────────────────────────────────────────────────────
  // AvgPool/MaxPool: require strides, pads_begin, pads_end, kernel attributes
  // that need complex arg-to-attribute mapping. Route to native execution.
  // if (opName == "avgpool2d" || opName == "avgpool" || opName == "AvgPool") return dg::op::kind::AvgPool;
  // if (opName == "maxpool2d" || opName == "maxpool" || opName == "MaxPool") return dg::op::kind::MaxPool;

  // ── Type casting ─────────────────────────────────────────────────────────
  if (opName == "cast" || opName == "Cast") return dg::op::kind::TypeCast;

  // ── Conditional selection ────────────────────────────────────────────────
  if (opName == "where" || opName == "Where" || opName == "select" || opName == "Select" || opName == "where_np") return dg::op::kind::Select;

  // ── Comparison (only GreaterEqual available in this oneDNN version) ──────
  if (opName == "greater_equal" || opName == "GreaterEqual") return dg::op::kind::GreaterEqual;

  // ── Bias addition (fusible into matmul/conv) ────────────────────────────
  if (opName == "biasadd" || opName == "BiasAdd" || opName == "bias_add") return dg::op::kind::BiasAdd;

  // ── Squared difference ──────────────────────────────────────────────────
  if (opName == "squared_difference" || opName == "SquaredDifference" || opName == "squareddifference") return dg::op::kind::SquaredDifference;

  // ── Interpolation / resize ──────────────────────────────────────────────
  // Interpolate: requires 'mode' string attribute that needs complex mapping.
  // if (opName == "interpolate" || opName == "Interpolate" || opName == "resize" || opName == "Resize") return dg::op::kind::Interpolate;

  // Not mappable — will be executed natively.
  // Do NOT use trait-based fallback: traits classify op CATEGORY (e.g. MATMUL trait
  // on gated_delta_rule) but oneDNN requires exact op implementations, not categories.
  return dg::op::kind::LastSymbol;
}

// ─── Trait-based op kind fallback (diagnostic only) ────────────────────────
//
// NOT used for compilation — only for reporting what CATEGORY an unknown op
// belongs to. Triton can use this because it generates custom kernels; oneDNN
// uses pre-built kernels and can only execute ops it specifically knows about.

dg::op::kind OneDnnGraphBackend::mapOpKindFromTraits(const std::string& opName) {
  // Diagnostic-only: returns what oneDNN kind we WOULD map to based on traits.
  // Caller must not use this for actual compilation decisions.
  using sd::ops::OpTraits;

  auto* op = sd::ops::OpRegistrator::getInstance().getOperation(opName.c_str());
  if (op == nullptr) return dg::op::kind::LastSymbol;
  auto* desc = op->getOpDescriptor();
  if (desc == nullptr) return dg::op::kind::LastSymbol;
  uint32_t traits = desc->getTraits();
  if (traits == 0) return dg::op::kind::LastSymbol;

  if (traits & OpTraits::OP_TRAIT_MATMUL)              return dg::op::kind::MatMul;
  if (traits & OpTraits::OP_TRAIT_NORMALIZATION)        return dg::op::kind::LayerNorm;
  if (traits & OpTraits::OP_TRAIT_ACTIVATION)           return dg::op::kind::ReLU;
  if (traits & OpTraits::OP_TRAIT_IDENTITY)            return dg::op::kind::StaticReshape;
  if (traits & OpTraits::OP_TRAIT_CAST)                return dg::op::kind::TypeCast;
  if (traits & OpTraits::OP_TRAIT_REDUCTION)           return dg::op::kind::ReduceSum;

  return dg::op::kind::LastSymbol;
}

// ─── Data type mapping ──────────────────────────────────────────────────────

dg::logical_tensor::data_type OneDnnGraphBackend::mapDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32: return dg::logical_tensor::data_type::f32;
    case DataType::BFLOAT16: return dg::logical_tensor::data_type::bf16;
    case DataType::HALF: return dg::logical_tensor::data_type::f16;
    case DataType::INT32: return dg::logical_tensor::data_type::s32;
    case DataType::INT8: return dg::logical_tensor::data_type::s8;
    case DataType::UINT8: return dg::logical_tensor::data_type::u8;
    case DataType::BOOL: return dg::logical_tensor::data_type::boolean;
    // oneDNN Graph API does NOT support INT64, FLOAT64, UINT16, UINT32, UINT64, etc.
    // Return undef so callers can detect and skip ops with unsupported types.
    default: return dg::logical_tensor::data_type::undef;
  }
}

// ─── Segment fusibility check ───────────────────────────────────────────────

// Anchor ops: compute-intensive ops where oneDNN provides real optimization.
// A segment is only worth compiling if it contains at least one anchor.
// Elementwise ops alone are NOT anchors — they're only useful as post-ops
// fused INTO an anchor (e.g., matmul + relu).
static bool isAnchorOp(dg::op::kind kind) {
  switch (kind) {
    case dg::op::kind::MatMul:
    case dg::op::kind::Convolution:
    case dg::op::kind::SoftMax:
    case dg::op::kind::LogSoftmax:
    case dg::op::kind::LayerNorm:
    case dg::op::kind::BatchNormInference:
    case dg::op::kind::GroupNorm:
    case dg::op::kind::AvgPool:
    case dg::op::kind::MaxPool:
    case dg::op::kind::ReduceSum:
    case dg::op::kind::ReduceMean:
    case dg::op::kind::ReduceMin:
    case dg::op::kind::ReduceMax:
    case dg::op::kind::ReduceProd:
      return true;
    default:
      return false;
  }
}

bool OneDnnGraphBackend::canFuseSegment(NativeSlot* slots, int start, int end) {
  if (!isAvailable()) {
    DSP_DIAG(BACKEND, "OneDnnGraphBackend::canFuseSegment: oneDNN not available");
    return false;
  }

  int mappableOps = 0;
  int anchorOps = 0;
  int totalOps = end - start + 1;
  bool hasCast = false;

  for (int i = start; i <= end; i++) {
    const auto& opName = slots[i].ident.opName;
    auto kind = mapOpKind(opName);
    if (kind != dg::op::kind::LastSymbol) {
      mappableOps++;
      if (isAnchorOp(kind)) anchorOps++;
    }
    // OneDNN Graph's add_op fails on TypeCast ops — skip segments containing
    // cast to avoid wasted compile attempts that always cascade to OpenVINO.
    if (kind == dg::op::kind::TypeCast) hasCast = true;
  }

  if (hasCast) {
    DSP_DIAG(SEGMENT, "OneDnnGraphBackend::canFuseSegment [%d-%d]: contains cast ops "
             "— skipping (oneDNN add_op fails on TypeCast)",
             start, end);
    return false;
  }

  // Require at least one anchor op. A segment of pure elementwise ops
  // gets no benefit from oneDNN — its optimization is in FUSING elementwise
  // into anchors (matmul+relu, conv+bias+gelu, etc.), not running them standalone.
  if (anchorOps < 1) {
    DSP_DIAG(SEGMENT, "OneDnnGraphBackend::canFuseSegment [%d-%d]: no anchor ops "
             "(mappable=%d totalOps=%d) — skipping",
             start, end, mappableOps, totalOps);
    return false;
  }

  // Accept: at least one anchor op exists, and enough mappable ops for fusion benefit.
  bool coverageOk = mappableOps >= MIN_MAPPABLE_OPS;
  DSP_DIAG(SEGMENT, "OneDnnGraphBackend::canFuseSegment [%d-%d]: "
           "anchors=%d mappable=%d/%d canFuse=%s",
           start, end, anchorOps, mappableOps, totalOps,
           coverageOk ? "true" : "false");
  return coverageOk;
}

// ─── Thread-local native slot executor ─────────────────────────────────────

thread_local OneDnnGraphBackend::NativeSlotExecutor OneDnnGraphBackend::nativeExecutor_ = nullptr;

void OneDnnGraphBackend::setNativeSlotExecutor(NativeSlotExecutor executor) {
  nativeExecutor_ = std::move(executor);
}

void OneDnnGraphBackend::clearNativeSlotExecutor() {
  nativeExecutor_ = nullptr;
}

// ─── Graph building ─────────────────────────────────────────────────────────
//
// Supports two modes:
//   Pure-OneDNN:  All ops in [startSlot, endSlot] are mappable. One dg::graph
//                 covering all slots, partitioned + compiled by oneDNN.
//   Mixed:        Some ops are unmappable (e.g. gather, rope, stridedslice).
//                 We split the range into consecutive runs of:
//                   - "OneDNN islands": contiguous mappable ops compiled as sub-graphs
//                   - "native ranges": contiguous unmappable ops recorded for
//                     slot-by-slot execution via nativeExecutor_
//                 The executionSchedule vector records the interleaved order.
//
// Unmappable ops that appear between mappable ops are not added to any dg::graph.
// Their input tensors are treated as external inputs for the next OneDNN island
// (since native execution will write those outputs before the next island runs).

OneDnnGraphBackend::CompiledSegment OneDnnGraphBackend::buildGraph(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots) {

  CompiledSegment result;
  result.valid = false;

  // Helper: resolve an NDArray* from a wiring source index.
  auto resolveWiringArray = [&](int srcIdx) -> NDArray* {
    if (srcIdx >= 0 && srcIdx < totalOutputSlots) return outputSlots[srcIdx];
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs) return externalInputs[extIdx];
    }
    return nullptr;
  };

  // Helper: check if a slot is fully mappable to oneDNN.
  // An op is mappable only if:
  //   (1) its name maps to an oneDNN kind
  //   (2) all its input/output arrays have data types supported by oneDNN Graph API
  //   (3) for multi-input ops, all inputs have the SAME data type (oneDNN binary
  //       ops reject mixed types like f32+f16; the partitioner marks such ops as
  //       unsupported, which blocks fusion of the entire connected sub-graph)
  auto isSlotMappable = [&](int s) -> bool {
    if (mapOpKind(slots[s].ident.opName) == dg::op::kind::LastSymbol) return false;

    // Pow semantic check: oneDNN Pow takes 1 input + scalar beta attribute.
    // nd4j pow takes 2 tensor inputs (base, exponent). Only mappable if the
    // exponent (input[1]) is a scalar constant whose value we can extract.
    auto kind = mapOpKind(slots[s].ident.opName);
    if (kind == dg::op::kind::Pow && slots[s].wiring.numInputs >= 2) {
      NDArray* expArr = resolveWiringArray(slots[s].wiring.inputSourceIndices[1]);
      if (expArr == nullptr || !expArr->isScalar()) {
        DSP_DIAG(COMPILE, "OneDnnGraphBackend: slot %d op '%s' has non-scalar exponent — "
                 "routing to native (oneDNN Pow requires scalar beta attribute)",
                 s, slots[s].ident.opName.c_str());
        return false;
      }
    }

    // Collect input data types, checking for unsupported types
    dg::logical_tensor::data_type firstInputDtype = dg::logical_tensor::data_type::undef;
    for (int i = 0; i < slots[s].wiring.numInputs; i++) {
      NDArray* arr = resolveWiringArray(slots[s].wiring.inputSourceIndices[i]);
      if (arr == nullptr) continue;
      auto dt = mapDataType(arr->dataType());
      if (dt == dg::logical_tensor::data_type::undef) {
        DSP_DIAG(COMPILE, "OneDnnGraphBackend: slot %d op '%s' input %d has unsupported dtype %d",
                 s, slots[s].ident.opName.c_str(), i, static_cast<int>(arr->dataType()));
        return false;
      }
      if (firstInputDtype == dg::logical_tensor::data_type::undef) {
        firstInputDtype = dt;
      } else if (dt != firstInputDtype && slots[s].wiring.numInputs > 1) {
        // Mixed input types: oneDNN binary ops require matching types.
        // Route to native execution instead of building a graph that will
        // produce unsupported partitions and block fusion.
        DSP_DIAG(COMPILE, "OneDnnGraphBackend: slot %d op '%s' has mixed input types "
                 "(input 0 dtype=%d, input %d dtype=%d) — routing to native",
                 s, slots[s].ident.opName.c_str(),
                 static_cast<int>(firstInputDtype), i, static_cast<int>(dt));
        return false;
      }
    }

    // Check output data types
    for (int i = 0; i < slots[s].wiring.numOutputs; i++) {
      int outIdx = slots[s].wiring.outputSlotIndices[i];
      if (outIdx >= 0 && outIdx < totalOutputSlots) {
        if (outputSlots[outIdx] == nullptr) {
          // Output array is null (e.g. in-place-fused VIEW_OF_SLOT at compile time).
          // For cast ops this is fatal: buildGraph() would create a logical_tensor
          // with unknown shape and oneDNN's add_op() would throw.  Route to native.
          if (kind == dg::op::kind::TypeCast) {
            DSP_DIAG(COMPILE, "OneDnnGraphBackend: slot %d op '%s' output %d is nullptr "
                     "— routing cast to native (cannot build logical_tensor without shape)",
                     s, slots[s].ident.opName.c_str(), i);
            return false;
          }
          continue;
        }
        if (mapDataType(outputSlots[outIdx]->dataType()) == dg::logical_tensor::data_type::undef) {
          DSP_DIAG(COMPILE, "OneDnnGraphBackend: slot %d op '%s' output %d has unsupported dtype %d",
                   s, slots[s].ident.opName.c_str(), i, static_cast<int>(outputSlots[outIdx]->dataType()));
          return false;
        }
      }
    }
    return true;
  };

  // Determine if this is a pure or mixed segment
  int totalOps = endSlot - startSlot + 1;
  int mappableOps = 0;
  for (int s = startSlot; s <= endSlot; s++) {
    if (isSlotMappable(s)) mappableOps++;
  }
  result.isMixedSegment = (mappableOps != totalOps);

  DSP_DIAG(COMPILE, "OneDnnGraphBackend::buildGraph [%d-%d]: totalOps=%d mappable=%d mixed=%s",
           startSlot, endSlot, totalOps, mappableOps,
           result.isMixedSegment ? "true" : "false");

  // ── Identify sub-ranges: runs of mappable and unmappable ops ────────────
  // A "sub-range" is [first, last] inclusive plus whether it's native or OneDNN.
  struct SubRange {
    int first, last;
    bool isNative;  // true = unmappable ops for native execution
  };
  std::vector<SubRange> subRanges;
  {
    int cur = startSlot;
    while (cur <= endSlot) {
      bool curMappable = isSlotMappable(cur);
      int runEnd = cur;
      while (runEnd + 1 <= endSlot && isSlotMappable(runEnd + 1) == curMappable) {
        runEnd++;
      }
      subRanges.push_back({cur, runEnd, !curMappable});
      cur = runEnd + 1;
    }
  }

  // ── Compile each OneDNN sub-range into a separate dg::graph ─────────────
  // Each OneDNN island is independent: inputs from prior native ranges are treated
  // as external inputs with pre-known shapes.
  //
  // IMPORTANT: Use a single global tensorId counter across all islands.
  // Each island builds its own dg::graph but they all share result.tensorIdToSlotMap.
  // If each island reset tensorId=0, different islands would assign the same numeric
  // ID to different slots/externals, corrupting the tensorIdToSlotMap lookups used
  // at execution time to resolve NDArray* pointers from PartitionEntry tensor IDs.
  size_t globalTensorId = 0;

  for (auto& sr : subRanges) {
    if (sr.isNative) {
      // Record native range: will be executed by nativeExecutor_ at execution time
      int nativeIdx = static_cast<int>(result.nativeRanges.size());
      result.nativeRanges.push_back({sr.first, sr.last});
      result.executionSchedule.push_back({true, nativeIdx});

      // Audit: mark all slots in native range as natively handled
      for (int s = sr.first; s <= sr.last; s++) {
        CompilationAuditEntry entry;
        entry.slotIndex = s;
        entry.opName = slots[s].ident.opName;
        entry.wasCompiled = false;
        entry.isNativeHandled = true;
        entry.reason = "unmappable op — executed natively via NativeSlotExecutor";
        result.compilationAudit.push_back(std::move(entry));
      }
      DSP_DIAG(COMPILE, "OneDnnGraphBackend::buildGraph: native range [%d-%d] (%d ops) added to schedule",
               sr.first, sr.last, sr.last - sr.first + 1);
      continue;
    }

    // ── Build one dg::graph for this OneDNN island ──────────────────────
    try {
      dg::graph g(dnnl::engine::kind::cpu);
      // Use globalTensorId (NOT a local reset-to-zero counter) so IDs are unique
      // across all islands within this segment. All islands share result.tensorIdToSlotMap,
      // so collisions would cause wrong NDArray* resolution at execute time.
      size_t& tensorId = globalTensorId;

      std::unordered_map<int, size_t> slotToTensorId;
      std::unordered_map<int, size_t> extToTensorId;
      std::unordered_map<size_t, dg::logical_tensor> logicalTensors;

      // Helper: create a logical tensor with the right layout.
      // - rank >= 2: layout_type::strided (gives oneDNN freedom to optimize)
      // - rank == 1: explicit strides {1} (oneDNN crashes on 1D + layout_type::strided)
      // - rank == 0: layout_type::strided (scalars work fine)
      // Actual strides are provided only at execution time (via dg::tensor constructor).
      // This matches the working pattern in sdpa.cpp.
      auto makeLT = [](size_t id, dg::logical_tensor::data_type dtype,
                       const std::vector<int64_t>& shape) -> dg::logical_tensor {
        if (shape.size() == 1) {
          // oneDNN Graph API bug: 1D tensors with layout_type::strided throw
          // "could not create logical_tensor with property". Use explicit strides.
          return dg::logical_tensor(id, dtype, shape, std::vector<int64_t>{1});
        }
        return dg::logical_tensor(id, dtype, shape, dg::logical_tensor::layout_type::strided);
      };

      auto getExternalInputTensor = [&](int extIdx) -> dg::logical_tensor {
        auto it = extToTensorId.find(extIdx);
        if (it != extToTensorId.end()) return logicalTensors.at(it->second);

        NDArray* arr = externalInputs[extIdx];
        if (arr == nullptr) THROW_EXCEPTION("OneDnnGraphBackend: null external input");

        size_t id = tensorId++;
        auto dtype = mapDataType(arr->dataType());
        if (dtype == dg::logical_tensor::data_type::undef) dtype = dg::logical_tensor::data_type::f32;
        int rank = arr->rankOf();
        std::vector<int64_t> shape(rank);
        for (int d = 0; d < rank; d++) { shape[d] = arr->sizeAt(d); }
        auto lt = makeLT(id, dtype, shape);
        logicalTensors.emplace(id, lt);
        extToTensorId[extIdx] = id;
        result.tensorIdToSlotMap[id] = -(extIdx + 1);
        return lt;
      };

      auto getSlotOutputTensor = [&](int slotIdx, NDArray* arr) -> dg::logical_tensor {
        auto it = slotToTensorId.find(slotIdx);
        if (it != slotToTensorId.end()) return logicalTensors.at(it->second);

        size_t id = tensorId++;
        auto dtype = mapDataType(arr != nullptr ? arr->dataType() : DataType::FLOAT32);
        if (dtype == dg::logical_tensor::data_type::undef) dtype = dg::logical_tensor::data_type::f32;
        dg::logical_tensor lt;
        if (arr != nullptr) {
          int rank = arr->rankOf();
          std::vector<int64_t> shape(rank);
          for (int d = 0; d < rank; d++) { shape[d] = arr->sizeAt(d); }
          lt = makeLT(id, dtype, shape);
        } else {
          lt = dg::logical_tensor(id, dtype, dg::logical_tensor::layout_type::strided);
        }
        logicalTensors.emplace(id, lt);
        slotToTensorId[slotIdx] = id;
        result.tensorIdToSlotMap[id] = slotIdx;
        return lt;
      };

      int opsAdded = 0;
      // Map from dg::op ID → slot index, so we can recover which slots belong
      // to each partition after oneDNN partitions the graph.
      std::unordered_map<size_t, int> opIdToSlot;

      for (int s = sr.first; s <= sr.last; s++) {
        NativeSlot& slot = slots[s];
        auto kind = mapOpKind(slot.ident.opName);
        // All slots in this sub-range are mappable — kind != LastSymbol guaranteed

        size_t opId = tensorId++;
        opIdToSlot[opId] = s;
        dg::op dgOp(opId, kind, slot.ident.opName);

        // Set op-specific attributes (same as before)
        if (kind == dg::op::kind::MatMul) {
          bool transposeA = (slot.args.numIArgs > 0 && slot.args.iArgs[0] != 0);
          bool transposeB = (slot.args.numIArgs > 1 && slot.args.iArgs[1] != 0);
          dgOp.set_attr<bool>(dg::op::attr::transpose_a, transposeA);
          dgOp.set_attr<bool>(dg::op::attr::transpose_b, transposeB);
        } else if (kind == dg::op::kind::SoftMax || kind == dg::op::kind::LogSoftmax) {
          int64_t axis = -1;
          if (slot.args.numIArgs > 0) axis = static_cast<int64_t>(slot.args.iArgs[0]);
          dgOp.set_attr<int64_t>(dg::op::attr::axis, axis);
        } else if (kind == dg::op::kind::Concat) {
          int64_t axis = 0;
          if (slot.args.numIArgs > 0) axis = static_cast<int64_t>(slot.args.iArgs[0]);
          dgOp.set_attr<int64_t>(dg::op::attr::axis, axis);
        } else if (kind == dg::op::kind::Elu) {
          float alpha = 1.0f;
          if (slot.args.numTArgs > 0) alpha = static_cast<float>(slot.args.tArgs[0]);
          dgOp.set_attr<float>(dg::op::attr::alpha, alpha);
        } else if (kind == dg::op::kind::Clamp) {
          float minVal = -std::numeric_limits<float>::infinity();
          float maxVal = std::numeric_limits<float>::infinity();
          if (slot.args.numTArgs > 0) minVal = static_cast<float>(slot.args.tArgs[0]);
          if (slot.args.numTArgs > 1) maxVal = static_cast<float>(slot.args.tArgs[1]);
          dgOp.set_attr<float>(dg::op::attr::min, minVal);
          dgOp.set_attr<float>(dg::op::attr::max, maxVal);
        } else if (kind == dg::op::kind::Pow) {
          // oneDNN Pow: x^beta (single input, scalar beta attribute).
          // nd4j pow has 2 tensor inputs — isSlotMappable verified input[1] is scalar.
          // Extract its value as the beta attribute.
          NDArray* expArr = resolveWiringArray(slot.wiring.inputSourceIndices[1]);
          float beta = expArr->e<float>(0);
          dgOp.set_attr<float>(dg::op::attr::beta, beta);
        } else if (kind == dg::op::kind::LeakyReLU) {
          float alpha = 0.01f;
          if (slot.args.numTArgs > 0) alpha = static_cast<float>(slot.args.tArgs[0]);
          dgOp.set_attr<float>(dg::op::attr::alpha, alpha);
        } else if (kind == dg::op::kind::HardSigmoid) {
          float alpha = 1.0f / 6.0f;
          float beta = 0.5f;
          if (slot.args.numTArgs > 0) alpha = static_cast<float>(slot.args.tArgs[0]);
          if (slot.args.numTArgs > 1) beta = static_cast<float>(slot.args.tArgs[1]);
          dgOp.set_attr<float>(dg::op::attr::alpha, alpha);
          dgOp.set_attr<float>(dg::op::attr::beta, beta);
        } else if (kind == dg::op::kind::BatchNormInference) {
          float epsilon = 1e-5f;
          if (slot.args.numTArgs > 0) epsilon = static_cast<float>(slot.args.tArgs[0]);
          dgOp.set_attr<float>(dg::op::attr::epsilon, epsilon);
        } else if (kind == dg::op::kind::GroupNorm) {
          int64_t groups = 1;
          if (slot.args.numIArgs > 0) groups = static_cast<int64_t>(slot.args.iArgs[0]);
          dgOp.set_attr<int64_t>(dg::op::attr::groups, groups);
          float epsilon = 1e-5f;
          if (slot.args.numTArgs > 0) epsilon = static_cast<float>(slot.args.tArgs[0]);
          dgOp.set_attr<float>(dg::op::attr::epsilon, epsilon);
        } else if (kind == dg::op::kind::LayerNorm) {
          float epsilon = 1e-5f;
          if (slot.args.numTArgs > 0) epsilon = static_cast<float>(slot.args.tArgs[0]);
          dgOp.set_attr<float>(dg::op::attr::epsilon, epsilon);
          dgOp.set_attr<bool>(dg::op::attr::keep_stats, false);
        } else if (kind == dg::op::kind::ReduceSum || kind == dg::op::kind::ReduceMean ||
                   kind == dg::op::kind::ReduceMin || kind == dg::op::kind::ReduceMax ||
                   kind == dg::op::kind::ReduceProd) {
          if (slot.args.numIArgs > 0) {
            std::vector<int64_t> axes(slot.args.numIArgs);
            for (int i = 0; i < slot.args.numIArgs; i++) {
              axes[i] = static_cast<int64_t>(slot.args.iArgs[i]);
            }
            dgOp.set_attr<std::vector<int64_t>>(dg::op::attr::axes, axes);
          }
          dgOp.set_attr<bool>(dg::op::attr::keep_dims, false);
          if (slot.args.numBArgs > 0 && slot.args.bArgs[0]) {
            dgOp.set_attr<bool>(dg::op::attr::keep_dims, true);
          }
        }

        // Wire inputs
        std::vector<dg::logical_tensor> inputTensors;
        for (int i = 0; i < slot.wiring.numInputs; i++) {
          // Pow: skip second input — exponent is set as beta attribute, not a tensor input.
          // oneDNN Pow expects exactly 1 input (the base).
          if (kind == dg::op::kind::Pow && i == 1) continue;

          int srcIdx = slot.wiring.inputSourceIndices[i];
          if (srcIdx >= 0) {
            NDArray* arr = (srcIdx < totalOutputSlots) ? outputSlots[srcIdx] : nullptr;
            inputTensors.push_back(getSlotOutputTensor(srcIdx, arr));
          } else {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExternalInputs) {
              inputTensors.push_back(getExternalInputTensor(extIdx));
            }
          }
        }
        dgOp.add_inputs(inputTensors);

        // Wire outputs
        std::vector<dg::logical_tensor> outputTensors;
        for (int i = 0; i < slot.wiring.numOutputs; i++) {
          int outSlotIdx = slot.wiring.outputSlotIndices[i];
          NDArray* arr = (outSlotIdx >= 0 && outSlotIdx < totalOutputSlots)
                             ? outputSlots[outSlotIdx] : nullptr;
          outputTensors.push_back(getSlotOutputTensor(outSlotIdx, arr));
        }
        dgOp.add_outputs(outputTensors);

        // Dump detailed logical tensor info for debugging partition support
        for (size_t ti = 0; ti < inputTensors.size(); ti++) {
          auto& lt = inputTensors[ti];
          auto dims = lt.get_dims();
          std::string shapeStr = "[";
          for (size_t d = 0; d < dims.size(); d++) {
            if (d > 0) shapeStr += ",";
            shapeStr += std::to_string(dims[d]);
          }
          shapeStr += "]";
          DSP_DIAG(COMPILE, "    input[%d] tensorId=%zu dtype=%d ndims=%d shape=%s",
                   static_cast<int>(ti), lt.get_id(),
                   static_cast<int>(lt.get_data_type()),
                   static_cast<int>(dims.size()), shapeStr.c_str());
        }
        for (size_t ti = 0; ti < outputTensors.size(); ti++) {
          auto& lt = outputTensors[ti];
          auto dims = lt.get_dims();
          std::string shapeStr = "[";
          for (size_t d = 0; d < dims.size(); d++) {
            if (d > 0) shapeStr += ",";
            shapeStr += std::to_string(dims[d]);
          }
          shapeStr += "]";
          DSP_DIAG(COMPILE, "    output[%d] tensorId=%zu dtype=%d ndims=%d shape=%s",
                   static_cast<int>(ti), lt.get_id(),
                   static_cast<int>(lt.get_data_type()),
                   static_cast<int>(dims.size()), shapeStr.c_str());
        }
        DSP_DIAG(COMPILE, "  slot %d op '%s' kind=%d numInputs=%d numOutputs=%d opId=%zu",
                 s, slot.ident.opName.c_str(), static_cast<int>(kind),
                 static_cast<int>(inputTensors.size()), static_cast<int>(outputTensors.size()),
                 opId);

        try {
          g.add_op(dgOp);
          opsAdded++;
        } catch (const std::exception& e) {
          DSP_DIAG(COMPILE, "OneDNN Graph: add_op failed for slot %d op '%s': %s",
                   s, slot.ident.opName.c_str(), e.what());
          return result;
        }

        CompilationAuditEntry auditEntry;
        auditEntry.slotIndex = s;
        auditEntry.opName = slot.ident.opName;
        auditEntry.wasCompiled = true;
        result.compilationAudit.push_back(std::move(auditEntry));
      }

      if (opsAdded < 1) continue;  // Empty island — shouldn't happen

      g.finalize();

      auto partitions = g.get_partitions();
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: OneDNN island [%d-%d] → %d partitions (%d ops)",
               sr.first, sr.last, static_cast<int>(partitions.size()), opsAdded);
      if (partitions.empty()) {
        DSP_DIAG(COMPILE, "OneDnnGraphBackend: no partitions for island [%d-%d]", sr.first, sr.last);
        continue;
      }

      // ── Clone diagnostic: if first partition unsupported, reproduce from scratch ─
      // Build a completely independent graph with the same ops to determine
      // whether the issue is in our construction or the oneDNN library state.
      if (!partitions.empty() && !partitions[0].is_supported()) {
        // -fno-threadsafe-statics: use std::call_once for thread-safe initialization.
        static std::once_flag cloneDiagFlag;
        std::call_once(cloneDiagFlag, [&]() {
          try {
            // Rebuild same graph from scratch using local IDs
            dg::graph cloneG(dnnl::engine::kind::cpu);
            size_t cloneId = 80000;
            std::unordered_map<size_t, dg::logical_tensor> cloneLTs;
            std::unordered_map<size_t, size_t> origToClone;  // orig tensor ID -> clone ID

            for (int s2 = sr.first; s2 <= sr.last; s2++) {
              NativeSlot& slot2 = slots[s2];
              auto kind2 = mapOpKind(slot2.ident.opName);
              if (kind2 == dg::op::kind::LastSymbol) continue;
              size_t cloneOpId = cloneId++;
              dg::op cloneOp(cloneOpId, kind2, "clone_" + slot2.ident.opName);
              if (kind2 == dg::op::kind::MatMul) {
                cloneOp.set_attr<bool>(dg::op::attr::transpose_a, false);
                cloneOp.set_attr<bool>(dg::op::attr::transpose_b, false);
              }
              // Clone inputs
              std::vector<dg::logical_tensor> cloneInputs;
              for (int ci = 0; ci < slot2.wiring.numInputs; ci++) {
                int srcIdx = slot2.wiring.inputSourceIndices[ci];
                NDArray* arr = resolveWiringArray(srcIdx);
                if (!arr) continue;
                // Check if we already created a clone LT for this source
                size_t origId = 0;
                auto sit = slotToTensorId.find(srcIdx);
                if (sit != slotToTensorId.end()) origId = sit->second;
                else {
                  auto eit = extToTensorId.find(-(srcIdx + 1));
                  if (eit != extToTensorId.end()) origId = eit->second;
                  else origId = srcIdx + 50000;
                }
                if (origToClone.find(origId) == origToClone.end()) {
                  size_t cid = cloneId++;
                  origToClone[origId] = cid;
                  int rank = arr->rankOf();
                  std::vector<int64_t> sh(rank);
                  for (int d = 0; d < rank; d++) sh[d] = arr->sizeAt(d);
                  auto dt = mapDataType(arr->dataType());
                  if (dt == dg::logical_tensor::data_type::undef) dt = dg::logical_tensor::data_type::f32;
                  cloneLTs[cid] = makeLT(cid, dt, sh);
                }
                cloneInputs.push_back(cloneLTs[origToClone[origId]]);
              }
              cloneOp.add_inputs(cloneInputs);
              // Clone outputs
              std::vector<dg::logical_tensor> cloneOutputs;
              for (int co = 0; co < slot2.wiring.numOutputs; co++) {
                int outIdx = slot2.wiring.outputSlotIndices[co];
                NDArray* outArr = (outIdx >= 0 && outIdx < totalOutputSlots) ? outputSlots[outIdx] : nullptr;
                size_t origOutId = 0;
                auto osit = slotToTensorId.find(outIdx);
                if (osit != slotToTensorId.end()) origOutId = osit->second;
                else origOutId = outIdx + 60000;
                if (origToClone.find(origOutId) == origToClone.end()) {
                  size_t cid = cloneId++;
                  origToClone[origOutId] = cid;
                  if (outArr) {
                    int rank = outArr->rankOf();
                    std::vector<int64_t> sh(rank);
                    for (int d = 0; d < rank; d++) sh[d] = outArr->sizeAt(d);
                    auto dt = mapDataType(outArr->dataType());
                    if (dt == dg::logical_tensor::data_type::undef) dt = dg::logical_tensor::data_type::f32;
                    cloneLTs[cid] = makeLT(cid, dt, sh);
                  } else {
                    cloneLTs[cid] = dg::logical_tensor(cid, dg::logical_tensor::data_type::f32,
                                                        dg::logical_tensor::layout_type::strided);
                  }
                }
                cloneOutputs.push_back(cloneLTs[origToClone[origOutId]]);
              }
              cloneOp.add_outputs(cloneOutputs);
              cloneG.add_op(cloneOp);
            }
            cloneG.finalize();
            auto cloneParts = cloneG.get_partitions();
            DSP_DIAG(COMPILE, "OneDNN CLONE-DIAGNOSTIC: island [%d-%d] clone has %d partitions",
                     sr.first, sr.last, static_cast<int>(cloneParts.size()));
            for (size_t cp = 0; cp < cloneParts.size(); cp++) {
              DSP_DIAG(COMPILE, "  clone partition[%d] supported=%d numOps=%d",
                       static_cast<int>(cp), cloneParts[cp].is_supported() ? 1 : 0,
                       static_cast<int>(cloneParts[cp].get_ops_num()));
            }
          } catch (const std::exception& e) {
            DSP_DIAG(COMPILE, "OneDNN CLONE-DIAGNOSTIC: EXCEPTION: %s", e.what());
          }
        });
      }

      // ── Process partitions: compile supported, convert unsupported to native ─
      // oneDNN's partitioner may split our island into some supported fused
      // partitions and some unsupported ones. Instead of failing the entire
      // island on any unsupported partition, we convert those ops to native
      // ranges and interleave with the compiled partitions.
      //
      // To maintain correct execution order, we sort all partition entries
      // (both supported and unsupported) by their minimum slot index.

      struct PartitionSlotInfo {
        int minSlot, maxSlot;
        int partitionIdx;  // into partitions vector
        bool supported;
        // For supported partitions:
        std::vector<dg::logical_tensor> inputLTs, outputLTs;
      };
      std::vector<PartitionSlotInfo> partInfos;

      int supportedCount = 0, unsupportedCount = 0;
      for (size_t partIdx = 0; partIdx < partitions.size(); partIdx++) {
        auto& partition = partitions[partIdx];
        PartitionSlotInfo info;
        info.supported = partition.is_supported();
        info.partitionIdx = -1;

        // Determine which slots this partition covers
        auto opIds = partition.get_ops();
        info.minSlot = INT_MAX;
        info.maxSlot = INT_MIN;
        for (auto opId : opIds) {
          auto slotIt = opIdToSlot.find(opId);
          if (slotIt != opIdToSlot.end()) {
            info.minSlot = std::min(info.minSlot, slotIt->second);
            info.maxSlot = std::max(info.maxSlot, slotIt->second);
          }
        }
        DSP_DIAG(COMPILE, "  partition[%d] supported=%d numOps=%d opIds=[%s] minSlot=%d maxSlot=%d",
                 static_cast<int>(partIdx), info.supported ? 1 : 0,
                 static_cast<int>(opIds.size()),
                 opIds.empty() ? "" : std::to_string(opIds[0]).c_str(),
                 info.minSlot, info.maxSlot);
        if (info.minSlot == INT_MAX) {
          DSP_DIAG(COMPILE, "  partition[%d] SKIPPED: no slot mapping for any opId", static_cast<int>(partIdx));
          continue;  // No slot mapping found
        }

        if (info.supported) {
          supportedCount++;
          auto inPorts = partition.get_input_ports();
          auto outPorts = partition.get_output_ports();
          for (auto& lt : inPorts) {
            auto it = logicalTensors.find(lt.get_id());
            if (it != logicalTensors.end()) info.inputLTs.push_back(it->second);
          }
          for (auto& lt : outPorts) {
            auto it = logicalTensors.find(lt.get_id());
            if (it != logicalTensors.end()) info.outputLTs.push_back(it->second);
          }

          try {
            CompiledSegment::PartitionEntry entry;
            entry.compiledPartition = partition.compile(info.inputLTs, info.outputLTs, engine_);
            entry.startSlot = info.minSlot;
            entry.endSlot = info.maxSlot;
            for (auto& lt : info.inputLTs) entry.inputTensorIds.push_back(lt.get_id());
            for (auto& lt : info.outputLTs) entry.outputTensorIds.push_back(lt.get_id());
            info.partitionIdx = static_cast<int>(result.partitions.size());
            result.partitions.push_back(std::move(entry));
          } catch (const std::exception& e) {
            DSP_DIAG(COMPILE, "OneDnnGraphBackend: island [%d-%d] partition [%d-%d] compile failed: %s",
                     sr.first, sr.last, info.minSlot, info.maxSlot, e.what());
            // Treat compile failure as unsupported — convert to native range
            info.supported = false;
            supportedCount--;
            unsupportedCount++;
          }
        } else {
          unsupportedCount++;
        }

        partInfos.push_back(std::move(info));
      }

      // Sort by slot order so execution schedule is correct
      std::sort(partInfos.begin(), partInfos.end(),
                [](const PartitionSlotInfo& a, const PartitionSlotInfo& b) {
                  return a.minSlot < b.minSlot;
                });

      // Build interleaved execution schedule:
      // - Supported partitions → OneDNN execution
      // - Unsupported partitions → native slot ranges
      // - Merge adjacent unsupported ranges for efficiency
      for (auto& info : partInfos) {
        if (info.supported && info.partitionIdx >= 0) {
          result.executionSchedule.push_back({false, info.partitionIdx});
        } else {
          // Convert to native range
          int nativeIdx = static_cast<int>(result.nativeRanges.size());
          result.nativeRanges.push_back({info.minSlot, info.maxSlot});
          result.executionSchedule.push_back({true, nativeIdx});
          result.isMixedSegment = true;  // Island became mixed after partition rejection

          // Update audit: these ops are now natively handled
          for (auto& audit : result.compilationAudit) {
            if (audit.slotIndex >= info.minSlot && audit.slotIndex <= info.maxSlot) {
              audit.wasCompiled = false;
              audit.isNativeHandled = true;
              audit.reason = "oneDNN partition unsupported — executed natively";
            }
          }
          DSP_DIAG(COMPILE, "OneDnnGraphBackend: island [%d-%d] unsupported partition [%d-%d] "
                   "converted to native range",
                   sr.first, sr.last, info.minSlot, info.maxSlot);
        }
      }
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: island [%d-%d] compiled %d/%d partitions "
               "(%d converted to native)",
               sr.first, sr.last, supportedCount, supportedCount + unsupportedCount,
               unsupportedCount);

    } catch (const std::exception& e) {
      DSP_DIAG(COMPILE, "OneDnnGraphBackend: island [%d-%d] build failed: %s",
               sr.first, sr.last, e.what());
      return result;
    }
  }

  result.valid = !result.partitions.empty();
  if (result.isMixedSegment) {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend: mixed segment [%d-%d] compiled: "
             "%d partitions, %d native ranges, %d schedule steps, valid=%s",
             startSlot, endSlot,
             static_cast<int>(result.partitions.size()),
             static_cast<int>(result.nativeRanges.size()),
             static_cast<int>(result.executionSchedule.size()),
             result.valid ? "true" : "false");
  } else {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend: pure-OneDNN segment [%d-%d] compiled: "
             "%d partitions, valid=%s",
             startSlot, endSlot,
             static_cast<int>(result.partitions.size()),
             result.valid ? "true" : "false");
  }

  return result;
}

// ─── Compile segment ────────────────────────────────────────────────────────

bool OneDnnGraphBackend::compileSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    LongType shapeKey,
    int totalSlots,
    int* requestedOutputSlotIndices,
    int numRequestedOutputs) {

  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, shapeKey};

  std::lock_guard<std::mutex> lock(cacheMtx_);

  // Negative cache check: skip known-bad segment+shape combinations
  // (mirrors Triton's failedCache_ pattern to avoid re-compiling failures)
  if (failedCache_.count(key) > 0) {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: NEGATIVE cache HIT "
             "(shapeKey=0x%llx) — skipping known-bad compilation",
             seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);
    return false;
  }

  auto it = cache_.find(key);
  if (it != cache_.end() && it->second.valid) {
    DSP_DIAG(JIT, "OneDnnGraphBackend::compileSegment [%d-%d]: cache HIT (shapeKey=0x%llx)",
             seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);
    return true;  // Already compiled for this shape
  }

  DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: cache MISS, building graph (shapeKey=0x%llx)",
           seg.def.startSlot, seg.def.endSlot, (long long)shapeKey);

  auto compiled = buildGraph(slots, seg.def.startSlot, seg.def.endSlot,
                             externalInputs, numExternalInputs,
                             outputSlots, totalOutputSlots);
  compiled.shapeKey = shapeKey;

  // Store compilation audit for validation
  lastCompilationAudit_ = compiled.compilationAudit;

  if (compiled.valid) {
    DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: SUCCESS partitions=%d",
             seg.def.startSlot, seg.def.endSlot, (int)compiled.partitions.size());
    // Clear negative cache entry if a previous shape failed but this one succeeds
    failedCache_.erase(key);
    cache_[key] = std::move(compiled);
    return true;
  }

  // Insert into negative cache so we don't re-attempt this segment+shape
  failedCache_.insert(key);
  DSP_DIAG(COMPILE, "OneDnnGraphBackend::compileSegment [%d-%d]: FAILED (added to negative cache)",
           seg.def.startSlot, seg.def.endSlot);
  return false;
}

// ─── Compilation audit ──────────────────────────────────────────────────────

std::vector<CompilationAuditEntry> OneDnnGraphBackend::getLastCompilationAudit() const {
  return lastCompilationAudit_;
}

// ─── Execute segment ────────────────────────────────────────────────────────

Status OneDnnGraphBackend::executeSegment(
    GraphSegment& seg, NativeSlot* slots,
    NDArray** externalInputs, int numExternalInputs,
    NDArray** outputSlots, int totalOutputSlots,
    void* stream) {

  SegmentCacheKey key{seg.def.startSlot, seg.def.endSlot, seg.def.shapeKeyState.compiledShapeKey};

  CompiledSegment* compiled = nullptr;
  {
    std::lock_guard<std::mutex> lock(cacheMtx_);
    auto it = cache_.find(key);
    if (it == cache_.end() || !it->second.valid) {
      DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: no compiled segment found",
               seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
    compiled = &it->second;
  }

  auto& strm = getThreadStream();

  // Helper: resolve NDArray from slot index
  auto resolveArray = [&](int slotIdx) -> NDArray* {
    if (slotIdx < 0) {
      int extIdx = -(slotIdx + 1);
      return (extIdx < numExternalInputs) ? externalInputs[extIdx] : nullptr;
    }
    return (slotIdx < totalOutputSlots) ? outputSlots[slotIdx] : nullptr;
  };

  if (!compiled->isMixedSegment) {
    // ── Pure-OneDNN path: execute all partitions in order ────────────────
    DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: pure-OneDNN, %d partitions",
             seg.def.startSlot, seg.def.endSlot, (int)compiled->partitions.size());

    for (auto& part : compiled->partitions) {
      if (part.cachedInputTensors.empty()) {
        part.cachedInputTensors.resize(part.inputTensorIds.size());
        for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
          size_t tid = part.inputTensorIds[i];
          auto slotIt = compiled->tensorIdToSlotMap.find(tid);
          if (slotIt == compiled->tensorIdToSlotMap.end()) return Status::KERNEL_FAILURE;
          NDArray* arr = resolveArray(slotIt->second);
          if (!arr) return Status::KERNEL_FAILURE;
          int rank = arr->rankOf();
          std::vector<int64_t> shape(rank), strides(rank);
          for (int d = 0; d < rank; d++) { shape[d] = arr->sizeAt(d); strides[d] = arr->strideAt(d); }
          part.cachedInputTensors[i].lt = dg::logical_tensor(tid, mapDataType(arr->dataType()), shape, strides);
          part.cachedInputTensors[i].lastBuffer = arr->buffer();
        }
      }
      if (part.cachedOutputTensors.empty()) {
        part.cachedOutputTensors.resize(part.outputTensorIds.size());
        for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
          size_t tid = part.outputTensorIds[i];
          auto slotIt = compiled->tensorIdToSlotMap.find(tid);
          if (slotIt == compiled->tensorIdToSlotMap.end()) return Status::KERNEL_FAILURE;
          NDArray* arr = resolveArray(slotIt->second);
          if (!arr) return Status::KERNEL_FAILURE;
          int rank = arr->rankOf();
          std::vector<int64_t> shape(rank), strides(rank);
          for (int d = 0; d < rank; d++) { shape[d] = arr->sizeAt(d); strides[d] = arr->strideAt(d); }
          part.cachedOutputTensors[i].lt = dg::logical_tensor(tid, mapDataType(arr->dataType()), shape, strides);
          part.cachedOutputTensors[i].lastBuffer = arr->buffer();
        }
      }

      std::vector<dg::tensor> inputTensors;
      inputTensors.reserve(part.inputTensorIds.size());
      for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.inputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!arr) return Status::KERNEL_FAILURE;
        inputTensors.emplace_back(part.cachedInputTensors[i].lt, engine_, arr->buffer());
      }

      std::vector<dg::tensor> outputTensors;
      outputTensors.reserve(part.outputTensorIds.size());
      for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.outputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!arr) return Status::KERNEL_FAILURE;
        outputTensors.emplace_back(part.cachedOutputTensors[i].lt, engine_, arr->buffer());
      }

      try {
        part.compiledPartition.execute(strm, inputTensors, outputTensors);
      } catch (const std::exception& e) {
        DSP_DIAG(EXECUTE, "OneDnnGraphBackend: partition execute failed: %s", e.what());
        return Status::KERNEL_FAILURE;
      }
    }

    strm.wait();
    return Status::OK;
  }

  // ── Mixed-segment path: interleave OneDNN partitions and native ranges ──
  DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: mixed segment, "
           "%d schedule steps (%d partitions, %d native ranges)",
           seg.def.startSlot, seg.def.endSlot,
           (int)compiled->executionSchedule.size(),
           (int)compiled->partitions.size(),
           (int)compiled->nativeRanges.size());

  if (compiled->nativeRanges.empty() == false && !nativeExecutor_) {
    DSP_DIAG(EXECUTE, "OneDnnGraphBackend::executeSegment [%d-%d]: MISSING NativeSlotExecutor "
             "for mixed segment (%d native ranges). Call setNativeSlotExecutor() before execute.",
             seg.def.startSlot, seg.def.endSlot, (int)compiled->nativeRanges.size());
    return Status::KERNEL_FAILURE;
  }

  for (const auto& step : compiled->executionSchedule) {
    if (step.isNative) {
      // Execute native range via the plan's slot-by-slot executor
      const auto& nr = compiled->nativeRanges[step.index];
      DSP_DIAG(EXECUTE, "OneDnnGraphBackend: NATIVE range [%d-%d] (step.index=%d)",
               nr.startSlot, nr.endSlot, step.index);
      auto nativeStatus = nativeExecutor_(nr.startSlot, nr.endSlot);
      if (nativeStatus != Status::OK) {
        DSP_DIAG(EXECUTE, "OneDnnGraphBackend: native range [%d-%d] failed with status=%d",
                 nr.startSlot, nr.endSlot, static_cast<int>(nativeStatus));
        return nativeStatus;
      }
    } else {
      // Execute OneDNN partition
      auto& part = compiled->partitions[step.index];
      DSP_DIAG(EXECUTE, "OneDnnGraphBackend: OneDNN partition (index=%d) for island [%d-%d]",
               step.index, part.startSlot, part.endSlot);

      if (part.cachedInputTensors.empty()) {
        part.cachedInputTensors.resize(part.inputTensorIds.size());
        for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
          size_t tid = part.inputTensorIds[i];
          auto slotIt = compiled->tensorIdToSlotMap.find(tid);
          if (slotIt == compiled->tensorIdToSlotMap.end()) return Status::KERNEL_FAILURE;
          NDArray* arr = resolveArray(slotIt->second);
          if (!arr) return Status::KERNEL_FAILURE;
          int rank = arr->rankOf();
          std::vector<int64_t> shape(rank), strides(rank);
          for (int d = 0; d < rank; d++) { shape[d] = arr->sizeAt(d); strides[d] = arr->strideAt(d); }
          part.cachedInputTensors[i].lt = dg::logical_tensor(tid, mapDataType(arr->dataType()), shape, strides);
          part.cachedInputTensors[i].lastBuffer = arr->buffer();
        }
      }
      if (part.cachedOutputTensors.empty()) {
        part.cachedOutputTensors.resize(part.outputTensorIds.size());
        for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
          size_t tid = part.outputTensorIds[i];
          auto slotIt = compiled->tensorIdToSlotMap.find(tid);
          if (slotIt == compiled->tensorIdToSlotMap.end()) return Status::KERNEL_FAILURE;
          NDArray* arr = resolveArray(slotIt->second);
          if (!arr) return Status::KERNEL_FAILURE;
          int rank = arr->rankOf();
          std::vector<int64_t> shape(rank), strides(rank);
          for (int d = 0; d < rank; d++) { shape[d] = arr->sizeAt(d); strides[d] = arr->strideAt(d); }
          part.cachedOutputTensors[i].lt = dg::logical_tensor(tid, mapDataType(arr->dataType()), shape, strides);
          part.cachedOutputTensors[i].lastBuffer = arr->buffer();
        }
      }

      std::vector<dg::tensor> inputTensors;
      inputTensors.reserve(part.inputTensorIds.size());
      for (size_t i = 0; i < part.inputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.inputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!arr) return Status::KERNEL_FAILURE;
        inputTensors.emplace_back(part.cachedInputTensors[i].lt, engine_, arr->buffer());
      }

      std::vector<dg::tensor> outputTensors;
      outputTensors.reserve(part.outputTensorIds.size());
      for (size_t i = 0; i < part.outputTensorIds.size(); i++) {
        auto slotIt = compiled->tensorIdToSlotMap.find(part.outputTensorIds[i]);
        NDArray* arr = resolveArray(slotIt->second);
        if (!arr) return Status::KERNEL_FAILURE;
        outputTensors.emplace_back(part.cachedOutputTensors[i].lt, engine_, arr->buffer());
      }

      try {
        part.compiledPartition.execute(strm, inputTensors, outputTensors);
        // Flush completed OneDNN work before native ranges read the outputs
        strm.wait();
      } catch (const std::exception& e) {
        DSP_DIAG(EXECUTE, "OneDnnGraphBackend: mixed partition [%d-%d] execute failed: %s",
                 part.startSlot, part.endSlot, e.what());
        return Status::KERNEL_FAILURE;
      }
    }
  }

  return Status::OK;
}

// ─── Cache invalidation ─────────────────────────────────────────────────────

void OneDnnGraphBackend::invalidateCache() {
  std::lock_guard<std::mutex> lock(cacheMtx_);
  cache_.clear();
  failedCache_.clear();
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_ONEDNN
