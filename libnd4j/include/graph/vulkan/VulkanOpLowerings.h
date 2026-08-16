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

#ifndef LIBND4J_VULKAN_OP_LOWERINGS_H
#define LIBND4J_VULKAN_OP_LOWERINGS_H

#include <config.h>
#include <system/common.h>

#include <memory>

// ── Feature guards ────────────────────────────────────────────────────────────
// These lowering patterns require both the Vulkan runtime and MLIR compilation.
// config.h makes the numeric 0/1 feature values visible before this header's
// include guard is established, keeping declarations and definitions aligned.

#if defined(HAVE_VULKAN) && HAVE_VULKAN && defined(HAVE_MLIR) && HAVE_MLIR

// MLIR core
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Transforms/DialectConversion.h>


// Linalg ops we lower from
#include <mlir/Dialect/Linalg/IR/Linalg.h>

// Arith / Math for intermediate ops
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>

#include <mlir/Pass/Pass.h>

#include <string>

namespace sd {
namespace graph {

// ─────────────────────────────────────────────────────────────────────────────
//  MatmulToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers linalg.matmul to GPU/SCF/Arith IR. Each GPU invocation computes
// one output element and accumulates the K dimension in nd4j.accumulator_type.
//
class SD_LIB_EXPORT MatmulToSpirv : public mlir::OpRewritePattern<mlir::linalg::MatmulOp> {
 public:
  using OpRewritePattern<mlir::linalg::MatmulOp>::OpRewritePattern;

  /// Lower one linalg.matmul to GPU/SCF/Arith IR.
  mlir::LogicalResult matchAndRewrite(mlir::linalg::MatmulOp op,
                                      mlir::PatternRewriter& rewriter) const override;
};

/// Rank-3 counterpart of MatmulToSpirv.  The leading dimension is the batch
/// index and maps to the Vulkan z grid dimension.
class SD_LIB_EXPORT BatchMatmulToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::BatchMatmulOp> {
 public:
  using OpRewritePattern<mlir::linalg::BatchMatmulOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::BatchMatmulOp op,
      mlir::PatternRewriter& rewriter) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
//  RmsNormToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a descriptor-selected RMS normalization generic op to SCF/Arith/Math.
// The sum-of-squares reduction and all normalization arithmetic use
// nd4j.accumulator_type.
//
//   2. Normalise:
//        y[i] = x[i] * rsqrt(variance + epsilon)
//   3. Scale (if a gamma operand is present):
//        y[i] = y[i] * gamma[i]
//
// The variance reduction uses OpGroupIAdd / OpGroupFAdd with
// GroupOperation::Reduce in the Workgroup scope so that all invocations in the
// workgroup cooperate on the reduction in a single pass, without a global
// barrier.  For hidden dims larger than the workgroup size the loop over
// elements is unrolled in the IR.
//
// Expected MLIR op signature (emitted by SameDiff → MLIR bridge):
//   linalg.generic {
//       indexing_maps = [...], iterator_types = ["parallel", "reduction"],
//       nd4j.op_hash = <descriptor-derived hash> : i64, nd4j.epsilon = 1.0e-6 }
//
//
class SD_LIB_EXPORT RmsNormToSpirv : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  /// Pattern guard: only match linalg.generic ops tagged with a descriptor-derived nd4j.op_hash.
  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kEpsilonAttrName = "nd4j.epsilon";
  static constexpr float kDefaultEpsilon = 1.0e-6f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  RopeToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers descriptor-selected RoPE to SCF/Arith IR.
//
// Lowering strategy (per-token, per-head):
//   Given input X[B, S, H, D] and precomputed cos/sin tables indexed by
//   position:
//     x1 = X[b, s, h, 2*i]
//     x2 = X[b, s, h, 2*i+1]
//     out[b, s, h, 2*i]   = x1 * cos[s, i] - x2 * sin[s, i]
//     out[b, s, h, 2*i+1] = x2 * cos[s, i] + x1 * sin[s, i]
//
//   The cos/sin values are read from precomputed buffers.  SPIR-V built-ins
//   OpExtInst (GLSL.std.450 Cos / Sin) are NOT used because we require
//   deterministic output across invocations and Vulkan implementations are
//   permitted to return slightly different results from transcendental built-ins.
//   Instead we read from a frozen cos/sin table that is bound as a storage
//   buffer, guaranteeing bit-exact results across replay.
//
// Expected MLIR op signature:
//   linalg.generic {
//       nd4j.op_hash = <descriptor-derived hash> : i64,
//       nd4j.head_dim = <D>,
//       nd4j.rotary_dim = <D/2> }
//

//
class SD_LIB_EXPORT RopeToSpirv : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kHeadDimAttrName = "nd4j.head_dim";
  static constexpr const char* kRotaryDimAttrName = "nd4j.rotary_dim";
};

// ─────────────────────────────────────────────────────────────────────────────
//  StructuredComputeToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Trait-selected, single-pipeline emitters for structured indexing and fused
// compute schedules. The recipe chooses the actual arithmetic after routing.
// Every floating reduction and dot product uses nd4j.accumulator_type; no
// intermediate tensor or host calculation is used.
class SD_LIB_EXPORT StructuredComputeToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: ElementwiseBinaryToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Parameterized pattern that lowers same-shape elementwise binary ops
// (add, subtract, multiply, divide, residual_add) to SCF/Arith IR.
//
// Dispatch dims: (1,1,1) serial loop — correctness over speed (Wave 1 contract).
//
// Expected MLIR op signature:
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64, nd4j.binary = true,
//                   indexing_maps = [...], iterator_types = ["parallel", ...]}
//
class SD_LIB_EXPORT ElementwiseBinaryToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kBinaryAttr  = "nd4j.binary";
};

// Lowers descriptor-selected ternary elementwise operations such as select.
// The condition is represented by ND4J's byte-addressed BOOL storage ABI.
class SD_LIB_EXPORT ElementwiseTernaryToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kTernaryAttr = "nd4j.ternary";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: ElementwiseUnaryToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Parameterized pattern that lowers elementwise unary activation ops
// (silu/swish, gelu, tanh, sigmoid, relu) to SCF/Arith+Math IR.
//
// Dispatch dims: (1,1,1) serial loop (Wave 1).
//
// Expected MLIR op signature:
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64, nd4j.unary = true,
//                   indexing_maps = [...], iterator_types = ["parallel", ...]}
//
class SD_LIB_EXPORT ElementwiseUnaryToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kUnaryAttr  = "nd4j.unary";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: SoftmaxToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a row-wise softmax op (last-dim reduce) to SCF/Arith+Math IR.
// Serial three-pass algorithm (max-find → exp-sum → normalize) per row.
// Dispatch dims: (1,1,1) serial loop (Wave 1/Wave 6).
//
// Algorithm per row m in [0, rows):
//   maxVal = max(x[m, k], k in [0, D))
//   expSum = Σ exp(x[m, k] - maxVal)
//   y[m, k] = exp(x[m, k] - maxVal) / expSum
//
// Expected MLIR op signature:
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64,
//                   indexing_maps = [...], iterator_types = ["parallel", "reduction"]}
//
class SD_LIB_EXPORT SoftmaxToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 1: LayerNormToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Per-row serial algorithm matching the rms_norm pattern structure.
// Dispatch dims: (1,1,1) serial loop (Wave 1/Wave 6).
//
// Algorithm per row m in [0, rows):
//   mean     = Σ x[m, k] / D
//   variance = Σ (x[m, k] - mean)^2 / D
//   y[m, k]  = (x[m, k] - mean) / sqrt(variance + epsilon)
//              * gamma[k] + beta[k]   (if scale/bias present)
//
// Expected MLIR op signature:
//                   indexing_maps = [...], iterator_types = ["parallel", "reduction"]}
//
class SD_LIB_EXPORT LayerNormToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kEpsilonAttr   = "nd4j.epsilon";
  static constexpr float       kDefaultEpsilon = 1.0e-5f;
};

/**
 * Lowers reusable mixed-input, multi-output elementwise kernel contracts.
 * The first recipe is normalize_moments: count/sum/squared-sum to mean/variance.
 */
class SD_LIB_EXPORT MultiOutputElementwiseToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override;
};

/** One launch for a descriptor-defined list of rank-two GEMM pairs. */
class SD_LIB_EXPORT BatchedMatrixListToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override;
};

/** Race-free serial scatter/add schedule, including duplicate indices. */
class SD_LIB_EXPORT IndexedAccumulationToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override;
};

/**
 * Lowers descriptor-defined indexed TAD movement, including pullRows and
 * disjoint in-place shuffle, without reading structural arrays on the host.
 */
class SD_LIB_EXPORT IndexedTadMovementToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 2: GatherToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// by integer indices — to SCF/Arith IR.
//
// This covers the inner-loop of LLM token-embedding and KV position lookups.
// Constraint: axis == 0, indices are 1-D (rank-1), table is 2-D [V, D].
// Output is [I, D] where I = number of indices.
// Dispatch dims: (1,1,1) serial (Wave 2 — workgroup parallel is Wave 3).
//
// UNMAPPABLE variants (noted in VulkanOpMappability.h comments):
//
// Expected MLIR op signature:
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64, nd4j.axis = 0 : i64,
//                   indexing_maps = [...], iterator_types = ["parallel", "parallel"]}
//
class SD_LIB_EXPORT GatherToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kAxisAttr    = "nd4j.axis";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 2: ConcatToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a concat op (concatenation along one axis) to SCF/Arith IR.
//

// Dispatch dims: (1,1,1) serial (Wave 2).
// The number of inputs is encoded as nd4j.num_inputs attribute and their
// axis-dim contributions as nd4j.input_sizes (list of i64).
//
// Expected MLIR op signature (2-input example):
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64, nd4j.axis = 1 : i64,
//                   nd4j.num_inputs = 2 : i64,
//                   indexing_maps = [...], iterator_types = ["parallel", "parallel"]}
//
class SD_LIB_EXPORT ConcatToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kAxisAttr     = "nd4j.axis";
  static constexpr const char* kNumInputsAttr = "nd4j.num_inputs";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 2: TransposeToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a transpose (permute) op to SCF/Arith IR.
//
// Supports the BNSH↔BSNH 4-D pivot that appears in every attention block,
// plus general rank-N permutations up to rank 4.
// The permutation is encoded as nd4j.permutation (list of i64).
// Dispatch dims: (1,1,1) serial (Wave 2).
//
// Expected MLIR op signature:
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64, nd4j.permutation = [0,2,1,3] : vector<4xi64>,
//                   indexing_maps = [...], iterator_types = ["parallel",...]}
//
class SD_LIB_EXPORT TransposeToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;

 private:
  static constexpr const char* kPermutationAttr = "nd4j.permutation";
};

// Descriptor-selected, static/replay-safe rank-N movement kernels.  These
// lower gather_nd, tile, repeat, reverse, slice, and stack to one real GPU
// launch over the logical output.  Payload values are copied at their storage
// type, so no floating-only conversion or host-side calculation is involved.
class SD_LIB_EXPORT DataMovementToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp op,
      mlir::PatternRewriter& rewriter) const override;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceSumToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a reduce_sum op to SCF/Arith IR.
//
// Supports two forms via nd4j.reduce_axes attribute (list of i64):
//   - Last-dim reduce: input [rows, D] → output [rows, 1], axes = [1]
//   - Full reduce:     input [rows, D] → output [1],       axes = [-1] or empty
// keepDims is encoded as nd4j.keep_dims = true/false : BoolAttr.
// Dispatch dims: (1,1,1) serial loop.
//
// Expected MLIR op signature:
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64,
//                   nd4j.reduce_axes = [1] : vector<1xi64>,
//                   nd4j.keep_dims = false : bool,
//                   ...}
//
class SD_LIB_EXPORT ReduceSumToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;
 private:
  static constexpr const char* kAxesAttr   = "nd4j.reduce_axes";
  static constexpr const char* kKeepDimsAttr = "nd4j.keep_dims";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceMeanToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a reduce_mean op — identical structure to reduce_sum but divides by
// the number of reduced elements.
//
class SD_LIB_EXPORT ReduceMeanToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;
 private:
  static constexpr const char* kAxesAttr    = "nd4j.reduce_axes";
  static constexpr const char* kKeepDimsAttr = "nd4j.keep_dims";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceMaxToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a reduce_max op — uses a running-max accumulator initialized to -INF.
//
class SD_LIB_EXPORT ReduceMaxToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;
 private:
  static constexpr const char* kAxesAttr    = "nd4j.reduce_axes";
  static constexpr const char* kKeepDimsAttr = "nd4j.keep_dims";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceMinToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a reduce_min op — uses a running-min accumulator initialized to +INF.
//
class SD_LIB_EXPORT ReduceMinToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;
 private:
  static constexpr const char* kAxesAttr    = "nd4j.reduce_axes";
  static constexpr const char* kKeepDimsAttr = "nd4j.keep_dims";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 3: ReduceProdToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Lowers a reduce_prod op — uses a running-product accumulator initialized to 1.
//
class SD_LIB_EXPORT ReduceProdToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;
 private:
  static constexpr const char* kAxesAttr    = "nd4j.reduce_axes";
  static constexpr const char* kKeepDimsAttr = "nd4j.keep_dims";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Wave 4: ReduceNDToSpirv
// ─────────────────────────────────────────────────────────────────────────────
//
// Extends descriptor-selected reductions to arbitrary ND inputs.
//
// Lowering strategy:
//   - Compute output shape from the frozen reduction axes and keepDims.
//   - For each output element, scan the input elements that map to it and keep
//     the complete reduction in nd4j.accumulator_type.
//   - Convert to the output storage type exactly once, at the final store.
//
// Expected MLIR op signature (same as Wave 3 but with higher-rank input):
//   linalg.generic {nd4j.op_hash = <descriptor-derived hash> : i64, nd4j.reduce_axes = [...],
//                   nd4j.keep_dims = <bool>, nd4j.nd_reduce = true,
//                   indexing_maps = [...], iterator_types = [...]}

//
// The wave-3 patterns handle rank==2; this pattern handles rank>2.
// The nd4j.nd_reduce = true attribute distinguishes the ND path from Wave 3.
//
class SD_LIB_EXPORT ReduceNDToSpirv
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
 public:
  using OpRewritePattern<mlir::linalg::GenericOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(mlir::linalg::GenericOp op,
                                      mlir::PatternRewriter& rewriter) const override;
 private:
  static constexpr const char* kAxesAttr    = "nd4j.reduce_axes";
  static constexpr const char* kKeepDimsAttr = "nd4j.keep_dims";
  static constexpr const char* kNdReduceAttr = "nd4j.nd_reduce";
};

// ─────────────────────────────────────────────────────────────────────────────
//  Registration entry-point
// ─────────────────────────────────────────────────────────────────────────────

/// Populate active Vulkan lowering patterns. Semantic selection for generic
/// operations is descriptor-hash/callback based.
SD_LIB_EXPORT void populateVulkanLoweringPatterns(mlir::RewritePatternSet& patterns);

/// Create and return an MLIR pass that applies all Vulkan lowering patterns
/// as a single OperationPass<mlir::ModuleOp>.
///
/// Usage in a PassManager:
///   pm.addPass(sd::graph::createVulkanOpLoweringPass());
///
SD_LIB_EXPORT std::unique_ptr<mlir::Pass> createVulkanOpLoweringPass();

}  // namespace graph
}  // namespace sd

#endif  // HAVE_VULKAN && HAVE_MLIR
#endif  // LIBND4J_VULKAN_OP_LOWERINGS_H
