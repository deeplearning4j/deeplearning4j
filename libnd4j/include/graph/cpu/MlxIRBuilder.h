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

#ifndef LIBND4J_MLX_IR_BUILDER_H
#define LIBND4J_MLX_IR_BUILDER_H

#include <config.h>

#if HAVE_MLX

#include <array/NDArray.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/SegmentAnalysisTypes.h>
#include <system/common.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward-declare MLX types to avoid leaking MLX headers into non-C++20 TUs
namespace mlx::core {
class array;
enum class Dtype;
}  // namespace mlx::core

namespace sd {
namespace graph {

/**
 * MLX IR builder for Apple Silicon GPU compute.
 *
 * Builds lazy mlx::core::array computation graphs from NativeSlot segments.
 * The graphs are evaluated via mlx::core::eval() which dispatches to Metal
 * GPU shaders on Apple Silicon.
 *
 * All phases implemented:
 *   Phase 1: Element-wise (binary, unary, comparison, logical, ternary, identity, cast)
 *   Phase 2: Reductions, matmul, normalization, shape manipulation, data movement, constant gen
 *   Phase 3: Convolution, fused attention (via mlx::core::fast)
 */
class MlxIRBuilder {
 public:
  MlxIRBuilder() = default;
  ~MlxIRBuilder() = default;

  // ─── Static analysis methods ─────────────────────────────────────────

  /**
   * Check if an op name is supported by the MLX backend.
   * Uses the shared OpCategoryTable — accepts ALL categories except UNSUPPORTED.
   */
  static bool isMlxMappable(const std::string& opName);

  /**
   * Profile a segment — build dataflow graph, count categories.
   * Reuses OpCategoryTable from shared infrastructure.
   */
  static SegmentProfile profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                       NDArray** outputSlots, int totalOutputSlots);

  /**
   * Combined analysis: profile -> classify -> check compilability.
   */
  static SegmentAnalysis analyzeSegment(NativeSlot* slots, int startSlot, int endSlot,
                                        int totalSlots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots);

  // ─── Graph construction ──────────────────────────────────────────────

  struct MlxGraph {
    // Output arrays keyed by output slot index — lazy, not yet evaluated
    std::unordered_map<int, std::shared_ptr<void>> outputArrays;
    bool valid = false;
  };

  /**
   * Build an MLX lazy computation graph covering the full segment.
   * Walks slots in order, resolves inputs from SSA map, dispatches
   * to per-category emit helpers.
   */
  MlxGraph buildGraph(NativeSlot* slots, int startSlot, int endSlot,
                      int totalSlots,
                      NDArray** externalInputs, int numExternalInputs,
                      NDArray** outputSlots, int totalOutputSlots);

  /**
   * Compute the set of output slots that are externally visible.
   */
  static std::unordered_set<int> computeExternallyVisibleOutputs(
      NativeSlot* slots, int startSlot, int endSlot, int totalSlots);

  // ─── Memory bridge (public for MlxGraphBackend) ─────────────────────

  static std::shared_ptr<void> ndArrayToMlxArray(NDArray* arr);
  static void mlxArrayToNDArray(const std::shared_ptr<void>& mlxArr, NDArray* output);

 private:
  // ─── Phase 1: Element-wise emit helpers ──────────────────────────────

  static std::shared_ptr<void> emitBinaryElementwise(const std::string& opName,
                                                      const std::shared_ptr<void>& lhs,
                                                      const std::shared_ptr<void>& rhs);

  static std::shared_ptr<void> emitUnaryElementwise(const std::string& opName,
                                                     const std::shared_ptr<void>& input,
                                                     const double* tArgs, int numTArgs);

  static std::shared_ptr<void> emitComparisonOp(const std::string& opName,
                                                 const std::shared_ptr<void>& lhs,
                                                 const std::shared_ptr<void>& rhs);

  static std::shared_ptr<void> emitLogicalOp(const std::string& opName,
                                              const std::shared_ptr<void>& lhs,
                                              const std::shared_ptr<void>& rhs);

  static std::shared_ptr<void> emitTernaryOp(const std::string& opName,
                                              const std::shared_ptr<void>& cond,
                                              const std::shared_ptr<void>& ifTrue,
                                              const std::shared_ptr<void>& ifFalse);

  static std::shared_ptr<void> emitIdentityOp(const std::shared_ptr<void>& input);

  static std::shared_ptr<void> emitCastOp(const std::shared_ptr<void>& input,
                                           sd::DataType targetType);

  // ─── Phase 2: Reduction emit helpers ─────────────────────────────────

  /** Reduction: reduce_sum, reduce_max, reduce_min, reduce_mean, reduce_prod, etc. */
  static std::shared_ptr<void> emitReductionOp(const std::string& opName,
                                                const std::shared_ptr<void>& input,
                                                const LongType* iArgs, int numIArgs,
                                                bool keepDims);

  // ─── Phase 2: Matmul emit helpers ────────────────────────────────────

  /** Matmul: matmul, batch_matmul, tensormmul, xw_plus_b */
  static std::shared_ptr<void> emitMatmulOp(const std::string& opName,
                                             const std::vector<std::shared_ptr<void>>& inputs,
                                             const double* tArgs, int numTArgs,
                                             const LongType* iArgs, int numIArgs);

  // ─── Phase 2: Normalization emit helpers ─────────────────────────────

  /** Normalization: softmax, log_softmax, layer_norm, rms_norm, batch_norm */
  static std::shared_ptr<void> emitNormalizationOp(const std::string& opName,
                                                    const std::vector<std::shared_ptr<void>>& inputs,
                                                    const LongType* iArgs, int numIArgs,
                                                    const double* tArgs, int numTArgs);

  // ─── Phase 2: Shape manipulation emit helpers ────────────────────────

  /** Shape ops: reshape, permute, expand_dims, squeeze, flatten */
  static std::shared_ptr<void> emitShapeManipOp(const std::string& opName,
                                                 const std::shared_ptr<void>& input,
                                                 const LongType* iArgs, int numIArgs,
                                                 NDArray* outputArr);

  // ─── Phase 2: Data movement emit helpers ─────────────────────────────

  /** Data movement: gather, concat, split, stack, tile, scatter_nd, strided_slice, etc. */
  static std::shared_ptr<void> emitDataMovementOp(const std::string& opName,
                                                   const std::vector<std::shared_ptr<void>>& inputs,
                                                   const LongType* iArgs, int numIArgs,
                                                   const double* tArgs, int numTArgs,
                                                   NDArray* outputArr);

  // ─── Phase 2: Constant generation emit helpers ───────────────────────

  /** Constant generation: zeros_like, ones_like, range, shape_of, create, set_scalar */
  static std::shared_ptr<void> emitConstantGenOp(const std::string& opName,
                                                  const std::vector<std::shared_ptr<void>>& inputs,
                                                  const double* tArgs, int numTArgs,
                                                  const LongType* iArgs, int numIArgs,
                                                  NDArray* outputArr);

  // ─── Phase 3: Convolution emit helpers ───────────────────────────────

  /** Convolution: conv2d, conv3d, depthwise_conv2d */
  static std::shared_ptr<void> emitConvolutionOp(const std::string& opName,
                                                  const std::vector<std::shared_ptr<void>>& inputs,
                                                  const LongType* iArgs, int numIArgs,
                                                  NDArray* outputArr);

  // ─── Phase 3: Fused attention emit helpers ───────────────────────────

  /** Fused attention: multi_head_attention, dot_product_attention_v2 */
  static std::shared_ptr<void> emitFusedAttentionOp(const std::string& opName,
                                                     const std::vector<std::shared_ptr<void>>& inputs,
                                                     const double* tArgs, int numTArgs,
                                                     const LongType* iArgs, int numIArgs);

  // ─── LLM: Rotary Position Embedding ──────────────────────────────────

  /** RoPE: rope, fused_rope — uses mlx::core::fast::rope() */
  static std::shared_ptr<void> emitRopeOp(const std::string& opName,
                                           const std::vector<std::shared_ptr<void>>& inputs,
                                           const double* tArgs, int numTArgs,
                                           const LongType* iArgs, int numIArgs);

  // ─── LLM: Fused mega-kernels ───────────────────────────────────────

  /** Fused LLM ops: fused_rms_norm_swiglu, fused_bias_dropout_residual */
  static std::shared_ptr<void> emitFusedLlmOp(const std::string& opName,
                                                const std::vector<std::shared_ptr<void>>& inputs,
                                                const double* tArgs, int numTArgs,
                                                const LongType* iArgs, int numIArgs);

  // ─── Type mapping ────────────────────────────────────────────────────

  static int sdTypeToMlxDtype(sd::DataType dt);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLX
#endif  // LIBND4J_MLX_IR_BUILDER_H
