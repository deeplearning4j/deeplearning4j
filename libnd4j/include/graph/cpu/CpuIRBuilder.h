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

#ifndef LIBND4J_CPU_IR_BUILDER_H
#define LIBND4J_CPU_IR_BUILDER_H

#if HAVE_MLIR

#include <array/NDArray.h>
#include <graph/GraphAnalysisUtils.h>
#include <graph/NativeDynamicShapePlan.h>
#include <graph/SegmentAnalysisTypes.h>
#include <mlir/runtime/MLIREngine.h>
#include <system/common.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations for MLIR types
namespace mlir {
class MLIRContext;
class ModuleOp;
class OpBuilder;
class Value;
class Location;
class Type;
template <typename T> class OwningOpRef;
}  // namespace mlir

namespace sd {
namespace graph {

/**
 * CPU MLIR IR builder for fused kernel compilation.
 *
 * Emits memref/scf/arith/math/linalg MLIR from NativeSlots, targeting the
 * MLIREngine's existing CPU lowering pipeline (linalg→affine→vector→LLVM JIT).
 *
 * Unlike TritonIRBuilder (which emits tt.load/tt.store + GPU grid ops),
 * this builder emits:
 *   - func.func @kernel(memref<?xf32>, ...) for each unique buffer arg
 *   - scf.for loops for element-wise, reduction, matmul, data movement
 *   - memref.load/store for data access
 *   - arith/math ops for fused computation chains
 *
 * Supports ALL op categories from OpCategoryTable:
 *   - BINARY/UNARY_ELEMENTWISE, COMPARISON, LOGICAL, TERNARY, IDENTITY, CAST
 *   - MATMUL (nested loop matmul with K-accumulation)
 *   - REDUCTION (scf.for accumulator loop)
 *   - NORMALIZATION (softmax, layer_norm, rms_norm decomposed)
 *   - SHAPE_MANIPULATION (permute via index remapping, reshape as identity)
 *   - DATA_MOVEMENT (gather, concat, split, tile, scatter_nd)
 *   - CONSTANT_GENERATION (zeros_like, ones_like, range, shape_of)
 *   - CONVOLUTION (nested loop direct conv2d)
 */
class CpuIRBuilder {
 public:
  CpuIRBuilder() = default;
  ~CpuIRBuilder() = default;

  // ─── Static analysis methods (no MLIR allocation) ──────────────────────

  /**
   * Check if an op name is supported by the CPU MLIR backend.
   * Uses the shared OpCategoryTable — accepts ALL categories except UNSUPPORTED.
   */
  static bool isMlirMappable(const std::string& opName);

  /**
   * Pass 1: Profile a segment — build dataflow graph, count categories.
   * Delegates to GraphAnalysisUtils::profileSegment() (shared with TritonIRBuilder).
   */
  static SegmentProfile profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                       NDArray** outputSlots, int totalOutputSlots);

  /**
   * Combined analysis: profile → pattern matching → classify.
   * Mirrors TritonIRBuilder::analyzeSegment() — accepts all compilable patterns.
   */
  static SegmentAnalysis analyzeSegment(NativeSlot* slots, int startSlot, int endSlot,
                                        int totalSlots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots, int totalOutputSlots);

  // ─── IR emission ──────────────────────────────────────────────────────

  /**
   * Build an MLIR module for a fused kernel covering the full segment.
   *
   * The module contains a single func.func @fused_kernel that handles
   * all op categories: element-wise loops, matmul nested loops, reduction
   * accumulators, normalization decompositions, data movement, etc.
   *
   * Each op is emitted sequentially in slot order. Element-wise ops are
   * fused into a single outer loop. Non-element-wise ops (matmul, reduction,
   * data movement) break out of the element-wise loop and emit their own
   * loop nests.
   */
  mlir::OwningOpRef<mlir::ModuleOp> buildModule(
      mlir::MLIRContext* context,
      NativeSlot* slots, int startSlot, int endSlot,
      int totalSlots,
      NDArray** externalInputs, int numExternalInputs,
      NDArray** outputSlots, int totalOutputSlots);

  /**
   * Compute the set of output slots that are externally visible.
   * Public so MlirCpuGraphBackend can reconstruct the arg mapping.
   * Delegates to GraphAnalysisUtils::computeExternallyVisibleOutputs().
   */
  static std::unordered_set<int> computeExternallyVisibleOutputs(
      NativeSlot* slots, int startSlot, int endSlot, int totalSlots);

 private:
  // ─── Emit helpers (scalar operations inside loop bodies) ──────────────

  /** Binary element-wise: add, sub, mul, div, max, min, pow, atan2, etc. */
  static mlir::Value emitBinaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                           const std::string& opName,
                                           mlir::Value lhs, mlir::Value rhs,
                                           mlir::Type resultType);

  /** Unary element-wise: relu, sigmoid, tanh, gelu, exp, log, sqrt, etc. */
  static mlir::Value emitUnaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                          const std::string& opName,
                                          mlir::Value input, mlir::Type resultType,
                                          const double* tArgs, int numTArgs);

  /** Comparison: greater, less, equals, etc. → arith.cmpf → i1 */
  static mlir::Value emitComparisonOp(mlir::OpBuilder& builder, mlir::Location loc,
                                      const std::string& opName,
                                      mlir::Value lhs, mlir::Value rhs);

  /** Ternary: where/select → arith.select */
  static mlir::Value emitTernaryOp(mlir::OpBuilder& builder, mlir::Location loc,
                                   const std::string& opName,
                                   mlir::Value cond, mlir::Value ifTrue, mlir::Value ifFalse);

  // ─── Full-array emit helpers (emit their own loop nests) ──────────────

  /**
   * Emit a reduction op (reduce_sum, reduce_max, etc.) as scf.for accumulator.
   * Reads from inputMemref, writes scalar result to outputMemref[0..nOut].
   * If keepDims, output has same length as input with reduced dims = 1.
   * For full reduction: output is a single element.
   */
  static void emitReductionOp(mlir::OpBuilder& builder, mlir::Location loc,
                               const std::string& opName,
                               mlir::Value inputMemref, mlir::Value outputMemref,
                               int64_t nElements, mlir::Type elemType);

  /**
   * Emit a normalization op (softmax, layer_norm, rms_norm) as multi-pass loops.
   * softmax: max → shifted exp → sum → normalize
   * layer_norm: mean → variance → normalize
   * rms_norm: mean_sq → rsqrt → scale
   */
  static void emitNormalizationOp(mlir::OpBuilder& builder, mlir::Location loc,
                                   const std::string& opName,
                                   mlir::Value inputMemref, mlir::Value outputMemref,
                                   int64_t nElements, mlir::Type elemType);

  /**
   * Emit matmul as triple-nested loop: for i in M, for j in N, for k in K.
   * A[M,K] x B[K,N] → C[M,N]
   */
  static void emitMatmulOp(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value aMemref, mlir::Value bMemref, mlir::Value cMemref,
                            int64_t M, int64_t N, int64_t K, mlir::Type elemType);

  /**
   * Emit gather: output[i] = data[indices[i]] (1D simplification).
   */
  static void emitGatherOp(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value dataMemref, mlir::Value indicesMemref,
                            mlir::Value outputMemref,
                            int64_t nElements, int gatherAxis,
                            const std::vector<LongType>& dataShape,
                            mlir::Type elemType);

  /**
   * Emit concat along axis: sequential copies from each input.
   */
  static void emitConcatOp(mlir::OpBuilder& builder, mlir::Location loc,
                            const std::vector<mlir::Value>& inputMemrefs,
                            mlir::Value outputMemref,
                            const std::vector<int64_t>& inputLengths,
                            mlir::Type elemType);

  /**
   * Emit tile: output = repeated copies of input.
   */
  static void emitTileOp(mlir::OpBuilder& builder, mlir::Location loc,
                          mlir::Value inputMemref, mlir::Value outputMemref,
                          int64_t inputLen, int64_t outputLen, mlir::Type elemType);

  /**
   * Emit scatter_nd: copy data to output, then scatter updates at indices.
   */
  static void emitScatterNdOp(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value dataMemref, mlir::Value indicesMemref,
                               mlir::Value updatesMemref, mlir::Value outputMemref,
                               int64_t dataLen, int64_t numUpdates, int64_t sliceSize,
                               mlir::Type elemType);

  /**
   * Emit shape manipulation (permute with index remapping, or identity copy).
   */
  static void emitShapeManipOp(mlir::OpBuilder& builder, mlir::Location loc,
                                const std::string& opName,
                                mlir::Value inputMemref, mlir::Value outputMemref,
                                int64_t nElements,
                                const std::vector<LongType>& inputShape,
                                const std::vector<LongType>& outputShape,
                                const LongType* iArgs, int numIArgs,
                                mlir::Type elemType);

  /**
   * Emit constant generation (zeros_like, ones_like, range).
   */
  static void emitConstantGenOp(mlir::OpBuilder& builder, mlir::Location loc,
                                 const std::string& opName,
                                 mlir::Value outputMemref,
                                 int64_t nElements, mlir::Type elemType,
                                 const double* tArgs, int numTArgs);

  /**
   * Emit direct conv2d as nested loops over N,OC,OH,OW,IC,KH,KW.
   */
  static void emitConvolutionOp(mlir::OpBuilder& builder, mlir::Location loc,
                                 mlir::Value inputMemref, mlir::Value filterMemref,
                                 mlir::Value outputMemref,
                                 const std::vector<LongType>& inputShape,
                                 const std::vector<LongType>& filterShape,
                                 const std::vector<LongType>& outputShape,
                                 const LongType* iArgs, int numIArgs,
                                 mlir::Type elemType);

  // ─── Additional emitters (pooling, embedding, loss) ─────────────────

  /**
   * Emit max/avg pooling 2D as nested loops over batch, channel, output spatial dims.
   * Supports NCHW format with kernel, stride, padding, dilation parameters.
   */
  static void emitPooling2dOp(mlir::OpBuilder& builder, mlir::Location loc,
                               const std::string& opName,
                               mlir::Value inputMemref, mlir::Value outputMemref,
                               const std::vector<LongType>& inputShape,
                               const std::vector<LongType>& outputShape,
                               int kH, int kW, int sH, int sW,
                               int pH, int pW, int dH, int dW,
                               mlir::Type elemType);

  /**
   * Emit embedding lookup: output[i,:] = table[indices[i],:]
   */
  static void emitEmbeddingLookup(mlir::OpBuilder& builder, mlir::Location loc,
                                    mlir::Value tableMemref, mlir::Value indicesMemref,
                                    mlir::Value outputMemref,
                                    int64_t numLookups, int64_t embeddingDim,
                                    mlir::Type elemType);

  /**
   * Emit softmax cross entropy loss: -sum(labels * log(softmax(logits)))
   */
  static void emitSoftmaxCrossEntropy(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value logitsMemref, mlir::Value labelsMemref,
                                        mlir::Value outputMemref,
                                        int64_t nElements, mlir::Type elemType);

  /**
   * Emit mean squared error loss: mean((predictions - labels)^2)
   */
  static void emitMSELoss(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value predictionsMemref, mlir::Value labelsMemref,
                            mlir::Value outputMemref,
                            int64_t nElements, mlir::Type elemType);

  // ─── New emitters for extended op coverage ─────────────────────────

  /**
   * Emit slice: copy a contiguous sub-range [begin, begin+length) from input.
   */
  static void emitSliceOp(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value inputMemref, mlir::Value outputMemref,
                           int64_t begin, int64_t length, mlir::Type elemType);

  /**
   * Emit logical element-wise op: and, or, not, xor on float-encoded booleans.
   */
  static void emitLogicalOp(mlir::OpBuilder& builder, mlir::Location loc,
                              const std::string& opName,
                              mlir::Value lhsMemref, mlir::Value rhsMemref,
                              mlir::Value outputMemref,
                              int64_t nElements, mlir::Type elemType);

  /**
   * Emit Huber loss: piecewise MSE/MAE with delta threshold.
   */
  static void emitHuberLoss(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Value predictionsMemref, mlir::Value labelsMemref,
                              mlir::Value outputMemref,
                              int64_t nElements, double delta, mlir::Type elemType);

  /**
   * Emit hinge loss: sum(max(0, 1 - y*t)) / n
   */
  static void emitHingeLoss(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Value predictionsMemref, mlir::Value labelsMemref,
                              mlir::Value outputMemref,
                              int64_t nElements, mlir::Type elemType);

  /**
   * Emit log loss: -mean(y*log(p) + (1-y)*log(1-p))
   */
  static void emitLogLoss(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value predictionsMemref, mlir::Value labelsMemref,
                            mlir::Value outputMemref,
                            int64_t nElements, double epsilon, mlir::Type elemType);

  /**
   * Emit one-hot encoding: indices → one-hot vectors.
   */
  static void emitOneHot(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value indicesMemref, mlir::Value outputMemref,
                           int64_t numIndices, int64_t depth,
                           double onValue, double offValue, mlir::Type elemType);

  /**
   * Emit xw_plus_b: matmul(x, w) + bias.
   */
  static void emitXWPlusB(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::Value xMemref, mlir::Value wMemref,
                            mlir::Value biasMemref, mlir::Value outputMemref,
                            int64_t batch, int64_t inFeatures, int64_t outFeatures,
                            mlir::Type elemType);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_MLIR
#endif  // LIBND4J_CPU_IR_BUILDER_H
