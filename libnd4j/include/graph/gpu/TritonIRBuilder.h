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

#ifndef LIBND4J_TRITON_IR_BUILDER_H
#define LIBND4J_TRITON_IR_BUILDER_H

#include <config.h>

#if HAVE_TRITON

#include <array/NDArray.h>
#include <graph/NativeDynamicShapePlan.h>
#include <system/common.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward declarations for MLIR types used in helper signatures
namespace mlir {
class OpBuilder;
class Location;
class Value;
class Type;
class RankedTensorType;
}  // namespace mlir

namespace sd {
namespace graph {

/**
 * Classification of how a libnd4j op maps to Triton IR.
 */
enum class TritonOpCategory {
  BINARY_ELEMENTWISE,     // add, sub, mul, div, min, max
  UNARY_ELEMENTWISE,      // relu, sigmoid, tanh, exp, log, sqrt, etc.
  COMPARISON,             // greater, less, equals, etc. -> arith::CmpFOp/CmpIOp
  LOGICAL,                // boolean_and, boolean_or, etc. -> arith::AndIOp/OrIOp
  TERNARY,                // where, select -> arith::SelectOp (3 inputs)
  IDENTITY,               // identity, assign -> SSA value forwarding
  MATMUL,                 // matmul, batch_matmul -> tt.dot
  REDUCTION,              // reduce_sum, reduce_max, etc. -> tt.reduce
  NORMALIZATION,          // softmax, layer_norm -> multi-op fused pattern
  CAST,                   // type cast -> arith cast ops
  UNSUPPORTED             // cannot be mapped to Triton IR
};

/**
 * Kernel pattern classification for mixed-category segments.
 */
enum class SegmentKernelPattern {
  ELEMENTWISE_1D,   // All element-wise (including comparison, logical, ternary, cast)
  REDUCTION_1D,     // Contains reduction ops
  NORMALIZATION,    // Contains normalization ops (softmax, layer_norm)
  MATMUL_2D,        // Contains matmul ops
  MATMUL_EPILOGUE   // Matmul + element-wise epilogue (bias+activation)
};

/**
 * Mapping entry for a single libnd4j op to Triton IR.
 */
struct TritonOpMapping {
  std::string opName;
  TritonOpCategory category;
  std::string tritonIrOp;       // Primary Triton IR operation name
  bool requiresPattern;         // true if compound pattern (e.g. relu = max(x,0))
};

/**
 * Kernel argument descriptor for wiring NDArray buffers to kernel parameters.
 */
struct TritonKernelArg {
  int slotIndex;              // >=0: output slot, <0: -(externalIndex+1)
  int outputIndex;            // Which output of the slot (usually 0)
  bool isOutput;              // true if this is a kernel output (written)
  DataType dtype;
  std::vector<LongType> shape;
};

/**
 * Result of building Triton IR from a segment of NativeSlots.
 */
struct TritonIRModule {
  void* mlirModule;           // Opaque handle to mlir::ModuleOp
  std::string kernelName;     // Generated kernel function name
  std::vector<TritonKernelArg> args;   // Ordered kernel arguments
  int numWarps;               // Recommended warps per block
  int numStages;              // Recommended pipeline stages
  unsigned int gridX;         // Grid dimension X (num_elements / BLOCK_SIZE)
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;        // Block dimension X (BLOCK_SIZE for element-wise)
  unsigned int blockY;
  unsigned int blockZ;
  bool valid;                 // true if construction succeeded

  TritonIRModule() : mlirModule(nullptr), numWarps(4), numStages(3),
                     gridX(1), gridY(1), gridZ(1),
                     blockX(1), blockY(1), blockZ(1),
                     valid(false) {}
};

/**
 * Constructs Triton MLIR IR (TTIR) from sequences of NativeSlots.
 *
 * The key advantage over CUDA Graphs: fused ops share SSA values in the IR,
 * so the Triton compiler can eliminate intermediate global memory stores
 * and fuse multiple element-wise ops into a single kernel.
 *
 * Example fusion: add(x,y) -> relu(result) -> mul(result, z)
 * CUDA Graphs: 3 separate kernel launches, 2 intermediate buffers
 * Triton:      1 fused kernel, 0 intermediate buffers
 *
 * Tile size selection:
 * - Element-wise ops: BLOCK_SIZE=1024, 1D grid
 * - MatMul: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 2D grid
 * - Reductions: BLOCK_SIZE=1024, tree reduction within block
 */
class TritonIRBuilder {
 public:
  TritonIRBuilder();
  ~TritonIRBuilder();

  /**
   * Check if a specific op can be mapped to Triton IR.
   */
  static bool isTritonMappable(const std::string& opName);

  /**
   * Get the op category for a libnd4j op name.
   */
  static TritonOpCategory getOpCategory(const std::string& opName);

  /**
   * Classify a segment's dominant kernel pattern based on its op mix.
   */
  static SegmentKernelPattern classifySegment(NativeSlot* slots, int startSlot, int endSlot);

  /**
   * Check if an op category is element-wise compatible (can be fused into 1D kernel).
   */
  static bool isElementwiseCompatible(TritonOpCategory cat);

  /**
   * Build a Triton MLIR module from a contiguous range of slots.
   *
   * Constructs TTIR by:
   * 1. Creating tt.func with pointer arguments for each unique buffer
   * 2. Adding tt.load for each input
   * 3. Mapping each op to its Triton IR equivalent (arith/math/tt ops)
   * 4. Adding tt.store for outputs
   *
   * Fused ops share SSA values — no intermediate global stores.
   *
   * @param slots           All slots in the plan
   * @param startSlot       First slot in the segment
   * @param endSlot         Last slot in the segment (inclusive)
   * @param externalInputs  External input arrays (for shape/dtype info)
   * @param numExternalInputs  Count of external inputs
   * @param outputSlots     Current output slot arrays (for shape/dtype info)
   * @param totalOutputSlots  Total output slots
   * @return TritonIRModule with MLIR handle and kernel metadata
   */
  TritonIRModule buildModule(NativeSlot* slots, int startSlot, int endSlot,
                             NDArray** externalInputs, int numExternalInputs,
                             NDArray** outputSlots, int totalOutputSlots);

 private:
  // Op mapping table (populated in constructor)
  static const std::unordered_map<std::string, TritonOpMapping>& getOpTable();

  // Determine optimal tile sizes based on op categories in the segment
  void selectTileConfig(const std::vector<TritonOpCategory>& categories,
                        const std::vector<std::vector<LongType>>& shapes,
                        int& blockSize, int& numWarps, int& numStages);

  // Generate a unique kernel name from the segment's op sequence
  std::string generateKernelName(NativeSlot* slots, int startSlot, int endSlot);

  // ── MLIR emission helpers ──

  // Emit a binary element-wise op (add, sub, mul, div, min, max)
  static mlir::Value emitBinaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                           const TritonOpMapping& mapping,
                                           mlir::Value lhs, mlir::Value rhs);

  // Emit a unary element-wise op (relu, sigmoid, tanh, gelu, exp, log, etc.)
  // Some are compound patterns (e.g., relu = max(x, 0), sigmoid = 1/(1+exp(-x)))
  static mlir::Value emitUnaryElementwise(mlir::OpBuilder& builder, mlir::Location loc,
                                          const TritonOpMapping& mapping,
                                          const NativeSlot& slot, mlir::Value input,
                                          int blockSize);

  // Emit a comparison op (greater, less, equals, etc.)
  static mlir::Value emitComparisonOp(mlir::OpBuilder& builder, mlir::Location loc,
                                      const std::string& opName,
                                      mlir::Value lhs, mlir::Value rhs, int blockSize);

  // Emit a logical op (boolean_and, boolean_or, etc.)
  static mlir::Value emitLogicalOp(mlir::OpBuilder& builder, mlir::Location loc,
                                   const std::string& opName,
                                   mlir::Value lhs, mlir::Value rhs, int blockSize);

  // Emit a ternary select/where op
  static mlir::Value emitTernaryOp(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value condition, mlir::Value trueVal,
                                   mlir::Value falseVal, int blockSize);

  // Emit a reduction kernel pattern
  static mlir::Value emitReductionOp(mlir::OpBuilder& builder, mlir::Location loc,
                                     const std::string& opName,
                                     mlir::Value input, int reductionAxis,
                                     mlir::RankedTensorType outputType);

  // Emit a normalization kernel pattern (softmax, layer_norm, rms_norm)
  static mlir::Value emitNormalizationOp(mlir::OpBuilder& builder, mlir::Location loc,
                                         const std::string& opName,
                                         mlir::Value input, int axis,
                                         mlir::RankedTensorType outputType);

  // Emit a matmul kernel pattern using tt.dot
  static void emitMatmulKernel(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                               int M, int N, int K,
                               int blockM, int blockN, int blockK);

  // Create a splat constant float tensor: splat(val) -> tensor<BLOCKxf32>
  static mlir::Value splatConstantF32(mlir::OpBuilder& builder, mlir::Location loc,
                                      mlir::RankedTensorType tensorType, float val);

  // Map an nd4j DataType to an MLIR element type
  static mlir::Type getMLIRType(mlir::OpBuilder& builder, DataType dtype);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_IR_BUILDER_H
