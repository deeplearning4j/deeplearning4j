/* ******************************************************************************
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

//
// Pure C++ type definitions for the Triton IR builder.
// NO MLIR dependencies — safe to include from NVCC-compiled .cu files.
//
// Separated from TritonIRBuilder.h so that TritonGraphBackend.h (included by
// NativeOps_dsp.cu) can use these types without pulling in MLIR headers that
// NVCC cannot compile (std::vector<mlir::Value> requires complete type).
//

#ifndef LIBND4J_TRITON_IR_BUILDER_TYPES_H
#define LIBND4J_TRITON_IR_BUILDER_TYPES_H

#include <config.h>

#if HAVE_TRITON

#include <array/DataType.h>
#include <system/common.h>
#include <graph/gpu/OpCategoryTable.h>

#include <string>
#include <vector>

namespace sd {
namespace graph {

/**
 * Section type within a cooperative mega-kernel.
 * Each section handles one category of op(s) with its own grid mapping.
 */
enum class KernelSectionType {
  ELEMENTWISE,         // 1D grid, BLOCK_SIZE elements per block
  MATMUL,              // 2D tiled, tt.dot with K-loop
  FUSED_ATTENTION,     // Flash Attention, online softmax, tt.dot for QK^T and P@V
  REDUCTION,           // Tree reduction within block
  NORMALIZATION,       // Multi-op fused pattern (softmax, layer_norm)
  GATHER,              // Indexed loads from data using indices
  GATHER_ND,           // Multi-dimensional indexed loads
  CONCAT,              // Cascading select over N inputs
  SPLIT,               // Partitioned loads into multiple outputs
  SPLIT_V,             // Variable-size split
  STACK,               // Stack along new axis
  STRIDED_SLICE,       // Strided load with begin/end/stride
  TILE,                // Repeated load with modular indexing
  SCATTER_ND,          // Indexed stores
  SCATTER_ND_UPDATE,   // Indexed update
  SHAPE_MANIPULATION,  // reshape, permute, expand_dims, squeeze, flatten
  CONSTANT_GENERATION, // ones_like, zeros_like, range, shape_of
  CONVOLUTION,         // Spatial convolution with nested loops
  IDENTITY             // SSA forwarding (no IR ops)
};

/**
 * A section within the cooperative mega-kernel.
 */
struct KernelSection {
  KernelSectionType type;
  int startSlot;
  int endSlot;
  int numOps;
  int gridRequirement;

  int matmulM, matmulN, matmulK;
  int blockM, blockN, blockK;

  int batchSize, numHeads, seqQ, seqK, headDim;
  int numKvHeads = 0;
  float attentionScale;
  bool attnQIsBSHD = false;
  bool attnKIsBSHD = false;

  int gatherAxis;
  int concatAxis;

  int matmulEpilogueStartSlot = -1;
  int matmulEpilogueEndSlot = -1;
  int matmulEpilogueCount = 0;

  int normEpilogueStartSlot = -1;
  int normEpilogueEndSlot = -1;
  int normEpilogueCount = 0;

  bool hasTrailingPermute = false;
  int trailingPermuteSlot = -1;
  int trailingPermuteOutputSlotIdx = -1;
  int trailingPermuteInputSlotIdx = -1;
  std::vector<int> trailingPermutation;
  std::vector<LongType> trailingPermuteInputShape;
  std::vector<LongType> trailingPermuteOutputShape;

  KernelSection()
      : type(KernelSectionType::ELEMENTWISE), startSlot(-1), endSlot(-1),
        numOps(0), gridRequirement(1),
        matmulM(0), matmulN(0), matmulK(0), blockM(128), blockN(128), blockK(32),
        batchSize(0), numHeads(0), seqQ(0), seqK(0), headDim(0), attentionScale(1.0f),
        gatherAxis(0), concatAxis(0) {}
};

/**
 * Epilogue operation applied to the matmul accumulator in-register.
 * Pure C++ — no MLIR types.
 */
struct EpilogueOp {
  enum Type {
    BIAS_ADD,        // true bias: 1D [N] vector broadcast across M
    RESIDUAL_ADD,    // residual: 2D [M, N] tensor element-wise add
    RELU,
    GELU,
    SILU,
    TANH,
    SIGMOID,
    CAST,
    ELEMENTWISE,
  };
  Type type;
  std::string opName;
  int biasArgIndex = -1;
};

/**
 * Mapping entry for a single libnd4j op to Triton IR.
 */
struct TritonOpMapping {
  std::string opName;
  TritonOpCategory category;
  std::string tritonIrOp;
  bool requiresPattern;
};

/**
 * Kernel argument descriptor.
 */
struct TritonKernelArg {
  int slotIndex = 0;
  int outputIndex = 0;
  bool isOutput = false;
  DataType dtype = FLOAT32;
  std::vector<LongType> shape;
  std::vector<LongType> strides;

  bool isNonContiguous() const {
    if (strides.empty() || shape.empty()) return false;
    int rank = static_cast<int>(shape.size());
    LongType expected = 1;
    for (int d = rank - 1; d >= 0; d--) {
      if (strides[d] != expected) return true;
      expected *= shape[d];
    }
    return false;
  }

  /**
   * DEPRECATED: Use the set-based check in CompiledKernel::hasDynamicArgsInSegment instead.
   *
   * This range-based check compares slotIndex against op-slot boundaries, but
   * slotIndex holds OUTPUT slot indices (from wiring.outputSlotIndices), which
   * can exceed the op-slot range in graphs with multi-output or wide ops.
   * The set-based variant in TritonGraphBackend.h is the correct implementation.
   */
  bool isDynamicInSegment(int segStartSlot, int segEndSlot) const {
    if (slotIndex < 0) return true;                         // external input
    if (slotIndex >= segStartSlot && slotIndex <= segEndSlot) return true;  // segment intermediate
    return false;
  }
};

/**
 * Result of building Triton IR from a segment.
 */
struct TritonIRModule {
  void* mlirModule;
  void* mlirContext;
  std::string kernelName;
  std::vector<TritonKernelArg> args;
  int numWarps;
  int numStages;
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int blockZ;
  bool valid;
  bool useIndirectArgs;
  bool useCooperativeLaunch;
  bool useDynamicGrid;
  int requiredGrid;
  std::vector<KernelSection> sections;
  int estimatedSharedMemBytes;

  struct LaunchPhase {
    int startSection;
    int endSection;
    int gridX;
  };
  std::vector<LaunchPhase> launchPhases;
  bool useMultiPhaseLaunch;

  TritonIRModule() : mlirModule(nullptr), mlirContext(nullptr), numWarps(4), numStages(3),
                     gridX(1), gridY(1), gridZ(1),
                     blockX(1), blockY(1), blockZ(1),
                     valid(false), useIndirectArgs(false), useCooperativeLaunch(false),
                     useDynamicGrid(true),
                     requiredGrid(1), estimatedSharedMemBytes(0),
                     useMultiPhaseLaunch(false) {}
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_IR_BUILDER_TYPES_H
