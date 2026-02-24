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
#include <graph/gpu/OpCategoryTable.h>
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

// TritonOpCategory enum is defined in OpCategoryTable.h (shared with NVRTC path)

/**
 * Kernel pattern classification for mixed-category segments.
 */
enum class SegmentKernelPattern {
  ELEMENTWISE_1D,   // All element-wise (including comparison, logical, ternary, cast)
  REDUCTION_1D,     // Contains reduction ops
  NORMALIZATION,    // Contains normalization ops (softmax, layer_norm)
  MATMUL_2D,        // Contains matmul ops
  MATMUL_EPILOGUE,  // Matmul + element-wise epilogue (bias+activation)
  FUSED_ATTENTION,  // onnx_multi_head_attention -> Flash Attention kernel
  WHOLE_GRAPH       // Mixed mega-segment: matmul + elementwise + shape + data movement + etc.
};

/**
 * Per-op node in the segment dataflow graph, built during profiling.
 */
struct OpNode {
  int slotIndex;                        // Absolute slot index
  int localIndex;                       // 0-based within segment
  std::string opName;
  TritonOpCategory category;
  std::vector<int> inputLocalIndices;   // -1 for external inputs
  std::vector<int> consumerLocalIndices;// Which local ops consume this op's output
  bool hasExternalInput;

  // Output shape info from DSP's pre-calculated cache (NativeSlot.cachedOutputShapes)
  // Populated when outputSlots are available; empty otherwise
  std::vector<LongType> outputShape;    // First output's dimensions
  DataType outputDtype;                 // First output's data type
  bool hasOutputShape;                  // true if outputShape was populated

  OpNode() : slotIndex(-1), localIndex(-1), category(TritonOpCategory::IDENTITY),
             hasExternalInput(false), outputDtype(FLOAT32), hasOutputShape(false) {}
};

/**
 * Structured profile from Pass 1 — O(n) single walk over the segment.
 */
struct SegmentProfile {
  std::vector<OpNode> nodes;
  int categoryCounts[16] = {};          // Indexed by (int)TritonOpCategory (16 values)
  int totalOps;
  int numUniqueExternalInputs;
  int numUniqueOutputs;
  bool hasMatmul, hasReduction, hasNormalization, hasFusedAttention;
  bool hasShapeManip, hasDataMovement;

  SegmentProfile()
      : totalOps(0), numUniqueExternalInputs(0), numUniqueOutputs(0),
        hasMatmul(false), hasReduction(false), hasNormalization(false),
        hasFusedAttention(false), hasShapeManip(false), hasDataMovement(false) {}
};

/**
 * Detected composite pattern from Pass 2 pattern matching.
 */
struct PatternMatch {
  enum Type {
    PURE_ELEMENTWISE,
    PURE_MATMUL,
    PURE_REDUCTION,
    PURE_NORMALIZATION,
    MATMUL_EPILOGUE,
    SOFTMAX_DECOMPOSED,
    ATTENTION_QKV,
    FFN_BLOCK,
    FUSED_ATTENTION_OP,
    MIXED_MEGA_SEGMENT
  };
  Type type;
  int priority;                         // Higher wins
  std::vector<int> localIndices;        // Participating ops (local indices)
  std::string description;

  PatternMatch() : type(PURE_ELEMENTWISE), priority(0) {}
};

/**
 * Base class for pluggable pattern detectors.
 */
class PatternDetector {
 public:
  virtual ~PatternDetector() = default;
  virtual const char* name() const = 0;
  virtual std::vector<PatternMatch> detect(const SegmentProfile& profile,
                                            NativeSlot* slots, int startSlot) = 0;
};

/**
 * Collection of all matched patterns from Pass 2.
 */
struct MatchedPatterns {
  std::vector<PatternMatch> matches;
  const PatternMatch* bestMatch() const {
    if (matches.empty()) return nullptr;
    const PatternMatch* best = &matches[0];
    for (size_t i = 1; i < matches.size(); i++) {
      if (matches[i].priority > best->priority) best = &matches[i];
    }
    return best;
  }
};

/**
 * Pre-compilation analysis of a segment — computed WITHOUT allocating MLIR objects.
 * Used by Pass 1 of the 2-pass optimizer to validate and classify before IR emission.
 */
struct SegmentAnalysis {
  bool canCompile;                // true if segment passes all validation checks
  std::string failureReason;      // Human-readable reason if canCompile is false
  int totalInputArgs;             // Unique external/pre-segment input buffers
  int totalOutputArgs;            // Unique output buffers consumed post-segment
  int totalArgs;                  // totalInputArgs + totalOutputArgs + 1 (n_elements)
  SegmentKernelPattern pattern;   // Classified kernel pattern
  // Per-category op counts
  int numElementwise;
  int numMatmul;
  int numReduction;
  int numNormalization;
  int numAttention;
  int numShapeManip;
  int numDataMovement;
  int numConstGen;
  int numIdentity;
  int numCast;

  SegmentAnalysis()
      : canCompile(false), totalInputArgs(0), totalOutputArgs(0), totalArgs(0),
        pattern(SegmentKernelPattern::ELEMENTWISE_1D),
        numElementwise(0), numMatmul(0), numReduction(0), numNormalization(0),
        numAttention(0), numShapeManip(0), numDataMovement(0), numConstGen(0),
        numIdentity(0), numCast(0) {}
};

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
  SHAPE_MANIPULATION,  // reshape, permute, expand_dims, squeeze, flatten — stride recomputation
  CONSTANT_GENERATION, // ones_like, zeros_like, range, shape_of
  CONVOLUTION,         // Spatial convolution with nested loops
  IDENTITY             // SSA forwarding (no IR ops)
};

/**
 * A section within the cooperative mega-kernel.
 * Groups contiguous ops of compatible types. Between sections with cross-block
 * data dependencies, a cooperative grid sync barrier is inserted.
 */
struct KernelSection {
  KernelSectionType type;
  int startSlot;              // First slot (absolute) in this section
  int endSlot;                // Last slot (absolute, inclusive) in this section
  int numOps;                 // Number of ops in this section
  int gridRequirement;        // Number of blocks needed for this section

  // Matmul-specific dimensions (valid when type == MATMUL)
  int matmulM, matmulN, matmulK;
  int blockM, blockN, blockK;

  // Attention-specific dimensions (valid when type == FUSED_ATTENTION)
  int batchSize, numHeads, seqQ, seqK, headDim;
  float attentionScale;

  // Data movement dimensions
  int gatherAxis;             // Gather/scatter axis
  int concatAxis;             // Concat axis
  std::vector<int> splitSizes; // Split sizes for split_v

  KernelSection()
      : type(KernelSectionType::ELEMENTWISE), startSlot(-1), endSlot(-1),
        numOps(0), gridRequirement(1),
        matmulM(0), matmulN(0), matmulK(0), blockM(128), blockN(128), blockK(32),
        batchSize(0), numHeads(0), seqQ(0), seqK(0), headDim(0), attentionScale(1.0f),
        gatherAxis(0), concatAxis(0) {}
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
  void* mlirContext;          // Opaque handle to mlir::MLIRContext (owns module memory)
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
  bool useIndirectArgs;       // true if kernel uses indirect arg passing (pointer array)
                              // When true, the kernel signature is (argArray* : ptr<i64>, n_elements : i32)
                              // and all buffer pointers are packed into a device-side i64 array.
                              // The launch code must allocate the array and pass its device pointer.
  bool useCooperativeLaunch;  // true if kernel requires cooperative launch (grid sync barriers)
  int requiredGrid;           // Maximum grid size needed across all sections
  std::vector<KernelSection> sections;  // Section breakdown of the kernel

  TritonIRModule() : mlirModule(nullptr), mlirContext(nullptr), numWarps(4), numStages(3),
                     gridX(1), gridY(1), gridZ(1),
                     blockX(1), blockY(1), blockZ(1),
                     valid(false), useIndirectArgs(false), useCooperativeLaunch(false),
                     requiredGrid(1) {}
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
   * Pass 1: Profile a segment — build dataflow graph and category counts.
   * O(n) single walk, no MLIR allocation.
   * When outputSlots/totalOutputSlots are provided, populates OpNode shape info
   * from the DSP's pre-calculated shape cache.
   */
  static SegmentProfile profileSegment(NativeSlot* slots, int startSlot, int endSlot,
                                        NDArray** outputSlots = nullptr, int totalOutputSlots = 0);

  /**
   * Pass 2: Run registered pattern detectors against the profile.
   */
  static MatchedPatterns matchPatterns(const SegmentProfile& profile,
                                        NativeSlot* slots, int startSlot);

  /**
   * Pass 3: Map best pattern to SegmentAnalysis with arg counts and feasibility.
   */
  static SegmentAnalysis classifyAndAnalyze(const SegmentProfile& profile,
                                             const MatchedPatterns& patterns,
                                             NativeSlot* slots, int startSlot, int endSlot,
                                             int totalSlots,
                                             NDArray** externalInputs, int numExternalInputs,
                                             NDArray** outputSlots, int totalOutputSlots,
                                             int* requestedOutputSlotIndices = nullptr,
                                             int numRequestedOutputs = 0);

  /**
   * Full analysis: runs all 3 passes (profile → match → classify).
   * Call this before buildModule() to avoid LLVM assertion crashes from
   * segments with too many function arguments.
   */
  static SegmentAnalysis analyzeSegment(NativeSlot* slots, int startSlot, int endSlot,
                                         int totalSlots,
                                         NDArray** externalInputs, int numExternalInputs,
                                         NDArray** outputSlots, int totalOutputSlots,
                                         int* requestedOutputSlotIndices = nullptr,
                                         int numRequestedOutputs = 0);

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
                             int totalSlots,
                             NDArray** externalInputs, int numExternalInputs,
                             NDArray** outputSlots, int totalOutputSlots,
                             int* requestedOutputSlotIndices = nullptr,
                             int numRequestedOutputs = 0);

  /**
   * Build a sectioned cooperative mega-kernel for mixed segments.
   *
   * Uses identifySections() to break the segment into typed sections (elementwise,
   * matmul, attention, data movement, etc.), emits each section with the appropriate
   * emitter, and inserts cooperative grid sync barriers between sections that have
   * cross-block data dependencies.
   *
   * Requires cooperative launch (cuLaunchCooperativeKernel) so all blocks are
   * co-resident on the GPU for grid-wide synchronization.
   */
  TritonIRModule buildSectionedModule(NativeSlot* slots, int startSlot, int endSlot,
                                       int totalSlots,
                                       NDArray** externalInputs, int numExternalInputs,
                                       NDArray** outputSlots, int totalOutputSlots,
                                       int* requestedOutputSlotIndices = nullptr,
                                       int numRequestedOutputs = 0);

  /**
   * Build a dedicated Triton MLIR module for matmul segments (MATMUL_2D / MATMUL_EPILOGUE).
   *
   * Creates a 2D tiled matmul kernel with K-loop using tt.dot.
   * For MATMUL_EPILOGUE, element-wise ops after the matmul are fused into the
   * same kernel (applied to the 2D tile before final store).
   */
  TritonIRModule buildMatmulModule(NativeSlot* slots, int startSlot, int endSlot,
                                    int totalSlots,
                                    NDArray** externalInputs, int numExternalInputs,
                                    NDArray** outputSlots, int totalOutputSlots,
                                    int* requestedOutputSlotIndices = nullptr,
                                    int numRequestedOutputs = 0);

 private:
  // Create a splat constant float tensor: splat(val) -> tensor<BLOCKxf32>
  static mlir::Value splatConstantF32(mlir::OpBuilder& builder, mlir::Location loc,
                                      mlir::RankedTensorType tensorType, float val);
  static mlir::Value splatConstantI32(mlir::OpBuilder& builder, mlir::Location loc,
                                      mlir::RankedTensorType tensorType, int val);
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

  // Emit a fused attention kernel (Flash Attention pattern with online softmax)
  // Grid: (batch * num_heads, ceil(seqQ / BLOCK_M)) — 2D
  // Takes Q, K, V pointers and produces output; uses online softmax to avoid O(seq^2) storage
  static void emitFusedAttentionKernel(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value qPtr, mlir::Value kPtr,
                                        mlir::Value vPtr, mlir::Value outPtr,
                                        int batchSize, int numHeads, int seqQ, int seqK,
                                        int headDim, float scale,
                                        int blockM, int blockN);

  // Map an nd4j DataType to an MLIR element type
  static mlir::Type getMLIRType(mlir::OpBuilder& builder, DataType dtype);

 public:
  // ── Sectioned kernel helpers (public for TritonGraphBackend access) ──

  // Identify sections within a slot range: groups contiguous element-wise ops,
  // creates separate sections for matmul, attention, data movement, etc.
  static std::vector<KernelSection> identifySections(
      NativeSlot* slots, int startSlot, int endSlot,
      NDArray** outputSlots, int totalOutputSlots,
      NDArray** externalInputs, int numExternalInputs);

 private:
  // Compute the grid requirement for a single section
  static int computeSectionGrid(const KernelSection& section, int blockSize);

  // Emit a cooperative grid sync barrier (inline PTX via tt.elementwise_inline_asm)
  static void emitGridSync(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::Value syncCounterPtr, mlir::Value numBlocksVal);

  // ── Section emitters (inline within the mega-kernel) ──

  // Emit matmul section: 2D tiled K-loop with tt.dot, using pre-computed pid mapping
  static void emitMatmulSection(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value pid, const KernelSection& section,
                                mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr);

  // Emit fused attention section: Flash Attention with online softmax
  static void emitAttentionSection(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value pid, const KernelSection& section,
                                   mlir::Value qPtr, mlir::Value kPtr,
                                   mlir::Value vPtr, mlir::Value outPtr);

  // ── Data movement section emitters ──

  // Gather: idx = load(indices + offsets); result = load(data + idx * stride)
  static void emitGatherSection(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value pid, int blockSize,
                                mlir::Value dataPtr, mlir::Value indicesPtr,
                                mlir::Value outputPtr, int axis,
                                const std::vector<LongType>& dataShape,
                                const std::vector<LongType>& indicesShape,
                                int nElements);

  // Concat: cascading select over N inputs based on position ranges
  static void emitConcatSection(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value pid, int blockSize,
                                const std::vector<mlir::Value>& inputPtrs,
                                mlir::Value outputPtr, int axis,
                                const std::vector<std::vector<LongType>>& inputShapes,
                                int nElements);

  // Slice: result = load(input + start + offsets * stride)
  static void emitSliceSection(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value pid, int blockSize,
                               mlir::Value inputPtr, mlir::Value outputPtr,
                               const std::vector<int>& begins,
                               const std::vector<int>& ends,
                               const std::vector<int>& strides,
                               const std::vector<LongType>& inputShape,
                               int nElements);

  // Split: each output gets a portion of the input
  static void emitSplitSection(mlir::OpBuilder& builder, mlir::Location loc,
                               mlir::Value pid, int blockSize,
                               mlir::Value inputPtr,
                               const std::vector<mlir::Value>& outputPtrs,
                               int axis, int numSplits,
                               const std::vector<LongType>& inputShape,
                               int nElements);

  // Tile: result = load(input + (offsets % input_size))
  static void emitTileSection(mlir::OpBuilder& builder, mlir::Location loc,
                              mlir::Value pid, int blockSize,
                              mlir::Value inputPtr, mlir::Value outputPtr,
                              const std::vector<LongType>& inputShape,
                              const std::vector<int>& repeats,
                              int nElements);

  // ScatterNd: store(output + load(indices + offsets), load(updates + offsets))
  static void emitScatterNdSection(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value pid, int blockSize,
                                   mlir::Value dataPtr, mlir::Value indicesPtr,
                                   mlir::Value updatesPtr, mlir::Value outputPtr,
                                   const std::vector<LongType>& dataShape,
                                   int nElements);

  // Shape manipulation: proper stride/offset recomputation for non-contiguous views
  static void emitShapeManipulationSection(mlir::OpBuilder& builder, mlir::Location loc,
                                           mlir::Value pid, int blockSize,
                                           mlir::Value inputPtr, mlir::Value outputPtr,
                                           const std::string& opName,
                                           const std::vector<LongType>& inputShape,
                                           const std::vector<LongType>& outputShape,
                                           const std::vector<int>& permutation,
                                           int nElements);

  // Per-element fallback: matmul/attention via scalar K-loop (no tt.dot, no grid sync)
  static void emitPerElementMatmul(mlir::OpBuilder& builder, mlir::Location loc,
                                   mlir::Value pid, int blockSize,
                                   mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                                   int M, int N, int K);

  // Convolution: nested spatial loops (scf.for over kH, kW) with accumulation
  static void emitConvolutionSection(mlir::OpBuilder& builder, mlir::Location loc,
                                     mlir::Value pid, int blockSize,
                                     mlir::Value inputPtr, mlir::Value filterPtr,
                                     mlir::Value outputPtr,
                                     const std::vector<LongType>& inputShape,
                                     const std::vector<LongType>& filterShape,
                                     const std::vector<LongType>& outputShape,
                                     int strideH, int strideW,
                                     int padH, int padW,
                                     int nElements);

  // im2col: rearrange image patches to columns
  // Input: [bS, iC, iH, iW] (4D) → Output: [bS, iC, kH, kW, oH, oW] (6D)
  // Each output element maps to one input element (or zero-pad if out of bounds)
  static void emitIm2colSection(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value pid, int blockSize,
                                mlir::Value inputPtr, mlir::Value outputPtr,
                                const std::vector<LongType>& inputShape,
                                const std::vector<LongType>& outputShape,
                                int kH, int kW,
                                int sH, int sW,
                                int pH, int pW,
                                int dH, int dW,
                                int nElements);

  // col2im: rearrange columns back to image (inverse of im2col)
  // Input: [bS, iC, kH, kW, oH, oW] (6D) → Output: [bS, iC, iH, iW] (4D)
  // Each output pixel accumulates contributions from all overlapping column positions
  static void emitCol2imSection(mlir::OpBuilder& builder, mlir::Location loc,
                                mlir::Value pid, int blockSize,
                                mlir::Value inputPtr, mlir::Value outputPtr,
                                const std::vector<LongType>& inputShape,
                                const std::vector<LongType>& outputShape,
                                int kH, int kW,
                                int sH, int sW,
                                int pH, int pW,
                                int dH, int dW,
                                int nElements);

  // ── Diagnostics helpers ──

  // Dump section breakdown to stderr
  static void dumpSectionBreakdown(const std::vector<KernelSection>& sections,
                                   int startSlot, int endSlot,
                                   int maxSectionGrid, bool cooperativeLaunch);

  // Dump arg mapping to stderr
  static void dumpArgMapping(const std::vector<TritonKernelArg>& args,
                             int startSlot, int endSlot,
                             int eliminatedCount);
};

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
#endif  // LIBND4J_TRITON_IR_BUILDER_H
