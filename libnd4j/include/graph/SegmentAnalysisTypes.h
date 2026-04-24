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

#ifndef LIBND4J_SEGMENT_ANALYSIS_TYPES_H
#define LIBND4J_SEGMENT_ANALYSIS_TYPES_H

/**
 * Shared analysis types used by both Triton GPU and MLIR CPU graph backends.
 * Zero dependencies on MLIR or Triton — pure C++ structs and enums.
 *
 * Extracted from TritonIRBuilder.h so that CPU backends can reuse segment
 * analysis without pulling in GPU-specific headers.
 */

#include <config.h>

// This header is only meaningful when at least one graph backend that uses
// segment analysis is enabled. Guard against accidental inclusion.
#if defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX

#include <graph/gpu/OpCategoryTable.h>
#include <system/common.h>

#include <string>
#include <vector>

namespace sd {
namespace graph {

// Forward declaration
struct NativeSlot;

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
  int categoryCounts[20] = {};          // Indexed by (int)TritonOpCategory (18 values + headroom)
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
    TWO_LAYER_MLP,           // matmul→activation→matmul: candidate for fused kernel (FastVLA pattern)
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
 * Used by both Triton GPU and MLIR CPU backends to validate and classify before IR emission.
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

}  // namespace graph
}  // namespace sd

#endif  // defined(SD_CUDA) || defined(HAVE_MLIR) || HAVE_TRITON || HAVE_MLX
#endif  // LIBND4J_SEGMENT_ANALYSIS_TYPES_H
