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

//
// TritonIRBuilder — Section identification & emission:
//   identifySections, computeSectionGrid, emitGridSync,
//   emitThreadfenceBarrier, emitGatherSection, emitConcatSection,
//   emitSliceSection, emitTileSection, emitShapeManipulationSection,
//   emitPerElementMatmul, emitMatmulSection, emitAttentionSection,
//   emitSplitSection, emitScatterNdSection, emitConvolutionSection,
//   emitIm2colSection, emitCol2imSection,
//   dumpSectionBreakdown, dumpArgMapping
//

#include <config.h>

#if HAVE_TRITON

#include <graph/gpu/TritonIRBuilder.h>
#include <graph/gpu/TritonIRBuilder_internal.h>
#include <array/ArrayOptions.h>
#include <graph/DspDiagnostics.h>
#include <helpers/logger.h>
#include <helpers/shape.h>
#include <system/Environment.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#ifdef SD_CUDA
#include <cuda_runtime.h>
#endif

// MLIR core
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// Triton MLIR dialect
#include <triton/Dialect/Triton/IR/Dialect.h>
#include <triton/Dialect/Triton/IR/Types.h>

// Standard MLIR dialects
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

namespace sd {
namespace graph {

using namespace ir_builder_internal;

// ─── Section identification ─────────────────────────────────────────────────
// Walk ops in the segment and group into sections. A new section starts when:
// - Op type changes from element-wise to non-element-wise or vice versa
// - A non-element-wise op appears (matmul, attention each get their own section)
// - Contiguous element-wise ops fuse into one section

std::vector<KernelSection> TritonIRBuilder::identifySections(
    NativeSlot* slots, int startSlot, int endSlot,
    NDArray** outputSlots, int totalOutputSlots,
    NDArray** externalInputs, int numExternalInputs) {

  std::vector<KernelSection> sections;
  int segSize = endSlot - startSlot + 1;
  if (segSize == 0) return sections;

  // Helper: resolve source index to NDArray
  auto resolveArray = [&](int srcIdx) -> NDArray* {
    if (srcIdx < 0) {
      int extIdx = -(srcIdx + 1);
      if (extIdx < numExternalInputs && externalInputs) return externalInputs[extIdx];
    } else if (srcIdx < totalOutputSlots && outputSlots) {
      return outputSlots[srcIdx];
    }
    return nullptr;
  };

  // Helper: classify a category into a section type
  auto categoryToSectionType = [](TritonOpCategory cat, const std::string& opName) -> KernelSectionType {
    switch (cat) {
      case TritonOpCategory::BINARY_ELEMENTWISE:
      case TritonOpCategory::UNARY_ELEMENTWISE:
      case TritonOpCategory::COMPARISON:
      case TritonOpCategory::LOGICAL:
      case TritonOpCategory::TERNARY:
      case TritonOpCategory::CAST:
        return KernelSectionType::ELEMENTWISE;
      case TritonOpCategory::IDENTITY:
        return KernelSectionType::IDENTITY;
      case TritonOpCategory::MATMUL:
        return KernelSectionType::MATMUL;
      case TritonOpCategory::FUSED_ATTENTION:
        return KernelSectionType::FUSED_ATTENTION;
      case TritonOpCategory::REDUCTION:
        return KernelSectionType::REDUCTION;
      case TritonOpCategory::NORMALIZATION:
        return KernelSectionType::NORMALIZATION;
      case TritonOpCategory::ROPE:
        // ROPE is paired elementwise (no reductions, no cross-block communication).
        // Paired access (i, i+halfDim) is always within a single head (halfDim ≤ 32)
        // which fits inside any block (blockSize ≥ 256). Classifying as ELEMENTWISE
        // lets ROPE merge into mega-kernels with no barriers.
        return KernelSectionType::ELEMENTWISE;
      case TritonOpCategory::SHAPE_MANIPULATION:
        return KernelSectionType::SHAPE_MANIPULATION;
      case TritonOpCategory::CONSTANT_GENERATION:
        return KernelSectionType::CONSTANT_GENERATION;
      case TritonOpCategory::CONVOLUTION:
        return KernelSectionType::CONVOLUTION;
      case TritonOpCategory::DATA_MOVEMENT: {
        // Sub-classify data movement ops
        std::string lower = opName;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("gather_nd") != std::string::npos || lower == "gathernd")
          return KernelSectionType::GATHER_ND;
        if (lower.find("gather") != std::string::npos)
          return KernelSectionType::GATHER;
        if (lower.find("concat") != std::string::npos)
          return KernelSectionType::CONCAT;
        if (lower.find("split_v") != std::string::npos || lower == "splitv")
          return KernelSectionType::SPLIT_V;
        if (lower.find("split") != std::string::npos)
          return KernelSectionType::SPLIT;
        if (lower.find("stack") != std::string::npos)
          return KernelSectionType::STACK;
        if (lower.find("strided_slice") != std::string::npos)
          return KernelSectionType::STRIDED_SLICE;
        if (lower.find("tile") != std::string::npos)
          return KernelSectionType::TILE;
        if (lower.find("scatter_nd_update") != std::string::npos)
          return KernelSectionType::SCATTER_ND_UPDATE;
        if (lower.find("scatter_nd") != std::string::npos)
          return KernelSectionType::SCATTER_ND;
        return KernelSectionType::GATHER;  // Default data movement
      }
      case TritonOpCategory::UNSUPPORTED:
      case TritonOpCategory::FUSED_LLM:
        // Non-compilable ops must use SHAPE_MANIPULATION section type
        // (alwaysFallback=true) so they run via native slot-by-slot.
        return KernelSectionType::SHAPE_MANIPULATION;
      default:
        return KernelSectionType::ELEMENTWISE;
    }
  };

  // Helper: check if a section type can be merged with element-wise.
  // SHAPE_MANIPULATION (reshape, squeeze, expand_dims, permute) is merged into
  // element-wise sections and SSA-forwarded. For autoregressive decode (seq=1),
  // permute [0,2,1,3] on [1,1,H,D] is identity; for general seq>1, permute
  // would need actual reordering via a transposed store at the section boundary.
  //
  // CONSTANT_GENERATION (shape_of, create, set_scalar, ones_like, zeros_like) is
  // NOT merged because these ops produce outputs with different element counts than
  // data ops (e.g., shape_of produces [rank] while data ops produce [batch,seq,dim]).
  // A single n_elements kernel parameter can't cover both sizes — the smaller one
  // wins and larger outputs get partially written, causing accuracy bugs.
  // REDUCTION and NORMALIZATION are NOT merged into ELEMENTWISE sections.
  // They are multi-pass patterns with section-local synchronization/ordering
  // requirements and should be emitted via dedicated section handlers.
  //
  // SHAPE_MANIPULATION ops (reshape, squeeze, expand_dims, permute, transpose)
  // are NOT merged into ELEMENTWISE sections.  They run via native fallback
  // (zero-copy view creation) which is both correct and free.  Merging them
  // caused Bug 2: the Triton kernel SSA-forwarded reshape values and stored
  // to the reshape output buffer, but the stored data was zeros — likely
  // due to buffer pointer resolution issues between the Triton kernel and
  // ND4J's view-based DataBuffer sharing.
  auto canMergeWithElementwise = [](KernelSectionType type) -> bool {
    switch (type) {
      case KernelSectionType::ELEMENTWISE:
      case KernelSectionType::IDENTITY:
        return true;
      default:
        return false;
    }
  };

  // Helper: compute total output elements for an op's first output
  auto getOutputElements = [&](int slotIdx) -> LongType {
    if (slotIdx < 0 || slotIdx > endSlot) return 0;
    auto& slot = slots[slotIdx];
    if (slot.numOutputs <= 0) return 0;
    int outIdx = slot.outputSlotIndices[0];
    NDArray* outArr = resolveArray(outIdx);
    if (outArr) return outArr->lengthOf();
    return 0;
  };

  KernelSection currentSection;
  currentSection.startSlot = startSlot;
  currentSection.endSlot = startSlot;
  currentSection.numOps = 0;

  auto firstCat = getOpCategory(slots[startSlot].opName);
  // Reclassify TERNARY ops with < 3 inputs (e.g., 1-input Where = coordinate
  // extraction, not element-wise select).  These are data-dependent and cannot
  // be compiled by Triton — they must run via native fallback.
  if (firstCat == TritonOpCategory::TERNARY && slots[startSlot].numInputs < 3) {
    firstCat = TritonOpCategory::UNSUPPORTED;
  }
  currentSection.type = categoryToSectionType(firstCat, slots[startSlot].opName);

  // Track the element count for the current element-wise section.
  // All ops in the same section must have compatible element counts for
  // the 1D grid kernel to work correctly. When element count changes,
  // we must start a new section.
  LongType currentSectionElements = getOutputElements(startSlot);

  // Track whether the current section contains a shape manipulation op
  // (permute, transpose, reshape, squeeze, expand_dims).  When true,
  // the NEXT op must start a new section so each shape op runs via
  // native fallback (zero-copy view or permute reindexing).
  bool currentSectionHasShapeOp = false;

  // Helper: detect whether an op is any shape manipulation op.
  // ALL shape manipulation ops are isolated into their own sections and
  // run via native fallback.  This prevents:
  //  - Permute/transpose: incorrect SSA-forwarding in 1D skeleton
  //  - Reshape/squeeze/expand_dims: zero-output bug from buffer pointer
  //    resolution issues between Triton kernels and ND4J view-based DataBuffers
  auto isShapeManipOp = [](const NativeSlot& slot) -> bool {
    auto cat = getOpCategory(slot.opName);
    return cat == TritonOpCategory::SHAPE_MANIPULATION;
  };

  auto shapeInfoToVector = [](const LongType* shapeInfo) -> std::vector<LongType> {
    std::vector<LongType> shapeVec;
    if (shapeInfo == nullptr) return shapeVec;
    int rank = shape::rank(shapeInfo);
    shapeVec.reserve(rank);
    const LongType* dims = shape::shapeOf(shapeInfo);
    for (int d = 0; d < rank; d++) {
      shapeVec.push_back(dims[d]);
    }
    return shapeVec;
  };

  auto resolveShapeVector = [&](int srcIdx) -> std::vector<LongType> {
    NDArray* arr = resolveArray(srcIdx);
    if (arr != nullptr) {
      std::vector<LongType> shapeVec;
      shapeVec.reserve(arr->rankOf());
      for (int d = 0; d < arr->rankOf(); d++) {
        shapeVec.push_back(arr->sizeAt(d));
      }
      return shapeVec;
    }

    if (srcIdx >= 0) {
      for (int s = startSlot; s <= endSlot; s++) {
        auto& producerSlot = slots[s];
        if (!producerSlot.shapeCacheValid() || producerSlot.cachedOutputShapes.empty()) continue;
        for (int o = 0; o < producerSlot.numOutputs; o++) {
          if (o >= static_cast<int>(producerSlot.cachedOutputShapes.size())) break;
          if (producerSlot.outputSlotIndices[o] == srcIdx) {
            return shapeInfoToVector(producerSlot.cachedOutputShapes[o]);
          }
        }
      }
    }

    return {};
  };

  auto normalizedAxisForSectionType = [&](KernelSectionType type,
                                          const NativeSlot& slot) -> int {
    if (slot.numIArgs <= 0 || slot.iArgs == nullptr) {
      return 0;
    }

    int axis = static_cast<int>(slot.iArgs[0]);
    std::vector<LongType> shapeVec;
    if (type == KernelSectionType::CONCAT) {
      if (slot.numOutputs > 0) {
        shapeVec = resolveShapeVector(slot.outputSlotIndices[0]);
      }
      if (shapeVec.empty() && slot.numInputs > 0) {
        shapeVec = resolveShapeVector(slot.inputSourceIndices[0]);
      }
    } else if (type == KernelSectionType::GATHER || type == KernelSectionType::GATHER_ND) {
      if (slot.numInputs > 0) {
        shapeVec = resolveShapeVector(slot.inputSourceIndices[0]);
      }
    }

    int rank = static_cast<int>(shapeVec.size());
    if (rank > 0 && axis < 0) {
      axis += rank;
    }
    if (rank > 0 && (axis < 0 || axis >= rank)) {
      axis = 0;
    }
    return axis;
  };

  auto canAppendNonElementwiseSlot = [&](const KernelSection& current,
                                         KernelSectionType nextType,
                                         const NativeSlot& nextSlot) -> bool {
    if (current.type != nextType) {
      return false;
    }
    if (nextType == KernelSectionType::CONCAT) {
      return current.concatAxis == normalizedAxisForSectionType(nextType, nextSlot);
    }
    if (nextType == KernelSectionType::GATHER || nextType == KernelSectionType::GATHER_ND) {
      return current.gatherAxis == normalizedAxisForSectionType(nextType, nextSlot);
    }
    return true;
  };

  auto getPermutationForSlot = [](const NativeSlot& slot, const std::string& opLower,
                                  int rank) -> std::vector<int> {
    std::vector<int> permutation;
    if (slot.numIArgs > 0 && slot.iArgs != nullptr) {
      permutation.reserve(slot.numIArgs);
      for (int d = 0; d < slot.numIArgs; d++) {
        permutation.push_back(static_cast<int>(slot.iArgs[d]));
      }
    }
    if (permutation.empty() && opLower == "transpose") {
      permutation.reserve(rank);
      for (int d = rank - 1; d >= 0; d--) {
        permutation.push_back(d);
      }
    }
    return permutation;
  };

  auto isLayoutIdentityPermute = [](const std::vector<LongType>& inputShape,
                                    const std::vector<int>& permutation) -> bool {
    if (inputShape.empty() || inputShape.size() != permutation.size()) return false;

    int nextPermIdx = 0;
    for (int inputDim = 0; inputDim < static_cast<int>(inputShape.size()); inputDim++) {
      if (inputShape[inputDim] == 1) continue;
      while (nextPermIdx < static_cast<int>(permutation.size()) &&
             inputShape[permutation[nextPermIdx]] == 1) {
        nextPermIdx++;
      }
      if (nextPermIdx >= static_cast<int>(permutation.size()) ||
          permutation[nextPermIdx] != inputDim) {
        return false;
      }
      nextPermIdx++;
    }
    return true;
  };

  // Helper: detect whether a shape manipulation op is an IDENTITY (true no-op).
  // reshape/squeeze/expand_dims are identity only when their logical shape is unchanged.
  // permute/transpose are identity only when they move singleton dimensions and preserve
  // the relative order of all non-singleton axes (the seq=1 decode fast path).
  // Identity shape ops are UNCONDITIONALLY SSA-forwarded. These are true no-ops
  // (same physical layout) and fusing them into elementwise sections is always safe.
  // Previously gated behind tritonFuseIdentityShapes() — now always on because:
  // 1. Element count is unchanged by definition
  // 2. No data movement — same buffer, just different view metadata
  // 3. Reduces section breaks and kernel launches for no cost
  auto isIdentityShapeOp = [&](const NativeSlot& slot) -> bool {
    auto cat = getOpCategory(slot.opName);
    if (cat != TritonOpCategory::SHAPE_MANIPULATION) return false;

    if (slot.numInputs <= 0 || slot.numOutputs <= 0) return false;
    int inputSrcIdx = slot.inputSourceIndices[0];
    std::vector<LongType> inputShape = resolveShapeVector(inputSrcIdx);
    std::vector<LongType> outputShape = resolveShapeVector(slot.outputSlotIndices[0]);
    if (outputShape.empty() && slot.shapeCacheValid() && !slot.cachedOutputShapes.empty()) {
      outputShape = shapeInfoToVector(slot.cachedOutputShapes[0]);
    }
    if (inputShape.empty() || outputShape.empty() || inputShape.size() != outputShape.size()) {
      return false;
    }

    std::string opLower = slot.opName;
    std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);
    bool isPermuteLike = (opLower == "permute" || opLower == "transpose");

    if (!isPermuteLike) {
      return inputShape == outputShape;
    }

    std::vector<int> permutation = getPermutationForSlot(slot, opLower,
                                                         static_cast<int>(inputShape.size()));
    if (permutation.size() != inputShape.size()) return false;

    bool exactIdentity = true;
    for (int d = 0; d < static_cast<int>(permutation.size()); d++) {
      if (permutation[d] != d) {
        exactIdentity = false;
        break;
      }
    }
    if (exactIdentity) return true;

    if (!sd::Environment::getInstance().tritonSpecializePermuteSeq1()) {
      return false;
    }

    return isLayoutIdentityPermute(inputShape, permutation);
  };

  // Check if the very first op is a shape manipulation op
  if (isShapeManipOp(slots[startSlot])) {
    currentSectionHasShapeOp = true;
  }

  for (int i = startSlot; i <= endSlot; i++) {
    auto cat = getOpCategory(slots[i].opName);
    // Reclassify TERNARY ops with < 3 inputs as non-compilable
    if (cat == TritonOpCategory::TERNARY && slots[i].numInputs < 3) {
      cat = TritonOpCategory::UNSUPPORTED;
    }
    auto sectionType = categoryToSectionType(cat, slots[i].opName);

    // Check if this is a shape manipulation op
    bool isShapeOp = (sectionType == KernelSectionType::SHAPE_MANIPULATION);
    
    // Check if it's an IDENTITY shape op (true no-op, SSA forward only)
    bool isIdentityShape = isIdentityShapeOp(slots[i]);
    
    // Identity shape ops are treated as element-wise for section merging purposes
    if (isIdentityShape) {
      sectionType = KernelSectionType::ELEMENTWISE;
      isShapeOp = false;
    }

    bool startNewSection = false;
    bool currentIsElementwiseLike = canMergeWithElementwise(currentSection.type);
    bool sectionIsElementwiseLike = canMergeWithElementwise(sectionType);

    if (i == startSlot) {
      // First op — always part of current section. Reassign the section type after
      // identity-shape detection so the seed type cannot remain SHAPE_MANIPULATION.
      currentSection.type = sectionType;
      currentSectionElements = getOutputElements(i);
      currentSectionHasShapeOp = isShapeOp;
      startNewSection = false;
    } else if (isShapeOp && !isIdentityShape) {
      // Non-identity shape manipulation ops always get their own isolated section.
      // Both starting a shape op AND the op after a shape op section must break.
      // Permute/transpose require non-linear index reindexing; reshape/squeeze/
      // expand_dims cause zero-output bugs when SSA-forwarded in Triton kernels
      // due to buffer pointer resolution issues with ND4J view-based DataBuffers.
      // All shape ops run via native fallback (alwaysFallback=true).
      startNewSection = true;
      currentSectionHasShapeOp = true;
    } else if (isIdentityShape) {
      // Identity shape ops (reshape where input shape == output shape) are no-ops.
      // They can be fused into element-wise sections WITHOUT data movement.
      // The Triton kernel just forwards the SSA value, no actual reshape happens.
      // This is safe because:
      // 1. Element count doesn't change (identity by definition)
      // 2. No data movement needed (same buffer, just different view metadata)
      // 3. SSA forwarding handles it correctly (no buffer pointer issues)
      // Continue with current section, don't start new one
      startNewSection = false;
    } else {
      // Check for cast chain: consecutive cast ops can fuse into single kernel
      // Controlled by tritonFuseCastChains flag (default: true).
      bool isCastChain = false;
      if (sd::Environment::getInstance().tritonFuseCastChains()) {
        auto currentCat = getOpCategory(slots[i].opName);
        bool currentIsCast = (currentCat == TritonOpCategory::CAST);
        bool sectionIsCast = (currentIsElementwiseLike && currentSection.numOps > 0);
        // Check if previous op in section was also a cast
        if (currentIsCast && sectionIsCast && currentSection.numOps > 0) {
          int prevSlot = i - 1;
          if (prevSlot >= startSlot) {
            auto prevCat = getOpCategory(slots[prevSlot].opName);
            if (prevCat == TritonOpCategory::CAST) {
              isCastChain = true;
            }
          }
        }
      }
      
      if (isCastChain) {
        // Cast chain continuation: consecutive cast ops fuse into single kernel.
        // This is safe because cast doesn't change element count.
        // The fused cast kernel does direct dtype conversion (A→C) instead of
        // intermediate (A→B→C), saving kernel launch overhead.
        startNewSection = false;
      } else if (sectionType == KernelSectionType::MATMUL ||
                 sectionType == KernelSectionType::FUSED_ATTENTION ||
                 sectionType == KernelSectionType::CONVOLUTION) {
        // Non-element-wise ops always get their own section
        startNewSection = true;
      } else if (currentSection.type == KernelSectionType::MATMUL ||
                 currentSection.type == KernelSectionType::FUSED_ATTENTION ||
                 currentSection.type == KernelSectionType::CONVOLUTION) {
        // After a non-element-wise section, start a new one
        startNewSection = true;
      } else if (!currentIsElementwiseLike) {
        // Non-mergeable section types can still absorb consecutive ops of the SAME type.
        // Keep section-level metadata stable for data-movement ops that later
        // consult section fields such as concatAxis.
        startNewSection = !canAppendNonElementwiseSlot(currentSection, sectionType, slots[i]);
      } else if (!sectionIsElementwiseLike && currentSection.type != sectionType) {
        // Non-elementwise sections must stay homogeneous. Attention-neighborhood
        // preferences are handled later by fusion scoring, not by creating mixed
        // section types that the emitters cannot execute safely.
        startNewSection = true;
      } else if (currentIsElementwiseLike && sectionIsElementwiseLike) {
        // Both share the 1D elementwise skeleton — check element count compatibility.
        // The 1D kernel uses a single n_elements for the grid and mask.
        // Ops with different output element counts cannot share the same grid:
        // - If n_elements is too small, larger outputs get partially written (stale data)
        // - If n_elements is too large, smaller outputs get buffer overflows
        // Identity ops don't change element count, so they're always safe to merge.
        // Shape manipulation ops (reshape, squeeze, expand_dims, permute) are now
        // excluded from ELEMENTWISE sections entirely — they run via native fallback.
        LongType opElements = getOutputElements(i);
        if (opElements > 0 && currentSectionElements > 0 && opElements != currentSectionElements) {
          // Element count changed — check if broadcast-compatible.
          // The 1D kernel uses a single n_elements (the larger of the two).
          // Broadcast-compatible cases:
          //   1. Scalar (1 element) is always broadcast-compatible
          //   2. One count evenly divides the other (e.g., [N] op on [B,N] data
          //      where output is [B,N]) — the larger count is used for the grid
          //      and the smaller operand is broadcast via modular indexing.
          bool broadcastCompatible = false;
          if (opElements == 1 || currentSectionElements == 1) {
            broadcastCompatible = true;
          } else {
            // Check if one divides the other (broadcast along batch dimensions)
            LongType larger = std::max(opElements, currentSectionElements);
            LongType smaller = std::min(opElements, currentSectionElements);
            if (smaller > 0 && larger % smaller == 0) {
              broadcastCompatible = true;
            }
          }
          if (!broadcastCompatible) {
            startNewSection = true;
          }
        }
      }
    }

    if (startNewSection && currentSection.numOps > 0) {
      // Finalize current section and start new one
      sections.push_back(currentSection);
      currentSection = KernelSection();
      currentSection.startSlot = i;
      currentSection.type = sectionType;
      // Reset element count for the new section
      currentSectionElements = getOutputElements(i);
      // Reset shape op tracking for the new section
      currentSectionHasShapeOp = isShapeOp;
    }

    // Track shape ops in current section (handles case when first op IS a shape op
    // and no new section was started because it was the first op overall)
    if (isShapeOp) {
      currentSectionHasShapeOp = true;
    }

    currentSection.endSlot = i;
    currentSection.numOps++;

    // Update section element count — use the largest non-scalar output
    // to handle cases where the first op in a section is a scalar
    if (canMergeWithElementwise(currentSection.type)) {
      LongType opElements = getOutputElements(i);
      if (opElements > currentSectionElements) {
        currentSectionElements = opElements;
      }
    }

    // Extract matmul dimensions
    if (sectionType == KernelSectionType::MATMUL) {
      currentSection.type = KernelSectionType::MATMUL;
      if (slots[i].numInputs >= 2) {
        NDArray* aArr = resolveArray(slots[i].inputSourceIndices[0]);
        NDArray* bArr = resolveArray(slots[i].inputSourceIndices[1]);
        if (aArr && aArr->rankOf() >= 2) {
          currentSection.matmulM = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 2));
          currentSection.matmulK = static_cast<int>(aArr->sizeAt(aArr->rankOf() - 1));
        }
        if (bArr && bArr->rankOf() >= 2) {
          currentSection.matmulN = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 1));
          if (currentSection.matmulK == 0)
            currentSection.matmulK = static_cast<int>(bArr->sizeAt(bArr->rankOf() - 2));
        }
        // Fallback: resolve M from input A's producing slot's cached output shape
        if (currentSection.matmulM == 0) {
          int aSrc = slots[i].inputSourceIndices[0];
          if (aSrc >= 0 && aSrc < totalOutputSlots) {
            for (int s = startSlot; s <= endSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid() && !slots[s].cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].cachedOutputShapes.size())) {
                  const LongType* si = slots[s].cachedOutputShapes[o];
                  if (si && shape::rank(si) >= 2) {
                    currentSection.matmulM = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 2]);
                    break;
                  }
                }
              }
              if (currentSection.matmulM > 0) break;
            }
          }
        }
        // Fallback: resolve M from output shape (output of matmul is [..., M, N])
        if (currentSection.matmulM == 0 && slots[i].numOutputs > 0) {
          int outIdx = slots[i].outputSlotIndices[0];
          NDArray* outArr = resolveArray(outIdx);
          if (outArr && outArr->rankOf() >= 2) {
            currentSection.matmulM = static_cast<int>(outArr->sizeAt(outArr->rankOf() - 2));
            if (currentSection.matmulN == 0)
              currentSection.matmulN = static_cast<int>(outArr->sizeAt(outArr->rankOf() - 1));
          } else if (outIdx >= 0 && outIdx < totalOutputSlots) {
            // Try the producing slot's cached output shape for outIdx
            for (int s = startSlot; s <= endSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == outIdx &&
                    slots[s].shapeCacheValid() && !slots[s].cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].cachedOutputShapes.size())) {
                  const LongType* si = slots[s].cachedOutputShapes[o];
                  if (si && shape::rank(si) >= 2) {
                    currentSection.matmulM = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 2]);
                    if (currentSection.matmulN == 0)
                      currentSection.matmulN = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 1]);
                  }
                }
              }
              if (currentSection.matmulM > 0) break;
            }
          }
        }
        // Fallback: resolve K from input A's producing slot's cached output shape
        if (currentSection.matmulK == 0) {
          int aSrc = slots[i].inputSourceIndices[0];
          if (aSrc >= 0 && aSrc < totalOutputSlots) {
            for (int s = startSlot; s <= endSlot; s++) {
              for (int o = 0; o < slots[s].numOutputs; o++) {
                if (slots[s].outputSlotIndices[o] == aSrc &&
                    slots[s].shapeCacheValid() && !slots[s].cachedOutputShapes.empty() &&
                    o < static_cast<int>(slots[s].cachedOutputShapes.size())) {
                  const LongType* si = slots[s].cachedOutputShapes[o];
                  if (si && shape::rank(si) >= 2) {
                    currentSection.matmulK = static_cast<int>(shape::shapeOf(si)[shape::rank(si) - 1]);
                  }
                }
              }
              if (currentSection.matmulK > 0) break;
            }
          }
        }
      }
    }

    // Extract attention dimensions
    if (sectionType == KernelSectionType::FUSED_ATTENTION) {
      currentSection.type = KernelSectionType::FUSED_ATTENTION;
      if (slots[i].numInputs >= 1) {
        NDArray* qArr = resolveArray(slots[i].inputSourceIndices[0]);
        if (qArr && qArr->rankOf() >= 3) {
          int rank = qArr->rankOf();
          if (rank >= 4) {
            // 4D BHSD: [batch, numHeads, seqQ, headDim]
            currentSection.headDim = static_cast<int>(qArr->sizeAt(rank - 1));
            currentSection.seqQ = static_cast<int>(qArr->sizeAt(rank - 2));
            currentSection.numHeads = static_cast<int>(qArr->sizeAt(rank - 3));
            currentSection.batchSize = 1;
            for (int d = 0; d < rank - 3; d++)
              currentSection.batchSize *= static_cast<int>(qArr->sizeAt(d));
            currentSection.attnQIsBSHD = false;
          } else {
            // 3D BSHD: [batch, seqQ, numHeads*headDim]
            currentSection.batchSize = static_cast<int>(qArr->sizeAt(0));
            currentSection.seqQ = static_cast<int>(qArr->sizeAt(1));
            int hidden = static_cast<int>(qArr->sizeAt(2));
            int nh = (slots[i].numIArgs > 0 && slots[i].iArgs) ? static_cast<int>(slots[i].iArgs[0]) : 1;
            if (nh <= 0) nh = 1;
            currentSection.numHeads = nh;
            currentSection.headDim = hidden / nh;
            currentSection.attnQIsBSHD = true;
          }
          currentSection.attentionScale = 1.0f / std::sqrt(static_cast<float>(currentSection.headDim));
        }

        // Determine effective K source: use past_key (input 4) if available
        bool hasPastKv = false;
        int effectiveKInputIdx = 1;
        if (slots[i].numInputs > 4) {
          NDArray* pastKeyArr = resolveArray(slots[i].inputSourceIndices[4]);
          if (pastKeyArr && pastKeyArr->rankOf() == 4) {
            auto pastKeyLen = pastKeyArr->sizeAt(pastKeyArr->rankOf() - 2);
            if (pastKeyLen > 1) {
              hasPastKv = true;
              effectiveKInputIdx = 4;
            }
          }
        }

        NDArray* kArr = (effectiveKInputIdx < slots[i].numInputs) ?
            resolveArray(slots[i].inputSourceIndices[effectiveKInputIdx]) : nullptr;
        if (kArr && kArr->rankOf() >= 2) {
          currentSection.seqK = static_cast<int>(kArr->sizeAt(kArr->rankOf() - 2));
          currentSection.attnKIsBSHD = hasPastKv ? false : currentSection.attnQIsBSHD;
          // GQA: extract KV head count from past_key (4D: [B, KvHeads, seqK, HD])
          if (hasPastKv && kArr->rankOf() == 4) {
            currentSection.numKvHeads = static_cast<int>(kArr->sizeAt(1));
            currentSection.headDim = static_cast<int>(kArr->sizeAt(3));
          }
        }
        // Default: MHA (numKvHeads = numHeads)
        if (currentSection.numKvHeads <= 0) currentSection.numKvHeads = currentSection.numHeads;
      }
    }

    // Extract gather axis from iArgs
    if (sectionType == KernelSectionType::GATHER || sectionType == KernelSectionType::GATHER_ND) {
      currentSection.gatherAxis = normalizedAxisForSectionType(sectionType, slots[i]);
    }

    // Extract concat axis
    if (sectionType == KernelSectionType::CONCAT) {
      currentSection.concatAxis = normalizedAxisForSectionType(sectionType, slots[i]);
    }
  }

  // Don't forget the last section
  if (currentSection.numOps > 0) {
    sections.push_back(currentSection);
  }

  // Post-processing: merge consecutive sections of the same type.
  // Only 1D elementwise-like sections are post-merged here. Other section types
  // carry section-specific metadata and grid semantics that must stay isolated
  // unless they are re-analyzed explicitly.
  if (sd::Environment::getInstance().tritonSectionFusion() && sections.size() > 1) {
    auto sectionMergeElements = [&](const KernelSection& sec) -> LongType {
      LongType maxElements = 0;
      for (int si = sec.startSlot; si <= sec.endSlot; si++) {
        LongType opElements = getOutputElements(si);
        if (opElements > maxElements) {
          maxElements = opElements;
        }
      }
      return maxElements;
    };

    auto canPostMergeSections = [&](const KernelSection& a, const KernelSection& b) -> bool {
      if (a.type != b.type || !canMergeWithElementwise(a.type)) {
        return false;
      }

      LongType aElements = sectionMergeElements(a);
      LongType bElements = sectionMergeElements(b);
      if (aElements <= 0 || bElements <= 0) {
        return true;
      }
      if (aElements == bElements) {
        return true;
      }
      return aElements == 1 || bElements == 1;
    };

    std::vector<KernelSection> merged;
    merged.reserve(sections.size());
    
    KernelSection* current = nullptr;
    for (auto& sec : sections) {
      if (current == nullptr) {
        merged.push_back(sec);
        current = &merged.back();
      } else if (canPostMergeSections(*current, sec)) {
        // Same 1D elementwise skeleton and compatible element counts.
        current->endSlot = sec.endSlot;
        current->numOps += sec.numOps;
        // Grid requirement will be recomputed below
      } else {
        merged.push_back(sec);
        current = &merged.back();
      }
    }
    
    if (merged.size() < sections.size()) {
      DSP_DIAG(COMPILE, "TritonIRBuilder::identifySections: post-merge reduced %zu sections to %zu",
               sections.size(), merged.size());
      sections = std::move(merged);
    }
  }

  // ── Matmul/Normalization epilogue absorption ──
  // When a MATMUL or NORMALIZATION section is immediately followed by an
  // ELEMENTWISE section whose ops ONLY consume the prior section's output,
  // absorb those elementwise ops as an epilogue. This eliminates the global
  // memory round-trip: the Triton emitter can fuse bias-add/activation/cast
  // into the matmul tile loop or normalization epilogue.
  //
  // Conditions for absorption:
  //   1. Next section is ELEMENTWISE
  //   2. ALL inputs of each epilogue op either come from the prior section's
  //      output slots OR from external inputs (constants/weights)
  //   3. The prior section's output is consumed ONLY by the epilogue ops
  //      (no other sections or ops outside this range consume it)
  //   4. Epilogue chain is short (≤ 8 ops) to avoid register pressure
  if (sections.size() > 1) {
    // Build a set of all output slot indices produced by each section
    auto sectionOutputSlots = [&](const KernelSection& sec) -> std::unordered_set<int> {
      std::unordered_set<int> outputs;
      for (int s = sec.startSlot; s <= sec.endSlot; s++) {
        for (int o = 0; o < slots[s].numOutputs; o++) {
          outputs.insert(slots[s].outputSlotIndices[o]);
        }
      }
      return outputs;
    };

    // Check if all inputs of an elementwise section come from the producer
    // section's outputs or from external inputs (negative source indices)
    auto allInputsFromProducerOrExternal = [&](const KernelSection& ewSec,
                                                const std::unordered_set<int>& producerOutputs) -> bool {
      for (int s = ewSec.startSlot; s <= ewSec.endSlot; s++) {
        for (int inp = 0; inp < slots[s].numInputs; inp++) {
          int srcIdx = slots[s].inputSourceIndices[inp];
          if (srcIdx < 0) continue;  // External input (constant/weight) — always OK
          if (producerOutputs.count(srcIdx) == 0) {
            // This input comes from a slot output NOT produced by the producer section.
            // Check if it's produced within the epilogue section itself (chain dependency).
            bool producedWithinEpilogue = false;
            for (int es = ewSec.startSlot; es < s; es++) {
              for (int eo = 0; eo < slots[es].numOutputs; eo++) {
                if (slots[es].outputSlotIndices[eo] == srcIdx) {
                  producedWithinEpilogue = true;
                  break;
                }
              }
              if (producedWithinEpilogue) break;
            }
            if (!producedWithinEpilogue) return false;
          }
        }
      }
      return true;
    };

    // Check that the producer section's outputs are not consumed by any section
    // other than the epilogue candidate
    auto outputsOnlyConsumedByNext = [&](const KernelSection& producerSec,
                                          const KernelSection& nextSec,
                                          const std::unordered_set<int>& producerOutputs) -> bool {
      for (int outSlot : producerOutputs) {
        // Check all slots outside [producerSec.start, nextSec.end]
        for (int s = startSlot; s <= endSlot; s++) {
          if (s >= producerSec.startSlot && s <= nextSec.endSlot) continue;
          for (int inp = 0; inp < slots[s].numInputs; inp++) {
            if (slots[s].inputSourceIndices[inp] == outSlot) {
              return false;  // Consumed outside the fused range
            }
          }
        }
      }
      return true;
    };

    static constexpr int MAX_EPILOGUE_OPS = 8;

    std::vector<KernelSection> absorbed;
    absorbed.reserve(sections.size());
    std::unordered_set<int> skipIndices;

    for (size_t si = 0; si < sections.size(); si++) {
      if (skipIndices.count(static_cast<int>(si))) continue;

      auto& sec = sections[si];
      bool isMatmul = (sec.type == KernelSectionType::MATMUL);
      bool isNorm = (sec.type == KernelSectionType::NORMALIZATION);

      if ((isMatmul || isNorm) && si + 1 < sections.size()) {
        auto& nextSec = sections[si + 1];
        if (nextSec.type == KernelSectionType::ELEMENTWISE &&
            nextSec.numOps <= MAX_EPILOGUE_OPS) {
          auto prodOutputs = sectionOutputSlots(sec);
          if (allInputsFromProducerOrExternal(nextSec, prodOutputs) &&
              outputsOnlyConsumedByNext(sec, nextSec, prodOutputs)) {
            // Absorb: extend the section to cover the epilogue ops
            KernelSection fused = sec;
            if (isMatmul) {
              fused.matmulEpilogueStartSlot = nextSec.startSlot;
              fused.matmulEpilogueEndSlot = nextSec.endSlot;
              fused.matmulEpilogueCount = nextSec.numOps;
              DSP_DIAG(COMPILE, "identifySections: absorbed %d ELEMENTWISE epilogue ops into MATMUL section [%d-%d] -> [%d-%d]",
                       nextSec.numOps, sec.startSlot, sec.endSlot, sec.startSlot, nextSec.endSlot);
            } else {
              fused.normEpilogueStartSlot = nextSec.startSlot;
              fused.normEpilogueEndSlot = nextSec.endSlot;
              fused.normEpilogueCount = nextSec.numOps;
              DSP_DIAG(COMPILE, "identifySections: absorbed %d ELEMENTWISE epilogue ops into NORMALIZATION section [%d-%d] -> [%d-%d]",
                       nextSec.numOps, sec.startSlot, sec.endSlot, sec.startSlot, nextSec.endSlot);
            }
            fused.endSlot = nextSec.endSlot;
            fused.numOps += nextSec.numOps;
            absorbed.push_back(fused);
            skipIndices.insert(static_cast<int>(si + 1));
            continue;
          }
        }
      }

      absorbed.push_back(sec);
    }

    if (absorbed.size() < sections.size()) {
      DSP_DIAG(COMPILE, "identifySections: epilogue absorption reduced %zu sections to %zu",
               sections.size(), absorbed.size());
      sections = std::move(absorbed);
    }
  }

  // Compute grid requirement for each section
  int defaultBlockSize = 1024;
  for (auto& sec : sections) {
    sec.gridRequirement = computeSectionGrid(sec, defaultBlockSize);
  }

  return sections;
}

// ─── Section grid computation ───────────────────────────────────────────────

int TritonIRBuilder::computeSectionGrid(const KernelSection& section, int blockSize) {
  switch (section.type) {
    case KernelSectionType::MATMUL: {
      int gridM = (section.matmulM + section.blockM - 1) / section.blockM;
      int gridN = (section.matmulN + section.blockN - 1) / section.blockN;
      return gridM * gridN;
    }
    case KernelSectionType::FUSED_ATTENTION: {
      int batchSize = std::max(1, section.batchSize);
      int numHeads = std::max(1, section.numHeads);
      int seqQ = std::max(1, section.seqQ);
      int seqK = std::max(1, section.seqK);
      int headDim = std::max(1, section.headDim);
      auto attnTile = chooseFusedAttentionTileConfig(
          batchSize, numHeads, seqQ, seqK, headDim);
      int batchHeads = batchSize * numHeads;
      int blockM = std::max(1, attnTile.blockM);
      int gridQ = (seqQ + blockM - 1) / blockM;
      return batchHeads * gridQ;
    }
    case KernelSectionType::ELEMENTWISE:
    case KernelSectionType::IDENTITY:
    case KernelSectionType::CONSTANT_GENERATION:
    case KernelSectionType::SHAPE_MANIPULATION:
    case KernelSectionType::REDUCTION:
    case KernelSectionType::NORMALIZATION:
    case KernelSectionType::GATHER:
    case KernelSectionType::GATHER_ND:
    case KernelSectionType::CONCAT:
    case KernelSectionType::SPLIT:
    case KernelSectionType::SPLIT_V:
    case KernelSectionType::STACK:
    case KernelSectionType::STRIDED_SLICE:
    case KernelSectionType::TILE:
    case KernelSectionType::SCATTER_ND:
    case KernelSectionType::SCATTER_ND_UPDATE:
    case KernelSectionType::CONVOLUTION:
    default:
      // Placeholder. buildSectionedModule() recomputes launch grid from resolved
      // section shapes after tile selection.
      return 1;
  }
}

// ─── Grid sync emission ─────────────────────────────────────────────────────
// Emit a cooperative grid-wide synchronization barrier using inline PTX.
// Uses a global atomic counter + spin loop: each block atomically increments
// the counter, then spins until the counter reaches numBlocks.

static std::atomic<int> gridSyncCounter_{0};

void TritonIRBuilder::emitGridSync(mlir::OpBuilder& builder, mlir::Location loc,
                                    mlir::Value syncCounterPtr, mlir::Value numBlocksVal) {
  // Cooperative grid sync using atomic counter + spin barrier in inline PTX.
  // Each block's thread 0 atomically increments a global counter, then spins
  // until all blocks have arrived. Requires cuLaunchCooperativeKernel for
  // co-residency guarantee.
  //
  // Protocol:
  //   membar.gl;                            // Flush pending global stores
  //   bar.sync 0;                           // CTA barrier (all threads done)
  //   if (threadIdx.x == 0):
  //     atom.global.add counter, 1          // Arrive
  //     while (load(counter) < numBlocks);  // Spin wait
  //   bar.sync 0;                           // Propagate to all threads

  // Emit inline PTX for the full grid sync protocol.
  // syncCounterPtr is a pointer to a global u32 counter (passed as kernel arg).
  // numBlocksVal is the total number of blocks in the cooperative grid.
  int syncId = gridSyncCounter_++;
  std::string labelName = "GRID_SYNC_SPIN_" + std::to_string(syncId);
  // Operand numbering: $0 = dummy output (=r), $1 = syncCounterPtr (l), $2 = numBlocks (r)
  std::string asmStr =
      "{\n"
      "  .reg .pred %p_t0, %p_loop;\n"
      "  .reg .b32 %r_tid, %r_cnt;\n"
      "  membar.gl;\n"
      "  bar.sync 0;\n"
      "  mov.u32 %r_tid, %tid.x;\n"
      "  setp.eq.u32 %p_t0, %r_tid, 0;\n"
      //  Initialize %p_loop to false for ALL threads.
      // PTX registers are NOT zero-initialized (per PTX ISA spec).
      // Without this, non-thread-0 threads have undefined %p_loop,
      // and if it's stale-true they enter an infinite spin loop
      // (the setp inside is predicated on %p_t0 so never updates %p_loop
      // for non-thread-0 threads) → bar.sync 0 deadlocks.
      "  setp.eq.u32 %p_loop, 0, 1;\n"  // 0 == 1 is false → %p_loop = false
      "  @%p_t0 atom.global.add.u32 %r_cnt, [$1], 1;\n"
      "  @%p_t0 add.u32 %r_cnt, %r_cnt, 1;\n"  // atom returns old value, add 1
      + labelName + ":\n"
      "  @%p_t0 ld.global.acquire.gpu.u32 %r_cnt, [$1];\n"
      "  @%p_t0 setp.lt.u32 %p_loop, %r_cnt, $2;\n"
      "  @%p_loop bra " + labelName + ";\n"
      "  bar.sync 0;\n"
      "}\n";

  // Use tt.elementwise_inline_asm with the counter pointer and numBlocks as operands.
  // ElementwiseInlineAsmOp requires at least 1 result, so we provide a dummy scalar i32 output.
  // Constraints: "=r" for dummy output, "l" for 64-bit pointer, "r" for 32-bit integer
  auto i32Type = builder.getI32Type();
  builder.create<mlir::triton::ElementwiseInlineAsmOp>(
      loc, /*resultTypes=*/mlir::TypeRange{i32Type},
      asmStr,
      /*constraints=*/"=r,l,r",
      /*isPure=*/false, /*pack=*/1,
      mlir::ValueRange{syncCounterPtr, numBlocksVal});
}


void TritonIRBuilder::emitThreadfenceBarrier(mlir::OpBuilder& builder, mlir::Location loc) {
  // Lightweight inter-section barrier using membar.gl + bar.sync.
  // membar.gl flushes all pending global memory stores, making them visible
  // to all other blocks on the GPU. bar.sync 0 synchronizes threads within
  // the CTA so all threads in this block see the flushed state.
  //
  // Unlike emitGridSync(), this does NOT wait for other blocks to arrive.
  // The GPU scheduler will eventually run all blocks. Later sections that
  // read cross-block data will see committed writes because membar.gl
  // guarantees global visibility ordering.
  //
  // This allows arbitrary grid sizes without the cooperative launch
  // co-residency requirement (no cuLaunchCooperativeKernel needed).
  std::string asmStr =
      "{\n"
      "  membar.gl;\n"
      "  bar.sync 0;\n"
      "}\n";

  auto i32Type = builder.getI32Type();
  builder.create<mlir::triton::ElementwiseInlineAsmOp>(
      loc, /*resultTypes=*/mlir::TypeRange{i32Type},
      asmStr,
      /*constraints=*/"=r",
      /*isPure=*/false, /*pack=*/1,
      mlir::ValueRange{});
}


// ─── Gather section emitter ─────────────────────────────────────────────────
//
// Multi-dimensional gather on axis k:
//   data shape:    [D0, ..., D_{k-1}, D_k, D_{k+1}, ..., D_n]
//   indices shape: [I0, I1, ..., I_m]
//   output shape:  [D0, ..., D_{k-1}, I0, ..., I_m, D_{k+1}, ..., D_n]
//
// For each flat output element i:
//   innerDim = D_{k+1} * ... * D_n
//   numIndices = I0 * I1 * ... * I_m  (total number of index values)
//   indexSliceSize = numIndices * innerDim  (one "outer" slice of the output)
//
//   For axis=0: outerIdx=0, idxPos = i / innerDim, innerIdx = i % innerDim
//   General:    outerIdx = i / indexSliceSize
//               remaining = i % indexSliceSize
//               idxPos = remaining / innerDim
//               innerIdx = remaining % innerDim
//
//   dataOffset = outerIdx * (D_k * innerDim) + indices[idxPos] * innerDim + innerIdx

void TritonIRBuilder::emitGatherSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         mlir::Value dataPtr, mlir::Value indicesPtr,
                                         mlir::Value outputPtr, int axis,
                                         const std::vector<LongType>& dataShape,
                                         const std::vector<LongType>& indicesShape,
                                         int nElements, bool gatherNd) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  // Derive element types from actual pointer arguments
  auto idxPtrType = mlir::cast<mlir::triton::PointerType>(indicesPtr.getType());
  auto dataPtrType = mlir::cast<mlir::triton::PointerType>(dataPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto outElemType = outPtrType.getPointeeType();

  auto clampI32 = [](LongType v) -> int {
    if (v < static_cast<LongType>(std::numeric_limits<int>::min())) return std::numeric_limits<int>::min();
    if (v > static_cast<LongType>(std::numeric_limits<int>::max())) return std::numeric_limits<int>::max();
    return static_cast<int>(v);
  };

  // Compute flat output element offsets for this block
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);
  auto zeroI32 = splatConstantI32(builder, loc, i32TensorType, 0);

  if (gatherNd && !dataShape.empty() && !indicesShape.empty()) {
    // gather_nd semantics:
    // output = data[indices[..., 0:indexDepth], ...]
    int dataRank = static_cast<int>(dataShape.size());
    int indexDepth = clampI32(indicesShape.back());
    if (indexDepth <= 0) indexDepth = 1;
    if (indexDepth > dataRank) indexDepth = dataRank;

    LongType tupleCount64 = 1;
    for (size_t d = 0; d + 1 < indicesShape.size(); d++) {
      tupleCount64 *= std::max<LongType>(1, indicesShape[d]);
    }
    if (tupleCount64 <= 0) tupleCount64 = 1;
    int tupleCount = clampI32(tupleCount64);

    LongType sliceSize64 = 1;
    for (int d = indexDepth; d < dataRank; d++) {
      sliceSize64 *= std::max<LongType>(1, dataShape[d]);
    }
    if (sliceSize64 <= 0) sliceSize64 = 1;
    int sliceSize = clampI32(sliceSize64);

    std::vector<int> dataStrides(indexDepth, 1);
    LongType runningStride = 1;
    for (int d = dataRank - 1; d >= 0; d--) {
      if (d < indexDepth) dataStrides[d] = clampI32(runningStride);
      runningStride *= std::max<LongType>(1, dataShape[d]);
    }

    auto tupleCountSplat = splatConstantI32(builder, loc, i32TensorType, tupleCount);
    auto sliceSizeSplat = splatConstantI32(builder, loc, i32TensorType, sliceSize);
    auto indexDepthSplat = splatConstantI32(builder, loc, i32TensorType, indexDepth);

    auto tupleIdx = builder.create<mlir::arith::DivSIOp>(loc, offsets, sliceSizeSplat);
    auto slicePos = builder.create<mlir::arith::RemSIOp>(loc, offsets, sliceSizeSplat);

    auto tupleGe0 = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sge, tupleIdx, zeroI32);
    auto tupleLtCount = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, tupleIdx, tupleCountSplat);
    auto tupleMask = builder.create<mlir::arith::AndIOp>(loc, tupleGe0, tupleLtCount);
    auto inBounds = builder.create<mlir::arith::AndIOp>(loc, mask, tupleMask);
    auto safeTupleIdx = builder.create<mlir::arith::SelectOp>(loc, tupleMask, tupleIdx, zeroI32);

    mlir::Value baseOffset = zeroI32;
    for (int d = 0; d < indexDepth; d++) {
      auto dSplat = splatConstantI32(builder, loc, i32TensorType, d);
      auto flatIdxMul = builder.create<mlir::arith::MulIOp>(loc, safeTupleIdx, indexDepthSplat);
      auto flatIdx = builder.create<mlir::arith::AddIOp>(loc, flatIdxMul, dSplat);

      auto idxPtrTensorType = mlir::RankedTensorType::get({blockSize}, idxPtrType);
      auto splatIdxPtr = builder.create<mlir::triton::SplatOp>(loc, idxPtrTensorType, indicesPtr);
      auto idxPtrs = builder.create<mlir::triton::AddPtrOp>(loc, idxPtrTensorType, splatIdxPtr, flatIdx);
      auto rawCoord = builder.create<mlir::triton::LoadOp>(loc,
          idxPtrs.getResult(), mask.getResult(), mlir::Value(),
          mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
      auto coord = castTo(builder, loc, rawCoord, i32Type);

      int dim = clampI32(std::max<LongType>(1, dataShape[d]));
      auto dimSplat = splatConstantI32(builder, loc, i32TensorType, dim);
      auto coordNeg = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, coord, zeroI32);
      auto coordPlusDim = builder.create<mlir::arith::AddIOp>(loc, coord, dimSplat);
      auto coordNorm = builder.create<mlir::arith::SelectOp>(loc, coordNeg, coordPlusDim, coord);

      auto ge0 = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::sge, coordNorm, zeroI32);
      auto ltDim = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::slt, coordNorm, dimSplat);
      auto coordInRange = builder.create<mlir::arith::AndIOp>(loc, ge0, ltDim);
      inBounds = builder.create<mlir::arith::AndIOp>(loc, inBounds, coordInRange);

      auto safeCoord = builder.create<mlir::arith::SelectOp>(loc, coordInRange, coordNorm, zeroI32);
      auto strideSplat = splatConstantI32(builder, loc, i32TensorType, dataStrides[d]);
      auto coordContrib = builder.create<mlir::arith::MulIOp>(loc, safeCoord, strideSplat);
      baseOffset = builder.create<mlir::arith::AddIOp>(loc, baseOffset, coordContrib);
    }
    auto dataOffset = builder.create<mlir::arith::AddIOp>(loc, baseOffset, slicePos);
    auto safeDataOffset = builder.create<mlir::arith::SelectOp>(loc, inBounds, dataOffset, zeroI32);

    auto dataPtrTensorType = mlir::RankedTensorType::get({blockSize}, dataPtrType);
    auto splatDataPtr = builder.create<mlir::triton::SplatOp>(loc, dataPtrTensorType, dataPtr);
    auto dataPtrs = builder.create<mlir::triton::AddPtrOp>(loc, dataPtrTensorType, splatDataPtr, safeDataOffset);
    auto gathered = builder.create<mlir::triton::LoadOp>(loc,
        dataPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

    auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
    auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
    auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
    mlir::Value storeVal = castTo(builder, loc, gathered, outElemType);
    mlir::Value outZero = castTo(builder, loc, zeroI32, outElemType);
    storeVal = builder.create<mlir::arith::SelectOp>(loc, inBounds, storeVal, outZero);
    builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                          mlir::triton::CacheModifier::NONE,
                                          mlir::triton::EvictionPolicy::NORMAL);
    return;
  }

  int dataRank = static_cast<int>(dataShape.size());
  int axisNorm = axis;
  if (axisNorm < 0) axisNorm += dataRank;
  if (axisNorm < 0 || axisNorm >= dataRank) axisNorm = 0;

  LongType innerDim64 = 1;
  for (int d = axisNorm + 1; d < dataRank; d++) {
    innerDim64 *= std::max<LongType>(1, dataShape[d]);
  }
  if (innerDim64 <= 0) innerDim64 = 1;
  int innerDim = clampI32(innerDim64);

  LongType numIndices64 = 1;
  if (!indicesShape.empty()) {
    for (auto s : indicesShape) numIndices64 *= std::max<LongType>(1, s);
  } else {
    numIndices64 = std::max<LongType>(1, nElements);
  }
  if (numIndices64 <= 0) numIndices64 = 1;
  int numIndices = clampI32(numIndices64);

  LongType axisDim64 = (axisNorm < dataRank) ? std::max<LongType>(1, dataShape[axisNorm]) : 1;
  int axisDim = clampI32(axisDim64);

  LongType indexSliceSize64 = std::max<LongType>(1, numIndices64 * std::max<LongType>(1, innerDim64));
  int indexSliceSize = clampI32(indexSliceSize64);
  int axisStride = clampI32(std::max<LongType>(1, axisDim64 * std::max<LongType>(1, innerDim64)));

  auto innerDimSplat = splatConstantI32(builder, loc, i32TensorType, innerDim);
  auto numIndicesSplat = splatConstantI32(builder, loc, i32TensorType, numIndices);
  auto indexSliceSizeSplat = splatConstantI32(builder, loc, i32TensorType, indexSliceSize);
  auto axisDimSplat = splatConstantI32(builder, loc, i32TensorType, axisDim);
  auto axisStrideSplat = splatConstantI32(builder, loc, i32TensorType, axisStride);

  auto outerIdx = builder.create<mlir::arith::DivSIOp>(loc, offsets, indexSliceSizeSplat);
  auto remaining = builder.create<mlir::arith::RemSIOp>(loc, offsets, indexSliceSizeSplat);
  auto idxPos = builder.create<mlir::arith::DivSIOp>(loc, remaining, innerDimSplat);
  auto innerIdx = builder.create<mlir::arith::RemSIOp>(loc, remaining, innerDimSplat);
  auto outerContrib = builder.create<mlir::arith::MulIOp>(loc, outerIdx, axisStrideSplat);

  auto idxGe0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, idxPos, zeroI32);
  auto idxLtN = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, idxPos, numIndicesSplat);
  auto idxInRange = builder.create<mlir::arith::AndIOp>(loc, idxGe0, idxLtN);
  auto inBounds = builder.create<mlir::arith::AndIOp>(loc, mask, idxInRange);
  auto safeIdxPos = builder.create<mlir::arith::SelectOp>(loc, idxInRange, idxPos, zeroI32);

  auto idxPtrTensorType = mlir::RankedTensorType::get({blockSize}, idxPtrType);
  auto splatIdxPtr = builder.create<mlir::triton::SplatOp>(loc, idxPtrTensorType, indicesPtr);
  auto idxPtrs = builder.create<mlir::triton::AddPtrOp>(loc, idxPtrTensorType, splatIdxPtr, safeIdxPos);
  auto rawIndices = builder.create<mlir::triton::LoadOp>(loc,
      idxPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto gatheredIdx = castTo(builder, loc, rawIndices, i32Type);

  auto idxNeg = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, gatheredIdx, zeroI32);
  auto idxPlusDim = builder.create<mlir::arith::AddIOp>(loc, gatheredIdx, axisDimSplat);
  auto idxNorm = builder.create<mlir::arith::SelectOp>(loc, idxNeg, idxPlusDim, gatheredIdx);

  auto idxNormGe0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, idxNorm, zeroI32);
  auto idxNormLtDim = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, idxNorm, axisDimSplat);
  auto axisInRange = builder.create<mlir::arith::AndIOp>(loc, idxNormGe0, idxNormLtDim);
  inBounds = builder.create<mlir::arith::AndIOp>(loc, inBounds, axisInRange);
  auto safeAxisIdx = builder.create<mlir::arith::SelectOp>(loc, axisInRange, idxNorm, zeroI32);

  auto scaledIdx = builder.create<mlir::arith::MulIOp>(loc, safeAxisIdx, innerDimSplat);
  auto partialOffset = builder.create<mlir::arith::AddIOp>(loc, outerContrib, scaledIdx);
  auto dataOffset = builder.create<mlir::arith::AddIOp>(loc, partialOffset, innerIdx);
  auto safeDataOffset = builder.create<mlir::arith::SelectOp>(loc, inBounds, dataOffset, zeroI32);

  auto dataPtrTensorType = mlir::RankedTensorType::get({blockSize}, dataPtrType);
  auto splatDataPtr = builder.create<mlir::triton::SplatOp>(loc, dataPtrTensorType, dataPtr);
  auto dataPtrs = builder.create<mlir::triton::AddPtrOp>(loc, dataPtrTensorType, splatDataPtr, safeDataOffset);
  auto gathered = builder.create<mlir::triton::LoadOp>(loc,
      dataPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  mlir::Value storeVal = castTo(builder, loc, gathered, outElemType);
  mlir::Value outZero = castTo(builder, loc, zeroI32, outElemType);
  storeVal = builder.create<mlir::arith::SelectOp>(loc, inBounds, storeVal, outZero);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                        mlir::triton::CacheModifier::NONE,
                                        mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Concat section emitter ─────────────────────────────────────────────────
//
// Concatenates inputs along `axis` into output.  Supports N-D arrays.
//
// Strategy: For each output element at linear index `out_off`:
//   1. Unravel out_off into N-D coordinates using the output shape.
//   2. Determine which input owns this element by inspecting the concat axis coordinate.
//   3. Compute the corresponding input linear index and load from that input.
//
// This is fully unrolled across inputs (no ForOp) to avoid Triton compiler bugs.

void TritonIRBuilder::emitConcatSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         const std::vector<mlir::Value>& inputPtrs,
                                         mlir::Value outputPtr, int axis,
                                         const std::vector<std::vector<LongType>>& inputShapes,
                                         int nElements) {
  if (inputPtrs.empty() || inputShapes.empty()) return;

  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  // Derive element type from the output pointer
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto elemType = outPtrType.getPointeeType();
  auto elemTensorType = mlir::RankedTensorType::get({blockSize}, elemType);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Compute output shape from input shapes
  int ndim = static_cast<int>(inputShapes[0].size());
  // Normalize axis to positive
  int normAxis = (axis < 0) ? (ndim + axis) : axis;
  if (normAxis < 0 || normAxis >= ndim) normAxis = 0;  // fallback

  // Compute output shape
  std::vector<int> outShape(ndim);
  for (int d = 0; d < ndim; d++) outShape[d] = static_cast<int>(inputShapes[0][d]);
  for (size_t inp = 1; inp < inputShapes.size(); inp++) {
    outShape[normAxis] += static_cast<int>(inputShapes[inp][normAxis]);
  }

  // Compute output strides (C-order: stride[d] = product(outShape[d+1..ndim-1]))
  std::vector<int> outStrides(ndim, 1);
  for (int d = ndim - 2; d >= 0; d--) outStrides[d] = outStrides[d + 1] * outShape[d + 1];

  // Unravel out linear index to N-D coordinates
  // coord[d] = (offsets / outStrides[d]) % outShape[d]
  std::vector<mlir::Value> coords(ndim);
  mlir::Value rem = offsets;
  for (int d = 0; d < ndim; d++) {
    if (outStrides[d] > 1) {
      auto strideConst = builder.create<mlir::arith::ConstantIntOp>(loc, outStrides[d], 32);
      auto strideSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, strideConst);
      coords[d] = builder.create<mlir::arith::DivSIOp>(loc, rem, strideSplat);
      rem = builder.create<mlir::arith::RemSIOp>(loc, rem, strideSplat);
    } else {
      // Last dimension: coord = rem
      coords[d] = rem;
    }
    if (d < ndim - 1 && outShape[d] > 0) {
      // Clamp/mod to outShape[d] (for safety, in case blockSize padding elements are out of range)
      auto shapeConst = builder.create<mlir::arith::ConstantIntOp>(loc, outShape[d], 32);
      auto shapeSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, shapeConst);
      coords[d] = builder.create<mlir::arith::RemSIOp>(loc, coords[d], shapeSplat);
    }
  }

  // For each input, compute: is this output element from this input?
  // The concat axis coordinate tells us: cumAxisStart[i] <= coord[normAxis] < cumAxisEnd[i]
  // If yes, compute the input linear index and load from input[i].
  mlir::Value result = splatConstantF32(builder, loc, elemTensorType, 0.0f);

  int cumAxisOffset = 0;
  for (size_t inp = 0; inp < inputPtrs.size(); inp++) {
    if (inp >= inputShapes.size()) break;
    const auto& inShape = inputShapes[inp];
    int inAxisSize = static_cast<int>(inShape[normAxis]);

    // Compute input strides (C-order)
    std::vector<int> inStrides(ndim, 1);
    for (int d = ndim - 2; d >= 0; d--) inStrides[d] = inStrides[d + 1] * static_cast<int>(inShape[d + 1]);

    // Determine if this element belongs to input[inp]:
    // cumAxisOffset <= coord[normAxis] < cumAxisOffset + inAxisSize
    auto startConst = builder.create<mlir::arith::ConstantIntOp>(loc, cumAxisOffset, 32);
    auto endConst = builder.create<mlir::arith::ConstantIntOp>(loc, cumAxisOffset + inAxisSize, 32);
    auto splatStart = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, startConst);
    auto splatEnd = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, endConst);
    auto geStart = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sge, coords[normAxis], splatStart);
    auto ltEnd = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, coords[normAxis], splatEnd);
    auto inRange = builder.create<mlir::arith::AndIOp>(loc, geStart, ltEnd);
    auto loadMask = builder.create<mlir::arith::AndIOp>(loc, mask, inRange);

    // Compute the local axis coordinate within input[inp]
    // localAxisCoord = coord[normAxis] - cumAxisOffset
    mlir::Value localAxisCoord;
    if (cumAxisOffset == 0) {
      localAxisCoord = coords[normAxis];
    } else {
      auto startSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, startConst);
      localAxisCoord = builder.create<mlir::arith::SubIOp>(loc, coords[normAxis], startSplat);
    }

    // Compute input linear offset: sum over dims of coord[d] * inStrides[d]
    // Use localAxisCoord for normAxis dimension
    mlir::Value inOffset = nullptr;
    for (int d = 0; d < ndim; d++) {
      mlir::Value coordD = (d == normAxis) ? localAxisCoord : coords[d];
      if (inStrides[d] == 0) continue;
      mlir::Value contribution;
      if (inStrides[d] == 1) {
        contribution = coordD;
      } else {
        auto strideConst = builder.create<mlir::arith::ConstantIntOp>(loc, inStrides[d], 32);
        auto strideSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, strideConst);
        contribution = builder.create<mlir::arith::MulIOp>(loc, coordD, strideSplat);
      }
      if (!inOffset) {
        inOffset = contribution;
      } else {
        inOffset = builder.create<mlir::arith::AddIOp>(loc, inOffset, contribution);
      }
    }
    if (!inOffset) {
      auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
      inOffset = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zero);
    }

    // Load from input[inp] at inOffset (using loadMask so out-of-range elements load zeros)
    auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtrs[inp].getType());
    auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
    auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtrs[inp]);
    auto inPtrsOp = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, inOffset);
    auto loaded = builder.create<mlir::triton::LoadOp>(loc,
        inPtrsOp.getResult(), loadMask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

    // Cast to output element type
    mlir::Value castLoaded = castTo(builder, loc, loaded, elemType);

    // Select: if this element belongs to input[inp], use loaded; otherwise keep current result
    result = builder.create<mlir::arith::SelectOp>(loc, inRange, castLoaded, result);

    cumAxisOffset += inAxisSize;
  }

  // Store result
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, result, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Slice section emitter ──────────────────────────────────────────────────

void TritonIRBuilder::emitSliceSection(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value pid, int blockSize,
                                        mlir::Value inputPtr, mlir::Value outputPtr,
                                        const std::vector<int>& begins,
                                        const std::vector<int>& ends,
                                        const std::vector<int>& strides,
                                        const std::vector<LongType>& inputShape,
                                        int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // ND strided slice: for each output flat index, unravel to output ND coords,
  // compute input ND coords as: in_coord[d] = begin[d] + out_coord[d] * stride[d],
  // then ravel to input flat offset.
  int rank = static_cast<int>(inputShape.size());

  // Compute output shape per dimension: ceil((end[d] - begin[d]) / stride[d])
  std::vector<int> outShape(rank);
  for (int d = 0; d < rank; d++) {
    int b = (d < static_cast<int>(begins.size())) ? begins[d] : 0;
    int e = (d < static_cast<int>(ends.size())) ? ends[d] : static_cast<int>(inputShape[d]);
    int s = (d < static_cast<int>(strides.size())) ? strides[d] : 1;
    if (s == 0) s = 1;
    outShape[d] = (e - b + s - 1) / s;  // ceil division
    if (outShape[d] <= 0) outShape[d] = 1;
  }

  // Compute output strides (row-major)
  std::vector<int> outStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    outStrides[d] = outStrides[d + 1] * outShape[d + 1];

  // Compute input strides (row-major)
  std::vector<int> inStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    inStrides[d] = inStrides[d + 1] * static_cast<int>(inputShape[d + 1]);

  // Unravel output flat offset to ND coords, apply begin+stride, ravel to input offset
  mlir::Value srcOffsets = splatConstantI32(builder, loc, i32TensorType, 0);
  mlir::Value remaining = offsets;
  for (int d = 0; d < rank; d++) {
    auto oStrideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[d]);
    auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, oStrideConst);
    if (d < rank - 1)
      remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, oStrideConst);
    // input_coord = begin[d] + coord * stride[d]
    int b = (d < static_cast<int>(begins.size())) ? begins[d] : 0;
    int s = (d < static_cast<int>(strides.size())) ? strides[d] : 1;
    mlir::Value inCoord;
    if (s == 1) {
      auto beginSplat = splatConstantI32(builder, loc, i32TensorType, b);
      inCoord = builder.create<mlir::arith::AddIOp>(loc, coord, beginSplat);
    } else {
      auto strideSplat = splatConstantI32(builder, loc, i32TensorType, s);
      auto scaled = builder.create<mlir::arith::MulIOp>(loc, coord, strideSplat);
      auto beginSplat = splatConstantI32(builder, loc, i32TensorType, b);
      inCoord = builder.create<mlir::arith::AddIOp>(loc, scaled, beginSplat);
    }
    auto inStrideSplat = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
    auto contrib = builder.create<mlir::arith::MulIOp>(loc, inCoord, inStrideSplat);
    srcOffsets = builder.create<mlir::arith::AddIOp>(loc, srcOffsets, contrib);
  }

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, srcOffsets);
  auto loaded = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast loaded data to output element type if needed
  mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Tile section emitter ───────────────────────────────────────────────────

void TritonIRBuilder::emitTileSection(mlir::OpBuilder& builder, mlir::Location loc,
                                       mlir::Value pid, int blockSize,
                                       mlir::Value inputPtr, mlir::Value outputPtr,
                                       const std::vector<LongType>& inputShape,
                                       const std::vector<int>& repeats,
                                       int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Tile: for each output flat index, unravel to ND output coords,
  // apply modulo per-dimension (coord_d % inputShape[d]) to get input coords,
  // then ravel to input flat offset.
  // outputShape[d] = inputShape[d] * repeats[d]
  int rank = static_cast<int>(inputShape.size());

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // Compute output shape = inputShape * repeats
  std::vector<int> outShape(rank);
  for (int d = 0; d < rank; d++) {
    int rep = (d < static_cast<int>(repeats.size())) ? repeats[d] : 1;
    outShape[d] = static_cast<int>(inputShape[d]) * rep;
  }

  // Compute output strides (row-major)
  std::vector<int> outStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    outStrides[d] = outStrides[d + 1] * outShape[d + 1];

  // Compute input strides (row-major)
  std::vector<int> inStrides(rank, 1);
  for (int d = rank - 2; d >= 0; d--)
    inStrides[d] = inStrides[d + 1] * static_cast<int>(inputShape[d + 1]);

  // Unravel output flat offset, mod each coord by inputShape[d], ravel to input offset
  mlir::Value srcOffsets = splatConstantI32(builder, loc, i32TensorType, 0);
  mlir::Value remaining = offsets;
  for (int d = 0; d < rank; d++) {
    auto strideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[d]);
    auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, strideConst);
    if (d < rank - 1)
      remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, strideConst);
    // Wrap to input dimension
    auto inDimConst = splatConstantI32(builder, loc, i32TensorType, static_cast<int>(inputShape[d]));
    auto wrappedCoord = builder.create<mlir::arith::RemSIOp>(loc, coord, inDimConst);
    auto inStrideConst = splatConstantI32(builder, loc, i32TensorType, inStrides[d]);
    auto contrib = builder.create<mlir::arith::MulIOp>(loc, wrappedCoord, inStrideConst);
    srcOffsets = builder.create<mlir::arith::AddIOp>(loc, srcOffsets, contrib);
  }

  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, srcOffsets);
  auto loaded = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Shape manipulation section emitter ─────────────────────────────────────
// For contiguous data, reshape/flatten/expand_dims/squeeze are just copies.
// Permute/transpose requires stride recomputation.

void TritonIRBuilder::emitShapeManipulationSection(mlir::OpBuilder& builder, mlir::Location loc,
                                                    mlir::Value pid, int blockSize,
                                                    mlir::Value inputPtr, mlir::Value outputPtr,
                                                    const std::string& opName,
                                                    const std::vector<LongType>& inputShape,
                                                    const std::vector<LongType>& outputShape,
                                                    const std::vector<int>& permutation,
                                                    int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Derive pointer types from actual arguments
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  std::string opLower = opName;
  std::transform(opLower.begin(), opLower.end(), opLower.begin(), ::tolower);

  bool isPermute = (opLower == "permute" || opLower == "transpose");

  if (isPermute && !permutation.empty() && inputShape.size() >= 2) {
    // General ND permute: for each output flat offset, unravel to output coords,
    // apply inverse permutation to get input coords, ravel to input flat offset.
    // output[d0,d1,...] = input[d_perm[0], d_perm[1], ...]
    // For output flat index: unravel with outputShape strides, then
    // srcOffset = sum(coord[perm[i]] * inputStride[i])
    int rank = static_cast<int>(inputShape.size());

    // Compute output strides (row-major): stride[i] = product of outputShape[i+1..rank-1]
    std::vector<int> outStrides(rank, 1);
    for (int d = rank - 2; d >= 0; d--)
      outStrides[d] = outStrides[d + 1] * static_cast<int>(outputShape[d + 1]);

    // Compute input strides (row-major): stride[i] = product of inputShape[i+1..rank-1]
    std::vector<int> inStrides(rank, 1);
    for (int d = rank - 2; d >= 0; d--)
      inStrides[d] = inStrides[d + 1] * static_cast<int>(inputShape[d + 1]);

    // Unravel output flat offset into ND coords, then compute input flat offset
    // srcOffset = 0
    // for each output dim d: coord_d = (flat / outStride[d]) % outShape[d]
    //   srcOffset += coord_d * inStride[perm[d]]
    mlir::Value srcOffsets = splatConstantI32(builder, loc, i32TensorType, 0);
    mlir::Value remaining = offsets;
    for (int d = 0; d < rank; d++) {
      // coord_d = remaining / outStride[d]
      auto strideConst = splatConstantI32(builder, loc, i32TensorType, outStrides[d]);
      auto coord = builder.create<mlir::arith::DivSIOp>(loc, remaining, strideConst);
      if (d < rank - 1) {
        // remaining = remaining % outStride[d]  (for next dimension)
        remaining = builder.create<mlir::arith::RemSIOp>(loc, remaining, strideConst);
      }
      // coord_d in output corresponds to perm[d] in input
      int inputDim = permutation[d];
      auto inStrideConst = splatConstantI32(builder, loc, i32TensorType, inStrides[inputDim]);
      auto contrib = builder.create<mlir::arith::MulIOp>(loc, coord, inStrideConst);
      srcOffsets = builder.create<mlir::arith::AddIOp>(loc, srcOffsets, contrib);
    }

    auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
    auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, srcOffsets);
    auto loaded = builder.create<mlir::triton::LoadOp>(loc,
        inPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

    mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
    auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
    auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
    builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                           mlir::triton::CacheModifier::NONE,
                                           mlir::triton::EvictionPolicy::NORMAL);
    return;
  }

  // Default: straight copy (reshape, flatten, expand_dims, squeeze, or general permute fallback)
  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, offsets);
  auto loaded = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  mlir::Value storeVal = castTo(builder, loc, loaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Per-element matmul fallback ────────────────────────────────────────────
// When cooperative launch is infeasible, compute matmul per-element without tt.dot.

void TritonIRBuilder::emitPerElementMatmul(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value pid, int blockSize,
                                            mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr,
                                            int M, int N, int K) {
  auto i32Type = builder.getI32Type();
  auto f32Type = builder.getF32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  int totalElements = M * N;
  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, totalElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Each element of C[i,j] = sum_k A[i,k] * B[k,j]
  // Decompose offsets into row and column
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto splatNConst = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nConst);
  auto rowIndices = builder.create<mlir::arith::DivSIOp>(loc, offsets, splatNConst);
  auto colIndices = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatNConst);

  // K-loop: accumulate A[row, k] * B[k, col] for k in [0, K)
  auto accInit = splatConstantF32(builder, loc, f32TensorType, 0.0f);
  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep, mlir::ValueRange{accInit});

  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdx = forOp.getInductionVar();
  auto accIter = forOp.getBody()->getArgument(1);

  // A offset: row * K + k
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto splatKConst = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kConst);
  auto splatK = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kIdx);
  auto aOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, rowIndices, splatKConst), splatK);

  // B offset: k * N + col
  auto bOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, splatK, splatNConst), colIndices);

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto aPtrType = mlir::cast<mlir::triton::PointerType>(aPtr.getType());
  auto bPtrType = mlir::cast<mlir::triton::PointerType>(bPtr.getType());
  auto cPtrType = mlir::cast<mlir::triton::PointerType>(cPtr.getType());
  auto aPtrTensorType = mlir::RankedTensorType::get({blockSize}, aPtrType);
  auto bPtrTensorType = mlir::RankedTensorType::get({blockSize}, bPtrType);
  auto cPtrTensorType = mlir::RankedTensorType::get({blockSize}, cPtrType);

  auto splatAPtr = builder.create<mlir::triton::SplatOp>(loc, aPtrTensorType, aPtr);
  auto aPtrs = builder.create<mlir::triton::AddPtrOp>(loc, aPtrTensorType, splatAPtr, aOffset);
  auto aLoaded = builder.create<mlir::triton::LoadOp>(loc,
      aPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  auto splatBPtr = builder.create<mlir::triton::SplatOp>(loc, bPtrTensorType, bPtr);
  auto bPtrs = builder.create<mlir::triton::AddPtrOp>(loc, bPtrTensorType, splatBPtr, bOffset);
  auto bLoaded = builder.create<mlir::triton::LoadOp>(loc,
      bPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast loaded values to f32 for accumulation
  auto aVal = castTo(builder, loc, aLoaded, f32Type);
  auto bVal = castTo(builder, loc, bLoaded, f32Type);

  // acc += a * b
  auto prod = builder.create<mlir::arith::MulFOp>(loc, aVal, bVal);
  auto newAcc = builder.create<mlir::arith::AddFOp>(loc, accIter, prod);

  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});

  // After K-loop: store result (cast f32 accumulator to output type)
  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);
  mlir::Value storeVal = castTo(builder, loc, finalAcc, cPtrType.getPointeeType());

  auto splatCPtr = builder.create<mlir::triton::SplatOp>(loc, cPtrTensorType, cPtr);
  auto cPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cPtrTensorType, splatCPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, cPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

// ─── Matmul section emitter (inline in mega-kernel) ─────────────────────────
// Adapted emitMatmulKernel for use within a sectioned kernel. Uses pid to
// derive 2D tile coordinates instead of GetProgramIdOp.

void TritonIRBuilder::emitMatmulSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, const KernelSection& section,
                                         mlir::Value aPtr, mlir::Value bPtr, mlir::Value cPtr) {
  int M = section.matmulM, N = section.matmulN, K = section.matmulK;
  int blockM = section.blockM, blockN = section.blockN, blockK = section.blockK;

  if (M == 0 || N == 0 || K == 0) {
    DSP_DIAG(COMPILE, "TritonIRBuilder::emitMatmulSection: invalid dimensions M=%d N=%d K=%d", M, N, K);
    return;
  }

  // Derive 2D tile indices from 1D pid
  // gridN = ceil(N / blockN)
  auto i32Type = builder.getI32Type();
  int gridN = (N + blockN - 1) / blockN;
  int gridM = (M + blockM - 1) / blockM;

  auto gridNConst = builder.create<mlir::arith::ConstantIntOp>(loc, gridN, 32);
  auto pidM = builder.create<mlir::arith::DivSIOp>(loc, pid, gridNConst);
  auto pidN = builder.create<mlir::arith::RemSIOp>(loc, pid, gridNConst);

  // Guard: only execute if pidM < gridM && pidN < gridN
  auto gridMConst = builder.create<mlir::arith::ConstantIntOp>(loc, gridM, 32);
  auto validM = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, pidM, gridMConst);
  auto validN = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, pidN, gridNConst);
  auto valid = builder.create<mlir::arith::AndIOp>(loc, validM, validN);

  auto ifOp = builder.create<mlir::scf::IfOp>(loc, valid, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Now emit the matmul body using pidM/pidN as tile coordinates.
  // This is the same logic as emitMatmulKernel but without GetProgramIdOp.
  auto f32Type = builder.getF32Type();
  auto i1Type = builder.getI1Type();

  auto aPtrType = mlir::cast<mlir::triton::PointerType>(aPtr.getType());
  auto bPtrType = mlir::cast<mlir::triton::PointerType>(bPtr.getType());
  auto cPtrType = mlir::cast<mlir::triton::PointerType>(cPtr.getType());
  auto aElemType = aPtrType.getPointeeType();
  auto bElemType = bPtrType.getPointeeType();
  auto cElemType = cPtrType.getPointeeType();

  auto dotPrecision = mlir::triton::InputPrecision::TF32;
  if (!mlir::isa<mlir::Float32Type>(aElemType)) {
    dotPrecision = mlir::triton::InputPrecision::IEEE;
  }

  auto blockMConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockM, 32);
  auto blockNConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockN, 32);
  auto mOffset = builder.create<mlir::arith::MulIOp>(loc, pidM, blockMConst);
  auto nOffset = builder.create<mlir::arith::MulIOp>(loc, pidN, blockNConst);

  auto i32BmType = mlir::RankedTensorType::get({blockM}, i32Type);
  auto i32BnType = mlir::RankedTensorType::get({blockN}, i32Type);
  auto i32BkType = mlir::RankedTensorType::get({blockK}, i32Type);

  auto rangeM = builder.create<mlir::triton::MakeRangeOp>(loc, i32BmType, 0, blockM);
  auto rangeN = builder.create<mlir::triton::MakeRangeOp>(loc, i32BnType, 0, blockN);
  auto rangeK = builder.create<mlir::triton::MakeRangeOp>(loc, i32BkType, 0, blockK);

  auto splatMOffset = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mOffset);
  auto mIndices = builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM);
  auto splatNOffset = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nOffset);
  auto nIndices = builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN);

  auto accType = mlir::RankedTensorType::get({blockM, blockN}, f32Type);
  auto zeroAttr = builder.getFloatAttr(f32Type, 0.0);
  auto zeroScalar = builder.create<mlir::arith::ConstantOp>(loc, f32Type, zeroAttr);
  auto accInit = builder.create<mlir::triton::SplatOp>(loc, accType, zeroScalar);

  auto kStart = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto kEnd = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto kStep = builder.create<mlir::arith::ConstantIntOp>(loc, blockK, 32);

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, kStart, kEnd, kStep, mlir::ValueRange{accInit});

  builder.setInsertionPointToStart(forOp.getBody());
  auto kIdxI32 = forOp.getInductionVar();
  auto accIter = forOp.getBody()->getArgument(1);

  auto splatKOffset = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kIdxI32);
  auto kIndices = builder.create<mlir::arith::AddIOp>(loc, splatKOffset, rangeK);

  // Load A tile [BM, BK]
  auto kConst = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto kExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 0);

  auto i32BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i32Type);
  auto kSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), kConst);
  auto mTimesK = builder.create<mlir::arith::MulIOp>(loc, mExpanded, kSplat);
  auto mTimesKBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, mTimesK);
  auto kBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBkType, kExpanded);
  auto aOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesKBroadcast, kBroadcast);

  auto aPtrTensorType = mlir::RankedTensorType::get({blockM, blockK},
      mlir::triton::PointerType::get(aElemType, 1));
  auto aSplat = builder.create<mlir::triton::SplatOp>(loc, aPtrTensorType, aPtr);
  auto aPtrs = builder.create<mlir::triton::AddPtrOp>(loc, aPtrTensorType, aSplat, aOffsets);

  auto mConst = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto kConst2 = builder.create<mlir::arith::ConstantIntOp>(loc, K, 32);
  auto mConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConst);
  auto kConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BkType, kConst2);
  auto mMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatMOffset, rangeM), mConstSplat);
  auto kMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      kIndices, kConstSplat);
  auto i1BmBkType = mlir::RankedTensorType::get({blockM, blockK}, i1Type);
  auto mMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1D, 1);
  auto kMaskExp = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 0);
  auto mMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, mMaskExp);
  auto kMask2D = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBkType, kMaskExp);
  auto aMask = builder.create<mlir::arith::AndIOp>(loc, mMask2D, kMask2D);

  auto aLoaded = builder.create<mlir::triton::LoadOp>(loc,
      aPtrs.getResult(), aMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Load B tile [BK, BN]
  auto nConst = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto kExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, kIndices, 1);
  auto nExpandedB = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto i32BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i32Type);
  auto nSplat = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockK, 1}, i32Type), nConst);
  auto kTimesN = builder.create<mlir::arith::MulIOp>(loc, kExpandedB, nSplat);
  auto kTimesNBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, kTimesN);
  auto nBroadcastB = builder.create<mlir::triton::BroadcastOp>(loc, i32BkBnType, nExpandedB);
  auto bOffsets = builder.create<mlir::arith::AddIOp>(loc, kTimesNBroadcast, nBroadcastB);

  auto bPtrTensorType = mlir::RankedTensorType::get({blockK, blockN},
      mlir::triton::PointerType::get(bElemType, 1));
  auto bSplat = builder.create<mlir::triton::SplatOp>(loc, bPtrTensorType, bPtr);
  auto bPtrs = builder.create<mlir::triton::AddPtrOp>(loc, bPtrTensorType, bSplat, bOffsets);

  auto nConstSplat = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConst);
  auto nMask1D = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt,
      builder.create<mlir::arith::AddIOp>(loc, splatNOffset, rangeN), nConstSplat);
  auto i1BkBnType = mlir::RankedTensorType::get({blockK, blockN}, i1Type);
  auto kMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, kMask1D, 1);
  auto nMaskExpB = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1D, 0);
  auto kMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, kMaskExpB);
  auto nMask2DB = builder.create<mlir::triton::BroadcastOp>(loc, i1BkBnType, nMaskExpB);
  auto bMask = builder.create<mlir::arith::AndIOp>(loc, kMask2DB, nMask2DB);

  auto bLoaded = builder.create<mlir::triton::LoadOp>(loc,
      bPtrs.getResult(), bMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  auto dotResult = builder.create<mlir::triton::DotOp>(
      loc, accType, aLoaded, bLoaded, accIter,
      dotPrecision, 0);

  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{dotResult});

  builder.setInsertionPointAfter(forOp);
  auto finalAcc = forOp.getResult(0);

  // Cast and store C
  mlir::Value storeVal = finalAcc;
  if (cElemType != f32Type) {
    auto cTileType = mlir::RankedTensorType::get({blockM, blockN}, cElemType);
    if (mlir::isa<mlir::FloatType>(cElemType)) {
      storeVal = builder.create<mlir::arith::TruncFOp>(loc, cTileType, finalAcc);
    }
  }

  auto i32BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i32Type);
  auto mExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, mIndices, 1);
  auto nExpandedC = builder.create<mlir::triton::ExpandDimsOp>(loc, nIndices, 0);
  auto nSplatC = builder.create<mlir::triton::SplatOp>(loc, mlir::RankedTensorType::get({blockM, 1}, i32Type), nConst);
  auto mTimesNC = builder.create<mlir::arith::MulIOp>(loc, mExpandedC, nSplatC);
  auto mTimesNCBroadcast = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, mTimesNC);
  auto nBroadcastC = builder.create<mlir::triton::BroadcastOp>(loc, i32BmBnType, nExpandedC);
  auto cOffsets = builder.create<mlir::arith::AddIOp>(loc, mTimesNCBroadcast, nBroadcastC);

  auto cPtrTensorType = mlir::RankedTensorType::get({blockM, blockN},
      mlir::triton::PointerType::get(cElemType, 1));
  auto cSplat = builder.create<mlir::triton::SplatOp>(loc, cPtrTensorType, cPtr);
  auto cPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cPtrTensorType, cSplat, cOffsets);

  auto mConstC = builder.create<mlir::arith::ConstantIntOp>(loc, M, 32);
  auto nConstC = builder.create<mlir::arith::ConstantIntOp>(loc, N, 32);
  auto mConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BmType, mConstC);
  auto nConstSplatC = builder.create<mlir::triton::SplatOp>(loc, i32BnType, nConstC);
  auto mMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, mIndices, mConstSplatC);
  auto nMask1DC = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, nIndices, nConstSplatC);
  auto i1BmBnType = mlir::RankedTensorType::get({blockM, blockN}, i1Type);
  auto mMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, mMask1DC, 1);
  auto nMaskExpC = builder.create<mlir::triton::ExpandDimsOp>(loc, nMask1DC, 0);
  auto mMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, mMaskExpC);
  auto nMask2DC = builder.create<mlir::triton::BroadcastOp>(loc, i1BmBnType, nMaskExpC);
  auto cMask = builder.create<mlir::arith::AndIOp>(loc, mMask2DC, nMask2DC);

  builder.create<mlir::triton::StoreOp>(loc, cPtrs, storeVal, cMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  // Move insertion point after the if block
  builder.setInsertionPointAfter(ifOp);
}

// ─── Diagnostics: section breakdown dump ────────────────────────────────────

void TritonIRBuilder::dumpSectionBreakdown(const std::vector<KernelSection>& sections,
                                            int startSlot, int endSlot,
                                            int maxSectionGrid, bool cooperativeLaunch) {
  auto& env = Environment::getInstance();
  const bool dumpEnabled = env.tritonDumpSections() || env.tritonVerbose();
  if (!dumpEnabled) return;

  DSP_DIAG(SEGMENT, "=== Triton Kernel: seg[%d-%d] ===", startSlot, endSlot);
  for (size_t i = 0; i < sections.size(); i++) {
    auto& sec = sections[i];
    const char* typeName = "UNKNOWN";
    switch (sec.type) {
      case KernelSectionType::ELEMENTWISE:        typeName = "ELEMENTWISE"; break;
      case KernelSectionType::MATMUL:             typeName = "MATMUL"; break;
      case KernelSectionType::FUSED_ATTENTION:    typeName = "ATTENTION"; break;
      case KernelSectionType::REDUCTION:          typeName = "REDUCTION"; break;
      case KernelSectionType::NORMALIZATION:      typeName = "NORMALIZATION"; break;
      case KernelSectionType::GATHER:             typeName = "GATHER"; break;
      case KernelSectionType::GATHER_ND:          typeName = "GATHER_ND"; break;
      case KernelSectionType::CONCAT:             typeName = "CONCAT"; break;
      case KernelSectionType::SPLIT:              typeName = "SPLIT"; break;
      case KernelSectionType::SPLIT_V:            typeName = "SPLIT_V"; break;
      case KernelSectionType::STACK:              typeName = "STACK"; break;
      case KernelSectionType::STRIDED_SLICE:      typeName = "STRIDED_SLICE"; break;
      case KernelSectionType::TILE:               typeName = "TILE"; break;
      case KernelSectionType::SCATTER_ND:         typeName = "SCATTER_ND"; break;
      case KernelSectionType::SCATTER_ND_UPDATE:  typeName = "SCATTER_ND_UPDATE"; break;
      case KernelSectionType::SHAPE_MANIPULATION: typeName = "SHAPE_MANIP"; break;
      case KernelSectionType::CONSTANT_GENERATION:typeName = "CONST_GEN"; break;
      case KernelSectionType::CONVOLUTION:        typeName = "CONVOLUTION"; break;
      case KernelSectionType::IDENTITY:           typeName = "IDENTITY"; break;
    }
    if (sec.type == KernelSectionType::MATMUL) {
      DSP_DIAG(SEGMENT, "Section %d: %-15s slots[%d-%d]  %d ops, grid=%d, M=%d N=%d K=%d",
               (int)i, typeName, sec.startSlot, sec.endSlot, sec.numOps, sec.gridRequirement,
               sec.matmulM, sec.matmulN, sec.matmulK);
    } else if (sec.type == KernelSectionType::FUSED_ATTENTION) {
      DSP_DIAG(SEGMENT, "Section %d: %-15s slots[%d-%d]  %d ops, grid=%d, B=%d H=%d seqQ=%d seqKV=%d headDim=%d",
               (int)i, typeName, sec.startSlot, sec.endSlot, sec.numOps, sec.gridRequirement,
               sec.batchSize, sec.numHeads, sec.seqQ, sec.seqK, sec.headDim);
    } else {
      DSP_DIAG(SEGMENT, "Section %d: %-15s slots[%d-%d]  %d ops, grid=%d",
               (int)i, typeName, sec.startSlot, sec.endSlot, sec.numOps, sec.gridRequirement);
    }
  }
  DSP_DIAG(SEGMENT, "Max section grid: %d, Cooperative launch: %s",
           maxSectionGrid, cooperativeLaunch ? "YES" : "NO");
}

// ─── Diagnostics: arg mapping dump ──────────────────────────────────────────

void TritonIRBuilder::dumpArgMapping(const std::vector<TritonKernelArg>& args,
                                      int startSlot, int endSlot,
                                      int eliminatedCount) {
  auto& env = Environment::getInstance();
  const bool dumpEnabled = env.tritonDumpArgs() || env.tritonVerbose();
  if (!dumpEnabled) return;

  DSP_DIAG(SEGMENT, "=== Arg Mapping: seg[%d-%d] ===", startSlot, endSlot);
  for (size_t i = 0; i < args.size(); i++) {
    auto& arg = args[i];
    std::ostringstream shapeStr;
    for (size_t d = 0; d < arg.shape.size(); d++) {
      if (d > 0) shapeStr << ",";
      shapeStr << (long long)arg.shape[d];
    }
    DSP_DIAG(SEGMENT, "Arg %3d: slot %4d %s dtype=%d shape=[%s]",
             (int)i, arg.slotIndex, arg.isOutput ? "OUT" : "IN ", (int)arg.dtype,
             shapeStr.str().c_str());
  }
  DSP_DIAG(SEGMENT, "Eliminated: %d internal intermediates, Total args: %d%s",
           eliminatedCount, (int)args.size(),
           args.size() > 200 ? " (indirect)" : " (direct)");
}

// ─── Section emitter implementations ────────────────────────────────────────

void TritonIRBuilder::emitAttentionSection(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value pid, const KernelSection& section,
                                            mlir::Value qPtr, mlir::Value kPtr,
                                            mlir::Value vPtr, mlir::Value outPtr) {
  (void)pid;
  // Delegate to the existing emitFusedAttentionKernel, which creates its own
  // GetProgramIdOp. For the sectioned kernel, this is called within an scf.if
  // guard so only blocks in the attention section's pid range execute it.
  // Note: emitFusedAttentionKernel uses its own pid0/pid1 from GetProgramIdOp.
  // In the cooperative kernel, we remap pid to the attention section's range.
  auto attnTile = chooseFusedAttentionTileConfig(
      std::max(1, section.batchSize),
      std::max(1, section.numHeads),
      std::max(1, section.seqQ),
      std::max(1, section.seqK),
      std::max(1, section.headDim));
  int blockM = attnTile.blockM;
  int blockN = attnTile.blockN;
  emitFusedAttentionKernel(builder, loc, qPtr, kPtr, vPtr, outPtr,
                            section.batchSize, section.numHeads,
                            section.numKvHeads > 0 ? section.numKvHeads : section.numHeads,
                            section.seqQ, section.seqK,
                            section.headDim, section.attentionScale,
                            blockM, blockN,
                            section.attnQIsBSHD, section.attnKIsBSHD,
                            mlir::Value(), std::vector<LongType>(),
                            mlir::Value(), mlir::Value(), 0, 0);
}

void TritonIRBuilder::emitSplitSection(mlir::OpBuilder& builder, mlir::Location loc,
                                        mlir::Value pid, int blockSize,
                                        mlir::Value inputPtr,
                                        const std::vector<mlir::Value>& outputPtrs,
                                        int axis, int numSplits,
                                        const std::vector<LongType>& inputShape,
                                        int nElements) {
  // Split input along `axis` into numSplits equal chunks.
  // Each chunk s gets a slice: begin[axis]=s*chunkAlongAxis, end[axis]=(s+1)*chunkAlongAxis.
  if (numSplits <= 0 || inputShape.empty()) return;
  int rank = static_cast<int>(inputShape.size());
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank) axis = 0;

  int axisSize = static_cast<int>(inputShape[axis]);
  int chunkAlongAxis = axisSize / numSplits;
  if (chunkAlongAxis <= 0) chunkAlongAxis = 1;

  // Compute per-chunk output size
  int chunkTotalElements = 1;
  for (int d = 0; d < rank; d++) {
    chunkTotalElements *= (d == axis) ? chunkAlongAxis : static_cast<int>(inputShape[d]);
  }

  for (int s = 0; s < numSplits && s < static_cast<int>(outputPtrs.size()); s++) {
    // Build begins/ends: only the axis dimension differs
    std::vector<int> begins(rank, 0);
    std::vector<int> ends;
    for (int d = 0; d < rank; d++) ends.push_back(static_cast<int>(inputShape[d]));
    begins[axis] = s * chunkAlongAxis;
    ends[axis] = (s + 1) * chunkAlongAxis;
    std::vector<int> strides(rank, 1);
    emitSliceSection(builder, loc, pid, blockSize, inputPtr, outputPtrs[s],
                     begins, ends, strides, inputShape, chunkTotalElements);
  }
}

// ─── RoPE section emitter ────────────────────────────────────────────────────
//
// Rotary Position Embedding: paired elementwise rotation using precomputed cos/sin.
// Input shape: [B, S, H, D] (or any shape where the last dim is headDim).
// Cos/Sin shape: [B, S, D] or [1, S, D] or [S, D] — no head dimension.
//
// For each pair (i, i+halfDim) within a head:
//   output[i]          = x[i] * cos[i%halfDim] - x[i+halfDim] * sin[i%halfDim]
//   output[i+halfDim]  = x[i+halfDim] * cos[i%halfDim] + x[i] * sin[i%halfDim]
//
void TritonIRBuilder::emitRoPESection(mlir::OpBuilder& builder, mlir::Location loc,
                                       mlir::Value pid, int blockSize,
                                       mlir::Value inputPtr, mlir::Value cosPtr, mlir::Value sinPtr,
                                       mlir::Value outputPtr,
                                       const std::vector<LongType>& inputShape,
                                       const std::vector<LongType>& cosShape,
                                       int ropeType, int nElements) {
  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32Type = builder.getF32Type();
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);

  // Derive element types from pointer args
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto cosPtrType = mlir::cast<mlir::triton::PointerType>(cosPtr.getType());
  auto sinPtrType = mlir::cast<mlir::triton::PointerType>(sinPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto outElemType = outPtrType.getPointeeType();

  // Extract dimensions from input shape [B, S, H, D]
  int inputRank = static_cast<int>(inputShape.size());
  int headDim = (inputRank > 0) ? static_cast<int>(inputShape[inputRank - 1]) : 64;
  int halfDim = headDim / 2;
  int numHeads = (inputRank >= 3) ? static_cast<int>(inputShape[inputRank - 2]) : 1;

  // Compute cos/sin stride for the "no head" dimension mapping
  // Cos shape is typically [B, S, D] while input is [B, S, H, D]
  // cosStride = product of dims after seqLen in cos = D (last dim of cos)
  int cosDim = 1;
  if (!cosShape.empty()) {
    cosDim = static_cast<int>(cosShape.back());
  }

  // Compute flat offsets: pid * blockSize + make_range(0, blockSize)
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  // Bounds mask
  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Compute d = offsets % headDim (position within head)
  auto headDimConst = splatConstantI32(builder, loc, i32TensorType, headDim);
  auto d = builder.create<mlir::arith::RemSIOp>(loc, offsets, headDimConst);

  // Compute halfDimConst and isFirstHalf = (d < halfDim)
  auto halfDimConst = splatConstantI32(builder, loc, i32TensorType, halfDim);
  auto isFirstHalf = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, d, halfDimConst);

  // Compute paired offset: firstHalf -> offset + halfDim, secondHalf -> offset - halfDim
  auto halfDimSplat = splatConstantI32(builder, loc, i32TensorType, halfDim);
  auto offsetPlusHalf = builder.create<mlir::arith::AddIOp>(loc, offsets, halfDimSplat);
  auto offsetMinusHalf = builder.create<mlir::arith::SubIOp>(loc, offsets, halfDimSplat);
  auto pairedOffsets = builder.create<mlir::arith::SelectOp>(
      loc, isFirstHalf, offsetPlusHalf, offsetMinusHalf);

  // Compute pairIdx = d % halfDim (index into cos/sin within the pair)
  auto pairIdx = builder.create<mlir::arith::RemSIOp>(loc, d, halfDimConst);

  // Compute cos/sin flat index.
  // Input is [B, S, H, D], cos is [B, S, D] (no head dimension).
  // For input flat position `pos`:
  //   head_stride = headDim
  //   seq_head_stride = numHeads * headDim
  //   seq_idx = (pos / seq_head_stride) (position in B*S space)
  //   cos_flat = seq_idx * cosDim + pairIdx
  auto seqHeadStride = splatConstantI32(builder, loc, i32TensorType, numHeads * headDim);
  auto seqIdx = builder.create<mlir::arith::DivSIOp>(loc, offsets, seqHeadStride);
  auto cosDimConst = splatConstantI32(builder, loc, i32TensorType, cosDim);
  auto cosBase = builder.create<mlir::arith::MulIOp>(loc, seqIdx, cosDimConst);
  auto cosIdx = builder.create<mlir::arith::AddIOp>(loc, cosBase, pairIdx);

  // Load input[offsets] and input[pairedOffsets]
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, offsets);
  auto inPairedPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, pairedOffsets);

  auto xVal = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto xPaired = builder.create<mlir::triton::LoadOp>(loc,
      inPairedPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Load cos[cosIdx] and sin[cosIdx]
  auto cosPtrTensorType = mlir::RankedTensorType::get({blockSize}, cosPtrType);
  auto sinPtrTensorType = mlir::RankedTensorType::get({blockSize}, sinPtrType);
  auto splatCosPtr = builder.create<mlir::triton::SplatOp>(loc, cosPtrTensorType, cosPtr);
  auto splatSinPtr = builder.create<mlir::triton::SplatOp>(loc, sinPtrTensorType, sinPtr);
  auto cosPtrs = builder.create<mlir::triton::AddPtrOp>(loc, cosPtrTensorType, splatCosPtr, cosIdx);
  auto sinPtrs = builder.create<mlir::triton::AddPtrOp>(loc, sinPtrTensorType, splatSinPtr, cosIdx);

  auto cosVal = builder.create<mlir::triton::LoadOp>(loc,
      cosPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  auto sinVal = builder.create<mlir::triton::LoadOp>(loc,
      sinPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Promote all to f32 for computation
  auto x = castTo(builder, loc, xVal, f32Type);
  auto xp = castTo(builder, loc, xPaired, f32Type);
  auto cv = castTo(builder, loc, cosVal, f32Type);
  auto sv = castTo(builder, loc, sinVal, f32Type);

  // Compute rotation:
  //   firstHalf:  output = x * cos - xPaired * sin
  //   secondHalf: output = xPaired * sin + x * cos
  // Unified: output = x * cos + signFactor * xPaired * sin
  //   where signFactor = isFirstHalf ? -1.0 : 1.0
  auto negOne = splatConstantF32(builder, loc, f32TensorType, -1.0f);
  auto posOne = splatConstantF32(builder, loc, f32TensorType, 1.0f);
  auto signFactor = builder.create<mlir::arith::SelectOp>(loc, isFirstHalf, negOne, posOne);

  auto xTimesCos = builder.create<mlir::arith::MulFOp>(loc, x, cv);
  auto xpTimesSin = builder.create<mlir::arith::MulFOp>(loc, xp, sv);
  auto signedXpSin = builder.create<mlir::arith::MulFOp>(loc, signFactor, xpTimesSin);
  auto result = builder.create<mlir::arith::AddFOp>(loc, xTimesCos, signedXpSin);

  // Cast back to output element type and store
  mlir::Value storeVal = castTo(builder, loc, result, outElemType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                        mlir::triton::CacheModifier::NONE,
                                        mlir::triton::EvictionPolicy::NORMAL);
}

// ─── RoPE SSA emitter ────────────────────────────────────────────────────────
//
// Register-level RoPE using tt.reshape/tt.trans/tt.split/tt.join.
// Eliminates the store→reload round-trip by operating directly on the SSA tensor.
//
// Algorithm:
//   1D SSA: tensor<blockSize x f32>
//     → tt.reshape → tensor<numHeadsInBlock x 2 x halfDim>
//     → tt.trans {0,2,1} → tensor<numHeadsInBlock x halfDim x 2>
//     → tt.split → x_first, x_second : tensor<numHeadsInBlock x halfDim>
//     → rotation: out_first = x1*cos - x2*sin, out_second = x2*cos + x1*sin
//     → tt.join → tt.trans {0,2,1} → tt.reshape → tensor<blockSize x f32>
//
// Cos/sin are loaded from pointers (different shape, unavoidable): 2 memory ops.
// Pointer-based emitter uses 7 memory ops. This saves 5 memory ops per element.
//
// Preconditions (checked by caller):
//   - blockSize % headDim == 0
//   - blockSize <= numHeads * headDim (all elements in block from same seq position)
//
mlir::Value TritonIRBuilder::emitRoPESSA(mlir::OpBuilder& builder, mlir::Location loc,
                                          mlir::Value inputSSA,
                                          mlir::Value cosPtr, mlir::Value sinPtr,
                                          mlir::Value pid, int blockSize,
                                          int headDim, int numHeads,
                                          const std::vector<LongType>& cosShape,
                                          int nElements) {
  auto f32Type = builder.getF32Type();
  auto i32Type = builder.getI32Type();
  int halfDim = headDim / 2;
  int numHeadsInBlock = blockSize / headDim;

  // ── 1. Cast input to f32 for computation ──
  auto inputF32 = castTo(builder, loc, inputSSA, f32Type);

  // ── 2. Reshape: tensor<blockSize x f32> → tensor<numHeadsInBlock x 2 x halfDim x f32> ──
  auto reshapedType = mlir::RankedTensorType::get(
      {(int64_t)numHeadsInBlock, 2, (int64_t)halfDim}, f32Type);
  auto reshaped = builder.create<mlir::triton::ReshapeOp>(
      loc, reshapedType, inputF32, /*allowReorder=*/false);

  // ── 3. Transpose: [N, 2, H] → [N, H, 2] (move halves-dim to last for split) ──
  auto transOrder = builder.getDenseI32ArrayAttr({0, 2, 1});
  auto transposed = builder.create<mlir::triton::TransOp>(loc, reshaped, transOrder);

  // ── 4. Split along last dim (size 2) → x_first, x_second : tensor<N x halfDim> ──
  auto splitOp = builder.create<mlir::triton::SplitOp>(loc, transposed);
  auto xFirst = splitOp.getResult(0);   // tensor<numHeadsInBlock x halfDim x f32>
  auto xSecond = splitOp.getResult(1);  // tensor<numHeadsInBlock x halfDim x f32>

  // ── 5. Load cos/sin from pointers ──
  // Compute seq index for this block: seqIdx = (pid * blockSize) / (numHeads * headDim)
  int seqHeadStride = numHeads * headDim;
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto pidTimesBlock = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto seqHeadStrideConst = builder.create<mlir::arith::ConstantIntOp>(loc, seqHeadStride, 32);
  auto seqIdxScalar = builder.create<mlir::arith::DivSIOp>(loc, pidTimesBlock, seqHeadStrideConst);

  // cosDim = last dim of cosShape (typically headDim)
  int cosDim = cosShape.empty() ? halfDim : static_cast<int>(cosShape.back());
  auto cosDimConst = builder.create<mlir::arith::ConstantIntOp>(loc, cosDim, 32);
  auto cosBaseOffset = builder.create<mlir::arith::MulIOp>(loc, seqIdxScalar, cosDimConst);

  // Create index range [0, halfDim) + cosBaseOffset
  auto halfDimI32Type = mlir::RankedTensorType::get({(int64_t)halfDim}, i32Type);
  auto halfRange = builder.create<mlir::triton::MakeRangeOp>(loc, halfDimI32Type, 0, halfDim);
  auto baseSplat = builder.create<mlir::triton::SplatOp>(loc, halfDimI32Type, cosBaseOffset);
  auto cosIndices = builder.create<mlir::arith::AddIOp>(loc, baseSplat, halfRange);

  // Create all-true mask for cos/sin loads (in-bounds guaranteed)
  auto i1Type = builder.getI1Type();
  auto halfDimI1Type = mlir::RankedTensorType::get({(int64_t)halfDim}, i1Type);
  auto trueConst = builder.create<mlir::arith::ConstantOp>(loc, builder.getIntegerAttr(i1Type, 1));
  auto cosMask = builder.create<mlir::triton::SplatOp>(loc, halfDimI1Type, trueConst);

  // Load cos: tensor<halfDim x cosElemType>
  auto cosPtrType = mlir::cast<mlir::triton::PointerType>(cosPtr.getType());
  auto halfDimCosPtrType = mlir::RankedTensorType::get({(int64_t)halfDim}, cosPtrType);
  auto cosPtrSplat = builder.create<mlir::triton::SplatOp>(loc, halfDimCosPtrType, cosPtr);
  auto cosLoadPtrs = builder.create<mlir::triton::AddPtrOp>(loc, halfDimCosPtrType, cosPtrSplat, cosIndices);
  auto cosLoaded1D = builder.create<mlir::triton::LoadOp>(loc,
      cosLoadPtrs.getResult(), cosMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Load sin: tensor<halfDim x sinElemType>
  auto sinPtrType = mlir::cast<mlir::triton::PointerType>(sinPtr.getType());
  auto halfDimSinPtrType = mlir::RankedTensorType::get({(int64_t)halfDim}, sinPtrType);
  auto sinPtrSplat = builder.create<mlir::triton::SplatOp>(loc, halfDimSinPtrType, sinPtr);
  auto sinLoadPtrs = builder.create<mlir::triton::AddPtrOp>(loc, halfDimSinPtrType, sinPtrSplat, cosIndices);
  auto sinLoaded1D = builder.create<mlir::triton::LoadOp>(loc,
      sinLoadPtrs.getResult(), cosMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast cos/sin to f32
  auto cosF32 = castTo(builder, loc, cosLoaded1D, f32Type);
  auto sinF32 = castTo(builder, loc, sinLoaded1D, f32Type);

  // Broadcast cos/sin from tensor<halfDim> to tensor<numHeadsInBlock x halfDim>
  auto nhHdF32Type = mlir::RankedTensorType::get(
      {(int64_t)numHeadsInBlock, (int64_t)halfDim}, f32Type);
  auto cosExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, cosF32, 0);
  auto sinExpanded = builder.create<mlir::triton::ExpandDimsOp>(loc, sinF32, 0);
  auto cos2D = builder.create<mlir::triton::BroadcastOp>(loc, nhHdF32Type, cosExpanded);
  auto sin2D = builder.create<mlir::triton::BroadcastOp>(loc, nhHdF32Type, sinExpanded);

  // ── 6. Rotation computation ──
  // out_first  = x_first * cos - x_second * sin
  // out_second = x_second * cos + x_first * sin
  auto firstTimesCos = builder.create<mlir::arith::MulFOp>(loc, xFirst, cos2D);
  auto secondTimesSin = builder.create<mlir::arith::MulFOp>(loc, xSecond, sin2D);
  auto outFirst = builder.create<mlir::arith::SubFOp>(loc, firstTimesCos, secondTimesSin);

  auto secondTimesCos = builder.create<mlir::arith::MulFOp>(loc, xSecond, cos2D);
  auto firstTimesSin = builder.create<mlir::arith::MulFOp>(loc, xFirst, sin2D);
  auto outSecond = builder.create<mlir::arith::AddFOp>(loc, secondTimesCos, firstTimesSin);

  // ── 7. Reassemble: join → trans → reshape back to 1D ──
  // tt.join: tensor<N x halfDim> × 2 → tensor<N x halfDim x 2>
  auto joined = builder.create<mlir::triton::JoinOp>(loc, outFirst, outSecond);

  // tt.trans {0,2,1}: tensor<N x halfDim x 2> → tensor<N x 2 x halfDim>
  auto invTransOrder = builder.getDenseI32ArrayAttr({0, 2, 1});
  auto invTransposed = builder.create<mlir::triton::TransOp>(loc, joined, invTransOrder);

  // tt.reshape: tensor<N x 2 x halfDim> → tensor<blockSize>
  auto result1DType = mlir::RankedTensorType::get({(int64_t)blockSize}, f32Type);
  auto result1D = builder.create<mlir::triton::ReshapeOp>(
      loc, result1DType, invTransposed, /*allowReorder=*/false);

  // ── 8. Cast back to original element type ──
  auto inputElemType = mlir::cast<mlir::RankedTensorType>(inputSSA.getType()).getElementType();
  return castTo(builder, loc, result1D, inputElemType);
}

void TritonIRBuilder::emitScatterNdSection(mlir::OpBuilder& builder, mlir::Location loc,
                                            mlir::Value pid, int blockSize,
                                            mlir::Value dataPtr, mlir::Value indicesPtr,
                                            mlir::Value updatesPtr, mlir::Value outputPtr,
                                            const std::vector<LongType>& dataShape,
                                            int nElements) {
  // ScatterNd: copy data to output, then scatter updates at indexed positions.
  //
  // data:    [D0, D1, ..., Dn]       — base tensor (same shape as output)
  // indices: [numUpdates, indexDepth] — scatter positions
  // updates: [numUpdates, S0, S1, ...] — values to scatter (S = slice shape)
  // output:  [D0, D1, ..., Dn]       — result
  //
  // Phase 1: Copy all data[i] -> output[i]  (nElements = output length)
  // Phase 2: For each update element j (0..totalUpdateElems-1):
  //   updateIdx = j / sliceSize
  //   slicePos  = j % sliceSize
  //   flatIdx   = indices[updateIdx * indexDepth + 0] * stride0 + ... + slicePos
  //   output[flatIdx] = updates[j]
  //
  // For the simple 1D index case (indexDepth=1):
  //   flatIdx = indices[updateIdx] * sliceSize + slicePos

  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);

  auto dataPtrType = mlir::cast<mlir::triton::PointerType>(dataPtr.getType());
  auto idxPtrType = mlir::cast<mlir::triton::PointerType>(indicesPtr.getType());
  auto updPtrType = mlir::cast<mlir::triton::PointerType>(updatesPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto dataPtrTensorType = mlir::RankedTensorType::get({blockSize}, dataPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);
  auto idxPtrTensorType = mlir::RankedTensorType::get({blockSize}, idxPtrType);
  auto updPtrTensorType = mlir::RankedTensorType::get({blockSize}, updPtrType);

  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  // Phase 1: Copy data to output (nElements = output length)
  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  auto splatDataPtr = builder.create<mlir::triton::SplatOp>(loc, dataPtrTensorType, dataPtr);
  auto dataPtrs = builder.create<mlir::triton::AddPtrOp>(loc, dataPtrTensorType, splatDataPtr, offsets);
  auto dataLoaded = builder.create<mlir::triton::LoadOp>(loc,
      dataPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  mlir::Value dataStoreVal = castTo(builder, loc, dataLoaded, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, dataStoreVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  // Phase 2: Scatter updates
  // Compute indexDepth (last dim of indices shape) and sliceSize (product of data dims after indexDepth)
  // For simplicity, assume indexDepth=1 for now (covers most common cases).
  // sliceSize = product of dataShape[1:]
  LongType sliceSize = 1;
  for (size_t d = 1; d < dataShape.size(); d++) sliceSize *= dataShape[d];
  if (sliceSize <= 0) sliceSize = 1;

  // totalUpdateElems = numUpdates * sliceSize (this is updates.lengthOf())
  // numUpdates is unknown at compile time but we can derive it:
  // numUpdates * sliceSize must equal updates.length, which we get from the mask.
  // For the kernel, we iterate over update elements using the same grid as output,
  // but with a separate mask based on the total update element count.
  // We don't have updates.length directly, so we express the scatter in terms of
  // the output grid: for each flat update element j, compute output position.

  // The key insight: we use a SECOND pass over the SAME grid but with a different mask.
  // totalUpdateElems = (nElements is output, but updates is typically smaller)
  // We pass the total update elements as a separate constant.
  // Since we don't have it at IR build time, derive from the grid:
  // Actually we DO know it at compile time from the arrays. But the function signature
  // only receives dataShape and nElements. We need the updates length.
  //
  // Alternative approach: iterate over ALL output elements, and for each element,
  // check if it should be overwritten. This is O(nElements) per update, too expensive.
  //
  // Better: iterate over update elements. We know sliceSize from dataShape.
  // We'll use the grid to cover update elements: the grid covers max(nElements, updateElems).
  // For Phase 2, mask with updateElems limit.
  // Since we don't have updateElems explicitly, compute it in the kernel:
  // updateElems = (we'd need it passed in)
  //
  // Simplest correct approach: Phase 2 iterates over the same nElements grid,
  // but only activates for positions that correspond to update elements.
  // For scatter_nd with indexDepth=1:
  //   For flat output position p: check if p's "row" (p / sliceSize) matches any index
  //   This is O(nElements * numUpdates) - not great.
  //
  // Most practical: Phase 2 is a separate grid over totalUpdateElems.
  // Since Triton requires a single grid, we use the SAME grid (nElements) and
  // only process elements within the update range.
  //
  // Simple 1D-index scatter: indices are [numUpdates] (or [numUpdates,1])
  // Each update i writes a slice of sliceSize elements starting at indices[i] * sliceSize
  // Total update elements = numUpdates * sliceSize
  // For flat j in [0, totalUpdateElems):
  //   updateIdx = j / sliceSize
  //   slicePos = j % sliceSize
  //   outPos = indices[updateIdx] * sliceSize + slicePos
  //   output[outPos] = updates[j]
  //
  // We process this using offsets (0..blockSize-1 per block), masked to totalUpdateElems.
  // But we need totalUpdateElems at compile time... we'll use nElements as upper bound
  // and add bounds checking.

  // For correctness: Phase 2 iterates over indices directly.
  // Load index for the current position, compute scatter target.
  auto sliceSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, static_cast<int>(sliceSize), 32);
  auto splatSliceSize = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sliceSizeConst);

  // updateIdx = offsets / sliceSize
  auto updateIdx = builder.create<mlir::arith::DivSIOp>(loc, offsets, splatSliceSize);
  // slicePos = offsets % sliceSize
  auto slicePos = builder.create<mlir::arith::RemSIOp>(loc, offsets, splatSliceSize);

  // Load index: indices[updateIdx] (treat indices as flat array of index values)
  auto splatIdxPtr = builder.create<mlir::triton::SplatOp>(loc, idxPtrTensorType, indicesPtr);
  auto idxPtrs = builder.create<mlir::triton::AddPtrOp>(loc, idxPtrTensorType, splatIdxPtr, updateIdx);
  auto rawIndices = builder.create<mlir::triton::LoadOp>(loc,
      idxPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
  mlir::Value indices = castTo(builder, loc, rawIndices, i32Type);

  // outPos = indices[updateIdx] * sliceSize + slicePos
  auto scaledIdx = builder.create<mlir::arith::MulIOp>(loc, indices, splatSliceSize);
  auto outPos = builder.create<mlir::arith::AddIOp>(loc, scaledIdx, slicePos);

  // Bounds check: outPos must be in [0, nElements)
  auto outPosBoundsCheck = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, outPos, splatN);
  auto outPosGe0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, outPos,
      builder.create<mlir::triton::SplatOp>(loc, i32TensorType,
          builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32)));
  auto scatterMask = builder.create<mlir::arith::AndIOp>(loc,
      builder.create<mlir::arith::AndIOp>(loc, mask, outPosBoundsCheck),
      outPosGe0);

  // Load update values: updates[offsets] (flat indexing)
  auto splatUpdPtr = builder.create<mlir::triton::SplatOp>(loc, updPtrTensorType, updatesPtr);
  auto updPtrs = builder.create<mlir::triton::AddPtrOp>(loc, updPtrTensorType, splatUpdPtr, offsets);
  auto updateVals = builder.create<mlir::triton::LoadOp>(loc,
      updPtrs.getResult(), mask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Scatter: output[outPos] = updates[offsets]
  mlir::Value updStoreVal = castTo(builder, loc, updateVals, outPtrType.getPointeeType());
  auto scatterPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, outPos);
  builder.create<mlir::triton::StoreOp>(loc, scatterPtrs, updStoreVal, scatterMask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);
}

void TritonIRBuilder::emitConvolutionSection(mlir::OpBuilder& builder, mlir::Location loc,
                                              mlir::Value pid, int blockSize,
                                              mlir::Value inputPtr, mlir::Value filterPtr,
                                              mlir::Value outputPtr,
                                              const std::vector<LongType>& inputShape,
                                              const std::vector<LongType>& filterShape,
                                              const std::vector<LongType>& outputShape,
                                              int strideH, int strideW,
                                              int padH, int padW,
                                              int nElements, int wFormat) {
  // Direct conv2d: each output element independently computes its value by
  // iterating over the filter spatial dimensions and input channels.
  //
  // Input shape: [N, IC, IH, IW] (NCHW)
  // Filter shape: depends on wFormat:
  //   0=[kH,kW,iC,oC], 1=[oC,iC,kH,kW], 2=[oC,kH,kW,iC]
  // Output shape: [N, OC, OH, OW]
  //
  // out[n,oc,oh,ow] = sum_{ic,kh,kw} input[n,ic,oh*sH-pH+kh,ow*sW-pW+kw] * filter[oc,ic,kh,kw]

  if (inputShape.size() < 4 || filterShape.size() < 4 || outputShape.size() < 4) {
    sd_debug("TritonIRBuilder::emitConvolutionSection: shapes must be 4D, got input=%d filter=%d output=%d\n",
              (int)inputShape.size(), (int)filterShape.size(), (int)outputShape.size());
    return;
  }

  int N  = static_cast<int>(inputShape[0]);
  int IC = static_cast<int>(inputShape[1]);
  int IH = static_cast<int>(inputShape[2]);
  int IW = static_cast<int>(inputShape[3]);

  // Extract OC, KH, KW based on weight format
  int OC, KH, KW;
  if (wFormat == 1) {
    // [oC, iC, kH, kW]
    OC = static_cast<int>(filterShape[0]);
    KH = static_cast<int>(filterShape[2]);
    KW = static_cast<int>(filterShape[3]);
  } else if (wFormat == 2) {
    // [oC, kH, kW, iC]
    OC = static_cast<int>(filterShape[0]);
    KH = static_cast<int>(filterShape[1]);
    KW = static_cast<int>(filterShape[2]);
  } else {
    // wFormat == 0: [kH, kW, iC, oC]
    KH = static_cast<int>(filterShape[0]);
    KW = static_cast<int>(filterShape[1]);
    OC = static_cast<int>(filterShape[3]);
  }

  int OH = static_cast<int>(outputShape[2]);
  int OW = static_cast<int>(outputShape[3]);

  auto i32Type = builder.getI32Type();
  auto f32Type = builder.getF32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);

  // Derive pointer types from actual arguments (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto filtPtrType = mlir::cast<mlir::triton::PointerType>(filterPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto filtPtrTensorType = mlir::RankedTensorType::get({blockSize}, filtPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // Standard 1D offsets
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Unravel linear index to (n, oc, oh, ow).
  // C-order NCHW [N, OC, OH, OW] has OW as the fastest-varying dimension (stride 1),
  // then OH (stride OW), then OC (stride OH*OW), then N (stride OC*OH*OW).
  //   ow = offsets % OW          (fastest / innermost)
  //   oh = (offsets / OW) % OH
  //   oc = (offsets / (OW * OH)) % OC
  //   n  = offsets / (OW * OH * OC)
  auto owConst = builder.create<mlir::arith::ConstantIntOp>(loc, OW, 32);
  auto ohConst = builder.create<mlir::arith::ConstantIntOp>(loc, OH, 32);
  auto ocConst = builder.create<mlir::arith::ConstantIntOp>(loc, OC, 32);
  auto owSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, owConst);
  auto ohSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ohConst);
  auto ocSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ocConst);

  auto ow_idx = builder.create<mlir::arith::RemSIOp>(loc, offsets, owSplat);
  auto tmp1 = builder.create<mlir::arith::DivSIOp>(loc, offsets, owSplat);
  auto oh_idx = builder.create<mlir::arith::RemSIOp>(loc, tmp1.getResult(), ohSplat);
  auto tmp2 = builder.create<mlir::arith::DivSIOp>(loc, tmp1.getResult(), ohSplat);
  auto oc_idx = builder.create<mlir::arith::RemSIOp>(loc, tmp2.getResult(), ocSplat);
  auto n_idx = builder.create<mlir::arith::DivSIOp>(loc, tmp2.getResult(), ocSplat);

  // Compute base positions in input space
  // oh_base = oh_idx * strideH - padH
  // ow_base = ow_idx * strideW - padW
  auto sHConst = builder.create<mlir::arith::ConstantIntOp>(loc, strideH, 32);
  auto sWConst = builder.create<mlir::arith::ConstantIntOp>(loc, strideW, 32);
  auto pHConst = builder.create<mlir::arith::ConstantIntOp>(loc, padH, 32);
  auto pWConst = builder.create<mlir::arith::ConstantIntOp>(loc, padW, 32);
  auto sHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sHConst);
  auto sWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sWConst);
  auto pHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pHConst);
  auto pWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pWConst);

  auto oh_base = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, oh_idx, sHSplat), pHSplat);
  auto ow_base = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::MulIOp>(loc, ow_idx, sWSplat), pWSplat);

  // Initialize accumulator to 0.0
  mlir::Value acc = splatConstantF32(builder, loc, f32TensorType, 0.0f);

  // Fully unrolled convolution loop: eliminates scf::ForOp entirely to avoid
  // Triton compiler pass pipeline bugs that corrupt accumulator element ordering.
  // Each iteration is emitted as independent SSA operations.
  int totalIters = IC * KH * KW;

  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto ihConst = builder.create<mlir::arith::ConstantIntOp>(loc, IH, 32);
  auto iwConst = builder.create<mlir::arith::ConstantIntOp>(loc, IW, 32);
  auto ihSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ihConst);
  auto iwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iwConst);
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zero);
  auto icIhIw = builder.create<mlir::arith::ConstantIntOp>(loc, IC * IH * IW, 32);
  auto ihIw = builder.create<mlir::arith::ConstantIntOp>(loc, IH * IW, 32);
  auto icIhIwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, icIhIw);
  auto ihIwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ihIw);

  for (int iter = 0; iter < totalIters; iter++) {
    int ic_i = iter / (KH * KW);
    int kh_i = (iter / KW) % KH;
    int kw_i = iter % KW;

    auto kh_val = builder.create<mlir::arith::ConstantIntOp>(loc, kh_i, 32);
    auto kw_val = builder.create<mlir::arith::ConstantIntOp>(loc, kw_i, 32);
    auto khSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kh_val);
    auto kwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kw_val);

    // Compute input position: h_in = oh_base + kh, w_in = ow_base + kw
    auto h_in = builder.create<mlir::arith::AddIOp>(loc, oh_base, khSplat);
    auto w_in = builder.create<mlir::arith::AddIOp>(loc, ow_base, kwSplat);

    // Bounds check: 0 <= h_in < IH && 0 <= w_in < IW
    auto h_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, h_in, zeroSplat);
    auto h_lt_IH = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, h_in, ihSplat);
    auto w_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, w_in, zeroSplat);
    auto w_lt_IW = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, w_in, iwSplat);
    auto h_valid = builder.create<mlir::arith::AndIOp>(loc, h_ge_0, h_lt_IH);
    auto w_valid = builder.create<mlir::arith::AndIOp>(loc, w_ge_0, w_lt_IW);
    auto in_bounds = builder.create<mlir::arith::AndIOp>(loc, h_valid, w_valid);

    // Input offset: n * IC*IH*IW + ic * IH*IW + h_in * IW + w_in
    auto ic_val = builder.create<mlir::arith::ConstantIntOp>(loc, ic_i, 32);
    auto icSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, ic_val);

    auto inOffset = builder.create<mlir::arith::AddIOp>(loc,
        builder.create<mlir::arith::AddIOp>(loc,
            builder.create<mlir::arith::MulIOp>(loc, n_idx, icIhIwSplat),
            builder.create<mlir::arith::MulIOp>(loc, icSplat, ihIwSplat)),
        builder.create<mlir::arith::AddIOp>(loc,
            builder.create<mlir::arith::MulIOp>(loc, h_in, iwSplat),
            w_in));

    // Load input value (masked by bounds check AND element mask)
    auto combinedMask = builder.create<mlir::arith::AndIOp>(loc, in_bounds, mask);
    auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
    auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, inOffset);
    auto inLoaded = builder.create<mlir::triton::LoadOp>(loc,
        inPtrs.getResult(), combinedMask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto inVal = castTo(builder, loc, inLoaded, f32Type);

    // Filter offset depends on weight format:
    // wFormat 0: [kH,kW,iC,oC] → kh*kW*iC*oC + kw*iC*oC + ic*oC + oc
    // wFormat 1: [oC,iC,kH,kW] → oc*iC*kH*kW + ic*kH*kW + kh*kW + kw
    // wFormat 2: [oC,kH,kW,iC] → oc*kH*kW*iC + kh*kW*iC + kw*iC + ic
    mlir::Value filterOffset;
    if (wFormat == 0) {
      // [kH, kW, iC, oC]
      int fOff = kh_i * KW * IC * OC + kw_i * IC * OC + ic_i * OC;
      auto fOffConst = builder.create<mlir::arith::ConstantIntOp>(loc, fOff, 32);
      auto fOffSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, fOffConst);
      filterOffset = builder.create<mlir::arith::AddIOp>(loc, fOffSplat, oc_idx);
    } else if (wFormat == 2) {
      // [oC, kH, kW, iC]
      int fOff = kh_i * KW * IC + kw_i * IC + ic_i;
      auto fOffConst = builder.create<mlir::arith::ConstantIntOp>(loc, fOff, 32);
      auto fOffSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, fOffConst);
      // oc_idx * KH * KW * IC + fOff
      auto khKwIc = builder.create<mlir::arith::ConstantIntOp>(loc, KH * KW * IC, 32);
      auto khKwIcSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, khKwIc);
      filterOffset = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, oc_idx, khKwIcSplat), fOffSplat);
    } else {
      // wFormat 1: [oC, iC, kH, kW]
      // oc * IC*KH*KW + (ic*KH*KW + kh*KW + kw) — ic/kh/kw are compile-time constants
      int fOff = ic_i * KH * KW + kh_i * KW + kw_i;
      auto fOffConst = builder.create<mlir::arith::ConstantIntOp>(loc, fOff, 32);
      auto fOffSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, fOffConst);
      auto icKhKw = builder.create<mlir::arith::ConstantIntOp>(loc, IC * KH * KW, 32);
      auto icKhKwSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, icKhKw);
      filterOffset = builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, oc_idx, icKhKwSplat), fOffSplat);
    }

    // Load filter value (masked by element mask only, filter is always in bounds)
    auto splatFilterPtr = builder.create<mlir::triton::SplatOp>(loc, filtPtrTensorType, filterPtr);
    auto filterPtrs = builder.create<mlir::triton::AddPtrOp>(loc, filtPtrTensorType, splatFilterPtr, filterOffset);
    auto filterLoaded = builder.create<mlir::triton::LoadOp>(loc,
        filterPtrs.getResult(), mask.getResult(), mlir::Value(),
        mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);
    auto filterVal = castTo(builder, loc, filterLoaded, f32Type);

    // Accumulate: acc += input * filter (zero out-of-bounds input)
    auto prod = builder.create<mlir::arith::MulFOp>(loc, inVal, filterVal);
    acc = builder.create<mlir::arith::AddFOp>(loc, acc, prod);
  }

  auto finalAcc = acc;

  mlir::Value outStoreVal = castTo(builder, loc, finalAcc, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, outStoreVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_debug("TritonIRBuilder::emitConvolutionSection: conv2d N=%d IC=%d IH=%d IW=%d OC=%d KH=%d KW=%d "
            "OH=%d OW=%d sH=%d sW=%d pH=%d pW=%d\n",
            N, IC, IH, IW, OC, KH, KW, OH, OW, strideH, strideW, padH, padW);
}

// ─── im2col emission ─────────────────────────────────────────────────────────
// Rearranges image patches into columns for convolution.
// Input: [bS, iC, iH, iW] (4D)  →  Output: [bS, iC, kH, kW, oH, oW] (6D)
//
// For each output element at (b, c, kRow, kCol, colH, colW):
//   imRow = (-pH + kRow * dH) + colH * sH
//   imCol = (-pW + kCol * dW) + colW * sW
//   out = (in_bounds) ? input[b, c, imRow, imCol] : 0

void TritonIRBuilder::emitIm2colSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         mlir::Value inputPtr, mlir::Value outputPtr,
                                         const std::vector<LongType>& inputShape,
                                         const std::vector<LongType>& outputShape,
                                         int kH, int kW,
                                         int sH, int sW,
                                         int pH, int pW,
                                         int dH, int dW,
                                         int nElements) {
  // Input: [bS, iC, iH, iW], Output: [bS, iC, kH, kW, oH, oW]
  if (inputShape.size() < 4 || outputShape.size() < 6) {
    sd_debug("TritonIRBuilder::emitIm2colSection: input must be 4D (got %d) and output must be 6D (got %d)\n",
              (int)inputShape.size(), (int)outputShape.size());
    return;
  }

  int bS = static_cast<int>(inputShape[0]);
  int iC = static_cast<int>(inputShape[1]);
  int iH = static_cast<int>(inputShape[2]);
  int iW = static_cast<int>(inputShape[3]);
  int oH = static_cast<int>(outputShape[4]);
  int oW = static_cast<int>(outputShape[5]);

  auto i32Type = builder.getI32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  // Derive pointer types from actual MLIR args (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // 1D offsets into output (6D linearized)
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Unravel linear index to 6D: (b, c, kRow, kCol, colH, colW)
  // Layout: [bS, iC, kH, kW, oH, oW] in row-major
  // colW = offset % oW
  // colH = (offset / oW) % oH
  // kCol = (offset / (oW * oH)) % kW
  // kRow = (offset / (oW * oH * kW)) % kH
  // c    = (offset / (oW * oH * kW * kH)) % iC
  // b    = offset / (oW * oH * kW * kH * iC)
  auto oWConst = builder.create<mlir::arith::ConstantIntOp>(loc, oW, 32);
  auto oHConst = builder.create<mlir::arith::ConstantIntOp>(loc, oH, 32);
  auto kWConst = builder.create<mlir::arith::ConstantIntOp>(loc, kW, 32);
  auto kHConst = builder.create<mlir::arith::ConstantIntOp>(loc, kH, 32);
  auto iCConst = builder.create<mlir::arith::ConstantIntOp>(loc, iC, 32);
  auto oWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oWConst);
  auto oHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oHConst);
  auto kWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kWConst);
  auto kHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kHConst);
  auto iCSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iCConst);

  auto colW_idx = builder.create<mlir::arith::RemSIOp>(loc, offsets, oWSplat);
  auto t1 = builder.create<mlir::arith::DivSIOp>(loc, offsets, oWSplat);
  auto colH_idx = builder.create<mlir::arith::RemSIOp>(loc, t1.getResult(), oHSplat);
  auto t2 = builder.create<mlir::arith::DivSIOp>(loc, t1.getResult(), oHSplat);
  auto kCol_idx = builder.create<mlir::arith::RemSIOp>(loc, t2.getResult(), kWSplat);
  auto t3 = builder.create<mlir::arith::DivSIOp>(loc, t2.getResult(), kWSplat);
  auto kRow_idx = builder.create<mlir::arith::RemSIOp>(loc, t3.getResult(), kHSplat);
  auto t4 = builder.create<mlir::arith::DivSIOp>(loc, t3.getResult(), kHSplat);
  auto c_idx = builder.create<mlir::arith::RemSIOp>(loc, t4.getResult(), iCSplat);
  auto b_idx = builder.create<mlir::arith::DivSIOp>(loc, t4.getResult(), iCSplat);

  // Compute input coordinates:
  // imRow = (-pH + kRow * dH) + colH * sH
  // imCol = (-pW + kCol * dW) + colW * sW
  auto dHConst = builder.create<mlir::arith::ConstantIntOp>(loc, dH, 32);
  auto dWConst = builder.create<mlir::arith::ConstantIntOp>(loc, dW, 32);
  auto sHConst = builder.create<mlir::arith::ConstantIntOp>(loc, sH, 32);
  auto sWConst = builder.create<mlir::arith::ConstantIntOp>(loc, sW, 32);
  auto pHConst = builder.create<mlir::arith::ConstantIntOp>(loc, pH, 32);
  auto pWConst = builder.create<mlir::arith::ConstantIntOp>(loc, pW, 32);
  auto dHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dHConst);
  auto dWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dWConst);
  auto sHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sHConst);
  auto sWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sWConst);
  auto pHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pHConst);
  auto pWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pWConst);

  // kRow * dH
  auto kRowDH = builder.create<mlir::arith::MulIOp>(loc, kRow_idx, dHSplat);
  // colH * sH
  auto colHSH = builder.create<mlir::arith::MulIOp>(loc, colH_idx, sHSplat);
  // imRow = kRow * dH + colH * sH - pH
  auto imRow = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc, kRowDH, colHSH), pHSplat);

  // kCol * dW
  auto kColDW = builder.create<mlir::arith::MulIOp>(loc, kCol_idx, dWSplat);
  // colW * sW
  auto colWSW = builder.create<mlir::arith::MulIOp>(loc, colW_idx, sWSplat);
  // imCol = kCol * dW + colW * sW - pW
  auto imCol = builder.create<mlir::arith::SubIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc, kColDW, colWSW), pWSplat);

  // Bounds check: 0 <= imRow < iH && 0 <= imCol < iW
  auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto iHConst = builder.create<mlir::arith::ConstantIntOp>(loc, iH, 32);
  auto iWConst = builder.create<mlir::arith::ConstantIntOp>(loc, iW, 32);
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zero);
  auto iHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iHConst);
  auto iWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iWConst);

  auto h_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, imRow, zeroSplat);
  auto h_lt_iH = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, imRow, iHSplat);
  auto w_ge_0 = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, imCol, zeroSplat);
  auto w_lt_iW = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, imCol, iWSplat);
  auto h_valid = builder.create<mlir::arith::AndIOp>(loc, h_ge_0, h_lt_iH);
  auto w_valid = builder.create<mlir::arith::AndIOp>(loc, w_ge_0, w_lt_iW);
  auto inBounds = builder.create<mlir::arith::AndIOp>(loc, h_valid, w_valid);

  // Combined mask: element in range AND in bounds
  auto combinedMask = builder.create<mlir::arith::AndIOp>(loc, inBounds, mask);

  // Input offset: b * (iC*iH*iW) + c * (iH*iW) + imRow * iW + imCol
  auto iCiHiW = builder.create<mlir::arith::ConstantIntOp>(loc, iC * iH * iW, 32);
  auto iHiW = builder.create<mlir::arith::ConstantIntOp>(loc, iH * iW, 32);
  auto iCiHiWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iCiHiW);
  auto iHiWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iHiW);

  auto inOffset = builder.create<mlir::arith::AddIOp>(loc,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, b_idx, iCiHiWSplat),
          builder.create<mlir::arith::MulIOp>(loc, c_idx, iHiWSplat)),
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::MulIOp>(loc, imRow, iWSplat),
          imCol));

  // Load from input with bounds mask (out-of-bounds → 0)
  auto splatInPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto inPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatInPtr, inOffset);
  auto inVal = builder.create<mlir::triton::LoadOp>(loc,
      inPtrs.getResult(), combinedMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Store to output — cast if input and output element types differ
  mlir::Value storeVal = castTo(builder, loc, inVal, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_debug("TritonIRBuilder::emitIm2colSection: bS=%d iC=%d iH=%d iW=%d kH=%d kW=%d "
            "oH=%d oW=%d sH=%d sW=%d pH=%d pW=%d dH=%d dW=%d\n",
            bS, iC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
}

// ─── col2im emission ─────────────────────────────────────────────────────────
// Rearranges columns back to image (inverse of im2col).
// Input: [bS, iC, kH, kW, oH, oW] (6D)  →  Output: [bS, iC, iH, iW] (4D)
//
// For each output pixel at (b, c, h, w):
//   Iterate over kRow in [0, kH) and kCol in [0, kW):
//     colH = (h + pH - kRow * dH)
//     if colH >= 0 && colH % sH == 0: colH /= sH
//       colW = (w + pW - kCol * dW)
//       if colW >= 0 && colW % sW == 0: colW /= sW
//         if colH < oH && colW < oW: val += col[b, c, kRow, kCol, colH, colW]
//   out[b, c, h, w] = val

void TritonIRBuilder::emitCol2imSection(mlir::OpBuilder& builder, mlir::Location loc,
                                         mlir::Value pid, int blockSize,
                                         mlir::Value inputPtr, mlir::Value outputPtr,
                                         const std::vector<LongType>& inputShape,
                                         const std::vector<LongType>& outputShape,
                                         int kH, int kW,
                                         int sH, int sW,
                                         int pH, int pW,
                                         int dH, int dW,
                                         int nElements) {
  // Input (columns): [bS, iC, kH, kW, oH, oW], Output (image): [bS, iC, iH, iW]
  if (inputShape.size() < 6 || outputShape.size() < 4) {
    sd_debug("TritonIRBuilder::emitCol2imSection: input must be 6D (got %d) and output must be 4D (got %d)\n",
              (int)inputShape.size(), (int)outputShape.size());
    return;
  }

  int bS = static_cast<int>(outputShape[0]);
  int iC = static_cast<int>(outputShape[1]);
  int iH = static_cast<int>(outputShape[2]);
  int iW = static_cast<int>(outputShape[3]);
  int oH = static_cast<int>(inputShape[4]);
  int oW = static_cast<int>(inputShape[5]);

  auto i32Type = builder.getI32Type();
  auto f32Type = builder.getF32Type();
  auto i32TensorType = mlir::RankedTensorType::get({blockSize}, i32Type);
  auto f32TensorType = mlir::RankedTensorType::get({blockSize}, f32Type);
  // Derive pointer types from actual MLIR args (NOT hardcoded f32)
  auto inPtrType = mlir::cast<mlir::triton::PointerType>(inputPtr.getType());
  auto outPtrType = mlir::cast<mlir::triton::PointerType>(outputPtr.getType());
  auto inPtrTensorType = mlir::RankedTensorType::get({blockSize}, inPtrType);
  auto outPtrTensorType = mlir::RankedTensorType::get({blockSize}, outPtrType);

  // 1D offsets into output (4D linearized)
  auto blockSizeConst = builder.create<mlir::arith::ConstantIntOp>(loc, blockSize, 32);
  auto offsetBase = builder.create<mlir::arith::MulIOp>(loc, pid, blockSizeConst);
  auto range = builder.create<mlir::triton::MakeRangeOp>(loc, i32TensorType, 0, blockSize);
  auto splatBase = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, offsetBase);
  auto offsets = builder.create<mlir::arith::AddIOp>(loc, splatBase, range);

  auto nElemConst = builder.create<mlir::arith::ConstantIntOp>(loc, nElements, 32);
  auto splatN = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, nElemConst);
  auto mask = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, offsets, splatN);

  // Unravel linear index to 4D: (b, c, h, w)
  // w = offset % iW
  // h = (offset / iW) % iH
  // c = (offset / (iW * iH)) % iC
  // b = offset / (iW * iH * iC)
  auto iWConst = builder.create<mlir::arith::ConstantIntOp>(loc, iW, 32);
  auto iHConst = builder.create<mlir::arith::ConstantIntOp>(loc, iH, 32);
  auto iCConst = builder.create<mlir::arith::ConstantIntOp>(loc, iC, 32);
  auto iWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iWConst);
  auto iHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iHConst);
  auto iCSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, iCConst);

  auto w_idx = builder.create<mlir::arith::RemSIOp>(loc, offsets, iWSplat);
  auto u1 = builder.create<mlir::arith::DivSIOp>(loc, offsets, iWSplat);
  auto h_idx = builder.create<mlir::arith::RemSIOp>(loc, u1.getResult(), iHSplat);
  auto u2 = builder.create<mlir::arith::DivSIOp>(loc, u1.getResult(), iHSplat);
  auto c_idx = builder.create<mlir::arith::RemSIOp>(loc, u2.getResult(), iCSplat);
  auto b_idx = builder.create<mlir::arith::DivSIOp>(loc, u2.getResult(), iCSplat);

  // Padded coordinates: imH = h + pH, imW = w + pW
  auto pHConst = builder.create<mlir::arith::ConstantIntOp>(loc, pH, 32);
  auto pWConst = builder.create<mlir::arith::ConstantIntOp>(loc, pW, 32);
  auto pHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pHConst);
  auto pWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, pWConst);
  auto imH = builder.create<mlir::arith::AddIOp>(loc, h_idx, pHSplat);
  auto imW = builder.create<mlir::arith::AddIOp>(loc, w_idx, pWSplat);

  // Column buffer strides for 6D [bS, iC, kH, kW, oH, oW]
  // colStride5 = 1 (oW dim)
  // colStride4 = oW (oH dim)
  // colStride3 = oW * oH (kW dim)
  // colStride2 = oW * oH * kW (kH dim)
  // colStride1 = oW * oH * kW * kH (iC dim)
  // colStride0 = oW * oH * kW * kH * iC (bS dim)
  int colStride4 = oW;
  int colStride3 = oW * oH;
  int colStride2 = oW * oH * kW;
  int colStride1 = oW * oH * kW * kH;
  int colStride0 = oW * oH * kW * kH * iC;

  // Base offset into column buffer: b * colStride0 + c * colStride1
  auto colStr0Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride0, 32);
  auto colStr1Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride1, 32);
  auto colStr0Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr0Const);
  auto colStr1Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr1Const);
  auto bsOffset = builder.create<mlir::arith::MulIOp>(loc, b_idx, colStr0Splat);
  auto cOffset = builder.create<mlir::arith::MulIOp>(loc, c_idx, colStr1Splat);
  auto bcOffset = builder.create<mlir::arith::AddIOp>(loc, bsOffset, cOffset);

  // Initialize accumulator to 0.0
  auto accInit = splatConstantF32(builder, loc, f32TensorType, 0.0f);

  // Nested loops over kRow in [0, kH) and kCol in [0, kW)
  // These are compile-time-constant loop bounds, uniform across all elements
  auto zeroScalar = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto oneScalar = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
  auto kHEnd = builder.create<mlir::arith::ConstantIntOp>(loc, kH, 32);
  auto kWEnd = builder.create<mlir::arith::ConstantIntOp>(loc, kW, 32);

  // Stride/dilation constants for vectorized computation
  auto dHConst = builder.create<mlir::arith::ConstantIntOp>(loc, dH, 32);
  auto dWConst = builder.create<mlir::arith::ConstantIntOp>(loc, dW, 32);
  auto sHConst = builder.create<mlir::arith::ConstantIntOp>(loc, sH, 32);
  auto sWConst = builder.create<mlir::arith::ConstantIntOp>(loc, sW, 32);
  auto dHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dHConst);
  auto dWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, dWConst);
  auto sHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sHConst);
  auto sWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, sWConst);
  auto oHConst = builder.create<mlir::arith::ConstantIntOp>(loc, oH, 32);
  auto oWConst = builder.create<mlir::arith::ConstantIntOp>(loc, oW, 32);
  auto oHSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oHConst);
  auto oWSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, oWConst);
  auto zeroSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, zeroScalar);
  auto colStr2Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride2, 32);
  auto colStr3Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride3, 32);
  auto colStr4Const = builder.create<mlir::arith::ConstantIntOp>(loc, colStride4, 32);
  auto colStr2Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr2Const);
  auto colStr3Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr3Const);
  auto colStr4Splat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, colStr4Const);

  // Outer loop: kRow
  auto kRowLoop = builder.create<mlir::scf::ForOp>(
      loc, zeroScalar, kHEnd, oneScalar, mlir::ValueRange{accInit});
  builder.setInsertionPointToStart(kRowLoop.getBody());
  auto kRow_val = kRowLoop.getInductionVar();
  auto acc_kr = kRowLoop.getBody()->getArgument(1);

  // colH_raw = imH - kRow * dH  (per-element, signed)
  auto kRowSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kRow_val);
  auto kRowDH = builder.create<mlir::arith::MulIOp>(loc, kRowSplat, dHSplat);
  auto colH_raw = builder.create<mlir::arith::SubIOp>(loc, imH, kRowDH);

  // Valid if colH_raw >= 0 && colH_raw % sH == 0
  auto colH_ge0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, colH_raw, zeroSplat);
  auto colH_mod = builder.create<mlir::arith::RemSIOp>(loc, colH_raw, sHSplat);
  auto colH_aligned = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, colH_mod, zeroSplat);
  auto colH_valid1 = builder.create<mlir::arith::AndIOp>(loc, colH_ge0, colH_aligned);

  // colH = colH_raw / sH (only meaningful where valid)
  auto colH = builder.create<mlir::arith::DivSIOp>(loc, colH_raw, sHSplat);
  // Additional check: colH < oH
  auto colH_lt_oH = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, colH, oHSplat);
  auto colH_valid = builder.create<mlir::arith::AndIOp>(loc, colH_valid1, colH_lt_oH);

  // Inner loop: kCol
  auto kColLoop = builder.create<mlir::scf::ForOp>(
      loc, zeroScalar, kWEnd, oneScalar, mlir::ValueRange{acc_kr});
  builder.setInsertionPointToStart(kColLoop.getBody());
  auto kCol_val = kColLoop.getInductionVar();
  auto acc_kc = kColLoop.getBody()->getArgument(1);

  // colW_raw = imW - kCol * dW
  auto kColSplat = builder.create<mlir::triton::SplatOp>(loc, i32TensorType, kCol_val);
  auto kColDW = builder.create<mlir::arith::MulIOp>(loc, kColSplat, dWSplat);
  auto colW_raw = builder.create<mlir::arith::SubIOp>(loc, imW, kColDW);

  // Valid if colW_raw >= 0 && colW_raw % sW == 0
  auto colW_ge0 = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, colW_raw, zeroSplat);
  auto colW_mod = builder.create<mlir::arith::RemSIOp>(loc, colW_raw, sWSplat);
  auto colW_aligned = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, colW_mod, zeroSplat);
  auto colW_valid1 = builder.create<mlir::arith::AndIOp>(loc, colW_ge0, colW_aligned);

  // colW = colW_raw / sW
  auto colW = builder.create<mlir::arith::DivSIOp>(loc, colW_raw, sWSplat);
  // Additional check: colW < oW
  auto colW_lt_oW = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, colW, oWSplat);
  auto colW_valid = builder.create<mlir::arith::AndIOp>(loc, colW_valid1, colW_lt_oW);

  // Combined validity: colH valid AND colW valid AND element mask
  auto hw_valid = builder.create<mlir::arith::AndIOp>(loc, colH_valid, colW_valid);
  auto loadMask = builder.create<mlir::arith::AndIOp>(loc, hw_valid, mask);

  // Column buffer offset: bcOffset + kRow * colStride2 + kCol * colStride3 + colH * colStride4 + colW
  auto colOffset = builder.create<mlir::arith::AddIOp>(loc, bcOffset,
      builder.create<mlir::arith::AddIOp>(loc,
          builder.create<mlir::arith::AddIOp>(loc,
              builder.create<mlir::arith::MulIOp>(loc, kRowSplat, colStr2Splat),
              builder.create<mlir::arith::MulIOp>(loc, kColSplat, colStr3Splat)),
          builder.create<mlir::arith::AddIOp>(loc,
              builder.create<mlir::arith::MulIOp>(loc, colH, colStr4Splat),
              colW)));

  // Load column value (masked: invalid positions get 0)
  auto splatColPtr = builder.create<mlir::triton::SplatOp>(loc, inPtrTensorType, inputPtr);
  auto colPtrs = builder.create<mlir::triton::AddPtrOp>(loc, inPtrTensorType, splatColPtr, colOffset);
  auto colVal = builder.create<mlir::triton::LoadOp>(loc,
      colPtrs.getResult(), loadMask.getResult(), mlir::Value(),
      mlir::triton::CacheModifier::NONE, mlir::triton::EvictionPolicy::NORMAL, false);

  // Cast loaded value to f32 for accumulation if needed
  auto colValF32 = castTo(builder, loc, colVal, f32Type);

  // Accumulate
  auto newAcc = builder.create<mlir::arith::AddFOp>(loc, acc_kc, colValF32);

  // Yield from inner loop (kCol)
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{newAcc});

  // Yield from outer loop (kRow)
  builder.setInsertionPointAfter(kColLoop);
  builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{kColLoop.getResult(0)});

  // Store accumulated result
  builder.setInsertionPointAfter(kRowLoop);
  auto finalAcc = kRowLoop.getResult(0);

  // Cast f32 accumulator to output element type for store
  mlir::Value storeVal = castTo(builder, loc, finalAcc, outPtrType.getPointeeType());
  auto splatOutPtr = builder.create<mlir::triton::SplatOp>(loc, outPtrTensorType, outputPtr);
  auto outPtrs = builder.create<mlir::triton::AddPtrOp>(loc, outPtrTensorType, splatOutPtr, offsets);
  builder.create<mlir::triton::StoreOp>(loc, outPtrs, storeVal, mask,
                                         mlir::triton::CacheModifier::NONE,
                                         mlir::triton::EvictionPolicy::NORMAL);

  sd_debug("TritonIRBuilder::emitCol2imSection: bS=%d iC=%d iH=%d iW=%d kH=%d kW=%d "
            "oH=%d oW=%d sH=%d sW=%d pH=%d pW=%d dH=%d dW=%d\n",
            bS, iC, iH, iW, kH, kW, oH, oW, sH, sW, pH, pW, dH, dW);
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
