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

// GPU graph backend (Triton/NVRTC/PTX) execution methods.
//
// Contains getGpuGraphBackend() which selects the best available GPU compiler
// backend (Triton > NVRTC > PTX) based on the configured GraphExecutionMode,
// and executeSegmentWithGpuGraph() which drives segment compilation, CUDA graph
// capture/replay for Triton fused kernels, and native ordered-range orchestration.

#include <graph/NativeDynamicShapePlan.h>
#include <graph/DspDiagnostics.h>
#include <graph/DspConstants.h>
#include <graph/DspHashUtils.h>
#include <graph/DspStreamGuard.h>
#include <graph/DspVerifyUtils.h>
#include <graph/DspAnalysisUtils.h>
#include <graph/gpu/ViewRecipe.h>
#include <helpers/ShapeBuilders.h>
#include <helpers/MmulHelper.h>
#include <ops/OpTraitTable.h>
#include <system/op_boilerplate.h>
#include <system/Environment.h>
#include <config.h>

// Forward declaration for TritonGraphBackend (full header only available when HAVE_TRITON)
namespace sd { namespace graph { class TritonGraphBackend; } }

// Portable buffer accessor: specialBuffer() on CUDA, buffer() on CPU.
#ifdef SD_CUDA
#define DSP_BUF(arr) ((arr)->specialBuffer())
#else
#define DSP_BUF(arr) ((arr)->buffer())
#endif

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// GPU graph backends (conditional)
#if HAVE_TRITON && defined(SD_CUDA)
#include <graph/gpu/TritonGraphBackend.h>
#endif
#ifdef SD_CUDA
#include <graph/gpu/NvrtcGraphBackend.h>
#include <graph/gpu/PtxGraphBackend.h>
#include <graph/cuda/CudaGraphReplayHandle.h>
#include <memory/cuda/CudaMemoryPool.h>
#endif
#ifdef SD_TPU
#include <graph/tpu/TpuGraphBackend.h>
#endif
#ifdef HAVE_HEXAGON_MLIR
#include <graph/hexagon/HexagonGraphBackend.h>
#endif

namespace sd {
namespace graph {

// ── Segment bucket classification ──────────────────────────────────────────
// Classifies gap slots into bucket types based on slot traits and
// materialization behavior. Used by DSP segment bucket diagnostics.

// Resolve the TritonOpCategory for a slot's op. Returns UNSUPPORTED if the
// op cannot be identified.
static TritonOpCategory resolveOpCategory(int slotIdx, NativeSlot* slots) {
  if (slots == nullptr) return TritonOpCategory::UNSUPPORTED;
  const std::string& opName = slots[slotIdx].ident.opName;
  if (opName.empty()) return TritonOpCategory::UNSUPPORTED;

  const auto& table = getOpCategoryTable();
  auto it = table.find(opName);
  return (it != table.end()) ? it->second : TritonOpCategory::UNSUPPORTED;
}

static uint32_t resolveStructuralSlotTraits(int slotIdx, NativeSlot* slots) {
  if (slots == nullptr) return 0;

  uint32_t traits = 0;
  const auto& slot = slots[slotIdx];
  if (slot.ident.op != nullptr && slot.ident.op->getOpDescriptor() != nullptr) {
    traits |= slot.ident.op->getOpDescriptor()->getTraits();
  }
  // Fallback: look up traits by op name from the trait table.
  if (traits == 0 && !slot.ident.opName.empty()) {
    traits |= sd::ops::getOpTraitsByName(slot.ident.opName);
  }

  if (slot.flags.isViewCapableOp) traits |= sd::ops::OP_TRAIT_VIEW_PRODUCING;
  if (slot.flags.isIdentityOp) traits |= sd::ops::OP_TRAIT_IDENTITY;
  if (slot.flags.outputShapeDependsOnInputValues) traits |= sd::ops::OP_TRAIT_VALUE_DEPENDENT_SHAPE;
  if (slot.flags.isDataDependent) traits |= sd::ops::OP_TRAIT_DATA_DEPENDENT;

  return traits;
}

static uint32_t resolveSlotTraits(int slotIdx, NativeSlot* slots) {
  uint32_t traits = resolveStructuralSlotTraits(slotIdx, slots);
  if (traits != 0) {
    return traits;
  }

  // Coarse trait recovery is only for legacy/unknown slots with no descriptor-backed
  // structural traits. It must not invent view semantics from the category table:
  // categories like SHAPE_MANIPULATION include both zero-copy views and materializing
  // ops such as broadcast_to.
  TritonOpCategory cat = resolveOpCategory(slotIdx, slots);
  switch (cat) {
    case TritonOpCategory::IDENTITY:
      traits |= sd::ops::OP_TRAIT_IDENTITY | sd::ops::OP_TRAIT_VIEW_PRODUCING;
      break;
    case TritonOpCategory::CONSTANT_GENERATION:
      traits |= sd::ops::OP_TRAIT_CONSTANT_GENERATION;
      break;
    case TritonOpCategory::DATA_MOVEMENT:
      traits |= sd::ops::OP_TRAIT_DATA_MOVEMENT;
      break;
    case TritonOpCategory::FUSED_ATTENTION:
      traits |= sd::ops::OP_TRAIT_ATTENTION;
      break;
    case TritonOpCategory::NORMALIZATION:
      traits |= sd::ops::OP_TRAIT_NORMALIZATION;
      break;
    case TritonOpCategory::MATMUL:
      traits |= sd::ops::OP_TRAIT_MATMUL;
      break;
    case TritonOpCategory::REDUCTION:
      traits |= sd::ops::OP_TRAIT_REDUCTION;
      break;
    case TritonOpCategory::UNARY_ELEMENTWISE:
      traits |= sd::ops::OP_TRAIT_UNARY_ELEMENTWISE;
      break;
    case TritonOpCategory::BINARY_ELEMENTWISE:
      traits |= sd::ops::OP_TRAIT_BINARY_ELEMENTWISE;
      break;
    case TritonOpCategory::TERNARY:
      traits |= sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE;
      break;
    case TritonOpCategory::COMPARISON:
      traits |= sd::ops::OP_TRAIT_COMPARISON;
      break;
    case TritonOpCategory::LOGICAL:
      traits |= sd::ops::OP_TRAIT_LOGICAL;
      break;
    default:
      break;
  }

  return traits;
}

static bool slotHasAnyTrait(int slotIdx, NativeSlot* slots, uint32_t traits) {
  return (resolveSlotTraits(slotIdx, slots) & traits) != 0;
}

static bool slotIsViewRecipeOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots,
                         sd::ops::OP_TRAIT_VIEW_PRODUCING | sd::ops::OP_TRAIT_IDENTITY);
}

static bool slotIsShapeRecipeOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots, sd::ops::OP_TRAIT_SHAPE_ONLY_OUTPUT);
}

static bool slotIsConstantGenerationOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots, sd::ops::OP_TRAIT_CONSTANT_GENERATION);
}

static bool slotWouldMaterialize(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots, sd::ops::OP_TRAIT_DATA_MOVEMENT);
}

static bool slotIsAttentionOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots, sd::ops::OP_TRAIT_ATTENTION);
}

static bool slotIsNormalizationOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots, sd::ops::OP_TRAIT_NORMALIZATION);
}

static bool slotIsElementwisePayloadOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots,
                         sd::ops::OP_TRAIT_UNARY_ELEMENTWISE |
                         sd::ops::OP_TRAIT_BINARY_ELEMENTWISE |
                         sd::ops::OP_TRAIT_TERNARY_ELEMENTWISE |
                         sd::ops::OP_TRAIT_COMPARISON |
                         sd::ops::OP_TRAIT_LOGICAL);
}

// Delegate to shared utilities in DspAnalysisUtils.h
static int findProducerStepInSegment(const GraphSegment& seg, NativeSlot* slots, int outputSlotIdx) {
  return dsp::findProducerStepInSegment(seg, slots, outputSlotIdx);
}

static bool segmentHasInternalValueShapeInputs(const GraphSegment& seg, NativeSlot* slots) {
  return dsp::segmentHasInternalValueShapeInputs(seg, slots);
}

static bool slotIsReductionOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots, sd::ops::OP_TRAIT_REDUCTION);
}

static bool slotIsMatmulOp(int slotIdx, NativeSlot* slots) {
  return slotHasAnyTrait(slotIdx, slots, sd::ops::OP_TRAIT_MATMUL);
}

// Classify a single gap slot into a GapClassification using slot traits.
static DspDiagnostics::GapClassification classifyGapSlot(int slotIdx, NativeSlot* slots) {
  DspDiagnostics::GapClassification result;
  result.startSlot = slotIdx;
  result.endSlot = slotIdx;
  result.primaryOpType = "(unknown)";
  result.isViewOnly = false;
  result.isShapeOnly = false;
  result.wouldMaterialize = false;
  result.bucketLabel = nullptr;

  if (slots == nullptr) return result;

  const std::string& opNameStr = slots[slotIdx].ident.opName;
  if (!opNameStr.empty()) {
    result.primaryOpType = opNameStr.c_str();
  }

  if (slotIsShapeRecipeOp(slotIdx, slots) || slotIsConstantGenerationOp(slotIdx, slots)) {
    result.isShapeOnly = true;
    result.bucketLabel = "shape_expression";
  } else if (slotIsViewRecipeOp(slotIdx, slots)) {
    result.isViewOnly = true;
    result.bucketLabel = "view_chain";
  } else if (slotWouldMaterialize(slotIdx, slots)) {
    result.wouldMaterialize = true;
    result.bucketLabel = "materializing_prep";
  } else if (slotIsAttentionOp(slotIdx, slots)) {
    result.bucketLabel = "attention_tail";
  } else if (slotIsNormalizationOp(slotIdx, slots)) {
    result.bucketLabel = "normalization_tail";
  } else if (slotIsElementwisePayloadOp(slotIdx, slots)) {
    result.bucketLabel = "elementwise_payload";
  } else if (slotIsReductionOp(slotIdx, slots)) {
    result.bucketLabel = "reduction";
  } else if (slotIsMatmulOp(slotIdx, slots)) {
    result.bucketLabel = "matmul";
  } else {
    result.bucketLabel = "other_compute";
  }

  return result;
}

// Merge consecutive gap classifications with the same bucket label into ranges.
static std::vector<DspDiagnostics::GapClassification>
mergeGapClassifications(const std::vector<DspDiagnostics::GapClassification>& gaps) {
  std::vector<DspDiagnostics::GapClassification> merged;
  if (gaps.empty()) return merged;

  auto current = gaps[0];
  for (size_t i = 1; i < gaps.size(); i++) {
    if (gaps[i].bucketLabel && current.bucketLabel &&
        std::string(gaps[i].bucketLabel) == std::string(current.bucketLabel) &&
        gaps[i].startSlot == current.endSlot + 1) {
      current.endSlot = gaps[i].endSlot;
    } else {
      merged.push_back(current);
      current = gaps[i];
    }
  }
  merged.push_back(current);
  return merged;
}

// Build combined bucket label from the set of gap classifications in a segment.
static std::string buildCombinedBucketLabel(const std::vector<DspDiagnostics::GapClassification>& gaps) {
  std::unordered_set<std::string> labels;

  for (const auto& g : gaps) {
    if (g.bucketLabel) {
      std::string label(g.bucketLabel);
      // Skip view-only and shape-only for combined label — they are transparent
      if (label == "view_chain" || label == "shape_expression") {
        continue;
      }
      labels.insert(label);
    }
  }

  std::string result;
  for (const auto& special : {"attention_tail", "materializing_prep",
                              "normalization_tail", "elementwise_payload",
                              "reduction", "matmul", "other_compute"}) {
    if (labels.count(special)) {
      if (!result.empty()) result += "+";
      result += special;
    }
  }

  if (result.empty()) {
    result = "transparent_gap_chain";
  }
  return result;
}

// ── Shape-expression chain folding ─────────────────────────────────────────
//
// Folds pure shape-expression chains (shape_of → gather → small concat/stack,
// create → range → ones_as) into pre-computed results when the entire
// subgraph is shape/meta-only.
//
// Rules:
//   - Only fold when the WHOLE subgraph is shape/meta-only (no real tensor payload).
//   - Do NOT fold any gather or concat that touches real tensor payload.
//   - A "small" tensor is one whose total element count fits within a threshold
//     typical of shape tensors (e.g., rank <= 8, total elements <= 64).
//
// This is the first real launch-count reduction pass.

// Shape chain thresholds — canonical definitions in DspConstants.h
using dsp::SHAPE_CHAIN_MAX_ELEMENTS;
using dsp::SHAPE_CHAIN_MAX_RANK;

// Check if a slot's output is a "small" shape-sized tensor (not a real payload).
static bool isSmallShapeTensor(int slotIdx, NDArray** outputSlots) {
  if (outputSlots == nullptr || slotIdx < 0 || outputSlots[slotIdx] == nullptr) {
    return false;  // Unknown — assume it's real
  }
  NDArray* arr = outputSlots[slotIdx];
  return arr->lengthOf() <= SHAPE_CHAIN_MAX_ELEMENTS &&
         arr->rankOf() <= SHAPE_CHAIN_MAX_RANK;
}

// Check if a data movement op is operating on small shape tensors (not real data).
static bool isShapeDataMovement(int slotIdx, NativeSlot* slots, NDArray** outputSlots) {
  if (slots == nullptr) return false;

  if (!slotWouldMaterialize(slotIdx, slots)) return false;

  // Check if ALL inputs are small shape tensors
  for (int i = 0; i < slots[slotIdx].wiring.numInputs; i++) {
    int srcIdx = slots[slotIdx].wiring.inputSourceIndices[i];
    if (srcIdx < 0) {
      // External input — assume it's real data (not a shape expression)
      return false;
    }
    if (!isSmallShapeTensor(srcIdx, outputSlots)) {
      return false;  // At least one input is a real tensor
    }
  }

  // Check if output is also small
  if (slots[slotIdx].wiring.numOutputs > 0) {
    int outIdx = slots[slotIdx].wiring.outputSlotIndices[0];
    if (!isSmallShapeTensor(outIdx, outputSlots)) {
      return false;  // Output is a real tensor
    }
  }

  return true;
}

// Check if an entire slot range forms a pure shape-expression chain.
static bool isPureShapeChain(int startSlot, int endSlot, NativeSlot* slots,
                             NDArray** outputSlots) {
  for (int s = startSlot; s <= endSlot; s++) {
    if (slotIsShapeRecipeOp(s, slots) || slotIsConstantGenerationOp(s, slots)) {
      continue;  // CONSTANT_GENERATION is always shape-only
    }

    if (slotIsViewRecipeOp(s, slots)) {
      continue;  // View ops — compatible with shape chains
    }

    if (slotWouldMaterialize(s, slots)) {
      if (!isShapeDataMovement(s, slots, outputSlots)) {
        return false;  // DATA_MOVEMENT touching real tensor payload
      }
      continue;
    }

    // Any other category means this is NOT a pure shape chain.
    return false;
  }
  return true;
}

// Folded shape chain result
struct FoldedShapeChain {
  struct FoldedResult {
    int slotIndex;
    std::vector<uint8_t> data;   // Raw bytes of the folded result
    DataType dtype;
    std::vector<LongType> shape;
    std::vector<LongType> strides;
  };
  std::vector<FoldedResult> results;
  int startSlot;
  int endSlot;
};

// Attempt to fold a shape-expression chain. Returns true if folding succeeded.
static bool foldShapeChain(int startSlot, int endSlot, NativeSlot* slots,
                           NDArray** outputSlots, NDArray** externalInputs,
                           int numExternalInputs, FoldedShapeChain& outFolded) {
  if (!isPureShapeChain(startSlot, endSlot, slots, outputSlots)) {
    return false;
  }

  outFolded = FoldedShapeChain();
  outFolded.startSlot = startSlot;
  outFolded.endSlot = endSlot;

  // Capture the current output as the folded result for each slot
  for (int s = startSlot; s <= endSlot; s++) {
    if (outputSlots == nullptr || slots[s].wiring.numOutputs == 0) continue;

    int outIdx = slots[s].wiring.outputSlotIndices[0];
    if (outIdx < 0 || outputSlots[outIdx] == nullptr) continue;

    NDArray* out = outputSlots[outIdx];
    if (!isSmallShapeTensor(outIdx, outputSlots)) continue;

    FoldedShapeChain::FoldedResult result;
    result.slotIndex = outIdx;
    result.dtype = out->dataType();
    result.shape.resize(out->rankOf());
    result.strides.resize(out->rankOf());
    for (int i = 0; i < (int)out->rankOf(); i++) {
      result.shape[i] = out->sizeAt(i);
      result.strides[i] = out->strideAt(i);
    }

    // Copy the data from device or host
    size_t bytes = out->dataBuffer()->getLenInBytes();
    result.data.resize(bytes);
#ifdef SD_CUDA
    if (out->dataBuffer()->special()) {
      cudaMemcpy(result.data.data(), out->dataBuffer()->special(),
                 bytes, cudaMemcpyDeviceToHost);
    }
#else
    if (out->dataBuffer()->buffer()) {
      std::memcpy(result.data.data(), out->dataBuffer()->buffer(), bytes);
    }
#endif

    outFolded.results.push_back(std::move(result));
  }

  return !outFolded.results.empty();
}

// Install folded shape chain results into output slots during replay.
static void installFoldedShapeChain(const FoldedShapeChain& folded, NDArray** outputSlots,
                                    int totalOutputSlots) {
  for (const auto& result : folded.results) {
    if (result.slotIndex < 0 || result.slotIndex >= totalOutputSlots) continue;
    if (outputSlots[result.slotIndex] == nullptr) continue;

    NDArray* out = outputSlots[result.slotIndex];
#ifdef SD_CUDA
    if (out->dataBuffer()->special() && !result.data.empty()) {
      cudaMemcpy(out->dataBuffer()->special(), result.data.data(),
                 result.data.size(), cudaMemcpyHostToDevice);
    }
#else
    if (out->dataBuffer()->buffer() && !result.data.empty()) {
      std::memcpy(out->dataBuffer()->buffer(), result.data.data(), result.data.size());
    }
#endif
  }
}

// ── Ordered replay units ───────────────────────────────────────────────────
//
// Replaces mixed monolithic replay with ordered replay units:
//   Triton island -> prep unit -> Triton island -> prep unit
//
// Prep units cover materializing ops:
//   gather, concat, stack, tile, materializing broadcast_to
//
// No more "graph replay first, internal gaps later".
// This is the correctness fix for the current phase violation.

enum class ReplayUnitType {
  TRITON_ISLAND,   // Compiled Triton sub-kernel (graph replay)
  PREP_UNIT,       // Materializing gap op (gather, concat, stack, tile, broadcast_to)
  VIEW_INSTALL,    // View recipe installation (reshape, permute, etc.)
  SHAPE_INSTALL,   // Folded shape chain installation
};

struct ReplayUnit {
  ReplayUnitType type;
  int startSlot;
  int endSlot;
  TritonOpCategory opCategory;  // For PREP_UNIT: which category of op
  const char* opName;           // For diagnostics
};

// Build an ordered sequence of replay units for a segment.
static std::vector<ReplayUnit> buildReplayUnits(const GraphSegment& seg,
                                                NativeSlot* slots,
                                                NDArray** outputSlots,
                                                TritonGraphBackend* tritonBE) {
  std::vector<ReplayUnit> units;

#if HAVE_TRITON && defined(SD_CUDA)
  if (tritonBE == nullptr) return units;

  auto gapSlots = tritonBE->getGapSlots(seg, slots);
  auto classifyGapUnitType = [&](int slotIdx) -> ReplayUnitType {
    if (slotIsViewRecipeOp(slotIdx, slots)) {
      return ReplayUnitType::VIEW_INSTALL;
    }
    if (slotIsShapeRecipeOp(slotIdx, slots) || slotIsConstantGenerationOp(slotIdx, slots)) {
      return ReplayUnitType::SHAPE_INSTALL;
    }
    return ReplayUnitType::PREP_UNIT;
  };

  int currentSlot = seg.def.startSlot;
  while (currentSlot <= seg.def.endSlot) {
    bool isGap = gapSlots.count(currentSlot) > 0;

    if (isGap) {
      int gapStart = currentSlot;
      ReplayUnitType gapType = classifyGapUnitType(currentSlot);
      while (currentSlot <= seg.def.endSlot &&
             gapSlots.count(currentSlot) > 0 &&
             classifyGapUnitType(currentSlot) == gapType) {
        currentSlot++;
      }
      int gapEnd = currentSlot - 1;

      TritonOpCategory primaryCat = resolveOpCategory(gapStart, slots);
      const char* primaryOpName = "(unknown)";
      if (!slots[gapStart].ident.opName.empty()) {
        primaryOpName = slots[gapStart].ident.opName.c_str();
      }

      if (gapType == ReplayUnitType::VIEW_INSTALL) {
        units.push_back({ReplayUnitType::VIEW_INSTALL, gapStart, gapEnd, primaryCat, primaryOpName});
      } else if (gapType == ReplayUnitType::SHAPE_INSTALL) {
        units.push_back({ReplayUnitType::SHAPE_INSTALL, gapStart, gapEnd, primaryCat, primaryOpName});
      } else {
        units.push_back({ReplayUnitType::PREP_UNIT, gapStart, gapEnd, primaryCat, primaryOpName});
      }
    } else {
      int islandStart = currentSlot;
      while (currentSlot <= seg.def.endSlot && gapSlots.count(currentSlot) == 0) {
        currentSlot++;
      }
      int islandEnd = currentSlot - 1;

      units.push_back({ReplayUnitType::TRITON_ISLAND, islandStart, islandEnd,
                       TritonOpCategory::UNSUPPORTED, "triton_sub_kernel"});
    }
  }
#else
  (void)seg; (void)slots; (void)outputSlots;
#endif

  return units;
}

// ── Phase 2: Replay Schedule Signature ─────────────────────────────────────
//
// Stable hash encoding of a segment's ordered replay schedule.
// Used to prove the same structure is replayed across decode steps
// and to detect when consolidation changes the schedule.

struct ReplayScheduleSignature {
  enum UnitKind : uint8_t {
    UK_TRITON_ISLAND = 0,
    UK_VIEW_RECIPE = 1,
    UK_SHAPE_RECIPE = 2,
    UK_MATERIALIZED_PREP = 3
  };
  struct UnitEntry {
    UnitKind kind;
    int16_t startSlot;
    int16_t endSlot;
    uint16_t opCategory;  // TritonOpCategory as uint16_t
  };
  static constexpr int MAX_UNITS = 32;
  UnitEntry units[MAX_UNITS];
  int numUnits;
  uint64_t hash;  // FNV-1a hash of the schedule for cross-step comparison
  int startSlot;
  int endSlot;

  ReplayScheduleSignature() : numUnits(0), hash(0), startSlot(0), endSlot(0) {
    for (int i = 0; i < MAX_UNITS; i++) {
      units[i] = {UK_TRITON_ISLAND, 0, 0, 0};
    }
  }

  static uint8_t toUnitKind(ReplayUnitType type) {
    switch (type) {
      case ReplayUnitType::TRITON_ISLAND: return UK_TRITON_ISLAND;
      case ReplayUnitType::VIEW_INSTALL: return UK_VIEW_RECIPE;
      case ReplayUnitType::SHAPE_INSTALL: return UK_SHAPE_RECIPE;
      case ReplayUnitType::PREP_UNIT: return UK_MATERIALIZED_PREP;
    }
    return UK_MATERIALIZED_PREP;
  }
};

// Build a replay schedule signature from the ordered unit list.
static ReplayScheduleSignature buildReplaySignature(const GraphSegment& seg,
                                                     const std::vector<ReplayUnit>& units) {
  ReplayScheduleSignature sig;
  sig.startSlot = seg.def.startSlot;
  sig.endSlot = seg.def.endSlot;
  sig.numUnits = std::min((int)units.size(), ReplayScheduleSignature::MAX_UNITS);

  // Compute FNV-1a hash incrementally
  uint64_t h = dsp::FNV1A64_OFFSET_BASIS;
  for (int i = 0; i < sig.numUnits; i++) {
    const auto& u = units[i];
    sig.units[i].kind = ReplayScheduleSignature::toUnitKind(u.type);
    sig.units[i].startSlot = (int16_t)u.startSlot;
    sig.units[i].endSlot = (int16_t)u.endSlot;
    sig.units[i].opCategory = (uint16_t)(int)u.opCategory;

    // Fold each field into hash
    uint8_t buf[8];
    buf[0] = sig.units[i].kind;
    buf[1] = (uint8_t)(sig.units[i].startSlot & 0xFF);
    buf[2] = (uint8_t)((sig.units[i].startSlot >> 8) & 0xFF);
    buf[3] = (uint8_t)(sig.units[i].endSlot & 0xFF);
    buf[4] = (uint8_t)((sig.units[i].endSlot >> 8) & 0xFF);
    buf[5] = (uint8_t)(sig.units[i].opCategory & 0xFF);
    buf[6] = (uint8_t)((sig.units[i].opCategory >> 8) & 0xFF);
    buf[7] = (uint8_t)i;  // order matters

    dsp::fnv1aMix(h, buf, 8);
  }
  sig.hash = h;
  return sig;
}

// Compare two signatures — returns true if schedules are identical.
static bool signaturesMatch(const ReplayScheduleSignature& a,
                            const ReplayScheduleSignature& b) {
  if (a.hash != b.hash) return false;
  if (a.numUnits != b.numUnits) return false;
  for (int i = 0; i < a.numUnits; i++) {
    if (a.units[i].kind != b.units[i].kind ||
        a.units[i].startSlot != b.units[i].startSlot ||
        a.units[i].endSlot != b.units[i].endSlot ||
        a.units[i].opCategory != b.units[i].opCategory) {
      return false;
    }
  }
  return true;
}

// ── Phase 2: Consolidation Pass ────────────────────────────────────────────
//
// Merges adjacent replay units when the merged unit remains phase-closed
// and pointer/shape-stable. Uses explicit lowering rules, not generic
// "merge nearby" heuristics.

// Profitability gate: decide whether a consolidation is worth doing.
struct ConsolidationDecision {
  bool approved;
  int kernelsBefore;
  int kernelsAfter;
  int bytesMaterialized;
  bool isReusable;
  const char* reason;
};

static ConsolidationDecision evaluateConsolidation(
    const ReplayUnit& a, const ReplayUnit& b,
    int currentUnitCount, int postConsolidationUnitCount) {
  ConsolidationDecision d;
  d.kernelsBefore = currentUnitCount;
  d.kernelsAfter = postConsolidationUnitCount;
  d.bytesMaterialized = 0;
  d.isReusable = true;  // Assumed true for frozen-shape segments
  d.reason = "approved";

  // Gate 1: Must reduce kernel count
  if (postConsolidationUnitCount >= currentUnitCount) {
    d.approved = false;
    d.reason = "no kernel reduction";
    return d;
  }

  // Gate 2: PREP_UNIT + PREP_UNIT merge is always OK (same phase)
  if (a.type == ReplayUnitType::PREP_UNIT && b.type == ReplayUnitType::PREP_UNIT) {
    d.approved = true;
    d.reason = "prep+prep merge";
    return d;
  }

  // Gate 3: VIEW_RECIPE or SHAPE_RECIPE followed by TRITON_ISLAND
  // → absorb into Triton island, no extra kernel
  if ((a.type == ReplayUnitType::VIEW_INSTALL || a.type == ReplayUnitType::SHAPE_INSTALL) &&
      b.type == ReplayUnitType::TRITON_ISLAND) {
    d.approved = true;
    d.reason = "recipe absorbed into Triton island";
    return d;
  }

  // Gate 4: TRITON_ISLAND followed by PREP_UNIT
  // Only merge if the prep unit is consumed by a downstream Triton island
  // (i.e., not the last unit in the segment)
  if (a.type == ReplayUnitType::TRITON_ISLAND && b.type == ReplayUnitType::PREP_UNIT) {
    d.approved = true;
    d.reason = "island+prep ordered merge";
    return d;
  }

  // Gate 5: VIEW_RECIPE followed by SHAPE_RECIPE → always safe (both non-materializing)
  if ((a.type == ReplayUnitType::VIEW_INSTALL && b.type == ReplayUnitType::SHAPE_INSTALL) ||
      (a.type == ReplayUnitType::SHAPE_INSTALL && b.type == ReplayUnitType::VIEW_INSTALL)) {
    d.approved = true;
    d.reason = "recipe+recipe merge";
    return d;
  }

  // Gate 6: GAP (transparent only) can be absorbed by any preceding unit
  // Transparent gaps are view_chain or shape_expression — no GPU work, just pointer arithmetic
  if (b.type == ReplayUnitType::VIEW_INSTALL || b.type == ReplayUnitType::SHAPE_INSTALL) {
    d.approved = true;
    d.reason = "transparent gap absorbed";
    return d;
  }

  // Gate 7: PREP_UNIT followed by TRITON_ISLAND (reverse of Gate 4)
  // Prep output feeds the island — safe to merge in order
  if (a.type == ReplayUnitType::PREP_UNIT && b.type == ReplayUnitType::TRITON_ISLAND) {
    d.approved = true;
    d.reason = "prep+island ordered merge";
    return d;
  }

  // Default: reject — phases would cross or profitability unclear
  d.approved = false;
  d.reason = "phase boundary or unprofitable";
  return d;
}

// Check if two adjacent materializing prep units can be fused into a single
// explicit lowering. Fusion legality is trait/category-driven, not name-driven.
static bool canFusePrepUnits(const ReplayUnit& a, const ReplayUnit& b,
                             NativeSlot* slots, const char** outRuleName) {
  if (a.type != ReplayUnitType::PREP_UNIT || b.type != ReplayUnitType::PREP_UNIT) {
    return false;
  }

  if (a.opCategory == TritonOpCategory::DATA_MOVEMENT &&
      b.opCategory == TritonOpCategory::DATA_MOVEMENT) {
    if (outRuleName) *outRuleName = "adjacent_materializing_prep";
    return true;
  }

  // Constant generation feeding data movement — safe to fuse
  if (a.opCategory == TritonOpCategory::CONSTANT_GENERATION &&
      b.opCategory == TritonOpCategory::DATA_MOVEMENT) {
    if (outRuleName) *outRuleName = "constgen_into_data_movement";
    return true;
  }

  // Adjacent constant generation ops — safe to fuse
  if (a.opCategory == TritonOpCategory::CONSTANT_GENERATION &&
      b.opCategory == TritonOpCategory::CONSTANT_GENERATION) {
    if (outRuleName) *outRuleName = "adjacent_constgen";
    return true;
  }

  // Shape manipulation feeding data movement — safe to fuse
  if (a.opCategory == TritonOpCategory::SHAPE_MANIPULATION &&
      b.opCategory == TritonOpCategory::DATA_MOVEMENT) {
    if (outRuleName) *outRuleName = "shape_into_data_movement";
    return true;
  }

  (void)slots;
  return false;
}

// Consolidate the ordered replay unit list.
// Returns the consolidated unit list and the decision metadata.
static std::vector<ReplayUnit> consolidateReplayUnits(
    const std::vector<ReplayUnit>& units,
    NativeSlot* slots,
    int* outUnitsBefore,
    int* outUnitsAfter) {

  if (units.empty()) {
    if (outUnitsBefore) *outUnitsBefore = 0;
    if (outUnitsAfter) *outUnitsAfter = 0;
    return units;
  }

  std::vector<ReplayUnit> consolidated;
  consolidated.reserve(units.size());

  auto prev = units[0];
  int unitsBefore = (int)units.size();

  for (size_t i = 1; i < units.size(); i++) {
    const auto& curr = units[i];

    // Check profitability gate
    int projectedAfter = (int)consolidated.size() + (int)(units.size() - i);
    auto decision = evaluateConsolidation(prev, curr, unitsBefore, projectedAfter);

    if (decision.approved) {
      // Check explicit fusion rules for prep units
      const char* ruleName = nullptr;
      if (canFusePrepUnits(prev, curr, slots, &ruleName)) {
        DSP_DIAG(EXECUTE, "CONSOLIDATE: fused units [%d-%d]+[%d-%d] via rule '%s' -> single unit",
                 prev.startSlot, prev.endSlot, curr.startSlot, curr.endSlot,
                 ruleName ? ruleName : "generic");
      }

      // Merge: extend prev to cover curr's range
      prev.endSlot = curr.endSlot;
      // Keep the primary op category of the first unit for diagnostics
    } else {
      // Cannot merge — emit prev and start new unit
      consolidated.push_back(prev);
      prev = curr;
    }
  }
  // Emit final unit
  consolidated.push_back(prev);

  if (outUnitsBefore) *outUnitsBefore = unitsBefore;
  if (outUnitsAfter) *outUnitsAfter = (int)consolidated.size();

  if (unitsBefore > (int)consolidated.size()) {
    DSP_DIAG(EXECUTE, "CONSOLIDATION: %d units -> %d units (saved %d)",
             unitsBefore, (int)consolidated.size(), unitsBefore - (int)consolidated.size());
  }

  return consolidated;
}

// ── Attention tail specialization ──────────────────────────────────────────
//
// Special-cases attention tails after ordered replay is correct.
// Relevant op types:
//   onnx_multi_head_attention, gather, concat, stack, broadcast_to,
//   reshape_no_copy, permute, expand_dims, CONST_GEN
//
// Strategy:
//   1. Absorb view-only and shape-only prep into the attention lowering.
//   2. Only keep a materializing prep step if attention truly requires
//      dense repeated K/V.
//
// This is the high-ROI optimization — attention tails are the most common
// source of invalid segment buckets.

struct AttentionTailPrep {
  // View-only prep ops that can be absorbed into attention lowering
  std::vector<int> viewOpSlots;       // reshape, permute, expand_dims, squeeze
  std::vector<int> shapeOpSlots;      // shape_of, create, range, ones_as
  // Materializing prep ops that may need to execute before attention
  std::vector<int> materializeSlots;  // gather, concat, stack, tile, broadcast_to
  int attentionSlot;                   // The attention op slot index
};

// Check if a slot range contains an attention tail pattern.
// An attention tail is: [prep ops...] -> onnx_multi_head_attention -> [optional trailing ops]
static bool isAttentionTailPattern(int startSlot, int endSlot, NativeSlot* slots,
                                   AttentionTailPrep& outPrep) {
  outPrep = AttentionTailPrep();
  outPrep.attentionSlot = -1;

  // Find the attention op in this range
  for (int s = startSlot; s <= endSlot; s++) {
    if (slotIsAttentionOp(s, slots)) {
      outPrep.attentionSlot = s;
      break;
    }
  }

  if (outPrep.attentionSlot < 0) return false;

  // Classify all slots before the attention op as prep
  for (int s = startSlot; s < outPrep.attentionSlot; s++) {
    if (slotIsViewRecipeOp(s, slots)) {
      outPrep.viewOpSlots.push_back(s);
    } else if (slotIsShapeRecipeOp(s, slots) || slotIsConstantGenerationOp(s, slots)) {
      outPrep.shapeOpSlots.push_back(s);
    } else if (slotWouldMaterialize(s, slots)) {
      outPrep.materializeSlots.push_back(s);
    }
  }

  return true;
}

// Check if a materializing prep op is actually needed by attention.
// Until the attention ABI can prove a prep unit is absorbable, keep all
// materializing prep explicit and ordered.
static bool isMaterializingPrepNeeded(int slotIdx, NativeSlot* slots, NDArray** outputSlots) {
  (void)outputSlots;
  return slotWouldMaterialize(slotIdx, slots);
}

// ── View recipe capture, validation, and installation ──────────────────────

// Map an op name to a ViewRecipeType using the category table for semantic
// classification, with string matching only for sub-type disambiguation.
static ViewRecipeType opToViewRecipeType(const char* opName) {
  std::string lower(opName ? opName : "");
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

  if (lower == "reshape_no_copy") return ViewRecipeType::RESHAPE_NO_COPY;
  if (lower == "reshape" || lower == "flatten" || lower == "flatten_2d")
    return ViewRecipeType::RESHAPE;
  if (lower == "permute" || lower.find("transpose") != std::string::npos)
    return ViewRecipeType::PERMUTE;
  if (lower == "expand_dims") return ViewRecipeType::EXPAND_DIMS;
  if (lower == "squeeze") return ViewRecipeType::SQUEEZE;
  if (lower == "strided_slice") return ViewRecipeType::STRIDED_SLICE;

  return ViewRecipeType::RESHAPE;  // Default recipe kind for view-compatible ops
}

// Check if the input array is C-contiguous and suitable for view creation.
static bool isViewCompatibleInput(NDArray* input) {
  if (input == nullptr || input->dataBuffer() == nullptr) return false;
  // View requires C-contiguous layout for simple offset + stride math
  return input->ordering() == 'c';
}

static bool resolveViewRecipeOwnerSlot(int outputSlotIdx,
                                       const SlotBufferInfo* slotOwnership,
                                       int totalOutputSlots,
                                       int& outOwnerSlotIdx) {
  outOwnerSlotIdx = outputSlotIdx;
  if (slotOwnership == nullptr || outputSlotIdx < 0 || outputSlotIdx >= totalOutputSlots) {
    return false;
  }

  std::vector<uint8_t> seen(static_cast<size_t>(totalOutputSlots), 0);
  int current = outputSlotIdx;
  while (current >= 0 && current < totalOutputSlots &&
         slotOwnership[current].ownership == BufferOwnership::VIEW_OF_SLOT) {
    if (seen[current]) {
      return false;
    }
    seen[current] = 1;

    int parent = slotOwnership[current].parentSlotIdx;
    if (parent < 0 || parent >= totalOutputSlots) {
      return false;
    }
    current = parent;
  }

  outOwnerSlotIdx = current;
  return current >= 0 && current < totalOutputSlots;
}

// Capture a view recipe for a single view-producing slot.
// Returns true if the recipe was successfully captured, false if the op
// would materialize (hard error path — caller should fail the segment).
static bool captureViewRecipe(int slotIdx, NativeSlot* slots, NDArray** outputSlots,
                              NDArray** externalInputs, int numExternalInputs,
                              SlotBufferInfo* slotOwnership, int totalOutputSlots,
                              ViewRecipe& outRecipe, std::string& outError) {
  outRecipe = ViewRecipe();
  outRecipe.outputSlotIndex = -1;

  if (slots == nullptr) {
    outError = "null slots array";
    return false;
  }

  const std::string& opNameStr3 = slots[slotIdx].ident.opName;
  if (opNameStr3.empty()) {
    outError = "no op name";
    return false;
  }

  if (!slotIsViewRecipeOp(slotIdx, slots)) {
    outError = "op is not view-capable (would materialize)";
    return false;
  }

  outRecipe.type = opToViewRecipeType(opNameStr3.c_str());

  // Find the output slot index for this slot
  if (slots[slotIdx].wiring.numOutputs > 0) {
    outRecipe.outputSlotIndex = slots[slotIdx].wiring.outputSlotIndices[0];
  }

  // Resolve the source input for this view op. For replay we need the
  // actual owning buffer observed during warmup, not just the logical input
  // edge, otherwise view chains can reinstall against an intermediate array
  // whose buffer is smaller than the captured output view.
  int sourceIdx = slots[slotIdx].wiring.inputSourceIndices[0];
  if (outRecipe.outputSlotIndex >= 0 && outRecipe.outputSlotIndex < totalOutputSlots &&
      slotOwnership != nullptr &&
      slotOwnership[outRecipe.outputSlotIndex].ownership == BufferOwnership::VIEW_OF_SLOT) {
    int ownerSlotIdx = -1;
    if (!resolveViewRecipeOwnerSlot(outRecipe.outputSlotIndex, slotOwnership,
                                    totalOutputSlots, ownerSlotIdx)) {
      outError = "view ownership chain is invalid or cyclic";
      return false;
    }
    sourceIdx = ownerSlotIdx;
  }
  outRecipe.sourceSlotIndex = sourceIdx;

  NDArray* source = nullptr;
  if (sourceIdx < 0) {
    int extIdx = -(sourceIdx + 1);
    if (extIdx >= 0 && extIdx < numExternalInputs) {
      source = externalInputs[extIdx];
    }
  } else if (sourceIdx >= 0) {
    source = outputSlots[sourceIdx];
  }

  if (source == nullptr || source->dataBuffer() == nullptr) {
    outError = "source array is null or has no data buffer";
    return false;
  }

  if (!isViewCompatibleInput(source)) {
    outError = "source is not C-contiguous — view not possible, must materialize";
    return false;
  }

  NDArray* out = nullptr;
  if (outputSlots != nullptr && outRecipe.outputSlotIndex >= 0) {
    out = outputSlots[outRecipe.outputSlotIndex];
  }
  if (out == nullptr || out->dataBuffer() == nullptr) {
    outError = "output array is null; cannot capture exact replay view state";
    return false;
  }

  if (out->dataBuffer() != source->dataBuffer()) {
    outError = "observed output did not alias the captured source buffer";
    return false;
  }

  const LongType outputOffset = out->offset();
  const LongType outputLength = out->lengthOf();
  const LongType sourceBufferElems = source->dataBuffer()->getNumElements();
  if (outputOffset < 0 || outputLength < 0 || outputOffset > sourceBufferElems ||
      outputLength > (sourceBufferElems - outputOffset)) {
    outError = "observed output view exceeds source buffer capacity";
    return false;
  }

  // Capture the source buffer address and size only after proving the
  // observed output really was a zero-copy alias of that source buffer.
  outRecipe.capturedSourceAddr = DSP_BUF(source);
  outRecipe.capturedSourceBytes = source->dataBuffer()->getLenInBytes();

  // Capture output shape from the observed output array.
  outRecipe.rank = std::min((int)out->rankOf(), (int)ViewRecipe::MAX_SHAPE_RANK);
  outRecipe.outputOrder = out->ordering();
  outRecipe.outputEws = out->ews();
  outRecipe.outputOffset = outputOffset;
  outRecipe.outputExtra = ArrayOptions::extra(out->shapeInfo());
  for (int i = 0; i < outRecipe.rank; i++) {
    outRecipe.outputShape[i] = out->sizeAt(i);
    outRecipe.outputStrides[i] = out->strideAt(i);
  }
  // Note: slot-level shape info is not available here — rely on output array

  // For permute: capture the permutation vector from tArgs
  if (outRecipe.type == ViewRecipeType::PERMUTE && slots[slotIdx].args.tArgs != nullptr) {
    int permRank = std::min(slots[slotIdx].args.numTArgs, (int)ViewRecipe::MAX_SHAPE_RANK);
    for (int i = 0; i < permRank; i++) {
      outRecipe.perm[i] = (int)slots[slotIdx].args.tArgs[i];
    }
  }

  // For strided_slice: capture begin/end/stride
  if (outRecipe.type == ViewRecipeType::STRIDED_SLICE) {
    outRecipe.sliceRank = outRecipe.rank;
    for (int i = 0; i < outRecipe.sliceRank; i++) {
      outRecipe.sliceBegin[i] = (slots[slotIdx].args.iArgs != nullptr && i < slots[slotIdx].args.numIArgs)
                                ? slots[slotIdx].args.iArgs[i] : 0;
      outRecipe.sliceEnd[i] = (slots[slotIdx].args.tArgs != nullptr && i < slots[slotIdx].args.numTArgs)
                              ? (LongType)slots[slotIdx].args.tArgs[i] : outRecipe.outputShape[i];
      outRecipe.sliceStride[i] = 1;  // Default stride
    }
  }

  outRecipe.validated = false;  // Will be validated during POINTERS_STABLE
  return true;
}

static LongType* buildCapturedViewShapeInfo(const ViewRecipe& recipe, NDArray* source) {
  if (source == nullptr) return nullptr;

  LongType* shapeInfo = ShapeBuilders::createShapeInfo(
      source->dataType(),
      recipe.outputOrder,
      recipe.rank,
      recipe.outputShape,
      recipe.outputStrides,
      nullptr,
      recipe.outputExtra);
  if (shapeInfo == nullptr) return nullptr;

  auto len = shape::shapeInfoLength(shapeInfo);
  shapeInfo[len - 2] = recipe.outputEws;
  shape::setOrder(shapeInfo, recipe.outputOrder);
  return shapeInfo;
}

// Capture view recipes for all view-capable gap slots in a segment.
// Returns the number of recipes captured.
static int captureViewRecipesForSegment(GraphSegment& seg, NativeSlot* slots,
                                        NDArray** externalInputs, int numExternalInputs,
                                        NDArray** outputSlots,
                                        SlotBufferInfo* slotOwnership,
                                        int totalOutputSlots) {
  seg.exec.viewRecipes = ViewRecipeChain();
  seg.exec.viewRecipes.segmentStartSlot = seg.def.startSlot;
  seg.exec.viewRecipes.segmentEndSlot = seg.def.endSlot;

  int captured = 0;
  for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
    if (!slotIsViewRecipeOp(s, slots)) continue;

    ViewRecipe recipe;
    std::string error;
    if (captureViewRecipe(s, slots, outputSlots, externalInputs, numExternalInputs,
                          slotOwnership, totalOutputSlots,
                          recipe, error)) {
      seg.exec.viewRecipes.recipes.push_back(std::move(recipe));
      captured++;
    } else {
      DSP_DIAG_SEG(FALLBACK, s, "VIEW_RECIPE_FAIL: seg[%d-%d] slot %d: %s",
                   seg.def.startSlot, seg.def.endSlot, s, error.c_str());
    }
  }

  return captured;
}

// Validate view recipes during POINTERS_STABLE phase.
// Checks that source buffer addresses haven't changed since capture.
// Returns true if all recipes are valid, false if any source changed.
static bool validateViewRecipes(GraphSegment& seg, NDArray** outputSlots,
                                NDArray** externalInputs, int numExternalInputs) {
  for (auto& recipe : seg.exec.viewRecipes.recipes) {
    NDArray* source = nullptr;
    int sourceIdx = recipe.sourceSlotIndex;
    if (sourceIdx < 0) {
      int extIdx = -(sourceIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs) {
        source = externalInputs[extIdx];
      }
    } else {
      source = outputSlots[sourceIdx];
    }

    if (source == nullptr || source->dataBuffer() == nullptr) {
      DSP_DIAG(FALLBACK, "VIEW_RECIPE_VALIDATE: seg[%d-%d] slot %d source is null",
               seg.def.startSlot, seg.def.endSlot, recipe.outputSlotIndex);
      return false;
    }

    void* currentAddr = DSP_BUF(source);
    if (currentAddr != recipe.capturedSourceAddr) {
      DSP_DIAG(FALLBACK, "VIEW_RECIPE_VALIDATE: seg[%d-%d] slot %d source addr changed: "
               "captured=%p current=%p",
               seg.def.startSlot, seg.def.endSlot, recipe.outputSlotIndex,
               recipe.capturedSourceAddr, currentAddr);
      return false;
    }

    LongType currentSourceBytes = source->dataBuffer()->getLenInBytes();
    if (currentSourceBytes != recipe.capturedSourceBytes) {
      DSP_DIAG(EXECUTE, "VIEW_RECIPE_VALIDATE: seg[%d-%d] slot %d source bytes changed: "
               "captured=%lld current=%lld sourceSlot=%d",
               seg.def.startSlot, seg.def.endSlot, recipe.outputSlotIndex,
               static_cast<long long>(recipe.capturedSourceBytes),
               static_cast<long long>(currentSourceBytes),
               recipe.sourceSlotIndex);
      return false;
    }

    recipe.validated = true;
  }

  return true;
}

// Install view recipes during REPLAYING phase.
// Creates zero-copy views from captured recipes and installs them into
// outputSlots so that consumer replay ops see the correct buffers.
static void installViewRecipes(GraphSegment& seg, NDArray** outputSlots,
                               int totalOutputSlots, NDArray** externalInputs,
                               int numExternalInputs) {
  for (const auto& recipe : seg.exec.viewRecipes.recipes) {
    if (!recipe.validated) continue;

    NDArray* source = nullptr;
    int sourceIdx = recipe.sourceSlotIndex;
    if (sourceIdx < 0) {
      int extIdx = -(sourceIdx + 1);
      if (extIdx >= 0 && extIdx < numExternalInputs) {
        source = externalInputs[extIdx];
      }
    } else {
      source = outputSlots[sourceIdx];
    }

    if (source == nullptr || source->dataBuffer() == nullptr) continue;

    int outputSlotIdx = recipe.outputSlotIndex;
    if (outputSlotIdx < 0 || outputSlotIdx >= totalOutputSlots) continue;

    LongType* shapeInfo = buildCapturedViewShapeInfo(recipe, source);
    if (shapeInfo == nullptr) {
      DSP_DIAG(EXECUTE, "VIEW_RECIPE_INSTALL_FAIL: seg[%d-%d] slot %d shapeInfo build failed",
               seg.def.startSlot, seg.def.endSlot, outputSlotIdx);
      continue;
    }

    LongType viewOffset = recipe.outputOffset;
    LongType viewLength = shape::length(shapeInfo);
    LongType sourceBufferElems = source->dataBuffer()->getNumElements();
    if (viewOffset < 0 || viewOffset > sourceBufferElems ||
        viewLength > (sourceBufferElems - viewOffset)) {
      delete[] shapeInfo;
      std::string msg = "DSP replay phase violation: view recipe for seg[" +
          std::to_string(seg.def.startSlot) + "-" + std::to_string(seg.def.endSlot) +
          "] output slot " + std::to_string(outputSlotIdx) +
          " requires " + std::to_string(viewLength) +
          " elements at offset " + std::to_string(viewOffset) +
          " from source slot " + std::to_string(recipe.sourceSlotIndex) +
          ", but source buffer has only " + std::to_string(sourceBufferElems) +
          " elements";
      THROW_EXCEPTION(msg.c_str());
    }

    // Create the view array
    NDArray* view = new NDArray(source->dataBuffer(),
                                shapeInfo,
                                LaunchContext::defaultContext(),
                                viewOffset);
    delete[] shapeInfo;
    if (view != nullptr) {
      // Install into output slot
      if (outputSlots[outputSlotIdx] != nullptr &&
          outputSlots[outputSlotIdx] != source) {
        delete outputSlots[outputSlotIdx];
      }
      outputSlots[outputSlotIdx] = view;
    }
  }
}

// Default capture host workspace size for Triton path (32MB, same as non-Triton path).
// Configurable via ND4J_DSP_CAPTURE_HOST_WORKSPACE_MB env var.
#ifdef SD_CUDA
static size_t TRITON_CAPTURE_HOST_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureHostWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();

// Default capture workspace size for Triton graph capture (128MB).
// Configurable via ND4J_DSP_CAPTURE_WORKSPACE_MB env var.
static size_t TRITON_CAPTURE_WORKSPACE_SIZE = []() -> size_t {
  size_t mb = static_cast<size_t>(Environment::getInstance().dsp().captureWorkspaceMb());
  return mb * 1024ULL * 1024ULL;
}();
#endif

// Local helper: convert Status enum to human-readable string for diagnostics.
static const char* statusName_gpu(Status status) {
  switch (status) {
    case Status::OK: return "OK";
    case Status::BAD_INPUT: return "BAD_INPUT";
    case Status::BAD_SHAPE: return "BAD_SHAPE";
    case Status::BAD_RANK: return "BAD_RANK";
    case Status::BAD_PARAMS: return "BAD_PARAMS";
    case Status::BAD_OUTPUT: return "BAD_OUTPUT";
    case Status::BAD_RNG: return "BAD_RNG";
    case Status::BAD_EPSILON: return "BAD_EPSILON";
    case Status::BAD_GRADIENTS: return "BAD_GRADIENTS";
    case Status::BAD_BIAS: return "BAD_BIAS";
    case Status::VALIDATION: return "VALIDATION";
    case Status::BAD_GRAPH: return "BAD_GRAPH";
    case Status::BAD_LENGTH: return "BAD_LENGTH";
    case Status::BAD_DIMENSIONS: return "BAD_DIMENSIONS";
    case Status::BAD_ORDER: return "BAD_ORDER";
    case Status::BAD_ARGUMENTS: return "BAD_ARGUMENTS";
    case Status::DOUBLE_WRITE: return "DOUBLE_WRITE";
    case Status::DOUBLE_READ: return "DOUBLE_READ";
    case Status::KERNEL_FAILURE: return "KERNEL_FAILURE";
    case Status::EQ_TRUE: return "EQ_TRUE";
    case Status::EQ_FALSE: return "EQ_FALSE";
    case Status::MAYBE: return "MAYBE";
    default: return "UNKNOWN";
  }
}

// Helper: extract specialBuffer() device addresses from NDArray** into void** for
// address snapshot diagnostics. Thread-local to avoid repeated allocation.
static void extractDeviceAddrs(NDArray** arrays, int count, std::vector<void*>& out) {
  out.resize(count);
  for (int i = 0; i < count; i++) {
    out[i] = (arrays != nullptr && arrays[i] != nullptr)
             ? DSP_BUF(arrays[i]) : nullptr;
  }
}

/**
 * Compute FNV-1a hash of slot output specialBuffer() addresses for a segment.
 * Used to verify that output buffers haven't been reallocated between capture
 * and replay — stale addresses in a CUDA graph cause SIGSEGV or corruption.
 */
static LongType computeSlotAddrHash(NDArray** outputSlots, int startSlot, int endSlot, int totalSlots) {
  return dsp::computeSlotAddrHash(outputSlots, startSlot, endSlot, totalSlots,
      [](NDArray* a) -> void* { return DSP_BUF(a); });
}

#ifdef SD_CUDA
static bool isCurrentDevicePointer(void* ptr, int currentDeviceId) {
  if (ptr == nullptr) return false;

  cudaPointerAttributes attrs;
  auto res = cudaPointerGetAttributes(&attrs, ptr);
  if (res != cudaSuccess) {
    cudaGetLastError();
    return false;
  }

  return attrs.type == cudaMemoryTypeDevice && attrs.device == currentDeviceId;
}
#else
static bool isCurrentDevicePointer(void* /*ptr*/, int /*currentDeviceId*/) {
  return false;
}
#endif

// ── GPU CONTEXT PROBE ──────────────────────────────────────────────────────
// Shared helper that dumps multi-device memory state + CUDA context health.
// Called by all error handlers to detect downstream/pre-existing errors.
#ifdef SD_CUDA
static void dumpGpuContextState(int failedDeviceId, const char* errorType) {
  // 1. Check for pre-existing CUDA error (downstream error detection)
  cudaError_t preExisting = cudaPeekAtLastError();
  if (preExisting != cudaSuccess) {
    DSP_DIAG(MEMORY, "%s: PRE-EXISTING CUDA ERROR on device %d: %d (%s) — "
             "this may be a downstream error from a previous operation",
             errorType, failedDeviceId,
             static_cast<int>(preExisting), cudaGetErrorString(preExisting));
  }

  // 2. Failed device: full detail
  {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, failedDeviceId);
    size_t gpuFree = 0, gpuTotal = 0;
    cudaSetDevice(failedDeviceId);
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    DSP_DIAG(MEMORY, "%s: device %d '%s' (cc=%d.%d): free=%zuMB total=%zuMB used=%zuMB "
             "multiProcessorCount=%d",
             errorType, failedDeviceId, props.name,
             props.major, props.minor,
             gpuFree / (1024*1024), gpuTotal / (1024*1024),
             (gpuTotal - gpuFree) / (1024*1024),
             props.multiProcessorCount);
  }

  // 3. Other devices: one-line summary each
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  for (int d = 0; d < deviceCount; d++) {
    if (d == failedDeviceId) continue;
    cudaSetDevice(d);
    size_t otherFree = 0, otherTotal = 0;
    cudaMemGetInfo(&otherFree, &otherTotal);
    cudaError_t otherErr = cudaPeekAtLastError();
    DSP_DIAG(MEMORY, "%s: device %d: free=%zuMB total=%zuMB%s",
             errorType, d, otherFree / (1024*1024), otherTotal / (1024*1024),
             otherErr != cudaSuccess ?
               (std::string(" CUDA_ERROR=") + cudaGetErrorString(otherErr)).c_str() : "");
  }

  // 4. Restore original device
  cudaSetDevice(failedDeviceId);

  // 5. Report graph execution active state
  DSP_DIAG(MEMORY, "%s: tl_graphExecutionActive=%d",
           errorType, tl_graphExecutionActive ? 1 : 0);
}

// ── DISTINCT ERROR HANDLERS ────────────────────────────────────────────────

static Status reportOomError(GraphSegment& seg, const char* phase,
                             size_t requestedBytes, int deviceId) {
  dumpGpuContextState(deviceId, "OOM");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY,
    "OOM ERROR in seg[%d-%d] during '%s' on device %d: "
    "requested=%zuMB gpuFree=%zuMB gpuTotal=%zuMB gpuUsed=%zuMB "
    "executionCount=%d phase=%d",
    seg.def.startSlot, seg.def.endSlot, phase, deviceId,
    requestedBytes / (1024*1024), gpuFree / (1024*1024),
    gpuTotal / (1024*1024), (gpuTotal - gpuFree) / (1024*1024),
    seg.exec.executionCount, static_cast<int>(seg.exec.currentPhase));
  seg.exec.compilationFailed = true;
  return Status::KERNEL_FAILURE;
}

static Status reportCaptureError(GraphSegment& seg, const char* step,
                                 cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "CAPTURE");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
    "CAPTURE ERROR in seg[%d-%d] at step '%s' on device %d: "
    "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
    "executionCount=%d numOps=%d compiledBy='%s'",
    seg.def.startSlot, seg.def.endSlot, step, deviceId,
    static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
    gpuFree / (1024*1024), gpuTotal / (1024*1024),
    seg.exec.executionCount, seg.def.endSlot - seg.def.startSlot + 1,
    seg.exec.compiledByBackend.c_str());
  seg.exec.compilationFailed = true;
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

static Status reportReplayError(GraphSegment& seg, const char* step,
                                cudaError_t cudaErr, int deviceId) {
  dumpGpuContextState(deviceId, "REPLAY");
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(EXECUTE,
    "REPLAY ERROR in seg[%d-%d] at step '%s' on device %d: "
    "cudaError=%d (%s) gpuFree=%zuMB gpuTotal=%zuMB "
    "executionCount=%d hasReplayHandle=%d",
    seg.def.startSlot, seg.def.endSlot, step, deviceId,
    static_cast<int>(cudaErr), cudaGetErrorString(cudaErr),
    gpuFree / (1024*1024), gpuTotal / (1024*1024),
    seg.exec.executionCount,
    seg.exec.replayHandle != nullptr ? 1 : 0);
  seg.exec.compilationFailed = true;
  cudaGetLastError(); // clear error state
  return Status::KERNEL_FAILURE;
}

#if HAVE_TRITON && defined(SD_CUDA)
static bool findUnsupportedTritonReplayGap(TritonGraphBackend* tritonBackend,
                                           const GraphSegment& seg,
                                           NativeSlot* slots,
                                           int* firstGapSlot,
                                           int* lastCoveredSlot,
                                           int* gapSlotCount) {
  if (firstGapSlot != nullptr) *firstGapSlot = -1;
  if (lastCoveredSlot != nullptr) *lastCoveredSlot = -1;
  if (gapSlotCount != nullptr) *gapSlotCount = 0;
  if (tritonBackend == nullptr) return false;

  auto gapSlots = tritonBackend->getGapSlots(seg, slots);
  if (gapSlotCount != nullptr) *gapSlotCount = static_cast<int>(gapSlots.size());
  if (gapSlots.empty()) return false;

  int maxCoveredSlot = -1;
  for (int slot = seg.def.startSlot; slot <= seg.def.endSlot; slot++) {
    if (gapSlots.find(slot) == gapSlots.end()) {
      maxCoveredSlot = slot;
    }
  }
  if (lastCoveredSlot != nullptr) *lastCoveredSlot = maxCoveredSlot;
  if (maxCoveredSlot < 0) return false;

  int earliestUnsupportedGap = -1;
  for (int slot = seg.def.startSlot; slot <= seg.def.endSlot; slot++) {
    if (gapSlots.find(slot) != gapSlots.end() && slot < maxCoveredSlot) {
      earliestUnsupportedGap = slot;
      break;
    }
  }
  if (firstGapSlot != nullptr) *firstGapSlot = earliestUnsupportedGap;
  return earliestUnsupportedGap >= 0;
}

/**
 * Build an ordered replay schedule for a mixed Triton/gap segment.
 *
 * Given a segment like seg[200-399] with gaps at [298-312] and [347-369],
 * this produces:
 *   unit 0: TRITON_ISLAND [200-297]  islandIndex=0
 *   unit 1: GAP            [298-312] islandIndex=-1
 *   unit 2: TRITON_ISLAND [313-346]  islandIndex=1
 *   unit 3: GAP            [347-369] islandIndex=-1
 *   unit 4: TRITON_ISLAND [370-399]  islandIndex=2
 */
static ReplaySchedule buildCompositeReplaySchedule(const GraphSegment& seg,
                                                    NativeSlot* slots,
                                                    TritonGraphBackend* tritonBackend) {
  ReplaySchedule schedule;
  auto gap_slots = tritonBackend->getGapSlots(seg, slots);

  int islandIdx = 0;
  int rangeStart = seg.def.startSlot;
  bool inIsland = (gap_slots.find(seg.def.startSlot) == gap_slots.end());

  for (int slot = seg.def.startSlot; slot <= seg.def.endSlot + 1; slot++) {
    bool isGap = (slot <= seg.def.endSlot && gap_slots.find(slot) != gap_slots.end());
    bool atBoundary = (slot > seg.def.endSlot) || (inIsland && isGap) || (!inIsland && !isGap);

    if (atBoundary && slot > rangeStart) {
      if (inIsland) {
        schedule.units.emplace_back(REPLAY_UNIT_TRITON_ISLAND, rangeStart, slot - 1, islandIdx++);
      } else {
        schedule.units.emplace_back(REPLAY_UNIT_GAP, rangeStart, slot - 1, -1);
      }
      rangeStart = slot;
      inIsland = !isGap;
    }
    // If at seg.def.endSlot+1 boundary with pending range, the loop above handles it
    if (slot == seg.def.endSlot + 1 && rangeStart <= seg.def.endSlot) {
      if (inIsland) {
        schedule.units.emplace_back(REPLAY_UNIT_TRITON_ISLAND, rangeStart, seg.def.endSlot, islandIdx++);
      } else {
        schedule.units.emplace_back(REPLAY_UNIT_GAP, rangeStart, seg.def.endSlot, -1);
      }
    }
  }

  // Pre-allocate replay handles for each island
  schedule.compositeReplayHandles.resize(schedule.units.size());
  return schedule;
}
#endif

// ── LRU GRAPH EVICTION ──────────────────────────────────────────────────────
// Evicts captured graphs to free GPU memory. Returns number of graphs evicted.
// When dspLruEviction is true, evicts least-recently-replayed graphs first.
// Otherwise evicts smallest (fewest nodes) first (legacy behavior).
int NativeDynamicShapePlan::evictLruGraphs(int segIdx, size_t neededBytes, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  bool lruMode = Environment::getInstance().dspLruEviction();
  int maxEvictions = Environment::getInstance().dspCaptureOomMaxRetries();
  int numEvicted = 0;

  for (int evictAttempt = 0; evictAttempt < maxEvictions; evictAttempt++) {
    // Check if we have enough memory already
    size_t gpuFree = 0, gpuTotal = 0;
    cudaMemGetInfo(&gpuFree, &gpuTotal);
    if (gpuFree >= neededBytes) {
      DSP_DIAG(MEMORY, "evictLruGraphs: have enough memory after %d evictions (%zuMB free >= %zuMB needed)",
               numEvicted, gpuFree / (1024*1024), neededBytes / (1024*1024));
      break;
    }

    // Find the best candidate to evict
    int evictIdx = -1;
    if (lruMode) {
      // LRU: find segment with smallest lastReplayExecCount (least recently used)
      int lruExecCount = INT_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
        if (candidate.exec.lastReplayExecCount < lruExecCount) {
          lruExecCount = candidate.exec.lastReplayExecCount;
          evictIdx = static_cast<int>(si);
        }
      }
    } else {
      // Smallest-first: find segment with fewest CUDA graph nodes
      size_t smallestNodes = SIZE_MAX;
      for (size_t si = 0; si < segments_.size(); si++) {
        if (static_cast<int>(si) == segIdx) continue;
        auto& candidate = segments_[si];
        if (!candidate.exec.replayHandle || !candidate.exec.replayHandle->isReady()) continue;
        auto* cudaReplay = dynamic_cast<CudaGraphReplayHandle*>(candidate.exec.replayHandle.get());
        size_t nodeCount = cudaReplay ? cudaReplay->getNumNodes() : 1;
        if (nodeCount == 0) nodeCount = 1;
        if (nodeCount < smallestNodes) {
          smallestNodes = nodeCount;
          evictIdx = static_cast<int>(si);
        }
      }
    }

    if (evictIdx < 0) {
      DSP_DIAG(MEMORY, "evictLruGraphs: no more evictable segments (evicted %d)", numEvicted);
      break;
    }

    // Evict the selected segment
    auto& evictSeg = segments_[evictIdx];
    DSP_DIAG(MEMORY, "evictLruGraphs: evicting seg[%d-%d] (lruExec=%d, mode=%s) for seg idx=%d (attempt %d/%d)",
             evictSeg.def.startSlot, evictSeg.def.endSlot, evictSeg.exec.lastReplayExecCount,
             lruMode ? "LRU" : "smallest", segIdx, evictAttempt + 1, maxEvictions);

    evictSeg.exec.replayHandle->releaseWorkspace(nullptr, evictSeg.def.startSlot);

    // Free pinned host pointers
    evictSeg.exec.replayHandle->freeHostPointers();
    evictSeg.exec.replayHandle->clearExternalAddresses();

    // Destroy replay handle (frees cudaGraphExec + cudaGraph)
    evictSeg.exec.replayHandle.reset();

    // Reset evicted segment for future re-capture
    evictSeg.exec.cachedShapeKey = 0;
    evictSeg.exec.capturedInputAddrKey = 0;
    evictSeg.exec.capturedCreateValueKey = 0;
    evictSeg.exec.compilationFailed = false;
    evictSeg.exec.gapOpsCapturedInGraph = false;
    evictSeg.exec.argTableStable = false;
    evictSeg.exec.compiledByBackend.clear();
    evictSeg.exec.executionCount = 0;
    evictSeg.exec.lastReplayExecCount = 0;

    numEvicted++;

    // Sync to ensure GPU memory is freed
    if (cudaStr != nullptr) {
      cudaStreamSynchronize(cudaStr);
    }
    cudaGetLastError();
  }

  // Final pool trim after evictions
  if (numEvicted > 0) {
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
    DSP_DIAG(MEMORY, "evictLruGraphs: evicted %d segments, trimmed pool on device %d", numEvicted, deviceId);
  }

  return numEvicted;
}

// ── PROACTIVE PRE-CAPTURE MEMORY CLEANUP ───────────────────────────────────
// Called before workspace allocation when about to capture a graph.
// Frees cached-but-unused GPU memory and evicts LRU graphs if needed.
void NativeDynamicShapePlan::proactivePreCaptureMemoryCleanup(GraphSegment& seg, int segIdx, void* stream) {
  auto cudaStr = (stream != nullptr) ? *static_cast<cudaStream_t*>(stream) : nullptr;
  int deviceId = 0;
  cudaGetDevice(&deviceId);

  // 1. Trim CUDA memory pool — cheap, can reclaim hundreds of MB
  DSP_DIAG(MEMORY, "proactive cleanup: trimming pool on device %d for seg[%d-%d]",
           deviceId, seg.def.startSlot, seg.def.endSlot);
  memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  if (cudaStr != nullptr) {
    memory::CudaMemoryPool::getInstance().trimPoolOnStream(deviceId, cudaStr);
  }

  // 2. Check if we have enough memory
  size_t gpuFree = 0, gpuTotal = 0;
  cudaMemGetInfo(&gpuFree, &gpuTotal);

  // Estimate needed: capture workspace + cuBLAS workspace (if not allocated) + safety margin
  size_t neededBytes = 0;
  if (sharedCaptureWorkspace_ == nullptr) {
    neededBytes += TRITON_CAPTURE_WORKSPACE_SIZE;  // 128MB default
  }
  if (cublasWorkspaceBuffer_ == nullptr) {
    neededBytes += Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
  }
  neededBytes += Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;

  DSP_DIAG(MEMORY, "proactive cleanup: gpuFree=%zuMB, needed=%zuMB (ws=%zuMB, cublas=%zuMB, safety=%dMB) for seg[%d-%d]",
           gpuFree / (1024*1024), neededBytes / (1024*1024),
           (sharedCaptureWorkspace_ == nullptr ? TRITON_CAPTURE_WORKSPACE_SIZE : 0) / (1024*1024),
           (cublasWorkspaceBuffer_ == nullptr ? (size_t)(Environment::getInstance().dspCublasWorkspaceMb()) : 0),
           Environment::getInstance().dspGraphMetadataSafetyMb(),
           seg.def.startSlot, seg.def.endSlot);

  if (gpuFree >= neededBytes) {
    DSP_DIAG(MEMORY, "proactive cleanup: sufficient memory (%zuMB >= %zuMB), no eviction needed",
             gpuFree / (1024*1024), neededBytes / (1024*1024));
    return;
  }

  // 3. LRU eviction
  DSP_DIAG(MEMORY, "proactive cleanup: insufficient memory (%zuMB < %zuMB), starting LRU eviction",
           gpuFree / (1024*1024), neededBytes / (1024*1024));
  int numEvicted = evictLruGraphs(segIdx, neededBytes, stream);

  // 4. Final trim after evictions
  if (numEvicted > 0) {
    memory::CudaMemoryPool::getInstance().trimPool(deviceId);
  }

  // Log final state
  cudaMemGetInfo(&gpuFree, &gpuTotal);
  DSP_DIAG(MEMORY, "proactive cleanup complete: evicted=%d, gpuFree=%zuMB/%zuMB for seg[%d-%d]",
           numEvicted, gpuFree / (1024*1024), gpuTotal / (1024*1024),
           seg.def.startSlot, seg.def.endSlot);
}

#endif  // SD_CUDA

// isStrictNoFallbackMode_gpu removed — all modes now throw on failure.
// There is no "non-strict" mode. Failures crash loudly, never fall back.

// ─── DSP Verify Helpers ────────────────────────────────────────────────────

// Source type name for diagnostics
static const char* sourceTypeName(int8_t st) {
  switch (static_cast<NativeSourceType>(st)) {
    case SOURCE_CONSTANT: return "CONSTANT";
    case SOURCE_VARIABLE: return "VARIABLE";
    case SOURCE_PLACEHOLDER: return "PLACEHOLDER";
    case SOURCE_OP_OUTPUT: return "OP_OUTPUT";
    default: return "UNKNOWN";
  }
}

#ifdef SD_CUDA
// Templated helpers in DspVerifyUtils.h (dspVerifyCopyValues, dspMaxDiff, dspFormatValues, etc.)
#endif  // SD_CUDA

void NativeDynamicShapePlan::clearGpuBackendFailedCache() {
#if HAVE_TRITON && defined(SD_CUDA)
  TritonGraphBackend::getInstance().clearFailedSegmentCache();
#endif
}

GraphBackend* NativeDynamicShapePlan::getGpuGraphBackend() {
  if (gpuGraphBackendChecked_) return gpuGraphBackend_;
  gpuGraphBackendChecked_ = true;

  // If a specific backend is forced via setGraphExecutionMode(), use only that one.
  // SLOT_BY_SLOT and graph-replay-only modes don't use a GPU compiler backend —
  // they rely on the GraphReplayHandle (CUDA/HIP/L0/Vulkan/Metal) for capture/replay.
  if (graphExecutionMode_ == GraphExecutionMode::GEM_SLOT_BY_SLOT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_CUDA_GRAPHS ||
      graphExecutionMode_ == GraphExecutionMode::GEM_HIP_GRAPHS ||
      graphExecutionMode_ == GraphExecutionMode::GEM_LEVELZERO ||
      graphExecutionMode_ == GraphExecutionMode::GEM_VULKAN ||
      graphExecutionMode_ == GraphExecutionMode::GEM_METAL) {
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }

#if HAVE_TRITON && defined(SD_CUDA)
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& triton = TritonGraphBackend::getInstance();
    if (triton.isAvailable()) {
      gpuGraphBackend_ = &triton;
      DSP_DIAG(BACKEND, "using Triton GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
      DSP_DIAG(BACKEND, "Triton backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
    DSP_DIAG(BACKEND, "Triton unavailable in AUTO mode, trying NVRTC/PTX backends");
  }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TRITON) {
    DSP_DIAG(BACKEND, "Triton backend requested but not compiled (HAVE_TRITON=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
  if (graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    DSP_DIAG(BACKEND, "Triton not compiled (HAVE_TRITON=0); AUTO mode will try NVRTC/PTX/CUDA graphs");
  }
#endif

#ifdef SD_CUDA
  if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& nvrtc = NvrtcGraphBackend::getInstance();
    if (nvrtc.isAvailable()) {
      gpuGraphBackend_ = &nvrtc;
      DSP_DIAG(BACKEND, "using NVRTC GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_NVRTC_JIT) {
      DSP_DIAG(BACKEND, "NVRTC backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }

  if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& ptx = PtxGraphBackend::getInstance();
    if (ptx.isAvailable()) {
      gpuGraphBackend_ = &ptx;
      DSP_DIAG(BACKEND, "using PTX template GPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_PTX_JIT) {
      DSP_DIAG(BACKEND, "PTX backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#endif

#ifdef SD_TPU
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& tpu = TpuGraphBackend::getInstance();
    if (tpu.isAvailable()) {
      gpuGraphBackend_ = &tpu;
      DSP_DIAG(BACKEND, "using TPU HLO compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU) {
      DSP_DIAG(BACKEND, "TPU backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_TPU) {
    DSP_DIAG(BACKEND, "TPU backend requested but not compiled (SD_TPU=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

#ifdef HAVE_HEXAGON_MLIR
  if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON ||
      graphExecutionMode_ == GraphExecutionMode::GEM_AUTO) {
    auto& hexagon = HexagonGraphBackend::getInstance();
    if (hexagon.isAvailable()) {
      gpuGraphBackend_ = &hexagon;
      DSP_DIAG(BACKEND, "using Hexagon MLIR NPU compiler backend");
      return gpuGraphBackend_;
    }
    if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON) {
      DSP_DIAG(BACKEND, "Hexagon backend requested but not available");
      gpuGraphBackend_ = nullptr;
      return nullptr;
    }
  }
#else
  if (graphExecutionMode_ == GraphExecutionMode::GEM_HEXAGON) {
    DSP_DIAG(BACKEND, "Hexagon backend requested but not compiled (HAVE_HEXAGON_MLIR=0)");
    gpuGraphBackend_ = nullptr;
    return nullptr;
  }
#endif

  gpuGraphBackend_ = nullptr;
  return nullptr;
}

Status NativeDynamicShapePlan::executeSegmentWithGpuGraph(
    GraphSegment& seg, NDArray** externalArrays, int numExt, void* stream) {

  // Derive segIdx for proactive eviction and OOM retry.
  int segIdx = -1;
  for (int si = 0; si < static_cast<int>(segments_.size()); si++) {
    if (&segments_[si] == &seg) { segIdx = si; break; }
  }

  {
    const char* mode = "unknown";
    if (seg.exec.executionCount == 0) mode = "warmup";
    else if (seg.exec.replayHandle != nullptr && seg.exec.replayHandle->isReady()) mode = "replay";
    else if (seg.exec.compilationFailed) mode = "slot-by-slot(failed)";
    else if (seg.exec.executionCount >= 1) mode = "capture-candidate";
    DSP_DIAG_SEG(SHAPE, seg.def.startSlot,
                 "executeSegmentWithGpuGraph: ENTER seg[%d-%d] mode=%s execCount=%d capturable=%d",
                 seg.def.startSlot, seg.def.endSlot, mode, seg.exec.executionCount, seg.def.isCapturable ? 1 : 0);
  }

#ifdef SD_CUDA
  // ── Segment lifecycle: SEG_ENTER ──────────────────────────────────────
  if (Environment::getInstance().tritonVerifyKernels()) {
    // Ensure VERIFY diagnostic category is enabled and output level is FULL
    // when tritonVerifyKernels is on (may be set at runtime via Java, after
    // DspDiagnostics constructor)
    if (!DSP_DIAG_ENABLED(VERIFY)) {
      sd::graph::DspDiagnostics::getInstance().enableCategories(sd::graph::DSP_DIAG_VERIFY);
      sd::graph::DspDiagnostics::getInstance().setLevel(sd::graph::DSP_LEVEL_FULL);
    }
    const char* mode = "unknown";
    if (seg.exec.executionCount == 0) mode = "warmup";
    else if (seg.exec.executionCount == 1) mode = "compile";
    else if (seg.exec.replayHandle != nullptr) mode = "replay";
    else if (seg.exec.compilationFailed) mode = "slot-by-slot";
    else mode = "capture";
    DSP_DIAG(VERIFY, "SEG_ENTER seg[%d-%d] execCount=%d mode=%s",
              seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount, mode);
    // Dump external input actuality flags for first N inputs
    int detailLimit = sd::graph::DspDiagnostics::getInstance().diagDetailLimit();
    int dumpCount = std::min(numExt, detailLimit);
    for (int i = 0; i < dumpCount; i++) {
      if (externalArrays[i] != nullptr && externalArrays[i]->dataBuffer() != nullptr) {
        auto* db = externalArrays[i]->dataBuffer();
        DSP_DIAG(VERIFY, "  EXT_INPUT[%d] dtype=%s len=%lld pAct=%d sAct=%d addr=%p",
                  i, DataTypeUtils::asString(externalArrays[i]->dataType()).c_str(),
                  (long long)externalArrays[i]->lengthOf(),
                  db->isPrimaryActual() ? 1 : 0, db->isSpecialActual() ? 1 : 0,
                  DSP_BUF(externalArrays[i]));
      }
    }
    if (numExt > detailLimit) {
      DSP_DIAG(VERIFY, "  ... and %d more external inputs", numExt - detailLimit);
    }
  }
#endif

  auto* backend = getGpuGraphBackend();
  if (backend == nullptr) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: no GPU backend selected for seg[%d-%d]",
              seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;
  }
  const char* backendName = backend->name();
#if HAVE_TRITON
  auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
#else
  void* tritonBackend = nullptr;
#endif

  // If compilation previously failed validation, never try again
  if (seg.exec.compilationFailed) {
    return Status::KERNEL_FAILURE;
  }

  // Check if this segment can be compiled by the selected GPU backend
  if (!backend->canFuseSegment(slots_, seg.def.startSlot, seg.def.endSlot)) {
    DSP_DIAG(BACKEND, "executeSegmentWithGpuGraph: backend=%s cannot fuse seg[%d-%d]",
              backendName, seg.def.startSlot, seg.def.endSlot);
    return Status::KERNEL_FAILURE;  // Caller will fall back to CUDA Graphs
  }

  // First execution: run slot-by-slot warmup BEFORE compilation.
  if (seg.exec.executionCount == 0) {
#ifdef SD_CUDA
    // ── Plan structure dump (one-time, on first segment execution) ─────────
    if (Environment::getInstance().tritonVerifyKernels()) {
      DSP_DIAG(VERIFY, "=== PLAN STRUCTURE ===");
      DSP_DIAG(VERIFY, "Plan: %d steps, %d output slots, %d external inputs, %d segments",
                numSlots_, totalOutputSlots_, numExternalInputs_, (int)segments_.size());
      for (int si = 0; si < (int)segments_.size(); si++) {
        auto& s = segments_[si];
        DSP_DIAG(VERIFY, "Segment %d: slots [%d..%d] (%d ops) %s",
                  si, s.def.startSlot, s.def.endSlot, s.def.endSlot - s.def.startSlot + 1,
                  s.def.isCapturable ? "capturable" : "non-capturable");
      }
      // Per-step wiring
      std::unordered_map<std::string, int> opHistogram;
      for (int s = 0; s < numSlots_; s++) {
        auto& sl = slots_[s];
        opHistogram[sl.ident.opName]++;
        // Build input description
        std::string inputsStr;
        for (int i = 0; i < sl.wiring.numInputs; i++) {
          if (i > 0) inputsStr += ", ";
          int srcIdx = sl.wiring.inputSourceIndices[i];
          if (srcIdx >= 0) {
            inputsStr += "slot#" + std::to_string(srcIdx);
          } else {
            int extIdx = -(srcIdx + 1);
            inputsStr += "ext#" + std::to_string(extIdx);
            if (extIdx < (int)externalInputNames_.size() && !externalInputNames_[extIdx].empty()) {
              inputsStr += ":\"" + externalInputNames_[extIdx] + "\"";
            }
            if (sl.wiring.inputSourceTypes != nullptr) {
              inputsStr += ":";
              inputsStr += sourceTypeName(sl.wiring.inputSourceTypes[i]);
            }
          }
        }
        // Build output description
        std::string outputsStr;
        for (int i = 0; i < sl.wiring.numOutputs; i++) {
          if (i > 0) outputsStr += ",";
          outputsStr += std::to_string(sl.wiring.outputSlotIndices[i]);
        }
        DSP_DIAG(VERIFY, "STEP %4d: %-20s inputs:[%s] -> outputs:[%s]%s%s%s",
                  s, sl.ident.opName.c_str(), inputsStr.c_str(), outputsStr.c_str(),
                  sl.flags.isIdentityOp ? " [IDENTITY]" : "",
                  sl.frozenConstantSlot() ? " [FROZEN]" : "",
                  sl.fusedChain.isFusedChainTail ? " [FUSED_TAIL]" : "");
      }
      // Op histogram
      std::string histStr;
      std::vector<std::pair<std::string, int>> sorted(opHistogram.begin(), opHistogram.end());
      std::sort(sorted.begin(), sorted.end(),
                [](const auto& a, const auto& b) { return b.second < a.second; });
      for (auto& p : sorted) {
        if (!histStr.empty()) histStr += ", ";
        histStr += p.first + "=" + std::to_string(p.second);
      }
      DSP_DIAG(VERIFY, "Op histogram: %s", histStr.c_str());
      DSP_DIAG(VERIFY, "=== END PLAN STRUCTURE ===");
    }
#endif

    // Pre-warmup promotion: view-capable slots should enter FROZEN state BEFORE
    // warmup execution when shapes are already frozen. This ensures they take the
    // view installation path (which shares the input's DataBuffer) instead of
    // allocating new output buffers. Without this, gap slots materialize separate
    // buffers that later get H2D-copied during capture, corrupting downstream values.
    if (shapesFrozen_) {
      int promoted = 0;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        auto& sl = slots_[s];
        if (!sl.flags.isViewCapableOp || sl.state_ >= NativeSlot::SlotState::FROZEN) continue;
        sl.state_ = NativeSlot::SlotState::FROZEN;
        promoted++;
      }
      if (promoted > 0) {
        DSP_DIAG(EXECUTE, "pre-warmup view promotion: %d view-capable slots promoted to FROZEN for seg[%d-%d]",
                  promoted, seg.def.startSlot, seg.def.endSlot);
      }
    }

    auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    // NOTE: executeSegmentSlotBySlot already increments seg.exec.executionCount on OK
    // (NativeDynamicShapePlan_segments.cpp line 930). Do NOT increment again here
    // — double-increment causes executionCount to skip the capture window [0,2],
    // preventing CUDA graph capture entirely and causing OOM from leaked per-step
    // allocations.

    // When shapes are frozen and executionCount is 1, the next call would
    // trigger compilation (needsCompile = executionCount==1). If compilation
    // already succeeded during unfrozen execution, skip recompilation by
    // bumping executionCount to 2. But DON'T skip if compilation hasn't
    // happened yet — let it trigger on the next call so cross-segment
    // shapes (now backfilled) are available for the Triton IR builder.
    if (shapesFrozen_ && warmupStatus == Status::OK && seg.exec.executionCount == 1
        && !Environment::getInstance().dspFreezeRecompile()) {
      // Only skip recompilation if segment already has compiled kernels.
      // seg.def.shapeKey != 0 means compilation ran previously and cached the key.
      if (seg.def.shapeKey != 0) {
        seg.exec.executionCount = 2;
        seg.exec.cachedShapeKey = seg.def.shapeKey;
        DSP_DIAG(COMPILE, "Post-freeze warmup: skipping recompile for seg[%d-%d] "
                  "(already compiled, shapeKey=%lld, bumped executionCount to 2)",
                  seg.def.startSlot, seg.def.endSlot, seg.def.shapeKey);
      } else {
        // Segment was never compiled — let executionCount stay at 1 so the
        // next call triggers compilation with backfilled cross-segment shapes.
        // DO NOT set seg.def.shapeKey here — it must stay 0 so the next call's
        // shapeKey check correctly identifies this as "never compiled".
        DSP_DIAG(COMPILE, "Post-freeze warmup: NOT skipping compile for seg[%d-%d] "
                  "(never compiled, executionCount stays at 1)",
                  seg.def.startSlot, seg.def.endSlot);
      }
    }
    return warmupStatus;
  }

  // Compute shape key for cache lookup.
  // When shapes are frozen and the key was already computed, reuse it — the shapes
  // cannot change so the hash is stable. Saves iterating all cross-segment inputs.
  // EXCEPTION: segments with value-dependent ops must ALWAYS recompute the shape key
  // because input VALUES (hashed by computeSegmentShapeKey for small inputs ≤32 elements)
  // can change even when shapes are frozen. Without this guard, the cached key would
  // miss value changes in reshape targets, broadcast dims, etc., causing CUDA graph
  // replay with stale output shapes.
  //
  // REPLAY OPTIMIZATION: During stable replay (executionCount >= 3 with a valid replay
  // handle), skip shape key computation entirely — even for hasValueDepOps segments.
  // The shape key was validated at capture time. Value-dependent inputs that can force
  // a rebuild are tracked separately via createValueKey/address stability checks.
  // If a value change truly requires graph invalidation, the createValueKey mechanism
  // catches it. Skipping shape key here eliminates N syncToHost calls per step
  // (one per small INT/INT64 cross-segment input array).
  // ── Shape key: detect if segment needs recompilation ──
  // Frozen + cached key: reuse. Otherwise: compute once and cache.
  const bool hasInternalValueShapeInputs = segmentHasInternalValueShapeInputs(seg, slots_);
  LongType segShapeKey;
  if (shapesFrozen_ && seg.exec.cachedShapeKey != 0) {
    segShapeKey = seg.exec.cachedShapeKey;
  } else {
    segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
    if (shapesFrozen_) {
      seg.exec.cachedShapeKey = segShapeKey;
    }
  }

  // Diagnostic: scan all outputSlots_ entries for freed DataBuffers.
  // Java may have closed DSP output arrays between steps (e.g., prefill KV outputs via
  // setCloseable(true)+close()), deleting the C++ NDArray and leaving dangling pointers.
  //
  // Always run this scan: during warmup/transitions it handles invalidation gracefully;
  // after warmup with frozen shapes, stale buffers indicate a bug (hard error via REQUIRE_TRUE).
  //
  // REPLAY OPTIMIZATION: Skip during stable replay (executionCount >= 4). In frozen
  // replay, arrays persist and are never closed by Java. The scan iterates all slots
  // in the segment range + all external inputs (~1333). For 278 captured segments,
  // this is significant host-side iteration. After the first few replays validate
  // no stale entries exist, skip the scan.
  bool isStableReplay = shapesFrozen_ && seg.exec.cachedShapeKey != 0 &&
                         seg.exec.executionCount >= 3;
  if (seg.exec.executionCount < 4 || !isStableReplay) {
    int invalidCount = 0;
    for (int si = seg.def.startSlot; si <= seg.def.endSlot && si < totalOutputSlots_; si++) {
      NDArray* cached = outputSlots_[si];
      if (cached != nullptr && !cached->isEmpty()) {
        auto* db = cached->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG_SLOT(MEMORY, si, "STALE outputSlots_[%d] detected "
                    "(arr=%p, db=%p, dbValid=%d, frozenConst=%d). Invalidating.",
                    si, (void*)cached, (void*)db, db ? (db->isValid() ? 1 : 0) : -1,
                    slots_[si].frozenConstantSlot() ? 1 : 0);
          outputSlots_[si] = nullptr;
          if (si < numSlots_ && slots_[si].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[si].state_ = NativeSlot::SlotState::FROZEN;
          }
          invalidCount++;
        }
      }
    }
    for (int ei = 0; ei < numExt; ei++) {
      NDArray* ext = externalArrays[ei];
      if (ext != nullptr && !ext->isEmpty()) {
        auto* db = ext->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          DSP_DIAG(MEMORY, "STALE externalInput[%d] detected "
                    "(arr=%p, db=%p, dbValid=%d)",
                    ei, (void*)ext, (void*)db, db ? (db->isValid() ? 1 : 0) : -1);
          invalidCount++;
        }
      }
    }
    if (invalidCount > 0) {
      DSP_DIAG(MEMORY, "executeSegmentWithGpuGraph: found %d stale entries in slot/external arrays",
                invalidCount);
      if (shapesFrozen_ && seg.exec.executionCount > 1) {
        // After warmup with frozen shapes, stale buffers mean a bug in array lifecycle management
        REQUIRE_TRUE(false, 0, "Stale buffer detected after warmup (executionCount=%d, frozen=%d, "
                     "invalidCount=%d) in seg[%d-%d]. This indicates a bug in DSP array persistence.",
                     seg.exec.executionCount, (int)shapesFrozen_, invalidCount,
                     seg.def.startSlot, seg.def.endSlot);
      }
      // During warmup/transitions, invalidate and re-execute
#ifdef SD_CUDA
      platformCleanupSegmentForRebuild(seg);
      seg.exec.argTableStable = false;
      batchD2DCount_ = 0;
      seg.exec.cachedShapeKey = 0;
#endif
      seg.exec.compilationFailed = false;
      DSP_DIAG(FALLBACK, "invalidated graph for seg[%d-%d] "
                "due to %d stale entries - executing slot-by-slot this step",
                seg.def.startSlot, seg.def.endSlot, invalidCount);
      return executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
    }
  }

  // Pre-execution: ensure all output slots in the segment have live arrays.
  // The Triton kernel's arg mapping references outputSlots_ for both inputs
  // (from prior ops) and outputs (to write results). Slot-by-slot warmup may
  // have released intermediate arrays via releaseAtStep_, leaving entries null.
  // First restore from outputSlots_, then allocate any remaining nulls
  // using cached shape info from warmup.
  //
  //  This MUST happen BEFORE compilation. The compiler resolves
  // arg mappings from outputSlots_ — if intermediate slots are null (released
  // after warmup), the compiler omits them from the arg table, producing
  // sub-kernels with missing inputs that read stale/garbage data on first
  // execution. By populating all slots before compilation, the compiler sees
  // all arrays and builds correct arg mappings.
  //
  // IMPORTANT: Java may close() output arrays between execution steps (e.g.,
  // prefill KV outputs via setCloseable(true)+close()). This frees the underlying
  // DataBuffer while outputSlots_ still holds the NDArray*. Validate the
  // DataBuffer before reusing — invalidate entries pointing to freed buffers.
  //
  //  If any output slot within the segment is allocated at a NEW address
  // (different from capture time), the cached CUDA graph becomes invalid. Triton
  // arg tables are refreshed with new addresses, but native ops (cuBLAS matmul)
  // have addresses baked into the graph. This address inconsistency causes the
  // graph to read stale data from old addresses while Triton writes to new ones.
  // Track any new allocations and invalidate the graph if needed.
  //
  // OPTIMIZATION: Skip when shapes are frozen and we've already done this
  // restoration at least once (executionCount > 2). In steady-state decode,
  // outputSlots_ are stable — no arrays are released or freed between steps.
  // EXCEPTION: segments entering capture for the first time (no replay handle)
  // MUST always get pre-exec restoration — cleanup may have nulled cross-segment
  // input slots that the capture path still needs wired directly.
  int preExecAllocCount = 0;
  if (!(shapesFrozen_ && seg.exec.executionCount > 2 && seg.exec.replayHandle != nullptr)) {
  for (int stepIdx = seg.def.startSlot; stepIdx <= seg.def.endSlot; stepIdx++) {
    NativeSlot& slot = slots_[stepIdx];
    // Phase 2: outputSlots_ == outputSlots_ (unified). No restore needed.
    // Validate input DataBuffers — Java close() may have freed them.
    for (int i = 0; i < slot.wiring.numInputs; i++) {
      int srcIdx = slot.wiring.inputSourceIndices[i];
      if (srcIdx >= 0 && srcIdx < totalOutputSlots_ && outputSlots_[srcIdx] != nullptr) {
        auto* db = outputSlots_[srcIdx]->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          outputSlots_[srcIdx] = nullptr;
          if (srcIdx < numSlots_ && slots_[srcIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[srcIdx].state_ = NativeSlot::SlotState::FROZEN;
          }
        }
      }
    }
    // Validate or allocate output slot entries
    for (int i = 0; i < slot.wiring.numOutputs; i++) {
      int slotIdx = slot.wiring.outputSlotIndices[i];
      if (slotIdx < 0 || slotIdx >= totalOutputSlots_) continue;
      // DIAGNOSTIC: trace configured slot pre-exec validation (ND4J_DSP_TRACE_SLOT)
      {
        int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
        if (ts >= 0 && slotIdx == ts && shapesFrozen_) {
          auto* arr = outputSlots_[slotIdx];
          auto* db = arr != nullptr ? arr->dataBuffer() : nullptr;
          DSP_DIAG_SLOT(MEMORY, stepIdx,
              "PRE_EXEC_VALIDATE: slot=%d arr=%p db=%p valid=%d exec=%d",
              slotIdx, (void*)arr, (void*)db,
              db != nullptr && db->isValid() ? 1 : 0,
              seg.exec.executionCount);
        }
      }
      // Validate existing entry
      if (outputSlots_[slotIdx] != nullptr) {
        auto* db = outputSlots_[slotIdx]->dataBuffer();
        if (db == nullptr || !db->isValid()) {
          {
            int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
            if (ts >= 0 && slotIdx == ts) {
              DSP_DIAG_SLOT(MEMORY, stepIdx,
                  "PRE_EXEC_NULL: slot=%d db=%p was nullOrInvalid exec=%d",
                  slotIdx, (void*)db, seg.exec.executionCount);
            }
          }
          outputSlots_[slotIdx] = nullptr;
          if (stepIdx < numSlots_ && slots_[stepIdx].state_ == NativeSlot::SlotState::FROZEN_CONSTANT) {
            slots_[stepIdx].state_ = NativeSlot::SlotState::FROZEN;
          }
        }
      }
      if (outputSlots_[slotIdx] == nullptr) {
        // After warmup with frozen shapes, null output slots indicate a persistence bug.
        // Frozen constant slots are exempt (they never allocate output arrays).
        // Warn but continue — the allocation path below will recover.
        if (shapesFrozen_ && seg.exec.executionCount > 1 && !slot.frozenConstantSlot()) {
          DSP_DIAG_SLOT(VERIFY, slotIdx,
              "BUG: Null output slot %d (%s) after warmup with frozen shapes — persistence bug. execCount=%d",
              slotIdx, slot.ident.opName.c_str(), seg.exec.executionCount);
        }
        // Allocate from cached shape info (populated during warmup)
        const LongType* shapeInfo = nullptr;
        if (i < static_cast<int>(slot.shapeCache.cachedOutputShapes.size()) && slot.shapeCache.cachedOutputShapes[i]) {
          shapeInfo = slot.shapeCache.cachedOutputShapes[i];
        }
        // For identity/view-like ops that don't cache output shapes,
        // derive the shape from the first input source's existing array
        if (!shapeInfo && slot.wiring.numInputs > 0) {
          int srcIdx = slot.wiring.inputSourceIndices[0];
          NDArray* srcArr = nullptr;
          if (srcIdx < 0) {
            int extIdx = -(srcIdx + 1);
            if (extIdx < numExt) srcArr = externalArrays[extIdx];
          } else if (srcIdx < totalOutputSlots_) {
            srcArr = outputSlots_[srcIdx];
            // Phase 2: outputSlots_ == outputSlots_ (unified), no separate restore
          }
          if (srcArr) shapeInfo = srcArr->shapeInfo();
        }
        if (shapeInfo) {
          auto dt = ArrayOptions::dataType(shapeInfo);
          // For cast ops, the output type must match the declared target type,
          // not the input type. When cachedOutputShapes is empty and the
          // using the input source's shape, the dtype would be wrong
          // (e.g., INT64 input for a cast-to-FLOAT op).
          if (slot.ident.op && slot.ident.op->getOpDescriptor() &&
              slot.ident.op->getOpDescriptor()->hasAnyTrait(sd::ops::OP_TRAIT_CAST) &&
              slot.args.numIArgs > 0 && slot.args.iArgs) {
            auto castDt = static_cast<DataType>(slot.args.iArgs[0]);
            if (castDt != dt) {
              DSP_DIAG(EXECUTE, "PRE_EXEC_ALLOC: cast dtype override slot=%d from %s to %s",
                       slotIdx, DataTypeUtils::asString(dt).c_str(),
                       DataTypeUtils::asString(castDt).c_str());
              dt = castDt;
            }
          }
          auto order = shape::order(shapeInfo);
          LongType rank = shape::rank(shapeInfo);
          std::vector<LongType> shapeVec(rank);
          for (int d = 0; d < rank; d++) shapeVec[d] = shapeInfo[d + 1];
          auto* arr = new NDArray(order, shapeVec, dt);
          outputSlots_[slotIdx] = arr;
          // Phase 2: outputSlots_ == outputSlots_ (unified), no separate assignment needed
          preExecAllocCount++;
          {
            int ts = sd::graph::DspDiagnostics::getInstance().traceSlot();
            if (ts >= 0 && slotIdx == ts) {
              auto* newDb = arr != nullptr ? arr->dataBuffer() : nullptr;
              DSP_DIAG_SLOT(MEMORY, stepIdx,
                  "PRE_EXEC_ALLOC: slot=%d arr=%p db=%p exec=%d",
                  slotIdx, (void*)arr, (void*)newDb, seg.exec.executionCount);
            }
          }
          if (Environment::getInstance().tritonVerifyKernels()) {
            DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=ALLOC dtype=%s len=%lld addr=%p",
                      slotIdx, DataTypeUtils::asString(dt).c_str(),
                      (long long)arr->lengthOf(), DSP_BUF(arr));
          }
        }
      }
    }
  }
  } // end if (!(shapesFrozen_ && executionCount > 2))

  // Compile once per stable shape; skip cache probe on steady-state replay.
  // This keeps the hot path focused on dispatch instead of repeated compile checks.
  // NOTE: Pre-exec output slot allocation above ensures all slots are populated
  // before the compiler resolves arg mappings. Without this ordering, intermediate
  // slots released after warmup are null and get omitted from the arg table,
  // causing sub-kernels to read stale data on their first execution.
  bool needsCompile = (seg.exec.executionCount == 1) || (seg.def.shapeKey != segShapeKey);
  if (needsCompile) {
    // When recompiling due to shape change (not the first compile), outputSlots_
    // has stale shapes from the previous execution. The compiler reads these shapes
    // to derive kernel parameters (e.g., seqQ/seqK for FUSED_ATTENTION). Run a
    // slot-by-slot pass first to populate outputSlots_ with current shapes before
    // compiling. This is like a mini-warmup for the new shape configuration.
    bool isRecompileDueToShapeChange = (seg.exec.executionCount > 1) && (seg.def.shapeKey != segShapeKey);
    if (isRecompileDueToShapeChange) {
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape change detected for seg[%d-%d] "
                "(shapeKey %lld->%lld, executionCount=%d). Running slot-by-slot warmup to "
                "refresh outputSlots_ before recompilation.",
                seg.def.startSlot, seg.def.endSlot, seg.def.shapeKey, segShapeKey, seg.exec.executionCount);
      // Invalidate cached graph — addresses and shapes changed
#ifdef SD_CUDA
      platformCleanupSegmentForRebuild(seg);
      seg.exec.argTableStable = false;
      batchD2DCount_ = 0;
#endif
      auto warmupStatus = executeSegmentSlotBySlot(seg, externalArrays, numExt, stream);
      if (warmupStatus != Status::OK) {
        DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape-change warmup FAILED for seg[%d-%d] status=%d",
                  seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
        return warmupStatus;
      }
      // Recompute shape key after warmup — outputSlots_ now has correct shapes
      segShapeKey = computeSegmentShapeKey(seg, externalArrays, numExt);
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: shape-change warmup OK for seg[%d-%d], "
                "recomputed shapeKey=%lld", seg.def.startSlot, seg.def.endSlot, segShapeKey);
    }

    if (!backend->compileSegment(seg, slots_, externalArrays, numExt,
                                 outputSlots_, totalOutputSlots_, segShapeKey,
                                 numSlots_)) {
      DSP_DIAG(COMPILE, "executeSegmentWithGpuGraph: backend=%s compile failed for seg[%d-%d]",
                backendName, seg.def.startSlot, seg.def.endSlot);
      return Status::KERNEL_FAILURE;
    }
  }

  // On first compilation, validate coverage
  if (seg.exec.executionCount == 1) {
    auto audit = backend->getLastCompilationAudit();
    lastCompilationAudit_ = audit;
    int compiledCount = 0;
    int failedCount = 0;
    for (const auto& entry : audit) {
      if (entry.wasCompiled) {
        compiledCount++;
      } else {
        failedCount++;
        DSP_DIAG_SLOT(COMPILE, entry.slotIndex, "%s VALIDATION: slot %d (%s) was NOT compiled: %s",
                  backendName, entry.slotIndex, entry.opName.c_str(), entry.reason.c_str());
      }
    }
    if (compiledCount == 0 && failedCount > 0) {
      // All ops FAILED compilation — hard error.
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] has zero compiled ops "
                "(failed=%d). Compilation failures are errors, not fallbacks.",
                backendName, seg.def.startSlot, seg.def.endSlot, failedCount);
      seg.exec.compilationFailed = true;
      return Status::KERNEL_FAILURE;
    }
    if (compiledCount == 0 && failedCount == 0) {
      // All sections stay native-ordered for this segment.
      // The compiled segment has 0 sub-kernels; executeSegment will run
      // everything via the ordered range executor.
      // DO NOT set compilationFailed — allow these segments to be captured as
      // CUDA graphs. During Triton graph capture, native ordered ranges are
      // recorded into the graph via the ordered range executor, enabling
      // single-launch replay instead of per-op kernel dispatch overhead.
      DSP_DIAG(COMPILE, "%s: segment [%d-%d] has only native ordered sections (no Triton kernels needed). "
                "Segment remains eligible for CUDA graph capture.",
                backendName, seg.def.startSlot, seg.def.endSlot);
    }
    if (failedCount > 0) {
      // Partial compilation failure — hard error. Fix the kernel.
      DSP_DIAG(COMPILE, "%s COMPILE ERROR: segment [%d-%d] partial compile FAILED "
                "(compiled=%d failed=%d). Compilation failures are errors, not fallbacks.",
                backendName, seg.def.startSlot, seg.def.endSlot, compiledCount, failedCount);
      seg.exec.compilationFailed = true;
      return Status::KERNEL_FAILURE;
    }
  }

  // Execute via selected GPU backend
  seg.def.shapeKey = segShapeKey;

#ifdef SD_CUDA
  cudaStream_t cudaStr = (stream != nullptr)
      ? *static_cast<cudaStream_t*>(stream) : nullptr;

  // If any output slots were re-allocated at new addresses, the cached CUDA graph
  // is invalid — native ops (cuBLAS) have the old addresses baked in while Triton
  // arg tables were refreshed with new addresses. Invalidate and re-capture.
  if (preExecAllocCount > 0 && seg.exec.replayHandle != nullptr) {
    DSP_DIAG(EXECUTE, "GRAPH INVALIDATED: %d output slots re-allocated at new addresses "
              "(cache entries freed by Java). seg[%d-%d] will re-capture.",
              preExecAllocCount, seg.def.startSlot, seg.def.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    // Reset execution count to trigger warmup→capture flow
    seg.exec.executionCount = 0;
    seg.exec.compilationFailed = false;
  }

  bool allowTritonCudaGraphReplay = Environment::getInstance().tritonGraphCapture() &&
                                    shapesFrozen_;

  // BLATANT DIAGNOSTIC: Log the capture decision factors
  int captureMinExec = Environment::getInstance().tritonCaptureMinExec();
  bool forceRecaptureEnabled = Environment::getInstance().tritonForceRecapture();
  bool hasReplayHandle = (seg.exec.replayHandle != nullptr);
  bool replayHandleNull = (seg.exec.replayHandle == nullptr);
  bool notCaptureFailed = !seg.exec.compilationFailed;
  bool execCountInWindow = (seg.exec.executionCount >= captureMinExec) && 
                           (forceRecaptureEnabled || seg.exec.executionCount <= (captureMinExec + 2));
  bool hasCudaStream = (cudaStr != nullptr);
  bool requiresOrderedGapCapture = false;
  
  DSP_DIAG(EXECUTE, "=== CAPTURE DECISION CHECK seg[%d-%d] ===", seg.def.startSlot, seg.def.endSlot);
  DSP_DIAG(EXECUTE, "  tritonGraphCapture()=%d, shapesFrozen_=%d => allowTritonCudaGraphReplay=%d",
           Environment::getInstance().tritonGraphCapture() ? 1 : 0,
           shapesFrozen_ ? 1 : 0, allowTritonCudaGraphReplay ? 1 : 0);
  DSP_DIAG(EXECUTE, "  seg.exec.executionCount=%d, captureMinExec=%d, window=[%d,%d], inWindow=%d",
           seg.exec.executionCount, captureMinExec, captureMinExec, captureMinExec + 2,
           execCountInWindow ? 1 : 0);
  DSP_DIAG(EXECUTE, "  hasReplayHandle=%d, replayHandleNull=%d",
           hasReplayHandle ? 1 : 0, replayHandleNull ? 1 : 0);
  DSP_DIAG(EXECUTE, "  compilationFailed=%d, cudaStr!=nullptr=%d",
           seg.exec.compilationFailed ? 1 : 0, hasCudaStream ? 1 : 0);
  
  bool shouldCaptureTritonGraph = false;

  int firstUnsupportedTritonGap = -1;
  int lastTritonCoveredSlot = -1;
  int tritonGapSlotCount = 0;
  bool hasUnsupportedTritonReplayGaps = false;
#if HAVE_TRITON && defined(SD_CUDA)
  hasUnsupportedTritonReplayGaps =
      findUnsupportedTritonReplayGap(tritonBackend, seg, slots_,
                                     &firstUnsupportedTritonGap,
                                     &lastTritonCoveredSlot,
                                     &tritonGapSlotCount);
  if (hasUnsupportedTritonReplayGaps) {
    requiresOrderedGapCapture = true;
    seg.exec.compositeReplaySchedule = buildCompositeReplaySchedule(seg, slots_, tritonBackend);
    DSP_DIAG(SHAPE,
             "TRITON_REPLAY_CONTRACT: seg[%d-%d] gap slot %d precedes covered slot %d "
             "(gapSlots=%d) currentPhase=%d planPhase=%d",
             seg.def.startSlot, seg.def.endSlot, firstUnsupportedTritonGap,
             lastTritonCoveredSlot, tritonGapSlotCount,
             static_cast<int>(seg.exec.currentPhase), static_cast<int>(planPhase_));
    DSP_DIAG(SHAPE, "COMPOSITE_SCHEDULE_BUILT: seg[%d-%d] units=%d",
             seg.def.startSlot, seg.def.endSlot,
             static_cast<int>(seg.exec.compositeReplaySchedule.units.size()));
  } else {
    DSP_DIAG(SHAPE, "NO_UNSUPPORTED_GAPS: seg[%d-%d] hasUnsupported=%d",
             seg.def.startSlot, seg.def.endSlot, hasUnsupportedTritonReplayGaps ? 1 : 0);

    // ── DSP Segment Bucket Classification ──
    // Classify all gap slots and emit a structured diagnostic summary.
    // This maps each gap range to its bucket type (view-only, shape-only,
    // materializing) so every invalid segment is explainable.
    if (DSP_DIAG_ENABLED(SEGMENT_BUCKETS)) {
      auto gapSlots = tritonBackend->getGapSlots(seg, slots_);
      std::vector<DspDiagnostics::GapClassification> gapClassifications;
      gapClassifications.reserve(gapSlots.size());
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        if (gapSlots.count(s)) {
          gapClassifications.push_back(classifyGapSlot(s, slots_));
        }
      }
      auto merged = mergeGapClassifications(gapClassifications);
      std::string bucketLabel = buildCombinedBucketLabel(merged);

      // Convert merged vector to array for diagnostic call
      std::vector<DspDiagnostics::GapClassification> diagVec(merged.begin(), merged.end());
      DspDiagnostics::getInstance().reportSegmentBucketSummary(
          seg.def.startSlot, seg.def.endSlot,
          diagVec.data(), static_cast<int>(diagVec.size()),
          bucketLabel.c_str(),
          false /* valid when captured in-order */);
    }
  }
#else
  seg.exec.compositeReplaySchedule = ReplaySchedule();
#endif

  bool captureWindowSatisfied = execCountInWindow || requiresOrderedGapCapture;
  shouldCaptureTritonGraph = allowTritonCudaGraphReplay &&
                             !hasReplayHandle &&
                             replayHandleNull &&
                             notCaptureFailed &&
                             captureWindowSatisfied &&
                             hasCudaStream;

  if (requiresOrderedGapCapture) {
    DSP_DIAG(EXECUTE,
             "COMPOSITE_GAP_CAPTURE: seg[%d-%d] has %d interleaved gap slots "
             "(first=%d lastCovered=%d). Gap ops will be EXCLUDED from CUDA graph; "
             "composite replay will execute gaps fresh before Triton-only graph replay.",
             seg.def.startSlot, seg.def.endSlot, tritonGapSlotCount,
             firstUnsupportedTritonGap, lastTritonCoveredSlot);
  }

  DSP_DIAG(EXECUTE, "  => shouldCaptureTritonGraph=%d", shouldCaptureTritonGraph ? 1 : 0);
  if (!shouldCaptureTritonGraph) {
    if (!allowTritonCudaGraphReplay)
      DSP_DIAG(EXECUTE, "  BLOCKED: allowTritonCudaGraphReplay=false (tritonGraphCapture=%d OR shapesFrozen_=%d)",
               Environment::getInstance().tritonGraphCapture() ? 1 : 0, shapesFrozen_ ? 1 : 0);
    if (!replayHandleNull)
      DSP_DIAG(EXECUTE, "  BLOCKED: replayHandle already exists (capture already done or in progress)");
    if (seg.exec.compilationFailed)
      DSP_DIAG(EXECUTE, "  BLOCKED: compilationFailed=true (previous capture failed, warmup path only)");
    if (!captureWindowSatisfied)
      DSP_DIAG(EXECUTE, "  BLOCKED: executionCount=%d outside capture window [%d,%d]",
               seg.exec.executionCount, captureMinExec, captureMinExec + 2);
    if (!hasCudaStream)
      DSP_DIAG(EXECUTE, "  BLOCKED: cudaStr=nullptr (no CUDA stream available)");
  } else {
    DSP_DIAG(EXECUTE, "  >>> CAPTURE WILL BE ATTEMPTED <<<");
  }
  DSP_DIAG(EXECUTE, "=== END CAPTURE DECISION CHECK ===");
  
  // NOTE: shouldCaptureTritonGraph is ONLY checked when we don't have a captured graph.
  // Once captured, we use useFastReplay based on argTableStable, not executionCount.
  // The executionCount window check prevents repeated capture attempts after success.
  
  // OPTIMIZATION: When argTableStable, addresses and create-op values haven't changed
  // since last refresh — skip the expensive hash/comparison loops over all external inputs.
  LongType segInputAddrKey;
  bool extAddrsStable;
  LongType createValueKey;
  bool canSkipReplayInvariantRecompute =
      seg.exec.argTableStable && allowTritonCudaGraphReplay &&
      !hasInternalValueShapeInputs;
  if (canSkipReplayInvariantRecompute) {
    // Fast path: arg table is stable, all addresses are known-good
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "seg[%d-%d] argTableStable=true → FAST PATH (skip addr/createValue recompute)",
                 seg.def.startSlot, seg.def.endSlot);
    segInputAddrKey = seg.exec.capturedInputAddrKey;
    extAddrsStable = true;
    createValueKey = seg.exec.capturedCreateValueKey;
  } else {
    segInputAddrKey = computeSegmentInputAddrKey(seg, externalArrays, numExt);
    extAddrsStable = (seg.exec.replayHandle && !seg.exec.replayHandle->getCapturedExternalAddresses().empty())
        ? externalAddrsMatch(seg, externalArrays, numExt)
        : (seg.exec.capturedInputAddrKey != 0 && seg.exec.capturedInputAddrKey == segInputAddrKey);
    createValueKey = computeCreateOpValueKey(seg, externalArrays, numExt);
    DSP_DIAG(EXECUTE, "ADDR_CHECK_SLOW: seg[%d-%d] extAddrsStable=%d addrKey=%lld (cached=%lld)",
             seg.def.startSlot, seg.def.endSlot, extAddrsStable ? 1 : 0,
             (long long)segInputAddrKey, (long long)seg.exec.capturedInputAddrKey);
  }
  bool createValuesStable = (createValueKey == 0) ||  // no create ops
                            (seg.exec.capturedCreateValueKey == createValueKey);
  if (hasInternalValueShapeInputs) {
    const bool shapeKeyStable = (seg.exec.cachedShapeKey == 0) ||
                                (seg.exec.cachedShapeKey == segShapeKey);
    seg.exec.argTableStable = seg.exec.argTableStable &&
                              extAddrsStable &&
                              createValuesStable &&
                              shapeKeyStable;
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "INTERNAL_VALUE_SHAPE_TRACKING: seg[%d-%d] argStable=%d shapeStable=%d "
                 "createStable=%d extAddrsStable=%d",
                 seg.def.startSlot, seg.def.endSlot,
                 seg.exec.argTableStable ? 1 : 0,
                 shapeKeyStable ? 1 : 0,
                 createValuesStable ? 1 : 0,
                 extAddrsStable ? 1 : 0);
  }
  if (!createValuesStable && seg.exec.replayHandle) {
    DSP_DIAG(EXECUTE, "CREATE_VALUE_KEY mismatch: captured=%lld current=%lld → invalidating graph seg[%d-%d]",
             (long long)seg.exec.capturedCreateValueKey, (long long)createValueKey, seg.def.startSlot, seg.def.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.executionCount = 0;
    seg.exec.compilationFailed = false;
    extAddrsStable = false;  // Force re-capture path
  }

  // Triton graph replay conditions:
  // 1. Shape key matches (frozen shapes)
  // 2. Create op input values stable (ConstantOfShape shapes unchanged)
  // 3. Input addresses are unchanged since capture
  //  Only enter the Triton replay path for segments actually compiled by Triton.
  // Segments captured by the raw CUDA graph path (NativeDynamicShapePlan_cudagraph.cu)
  // have replayHandles but NO Triton arg tables. The Triton replay path's D2D copy +
  // arg table refresh is incompatible with raw CUDA graphs — it can corrupt cross-segment
  // data, causing downstream segments to read zeros instead of valid output → NaN.
  // compiledByBackend is set to backendName ONLY after a successful Triton execution.
  // Raw CUDA captures leave it empty → excluded from this path → fall through to
  // executeSegmentWithGraph() in cudagraph.cu which handles replay correctly.
  bool isTritonCompiled = (!seg.exec.compiledByBackend.empty() && seg.exec.compiledByBackend == backendName);

  // Invalidate stale graphs that have gap ops baked in. Gap ops must NOT be
  // captured into CUDA graphs — their baked addresses go stale on replay.
  // New captures exclude gap ops; this catches legacy pre-fix graphs.
  if (allowTritonCudaGraphReplay &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() &&
      isTritonCompiled &&
      hasUnsupportedTritonReplayGaps &&
      seg.exec.gapOpsCapturedInGraph) {
    DSP_DIAG(EXECUTE,
             "STALE_GAP_GRAPH_INVALIDATE: invalidating seg[%d-%d] replay handle "
             "because gap ops were baked into the graph (stale addresses on replay).",
             seg.def.startSlot, seg.def.endSlot);
    platformCleanupSegmentForRebuild(seg);
    seg.exec.argTableStable = false;
    batchD2DCount_ = 0;
    seg.exec.capturedInputAddrKey = 0;
    seg.exec.capturedCreateValueKey = 0;
    seg.exec.gapOpsCapturedInGraph = false;
    seg.exec.executionCount = captureMinExec;
    hasReplayHandle = false;
    replayHandleNull = true;
    isTritonCompiled = false;
    extAddrsStable = false;
  }

  if (allowTritonCudaGraphReplay && seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() && !isTritonCompiled) {
    DSP_DIAG(EXECUTE, "TRITON_REPLAY_SKIP: seg[%d-%d] has replayHandle but compiledBy='%s' (not %s) "
             "→ falling through to raw CUDA graph replay path",
             seg.def.startSlot, seg.def.endSlot,
             seg.exec.compiledByBackend.empty() ? "(empty)" : seg.exec.compiledByBackend.c_str(),
             backendName);
  }

  if (allowTritonCudaGraphReplay &&
      seg.exec.replayHandle != nullptr &&
      seg.exec.replayHandle->isReady() &&
      isTritonCompiled &&
      seg.exec.cachedShapeKey == segShapeKey &&
      createValuesStable &&
      extAddrsStable) {

#if HAVE_TRITON && defined(SD_CUDA)
    if (hasUnsupportedTritonReplayGaps) {
      DSP_DIAG(EXECUTE,
               "COMPOSITE_REPLAY_ENTER: seg[%d-%d] has %d gap slots — composite replay "
               "will execute gaps fresh then replay Triton-only graph (units=%d).",
               seg.def.startSlot, seg.def.endSlot, tritonGapSlotCount, seg.exec.replayUnitCount);
    }
#endif

    DSP_DIAG(EXECUTE, "TRITON_REPLAY_ENTER: seg[%d-%d] extAddrsStable=%d argTableStable=%d compositeGaps=%d",
             seg.def.startSlot, seg.def.endSlot, extAddrsStable ? 1 : 0,
             seg.exec.argTableStable ? 1 : 0, hasUnsupportedTritonReplayGaps ? 1 : 0);

    // ── Install view recipes before replay ──────────────────────────────
    // View-producing ops (reshape, permute, etc.) were captured as recipes
    // during SHAPES_FROZEN and validated during POINTERS_STABLE. Now install
    // them as zero-copy views before consumer replay executes.
    //
    // SKIP view recipe installation when composite replay has gap units.
    // Composite replay executes ALL gap slots fresh (including view-producing ops),
    // which is authoritative. View recipes would conflict: they install stale
    // views from capture-time source addresses, which the fresh gap execution
    // then overwrites. The fresh gap execution produces correct results; view
    // recipes are only needed for segments WITHOUT composite gap replay.
    if (!seg.exec.viewRecipes.recipes.empty() && planPhase_ >= PlanPhase::REPLAYING
        && !hasUnsupportedTritonReplayGaps) {
      installViewRecipes(seg, outputSlots_, totalOutputSlots_, externalArrays, numExt);
      DSP_DIAG(EXECUTE, "VIEW_RECIPE_INSTALL: seg[%d-%d] installed %d view recipes before replay",
               seg.def.startSlot, seg.def.endSlot, static_cast<int>(seg.exec.viewRecipes.recipes.size()));
    } else if (!seg.exec.viewRecipes.recipes.empty() && hasUnsupportedTritonReplayGaps) {
      DSP_DIAG(EXECUTE, "VIEW_RECIPE_SKIP: seg[%d-%d] skipping %d view recipes — composite "
               "replay gap execution handles view-producing ops",
               seg.def.startSlot, seg.def.endSlot, static_cast<int>(seg.exec.viewRecipes.recipes.size()));
    }

    bool useFastReplay = seg.exec.argTableStable &&
                         !hasInternalValueShapeInputs &&
                         !Environment::getInstance().tritonVerifyKernels();
    DSP_DIAG(EXECUTE, "REPLAY_PATH: seg[%d-%d] useFastReplay=%d argStable=%d verify=%d",
             seg.def.startSlot, seg.def.endSlot, useFastReplay ? 1 : 0,
             seg.exec.argTableStable ? 1 : 0,
             Environment::getInstance().tritonVerifyKernels() ? 1 : 0);

    // ── Phase 2: Replay Schedule Signature & Consolidation ──────────────
    // Build the replay schedule signature to track cross-step invariance
    // and apply consolidation diagnostics.
#if HAVE_TRITON && defined(SD_CUDA)
    auto* tritonBE = dynamic_cast<TritonGraphBackend*>(getGpuGraphBackend());
    if (tritonBE != nullptr && planPhase_ >= PlanPhase::SHAPES_FROZEN) {
      auto rawUnits = buildReplayUnits(seg, slots_, outputSlots_, tritonBE);
      ReplayScheduleSignature rawSig = buildReplaySignature(seg, rawUnits);

      // Compare with previous signature to detect schedule drift
      static thread_local ReplayScheduleSignature prevSig;
      static thread_local const NativeDynamicShapePlan* sigPlan = nullptr;
      if (sigPlan != this) {
        prevSig = ReplayScheduleSignature();  // Reset for new plan
        sigPlan = this;
      }
      if (prevSig.numUnits > 0 && !signaturesMatch(prevSig, rawSig)) {
        DSP_DIAG(EXECUTE, "SCHEDULE_DRIFT: seg[%d-%d] signature changed: hash %llx -> %llx, units %d -> %d",
                 seg.def.startSlot, seg.def.endSlot,
                 (unsigned long long)prevSig.hash, (unsigned long long)rawSig.hash,
                 prevSig.numUnits, rawSig.numUnits);
      }

      // Apply consolidation pass
      int unitsBefore = 0, unitsAfter = 0;
      auto consolidatedUnits = consolidateReplayUnits(rawUnits, slots_, &unitsBefore, &unitsAfter);
      ReplayScheduleSignature consSig = buildReplaySignature(seg, consolidatedUnits);

      // Store real replay signature and unit count in segment exec state
      seg.exec.replaySignatureHash = consSig.hash;
      seg.exec.replayUnitCount = consSig.numUnits;

      DSP_DIAG(EXECUTE, "REPLAY_SIG: seg[%d-%d] raw=%d consolidated=%d hash=%llx stable=%d",
               seg.def.startSlot, seg.def.endSlot, rawSig.numUnits, consSig.numUnits,
               (unsigned long long)consSig.hash,
               signaturesMatch(prevSig, rawSig) ? 1 : 0);

      prevSig = rawSig;  // Track raw (pre-consolidation) signature for drift detection
    }
#endif

    // ── cuBLAS workspace invariant assertion during REPLAYING ──────────────
    // During one continuous REPLAYING epoch, the cuBLAS workspace address and
    // size must not change. Captured graphs bake in that workspace pointer.
    // The baseline must reset when a plan is rebuilt or demoted out of replay,
    // otherwise a later plan/config on the same thread is compared against a
    // stale snapshot from an earlier replay epoch.
    static thread_local const NativeDynamicShapePlan* replayWorkspacePlan = nullptr;
    static thread_local void* replayWorkspaceAddr = nullptr;
    static thread_local size_t replayWorkspaceSize = 0;
    if (replayWorkspacePlan != this || planPhase_ < PlanPhase::REPLAYING) {
      replayWorkspacePlan = this;
      replayWorkspaceAddr = nullptr;
      replayWorkspaceSize = 0;
    }
    if (planPhase_ >= PlanPhase::REPLAYING && cublasWorkspaceBuffer_ != nullptr) {
      if (replayWorkspaceAddr == nullptr) {
        replayWorkspaceAddr = cublasWorkspaceBuffer_;
        replayWorkspaceSize = cublasWorkspaceSize_;
      } else if (cublasWorkspaceBuffer_ != replayWorkspaceAddr ||
                 cublasWorkspaceSize_ != replayWorkspaceSize) {
        char errMsg[256];
        snprintf(errMsg, sizeof(errMsg),
                 "DSP phase contract violation: cuBLAS workspace changed during REPLAYING phase. "
                 "addr %p → %p, size %zu → %zu. Captured graphs have stale workspace pointers.",
                 replayWorkspaceAddr, cublasWorkspaceBuffer_,
                 replayWorkspaceSize, cublasWorkspaceSize_);
        DSP_DIAG(FALLBACK, "CUBLAS_WORKSPACE_DRIFT: %s", errMsg);
        sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(
            static_cast<int>(Status::KERNEL_FAILURE));
        sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errMsg);
        return Status::KERNEL_FAILURE;
      }
    }

    // CRITICAL FIX: Set tl_dspExecutionStream for ALL Triton executions, not just capture replay.
    // Without this, syncToSpecial() calls fall back to stream 0 and do full cudaStreamSynchronize,
    // causing 657k sync calls per decode step. Setting tl_dspExecutionStream allows async H2D
    // copies on the same stream as compute, with stream ordering guaranteeing correctness.
    // RAII guard: restores previous tl_dspExecutionStream value when this function exits.
    sd::graph::DspStreamGuard dspStreamGuard(cudaStr);

    // CRITICAL FIX: Cross-stream ordering for stable-address variable inputs.
    //
    // When external inputs (embeddings, input_ids) use reusable fixed-address buffers,
    // .assign() writes new data to the SAME GPU address each step. The assign runs on
    // the default LaunchContext stream (stream A). The graph replay launches on the DSP
    // execution stream (stream B = cudaStr). Without explicit ordering, stream B can
    // launch the graph BEFORE stream A's assign completes — reading stale data.
    //
    // Fix: record a CUDA event on the default stream after all prior work, then make
    // the DSP stream wait on it. This creates a happens-before relationship:
    //   assign() on default stream → event → DSP stream graph launch
    //
    // Only needed when the DSP stream differs from the default stream (which it
    // always does in the replay path since DspStreamGuard sets tl_dspExecutionStream).
    {
      cudaStream_t defaultStream = nullptr;
      auto* defaultStreamPtr = LaunchContext::defaultContext()->getCudaStream();
      if (defaultStreamPtr != nullptr) {
        defaultStream = *defaultStreamPtr;
      }
      if (defaultStream != nullptr && defaultStream != cudaStr) {
        cudaEvent_t crossStreamEvt;
        cudaEventCreateWithFlags(&crossStreamEvt, cudaEventDisableTiming);
        cudaEventRecord(crossStreamEvt, defaultStream);
        cudaStreamWaitEvent(cudaStr, crossStreamEvt, 0);
        cudaEventDestroy(crossStreamEvt);
        DSP_DIAG(EXECUTE, "CROSS_STREAM_SYNC: DSP stream %p waiting on default stream %p event "
                 "for seg[%d-%d] — ensures .assign() data visible to graph replay",
                 (void*)cudaStr, (void*)defaultStream, seg.def.startSlot, seg.def.endSlot);
      }
    }

    if (useFastReplay) {
      // Fast path: arg table pointers are stable so skip refresh.
      cudaGetLastError();
      if (!variableIndicesCached_) {
        variableExternalInputIndices_.clear();
        for (int ei = 0; ei < numExt; ei++) {
          if (ei < static_cast<int>(externalInputIsVariable_.size()) &&
              externalInputIsVariable_[ei]) {
            variableExternalInputIndices_.push_back(ei);
          }
        }
        variableIndicesCached_ = true;
        DSP_DIAG(EXECUTE, "FAST_REPLAY_CACHED_VARIABLE_INDICES: %d variable inputs out of %d total",
                 static_cast<int>(variableExternalInputIndices_.size()), numExt);
      }
      int fastSynced = 0;
      for (int ei : variableExternalInputIndices_) {
        if (ei >= numExt || externalArrays[ei] == nullptr) continue;
        externalArrays[ei]->syncToDevice();
        fastSynced++;
      }
      DSP_DIAG(EXECUTE, "FAST_REPLAY_EXT_SYNC: %d H2D (of %d variable) execCount=%d",
               fastSynced, static_cast<int>(variableExternalInputIndices_.size()),
               seg.exec.executionCount);

      // CRITICAL FIX: Copy consolidated arg table to device during fast replay.
      // The captured graph's H2D memcpy nodes copy from consolidatedArgTableHostPinned
      // to consolidatedArgTableDevice. The host-pinned table was populated during
      // capture and contains correct pointers (since argTableStable=true). But the
      // device arg table must be updated before graph launch. Without this copy,
      // the device arg table may have stale data from a previous execution, causing
      // Triton kernels to read/write wrong buffers.
#if HAVE_TRITON && defined(SD_CUDA)
      {
        auto* tritonBackendFast = dynamic_cast<TritonGraphBackend*>(backend);
        if (tritonBackendFast != nullptr) {
          tritonBackendFast->copyConsolidatedArgTableToDevice(seg, stream);
        }
      }
#endif
    } else {
      DspDiagnostics::ExtInputSyncResult syncResult = {0, 0, 0};
      DSP_DIAG_DUMP_EXT_INPUTS(externalArrays, numExt, seg.exec.executionCount, syncResult);
      int synced = 0, skipped = 0;
      if (shapesFrozen_ && !externalInputIsVariable_.empty()) {
        // Frozen replay: only sync variable inputs
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] == nullptr) continue;
          if (ei < static_cast<int>(externalInputIsVariable_.size()) &&
              !externalInputIsVariable_[ei]) {
            skipped++;
            continue;
          }
          auto* db = externalArrays[ei]->dataBuffer();
          bool pAct = db ? db->isPrimaryActual() : false;
          bool sAct = db ? db->isSpecialActual() : false;
          if (pAct && !sAct) synced++;
          else skipped++;
          externalArrays[ei]->syncToDevice();
        }
      } else {
        // Non-frozen or no variable info: sync all
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] != nullptr) {
            auto* db = externalArrays[ei]->dataBuffer();
            bool pAct = db ? db->isPrimaryActual() : false;
            bool sAct = db ? db->isSpecialActual() : false;
            if (pAct && !sAct) synced++;
            else skipped++;
            externalArrays[ei]->syncToDevice();
          }
        }
      }
      DSP_DIAG(EXECUTE, "EXT_INPUT_SYNC replay: %d H2D, %d skip (device up-to-date) execCount=%d",
               synced, skipped, seg.exec.executionCount);

      // Dump SMALL variable external inputs (verify mode only)
      if (Environment::getInstance().tritonVerifyKernels()) {
      cudaDeviceSynchronize();
      for (int ei = 0; ei < numExt; ei++) {
        if (externalArrays[ei] == nullptr) continue;
        auto* arr = externalArrays[ei];
        bool isSmall = arr->lengthOf() <= 16;
        std::string name = (ei < (int)externalInputNames_.size()) ? externalInputNames_[ei] : "?";
        std::string vals = "?";
        if (isSmall && DSP_BUF(arr)) {
          int n = std::min((int)arr->lengthOf(), 4);
          int elemSize = DataTypeUtils::sizeOf(arr->dataType());
          std::vector<uint8_t> devBytes(n * elemSize);
          cudaMemcpy(devBytes.data(), DSP_BUF(arr), n * elemSize, cudaMemcpyDeviceToHost);
          vals = "";
          for (int j = 0; j < n; j++) {
            if (j > 0) vals += ",";
            if (arr->dataType() == INT64 || arr->dataType() == DataType::INT64) {
              int64_t v; std::memcpy(&v, devBytes.data() + j * 8, 8);
              vals += std::to_string(v);
            } else if (arr->dataType() == INT32) {
              int32_t v; std::memcpy(&v, devBytes.data() + j * 4, 4);
              vals += std::to_string(v);
            } else if (arr->dataType() == FLOAT32) {
              float v; std::memcpy(&v, devBytes.data() + j * 4, 4);
              vals += std::to_string(v);
            } else {
              vals += "?";
            }
          }
        }
        if (!isSmall || name.find("input") != std::string::npos ||
            name.find("position") != std::string::npos ||
            name.find("attention") != std::string::npos ||
            name.find("embed") != std::string::npos ||
            name.find("past") != std::string::npos) {
          DSP_DIAG(EXECUTE, "EXT_DATA[%d]:\"%s\" type=%d rank=%d len=%lld addr=%p vals=[%s] execCount=%d",
                   ei, name.c_str(), (int)arr->dataType(), (int)arr->rankOf(),
                   (long long)arr->lengthOf(),
                   DSP_BUF(arr), vals.c_str(), seg.exec.executionCount);
        }
      }
      } // end tritonVerifyKernels() EXT_DATA dump
    }
    // Snapshot buffer addresses BEFORE replay for comparison with capture-time addresses.
    // REPLAY OPTIMIZATION: Only compute address snapshots during first few replays
    // or when diagnostics are enabled. In stable replay (executionCount >= 4),
    // addresses don't change. Skipping saves 2 vector allocations + iteration
    // over all output slots + external inputs (~3000+ arrays) per segment.
    if (seg.exec.executionCount < 4 || DSP_DIAG_ENABLED(EXECUTE)) {
      std::vector<void*> outAddrs, extAddrs;
      extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
      extractDeviceAddrs(externalArrays, numExt, extAddrs);
      DSP_DIAG_SNAPSHOT_ADDRS("replay-entry", outAddrs.data(), totalOutputSlots_,
                               extAddrs.data(), numExt);
      int mismatches = DSP_DIAG_COMPARE_ADDRS("capture-entry", "replay-entry");
      if (mismatches > 0) {
        DSP_DIAG(EXECUTE, "WARNING: %d address mismatches between capture and replay!", mismatches);
      }
    }

#if HAVE_TRITON && defined(SD_CUDA)
    {
      auto* tritonBackend = dynamic_cast<TritonGraphBackend*>(backend);
      if (tritonBackend != nullptr) {
        auto refreshStatus = tritonBackend->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                 outputSlots_, totalOutputSlots_,
                                                 stream);
        if (refreshStatus != Status::OK) {
          platformCleanupSegmentForRebuild(seg);
          seg.exec.argTableStable = false;
          batchD2DCount_ = 0;
          seg.exec.cachedShapeKey = 0;
          seg.exec.compilationFailed = true;
          char buf[256];
          snprintf(buf, sizeof(buf),
                   "NativeDSP: refreshArgTablesForReplay FAILED for seg[%d-%d] "
                   "shapeKey=%lld execCount=%d. Arg table refresh must not fail during replay "
                   "— fix the root cause.",
                   seg.def.startSlot, seg.def.endSlot, (long long)seg.def.shapeKey,
                   seg.exec.executionCount);
          DSP_DIAG(COMPILE, "%s", buf);
          THROW_EXCEPTION(buf);
        }
      }

      // CRITICAL FIX: After refreshing arg tables on host, copy to device BEFORE graph launch.
      // The captured graph has arg table addresses baked in - we just need to update the content.
      // This consolidated copy replaces ~N per-kernel cudaMemcpyAsync calls with ONE copy.
      if (tritonBackend != nullptr) {
        tritonBackend->copyConsolidatedArgTableToDevice(seg, stream);
      }

    }
#endif
    cudaGetLastError();  // Clear any sticky errors

    // DIAGNOSTIC: Zero capture workspace before replay to test stale-data hypothesis.
    // If zeroing the workspace fixes divergence, stale workspace data is the root cause.
    // This is gated on tritonVerifyKernels to avoid performance impact in production.
    if (Environment::getInstance().tritonVerifyKernels() &&
        seg.exec.replayHandle && seg.exec.replayHandle->getWorkspacePtr() != nullptr &&
        seg.exec.replayHandle->getWorkspaceBytes() > 0) {
      cudaMemsetAsync(seg.exec.replayHandle->getWorkspacePtr(), 0,
                      seg.exec.replayHandle->getWorkspaceBytes(), cudaStr);
      cudaStreamSynchronize(cudaStr);
      DSP_DIAG(VERIFY, "REPLAY_DIAG: zeroed capture workspace (%zuMB) before replay execCount=%d",
               seg.exec.replayHandle->getWorkspaceBytes() / (1024*1024), seg.exec.executionCount);
    }

    // DIAGNOSTIC: Dump specific VARIABLE external inputs before replay to trace stale data.
    if (Environment::getInstance().tritonVerifyKernels()) {
      cudaDeviceSynchronize();
      for (int ei = 0; ei < numExt; ei++) {
        if (ei < (int)externalInputIsVariable_.size() && externalInputIsVariable_[ei] &&
            externalArrays[ei] != nullptr && externalArrays[ei]->lengthOf() <= 8) {
          auto* arr = externalArrays[ei];
          auto* db = arr->dataBuffer();
          int n = std::min((int)arr->lengthOf(), 8);
          int elemSize = DataTypeUtils::sizeOf(arr->dataType());
          std::vector<uint8_t> hostBytes(n * elemSize), devBytes(n * elemSize);
          if (db && db->primary()) std::memcpy(hostBytes.data(), static_cast<char*>(arr->buffer()), n * elemSize);
          if (DSP_BUF(arr)) cudaMemcpy(devBytes.data(), DSP_BUF(arr), n * elemSize, cudaMemcpyDeviceToHost);
          float hv[8]={0}, dv[8]={0};
          dspBytesToFloat(hostBytes.data(), arr->dataType(), hv, n);
          dspBytesToFloat(devBytes.data(), arr->dataType(), dv, n);
          std::string name = (ei < (int)externalInputNames_.size()) ? externalInputNames_[ei] : "?";
          DSP_DIAG(VERIFY, "PRE_REPLAY ext#%d:\"%s\" len=%d pAct=%d sAct=%d host=[%.0f,%.0f,%.0f,%.0f] dev=[%.0f,%.0f,%.0f,%.0f]",
                    ei, name.c_str(), n,
                    db ? (db->isPrimaryActual()?1:0) : -1,
                    db ? (db->isSpecialActual()?1:0) : -1,
                    hv[0],hv[1],hv[2],hv[3], dv[0],dv[1],dv[2],dv[3]);
        }
      }
    }

    // Pre-replay batch-zero: zero all output buffers OUTSIDE the graph.
    // Individual cudaMemsetAsync calls use dedicated fill engines (not SMs),
    // pipeline efficiently, and add 0 graph nodes (they run before cudaGraphLaunch).
    // Stream ordering guarantees all zeroing completes before graph launch.
    // NOTE: Do NOT use batchZeroKernel here — it runs on SMs (competition with
    // compute kernels) and has alignment requirements that cause accuracy issues.
    // Use per-segment batch-zero entries (saved during capture) instead of the
    // shared batchZeroEntries_ which only contains the LAST captured segment's data.
    auto& segBZ = seg.exec.segBatchZeroEntries;
    if (Environment::getInstance().dspBatchZero() && !segBZ.empty()) {
      // Refresh batch-zero pointers from current outputSlots_ entries.
      // During frozen replay, the pre-exec restoration may be skipped (optimization),
      // but outputSlots_ entries persist with stable shapes. Re-derive the GPU
      // pointer from the authoritative source to avoid stale pointers that cause
      // CUDA error 700 (illegal memory access) during cudaMemsetAsync.
      for (auto& entry : segBZ) {
        if (entry.outputSlotIndex >= 0 && entry.outputSlotIndex < totalOutputSlots_) {
          NDArray* cached = outputSlots_[entry.outputSlotIndex];
          if (cached != nullptr && DSP_BUF(cached) != nullptr) {
            entry.ptr = DSP_BUF(cached);
            entry.bytes = static_cast<int>(cached->dataBuffer()->getLenInBytes());
          }
        }
      }
      for (auto& entry : segBZ) {
        if (entry.ptr != nullptr && entry.bytes > 0) {
          cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
        }
      }
      DSP_DIAG(MEMORY, "pre-replay batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, outside graph) seg[%d-%d]",
                static_cast<int>(segBZ.size()), seg.def.startSlot, seg.def.endSlot);
    }

    // Replay strategy: configurable via ND4J_TRITON_GRAPH_REINSTANTIATE.
    // Default (OFF): direct replay of existing graphExec.
    // ON: destroy and re-instantiate graphExec from graph template before each replay.
    // Skip entirely if lineage validation or cross-segment size mismatch invalidated the graph.
    {
      // NOTE: Replay preserves the shared cuBLAS workspace. Capture zeroes it only
      // for the first captured segment in a fresh session; once graphs exist, later
      // captures and all replays preserve the accumulated plan/descriptor state.
      // Per-segment replay zeroing was destroying cuBLAS state that later segments
      // depend on but do not re-upload via explicit H2D nodes.

      // Pre-launch CUDA error check: detect accumulated errors from prior segments
      // that would manifest as hangs during this segment's cudaStreamSynchronize.
      {
        cudaError_t preLaunchErr = cudaPeekAtLastError();
        if (preLaunchErr != cudaSuccess) {
          DSP_DIAG(EXECUTE, "PRE_REPLAY_ERROR: seg[%d-%d] cudaPeekAtLastError=%d (%s) — clearing",
                   seg.def.startSlot, seg.def.endSlot, (int)preLaunchErr,
                   cudaGetErrorString(preLaunchErr));
          cudaGetLastError();  // clear it
        }
      }

      // ── TRIPWIRE: validate all pointers before graph launch ──────────────
      // NULL dereference in libcuda.so during cudaGraphLaunch means a kernel
      // arg or memcpy source/dest is NULL. Check everything we can reach.
      //
      // REPLAY OPTIMIZATION: Skip tripwire on stable replay (executionCount >= 4).
      // After 3+ successful replays, pointers are stable. Running the full
      // tripwire still adds host-side overhead, so only run during the first
      // few replays and when verify mode is on.
      if (seg.exec.replayHandle &&
          (seg.exec.executionCount < 4 || Environment::getInstance().tritonVerifyKernels())) {
        // Check output slot device pointers for this segment's range
        int nullSlots = 0;
        for (int si = seg.def.startSlot; si <= seg.def.endSlot && si < numSlots_; si++) {
          for (int oi = 0; oi < slots_[si].wiring.numOutputs; oi++) {
            int slotIdx = slots_[si].wiring.outputSlotIndices[oi];
            if (slotIdx >= 0 && slotIdx < totalOutputSlots_) {
              NDArray* slotArr = outputSlots_[slotIdx];
              if (slotArr == nullptr || DSP_BUF(slotArr) == nullptr) {
                nullSlots++;
                DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_SLOT: seg[%d-%d] step=%d "
                         "outputSlot=%d arr=%p devPtr=%p",
                         seg.def.startSlot, seg.def.endSlot, si, slotIdx,
                         (void*)slotArr,
                         slotArr ? DSP_BUF(slotArr) : nullptr);
              }
            }
          }
        }
        // Check workspace pointer
        void* wsPtr = seg.exec.replayHandle->getWorkspacePtr();
        size_t wsBytes = seg.exec.replayHandle->getWorkspaceBytes();
        if (wsBytes > 0 && wsPtr == nullptr) {
          DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_WORKSPACE: seg[%d-%d] wsBytes=%zu "
                   "wsPtr=NULL — graph H2D nodes will crash!",
                   seg.def.startSlot, seg.def.endSlot, wsBytes);
        }
        // Check captured host pointers
        auto& hostPtrs = seg.exec.replayHandle->getCapturedHostPtrs();
        int nullHostPtrs = 0;
        for (size_t hi = 0; hi < hostPtrs.size(); hi++) {
          if (hostPtrs[hi] == nullptr) {
            nullHostPtrs++;
            DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_HOSTPTR: seg[%d-%d] hostPtr[%zu]=NULL",
                     seg.def.startSlot, seg.def.endSlot, hi);
          }
        }
        // Check key external inputs that the segment uses
        int nullExtInputs = 0;
        for (int ei = 0; ei < numExt; ei++) {
          if (externalArrays[ei] != nullptr &&
              DSP_BUF(externalArrays[ei]) == nullptr) {
            nullExtInputs++;
            if (nullExtInputs <= 5) {
              std::string name = (ei < (int)externalInputNames_.size())
                                 ? externalInputNames_[ei] : "?";
              DSP_DIAG(EXECUTE, "TRIPWIRE_NULL_EXT_DEVPTR: seg[%d-%d] ext[%d]=\"%s\" "
                       "len=%lld dtype=%d — device pointer is NULL",
                       seg.def.startSlot, seg.def.endSlot, ei, name.c_str(),
                       (long long)externalArrays[ei]->lengthOf(),
                       (int)externalArrays[ei]->dataType());
            }
          }
        }
        // Summary
        if (nullSlots > 0 || nullHostPtrs > 0 || nullExtInputs > 0 ||
            (wsBytes > 0 && wsPtr == nullptr)) {
          DSP_DIAG(EXECUTE, "TRIPWIRE_SUMMARY: seg[%d-%d] DANGER — "
                   "nullSlots=%d nullHostPtrs=%d nullExtDevPtrs=%d "
                   "wsPtr=%p wsBytes=%zu",
                   seg.def.startSlot, seg.def.endSlot,
                   nullSlots, nullHostPtrs, nullExtInputs,
                   wsPtr, wsBytes);
        } else {
          DSP_DIAG(EXECUTE, "TRIPWIRE_OK: seg[%d-%d] %d hostPtrs, ws=%p/%zuMB — no NULL pointers detected",
                   seg.def.startSlot, seg.def.endSlot,
                   (int)hostPtrs.size(), wsPtr, wsBytes / (1024*1024));
        }
      }
      // ── END TRIPWIRE ─────────────────────────────────────────────────────

      // Output buffers are now captured directly. If any slot address changes,
      // the graph has stale baked-in pointers and must be rebuilt.
      //
      // REPLAY OPTIMIZATION: Skip fingerprinting during stable replay
      // (executionCount >= 4 with argTableStable). In frozen replay, buffer
      // addresses are persistent — they never get freed/reallocated. The hash
      // computation iterates all output slots in the segment range, adding
      // host-side overhead per segment per step.
      if (seg.exec.capturedSlotAddrHash != 0 &&
          (seg.exec.executionCount < 4 || !seg.exec.argTableStable)) {
        LongType currentAddrHash = computeSlotAddrHash(
            outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
        if (currentAddrHash != seg.exec.capturedSlotAddrHash) {
          platformCleanupSegmentForRebuild(seg);
          seg.exec.argTableStable = false;
          batchD2DCount_ = 0;
          seg.exec.capturedInputAddrKey = 0;
          seg.exec.capturedCreateValueKey = 0;
          seg.exec.compilationFailed = true;
          char buf[256];
          snprintf(buf, sizeof(buf),
                   "NativeDSP: SLOT ADDRESS DRIFT for seg[%d-%d]: "
                   "captured=0x%llx current=0x%llx. Output slot addresses changed after "
                   "capture — this indicates a buffer lifecycle bug. Fix the root cause.",
                   seg.def.startSlot, seg.def.endSlot,
                   (long long)seg.exec.capturedSlotAddrHash, (long long)currentAddrHash);
          DSP_DIAG(COMPILE, "%s", buf);
          THROW_EXCEPTION(buf);
        }
      }

      // COMPOSITE REPLAY: if this segment has gap units in its schedule, execute
      // gap slots BEFORE the monolithic graph replay. Gaps produce outputs at
      // stable outputSlots_ addresses (same as warmup), which the graph reads
      // from its captured arg table. This ensures gap outputs are fresh when
      // the graph's islands consume them.
      auto& sched = seg.exec.compositeReplaySchedule;
      DSP_DIAG(SHAPE, "COMPOSITE_SCHEDULE_CHECK: seg[%d-%d] units=%d handles=%d execCount=%d",
               seg.def.startSlot, seg.def.endSlot,
               static_cast<int>(sched.units.size()),
               static_cast<int>(sched.compositeReplayHandles.size()),
               seg.exec.executionCount);
      bool hasGapsInSchedule = false;
      if (!sched.units.empty()) {
        for (auto& u : sched.units) {
          if (u.kind == REPLAY_UNIT_GAP) {
            hasGapsInSchedule = true;
            break;
          }
        }
      }

      bool replayOk = false;
      if (hasGapsInSchedule) {
        // Execute gap units in order, then replay the monolithic graph.
        for (auto& u : sched.units) {
          if (u.kind == REPLAY_UNIT_GAP) {
            DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: executing gap[%d-%d] before graph for seg[%d-%d]",
                     u.startSlot, u.endSlot, seg.def.startSlot, seg.def.endSlot);
            for (int s = u.startSlot; s <= u.endSlot; s++) {
              auto gapStatus = executeSlot(s, externalArrays, numExt, stream);
              if (gapStatus != Status::OK) {
                DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: gap slot %d FAILED status=%d",
                         s, static_cast<int>(gapStatus));
                platformCleanupSegmentForRebuild(seg);
                return gapStatus;
              }
            }
          }
        }
        // Now replay the Triton-only graph (gap ops are NOT in the graph).
        if (!seg.exec.replayHandle) {
          DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: no replay handle seg[%d-%d]",
                   seg.def.startSlot, seg.def.endSlot);
          platformCleanupSegmentForRebuild(seg);
          return Status::KERNEL_FAILURE;
        }

        // CRITICAL FIX: Refresh arg tables AFTER gap execution but BEFORE graph replay.
        // Gap ops (e.g., view-producing reshape/permute) may replace output slot NDArray
        // wrappers via the frozen view fast path in executeSlot(). While the underlying
        // DataBuffer (GPU address) is shared with the input, the output slot's
        // specialBuffer() address changes when the slot's previous own DataBuffer is
        // replaced with a view of the input's DataBuffer. The captured graph's H2D memcpy
        // nodes bake in the arg table addresses from capture time. Without refreshing,
        // the graph replays with stale arg table addresses, causing Triton kernels to
        // read/write wrong buffers → garbage output → "User" repeating tokens.
#if HAVE_TRITON && defined(SD_CUDA)
        {
          auto* tritonBackend2 = dynamic_cast<TritonGraphBackend*>(backend);
          if (tritonBackend2 != nullptr) {
            auto refreshStatus2 = tritonBackend2->refreshArgTablesForReplay(seg, externalArrays, numExt,
                                                     outputSlots_, totalOutputSlots_,
                                                     stream);
            if (refreshStatus2 != Status::OK) {
              platformCleanupSegmentForRebuild(seg);
              seg.exec.argTableStable = false;
              batchD2DCount_ = 0;
              seg.exec.capturedInputAddrKey = 0;
              seg.exec.capturedCreateValueKey = 0;
              seg.exec.compilationFailed = true;
              char buf[256];
              snprintf(buf, sizeof(buf),
                       "NativeDSP: COMPOSITE_REPLAY refreshArgTablesForReplay FAILED for "
                       "seg[%d-%d] post-gap. Arg table refresh must not fail during "
                       "composite replay — fix the root cause.",
                       seg.def.startSlot, seg.def.endSlot);
              DSP_DIAG(COMPILE, "%s", buf);
              THROW_EXCEPTION(buf);
            }
            tritonBackend2->copyConsolidatedArgTableToDevice(seg, stream);
          }
        }
#endif

        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: launching graph for seg[%d-%d] after gaps",
                 seg.def.startSlot, seg.def.endSlot);
        replayOk = seg.exec.replayHandle->replay(stream);
        DSP_DIAG(EXECUTE, "COMPOSITE_REPLAY: graph replay %s for seg[%d-%d]",
                 replayOk ? "OK" : "FAILED", seg.def.startSlot, seg.def.endSlot);
      } else if (!seg.exec.replayHandle) {
        DSP_DIAG(EXECUTE, "REPLAY_SKIPPED: handle=%p seg[%d-%d]",
                 (void*)seg.exec.replayHandle.get(), seg.def.startSlot, seg.def.endSlot);
      } else if (Environment::getInstance().tritonGraphReinstantiate()) {
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                     "seg[%d-%d] REPLAY via reInstantiate path (execCount=%d replays=%d)",
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                     seg.exec.replayHandle->getStatistics().replayCount);
        auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
        if (!cudaReplay->getNativeHandle()->reInstantiate()) {
          DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph reInstantiate FAILED for seg[%d-%d]",
                    seg.def.startSlot, seg.def.endSlot);
        } else {
          replayOk = seg.exec.replayHandle->replay(stream);
          DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                       "seg[%d-%d] reInstantiate replay %s",
                       seg.def.startSlot, seg.def.endSlot, replayOk ? "OK" : "FAILED");
        }
      } else {
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                     "seg[%d-%d] REPLAY via direct path (execCount=%d replays=%d)",
                     seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount,
                     seg.exec.replayHandle->getStatistics().replayCount);
        replayOk = seg.exec.replayHandle->replay(stream);
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                     "seg[%d-%d] direct replay %s",
                     seg.def.startSlot, seg.def.endSlot, replayOk ? "OK" : "FAILED");
      }
      if (replayOk) {
        // LRU tracking: record when this segment was last replayed for eviction ordering
        seg.exec.lastReplayExecCount = executeCount_;

        // Find the ACTUAL final output slot index (not the step index)
        int finalOutputSlot = -1;
        if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
          finalOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
        }
        // Fallback to seg.def.endSlot if output slot lookup fails
        if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_) {
          finalOutputSlot = seg.def.endSlot;
        }

        // ── Timed sync with 30s timeout — BEFORE any D2H diagnostic copies ──
        // DSP_DIAG_DUMP_SLOT and DSP_DIAG_DUMP_SEG_OUTPUT internally call
        // cudaStreamSynchronize via safeDtoH(). If the GPU is hung, those calls
        // block forever. By syncing here first with a timeout, we can detect
        // and report the hang instead of blocking.
        bool replaySyncOk = true;
        if (DSP_DIAG_ENABLED(EXECUTE)) {
          // Graph node stats for the replayed graph
          auto* cudaReplayForDiag = dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
          if (cudaReplayForDiag && cudaReplayForDiag->getNativeHandle()) {
            auto stats = cudaReplayForDiag->getNativeHandle()->getStatistics();
            DSP_DIAG(EXECUTE, "REPLAY_GRAPH_STATS: seg[%d-%d] kernels=%d memcpyH2D=%d memsets=%d "
                     "memAllocs=%d memFrees=%d hostCbs=%d childGraphs=%d totalNodes=%zu",
                     seg.def.startSlot, seg.def.endSlot,
                     stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                     stats.numMemAllocs, stats.numMemFrees, stats.numHostCallbacks,
                     stats.numChildGraphs, cudaReplayForDiag->getNativeHandle()->getNumNodes());
          }
          fflush(stdout); fflush(stderr);

          // Timed sync: use event polling with 30s timeout
          cudaEvent_t syncEvt;
          cudaEventCreateWithFlags(&syncEvt, cudaEventDisableTiming);
          cudaEventRecord(syncEvt, cudaStr);

          auto syncStart = std::chrono::steady_clock::now();
          const int timeoutSec = 30;
          while (true) {
            cudaError_t evtErr = cudaEventQuery(syncEvt);
            if (evtErr == cudaSuccess) {
              break;
            } else if (evtErr == cudaErrorNotReady) {
              auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::steady_clock::now() - syncStart).count();
              if (elapsed >= timeoutSec) {
                replaySyncOk = false;
                DSP_DIAG(EXECUTE, "GPU_HANG_DETECTED: seg[%d-%d] cudaEventQuery not ready after %ds "
                         "— graph replay stuck! execCount=%d replays=%d",
                         seg.def.startSlot, seg.def.endSlot, timeoutSec,
                         seg.exec.executionCount, seg.exec.replayHandle->getStatistics().replayCount);
                // Check for CUDA errors that might explain the hang
                cudaError_t hangErr = cudaPeekAtLastError();
                if (hangErr != cudaSuccess) {
                  DSP_DIAG(EXECUTE, "GPU_HANG_CUDA_ERROR: %d (%s)", (int)hangErr,
                           cudaGetErrorString(hangErr));
                }
                // Log GPU memory state
                size_t freeMem = 0, totalMem = 0;
                cudaMemGetInfo(&freeMem, &totalMem);
                DSP_DIAG(EXECUTE, "GPU_HANG_MEM: free=%zuMB total=%zuMB used=%zuMB",
                         freeMem/(1024*1024), totalMem/(1024*1024),
                         (totalMem-freeMem)/(1024*1024));
                fflush(stdout); fflush(stderr);
                // Fatal: graph replay hang means GPU is stuck. Continuing
                // produces garbage and may cascade into further hangs.
                {
                  std::string msg = "CUDA graph replay hung for seg[" +
                      std::to_string(seg.def.startSlot) + "-" +
                      std::to_string(seg.def.endSlot) + "] after " +
                      std::to_string(timeoutSec) + "s — aborting execution";
                  THROW_EXCEPTION(msg.c_str());
                }
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
              DSP_DIAG(EXECUTE, "REPLAY_SYNC_ERROR: seg[%d-%d] cudaEventQuery returned %d (%s)",
                       seg.def.startSlot, seg.def.endSlot, (int)evtErr, cudaGetErrorString(evtErr));
              // Fatal: CUDA error during graph replay means the graph is corrupt
              {
                std::string msg = "CUDA graph replay error for seg[" +
                    std::to_string(seg.def.startSlot) + "-" +
                    std::to_string(seg.def.endSlot) + "]: cudaEventQuery returned " +
                    std::to_string((int)evtErr) + " (" +
                    cudaGetErrorString(evtErr) + ")";
                THROW_EXCEPTION(msg.c_str());
              }
            }
          }
          cudaEventDestroy(syncEvt);
        }

        // Only do D2H diagnostic copies if sync succeeded (GPU not hung)
        if (replaySyncOk && finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
            outputSlots_[finalOutputSlot] != nullptr) {
          auto* finalOut = outputSlots_[finalOutputSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("replay", finalOutputSlot,
                               DSP_BUF(finalOut), finalOut->lengthOf());
          }
          if (finalOut->dataType() == FLOAT32 && finalOut->lengthOf() > 0) {
            DSP_DIAG_DUMP_SEG_OUTPUT("GRAPH_REPLAY", finalOutputSlot, DSP_BUF(finalOut),
                                     finalOut->lengthOf(), seg.exec.executionCount, stream);
          }
          if (DSP_DIAG_ENABLED(EXECUTE)) {
            int replayArgmax = dspArgmax(DSP_BUF(finalOut), finalOut->dataType(),
                                         finalOut->lengthOf());
            std::string firstVals = dspDumpSlotValues(DSP_BUF(finalOut), finalOut->dataType(),
                                                       finalOut->lengthOf(), 4);
            DSP_DIAG(EXECUTE, "GRAPH_REPLAY ARGMAX: slot=%d argmax=%d len=%lld vals=%s execCount=%d",
                     finalOutputSlot, replayArgmax, (long long)finalOut->lengthOf(),
                     firstVals.c_str(), seg.exec.executionCount);
          }
        }

        seg.exec.executionCount++;
        totalGraphReplays_++;

        // ── Capture view recipes during SHAPES_FROZEN ───────────────────
        // After successful execution, capture view recipes for view-capable
        // ops in this segment. This happens once during SHAPES_FROZEN so that
        // subsequent REPLAYING phases can install views instead of executing gaps.
        if (shapesFrozen_ && seg.exec.viewRecipes.recipes.empty()) {
          int captured = captureViewRecipesForSegment(seg, slots_, externalArrays, numExt,
                                                      outputSlots_, slotOwnership_,
                                                      totalOutputSlots_);
          if (captured > 0) {
            DSP_DIAG(EXECUTE, "VIEW_RECIPE_CAPTURE: seg[%d-%d] captured %d view recipes",
                     seg.def.startSlot, seg.def.endSlot, captured);
          }

          // Also attempt to fold pure shape-expression chains in this segment.
          // Shape chains are folded once and replayed by direct buffer install —
          // no kernel launches needed for shape-only subgraphs.
          if (isPureShapeChain(seg.def.startSlot, seg.def.endSlot, slots_, outputSlots_)) {
            FoldedShapeChain folded;
            if (foldShapeChain(seg.def.startSlot, seg.def.endSlot, slots_, outputSlots_,
                               externalArrays, numExt, folded)) {
              DSP_DIAG(EXECUTE, "SHAPE_CHAIN_FOLD: seg[%d-%d] folded %d shape results",
                       seg.def.startSlot, seg.def.endSlot, static_cast<int>(folded.results.size()));
              // Store folded chain in segment exec state for replay install
              // (In production, this would be persisted in seg.exec or a plan-level cache)
            }
          }
        }

        // ── Validate view recipes during POINTERS_STABLE ────────────────
        if (planPhase_ >= PlanPhase::POINTERS_STABLE && !seg.exec.viewRecipes.recipes.empty()) {
          bool valid = validateViewRecipes(seg, outputSlots_, externalArrays, numExt);
          if (!valid) {
            std::string msg = "DSP replay phase violation: view recipe source drift for seg[" +
                std::to_string(seg.def.startSlot) + "-" + std::to_string(seg.def.endSlot) +
                "] after POINTERS_STABLE";
            THROW_EXCEPTION(msg.c_str());
          }
        }

        // ── REPLAY VERIFICATION ─────────────────────────────────────────
        if (replaySyncOk && Environment::getInstance().tritonVerifyKernels()) {
          cudaStreamSynchronize(cudaStr);
          performReplayVerify(seg, externalArrays, numExt, stream, "TRITON");
        }

        // Force re-capture every step (diagnostic mode).
        // Invalidates the cached graph after each replay so the next step
        // re-captures with fresh data.  Correct but slow.
        if (Environment::getInstance().tritonForceRecapture()) {
          platformCleanupSegmentForRebuild(seg);
          seg.exec.argTableStable = false;
          batchD2DCount_ = 0;
          seg.exec.capturedInputAddrKey = 0;
          DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after replay execCount=%d", seg.exec.executionCount);
        }

        // ── Replay contract: gap ops must NOT be baked into CUDA graphs ──
        // Gap ops (matmul, attention, etc.) are excluded from CUDA graph capture.
        // They are executed fresh via the composite replay schedule before graph
        // replay. If gapOpsCapturedInGraph is true, it means a stale graph from
        // before the fix — invalidate and recapture.
        if (replaySyncOk) {
#if HAVE_TRITON && defined(SD_CUDA)
          auto* tritonBE = dynamic_cast<TritonGraphBackend*>(backend);
          if (tritonBE != nullptr) {
            auto gapSlots = tritonBE->getGapSlots(seg, slots_);
            if (!gapSlots.empty() && seg.exec.gapOpsCapturedInGraph) {
              DSP_DIAG(FALLBACK, "STALE_GAP_GRAPH: seg[%d-%d] has %d gap slots baked into graph "
                       "— invalidating for recapture without gap ops",
                       seg.def.startSlot, seg.def.endSlot, static_cast<int>(gapSlots.size()));
              platformCleanupSegmentForRebuild(seg);
              seg.exec.argTableStable = false;
              batchD2DCount_ = 0;
              seg.exec.capturedInputAddrKey = 0;
              seg.exec.capturedCreateValueKey = 0;
              seg.exec.executionCount = captureMinExec;
              seg.exec.gapOpsCapturedInGraph = false;
              return Status::KERNEL_FAILURE;
            }
          }
#endif  // HAVE_TRITON
        }

        // Phase 2: outputSlots_ == outputSlots_ (unified).
        // Post-replay restoration is a no-op — arrays are already in place.

        if (Environment::getInstance().tritonVerifyKernels()) {
          DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=OK(replay) execCount=%d",
                    seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
        }
        return Status::OK;
      }
      // Launch failed — this is a fatal error. Graph replay failure means
      // the captured graph is corrupt or the CUDA runtime is in a bad state.
      {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        platformCleanupSegmentForRebuild(seg);
        return reportReplayError(seg, "graph_replay", cudaGetLastError(), deviceId);
      }
    }
  }
#endif

#if HAVE_TRITON && defined(SD_CUDA)
  struct TritonOrderedRangeGuard {
    bool active = false;
    ~TritonOrderedRangeGuard() {
      if (active) TritonGraphBackend::clearOrderedRangeExecutor();
    }
  } tritonOrderedRangeGuard;

  if (tritonBackend != nullptr) {
    TritonGraphBackend::setOrderedRangeExecutor(
        [this, &seg, externalArrays, numExt, stream](int startSlot, int endSlot) -> Status {
          if (startSlot > endSlot) return Status::OK;

          GraphSegment gapSeg;
          gapSeg.def.startSlot = startSlot;
          gapSeg.def.endSlot = endSlot;
          gapSeg.exec.executionCount = seg.exec.executionCount;
          gapSeg.exec.compilationFailed = seg.exec.compilationFailed;

          // Check if the Triton stream is currently being captured (CUDA graph recording).
          bool streamIsCapturing = false;
#ifdef SD_CUDA
          if (stream != nullptr) {
            cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
            cudaStreamIsCapturing(*static_cast<cudaStream_t*>(stream), &capStat);
            streamIsCapturing = (capStat != cudaStreamCaptureStatusNone);
          }

          cudaStream_t tritonStr = *static_cast<cudaStream_t*>(stream);
          auto* lcStream = LaunchContext::defaultContext()->getCudaStream();
          cudaStream_t gapStr = lcStream ? *lcStream : nullptr;
          bool streamsMatch = (tritonStr == gapStr);

          // One-time diagnostic: log whether streams match
          static bool streamDiagDone = false;
          if (!streamDiagDone) {
            DSP_DIAG(BACKEND, "stream diag: tritonStr=%p gapStr=%p match=%d capturing=%d",
                     (void*)tritonStr, (void*)gapStr, streamsMatch ? 1 : 0,
                     streamIsCapturing ? 1 : 0);
            streamDiagDone = true;
          }

          if (streamIsCapturing) {
            // ── CAPTURE PATH: SKIP gap ops entirely ──
            //
            // During CUDA graph capture, Triton kernels are recorded into the graph.
            // Gap ops (matmul, attention, etc.) must NOT execute because:
            //
            //  1. Executing on the capturing stream bakes stale addresses into the
            //     graph — on replay, gap ops read/write wrong buffers, producing
            //     garbage that accumulates across 30 transformer layers.
            //
            //  2. Executing on a separate stream also fails because native ops
            //     internally use the legacy stream (stream 0) for D2H copies,
            //     allocations, and syncs — all of which are illegal during capture
            //     (error 900/224: "operation would make the legacy stream depend
            //     on a capturing blocking stream").
            //
            // Solution: SKIP gap ops during capture. Warmup already executed them
            // and populated outputSlots_ at the correct addresses. The Triton arg
            // table snapshot will reference these warmup addresses. On replay, the
            // composite replay schedule executes gaps FRESH before graph replay.
            //
            // This is correct because:
            //  - Shapes are frozen (gap output shapes don't change)
            //  - Output buffer addresses are stable (same outputSlots_ from warmup)
            //  - Triton kernels reference buffer addresses via arg tables, which
            //    are refreshed from outputSlots_ before each replay
            //  - The captured graph contains ONLY Triton kernels

            DSP_DIAG(EXECUTE, "GAP_SKIP_DURING_CAPTURE: gap[%d-%d] SKIPPED (warmup outputs "
                     "already at stable addresses) for seg[%d-%d]",
                     startSlot, endSlot, seg.def.startSlot, seg.def.endSlot);

            // gapOpsCapturedInGraph stays false — gaps are NOT in the graph
            return Status::OK;
          }

          // ── NON-CAPTURE PATH: normal gap execution with stream sync ──
          // Triton kernels and gap ops run on different streams. Synchronize
          // to ensure gap ops see completed Triton outputs and vice versa.
          if (!streamsMatch && stream != nullptr) {
            cudaStreamSynchronize(tritonStr);
          }
#endif
          bool savedGraphActive = tl_graphExecutionActive;
          tl_graphExecutionActive = false;
          auto gapStatus = executeSegmentSlotBySlot(gapSeg, externalArrays, numExt, stream);
#ifdef SD_CUDA
          if (!streamsMatch && gapStr != nullptr) {
            cudaStreamSynchronize(gapStr);
          }
#endif
          tl_graphExecutionActive = savedGraphActive;
          return gapStatus;
        });
    tritonOrderedRangeGuard.active = true;
  }
#endif

  Status status = Status::KERNEL_FAILURE;
  bool usedTritonGraphCapture = false;

#ifdef SD_CUDA
  // Recompute shouldCaptureTritonGraph here (same logic as CAPTURE DECISION CHECK above).
  // This is the actual capture point - the diagnostic above just logs the decision.
  bool hasReplayHandleNow = (seg.exec.replayHandle != nullptr);
  bool replayHandleNullNow = (seg.exec.replayHandle == nullptr);
  bool execCountInWindowNow = (seg.exec.executionCount >= captureMinExec) &&
                              (forceRecaptureEnabled || seg.exec.executionCount <= (captureMinExec + 2));
  bool captureWindowSatisfiedNow = execCountInWindowNow || requiresOrderedGapCapture;
  bool shouldCaptureTritonGraphNow = allowTritonCudaGraphReplay &&
                                     !hasReplayHandleNow &&
                                     replayHandleNullNow &&
                                     !seg.exec.compilationFailed &&
                                     captureWindowSatisfiedNow &&
                                     hasCudaStream;
  // OOM retry deferred check: if a previous capture attempt failed with OOM and
  // we haven't reached the retry-after execution count, skip capture for this
  // execution and keep the warmup path active.
  if (seg.exec.captureOomRetries > 0 &&
      seg.exec.executionCount < seg.exec.captureRetryAfterExec) {
    DSP_DIAG_SEG(EXECUTE, seg.def.startSlot,
                 "OOM RETRY DEFERRED: seg[%d-%d] retries=%d execCount=%d retryAfter=%d — warmup path",
                 seg.def.startSlot, seg.def.endSlot, seg.exec.captureOomRetries,
                 seg.exec.executionCount, seg.exec.captureRetryAfterExec);
    shouldCaptureTritonGraphNow = false;
  }

  // Proactive memory cleanup before capture: trim pool, evict LRU graphs if needed.
  if (shouldCaptureTritonGraphNow && Environment::getInstance().dspProactiveEvictBeforeCapture()) {
    proactivePreCaptureMemoryCleanup(seg, segIdx, stream);
  }

  if (shouldCaptureTritonGraphNow) {
    DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                 "GRAPH CAPTURE BEGIN: seg[%d-%d] size=%d execCount=%d shapesFrozen=%d",
                 seg.def.startSlot, seg.def.endSlot, seg.def.endSlot - seg.def.startSlot + 1,
                 seg.exec.executionCount, shapesFrozen_ ? 1 : 0);
    seg.exec.gapOpsCapturedInGraph = false;

    // Set up capture workspace BEFORE beginCapture — cudaMalloc must be outside capture.
    // Native ordered range ops (matmul, attention, concat) need temporary buffers during execution.
    // With tl_graphExecutionActive=true, CudaMemoryPool allocates from this workspace
    // instead of calling cudaMallocAsync (which fails during capture).
    // TRITON_CAPTURE_WORKSPACE_SIZE is now at file scope (above).

    // Create the replayHandle BEFORE capture — it must exist to store workspace, host ptrs, etc.
    {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      seg.exec.replayHandle = GraphReplayFactory::create(deviceId);
    }

    if (seg.exec.replayHandle->getWorkspacePtr() == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);

      // Shared workspace: allocate once, reuse across all segments.
      // Segments execute sequentially (cudaGraphLaunch + cudaStreamSynchronize),
      // and workspace offset resets each capture, so sharing is safe.
      if (sharedCaptureWorkspace_ == nullptr) {
        // First segment — allocate the shared workspace
        cudaError_t err = cudaMalloc(&sharedCaptureWorkspace_, TRITON_CAPTURE_WORKSPACE_SIZE);
        if (err != cudaSuccess) {
          cudaGetLastError();
          sharedCaptureWorkspace_ = nullptr;
        }
        if (sharedCaptureWorkspace_ != nullptr) {
          sharedCaptureWorkspaceBytes_ = TRITON_CAPTURE_WORKSPACE_SIZE;
          sharedCaptureWorkspaceDevice_ = deviceId;
          memory::CudaMemoryPool::getInstance().registerCaptureWorkspace(
              sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                    "allocated SHARED capture workspace: %zuMB on device %d",
                    TRITON_CAPTURE_WORKSPACE_SIZE / (1024*1024), deviceId);
        } else {
          // Shared allocation failed — ABORT capture for this segment.
          platformCleanupSegmentForRebuild(seg);
          return reportOomError(seg, "shared_workspace_allocation",
                                TRITON_CAPTURE_WORKSPACE_SIZE, deviceId);
        }
      } else {
        DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                  "using shared workspace for seg[%d-%d]",
                  seg.def.startSlot, seg.def.endSlot);
      }

      // Point this segment's replay handle at the shared workspace
      seg.exec.replayHandle->useExternalWorkspace(
          sharedCaptureWorkspace_, sharedCaptureWorkspaceBytes_);
    }

    // Guard: if replay handle creation failed, crash immediately.
    // Silent fallthrough to slot-by-slot masks the real bug.
    if (seg.exec.replayHandle == nullptr) {
      int deviceId = 0;
      cudaGetDevice(&deviceId);
      char buf[256];
      snprintf(buf, sizeof(buf),
               "NativeDSP: GraphReplayFactory::create returned nullptr for seg[%d-%d] on device %d. "
               "Replay handle creation failed — fix the root cause.",
               seg.def.startSlot, seg.def.endSlot, deviceId);
      DSP_DIAG(COMPILE, "%s", buf);
      THROW_EXCEPTION(buf);
    } else {
    tl_captureWorkspace = seg.exec.replayHandle->getWorkspacePtr();
    tl_captureWorkspaceSize = seg.exec.replayHandle->getWorkspaceBytes();
    tl_captureWorkspaceOffset = 0;
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();

    // Allocate pinned host workspace for H2D source copies during capture.
    // During capture, DataBuffer::syncToSpecial and PointersManager need a persistent
    // pinned buffer as H2D memcpy source. Without this, they use _primaryBuffer directly.
    // Temporary arrays (axis/dimension params for gap ops) get freed after the op completes,
    // but the graph's H2D memcpy node bakes the source address — reading freed memory on
    // launch causes SIGSEGV. The pinned workspace persists for the graph's lifetime.
    void* captureHostWs = nullptr;
    {
      auto hostWsErr = cudaMallocHost(&captureHostWs, TRITON_CAPTURE_HOST_WORKSPACE_SIZE);
      if (hostWsErr != cudaSuccess) {
        cudaGetLastError();
        captureHostWs = nullptr;
        // Host workspace allocation failed — H2D copies during capture will use
        // non-pinned _primaryBuffer directly. When temporary arrays (axis/dimension
        // params for gap ops) are freed after the op completes, the graph's H2D
        // memcpy node still references the freed source address, causing SIGSEGV on
        // replay. This is a fatal error, not a degraded-but-correct path.
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "NativeDSP: cudaMallocHost failed for capture host workspace (%zuMB) "
                 "seg[%d-%d] device %d cudaError=%d (%s). "
                 "Without pinned host workspace, graph replay will SIGSEGV on freed source addresses.",
                 TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024),
                 seg.def.startSlot, seg.def.endSlot, deviceId,
                 static_cast<int>(hostWsErr), cudaGetErrorString(hostWsErr));
        DSP_DIAG(COMPILE, "%s", buf);
        // No TLS or context cleanup needed — capture hasn't started yet
        restoreCublasWorkspaceAfterCapture(stream);
        platformCleanupSegmentForRebuild(seg);
        THROW_EXCEPTION(buf);
      } else {
        DSP_DIAG(MEMORY, "allocated %zuMB pinned host workspace for Triton capture seg[%d-%d]",
                  TRITON_CAPTURE_HOST_WORKSPACE_SIZE / (1024*1024), seg.def.startSlot, seg.def.endSlot);
      }
    }
    tl_captureHostWorkspace = captureHostWs;
    tl_captureHostWorkspaceSize = (captureHostWs != nullptr) ? TRITON_CAPTURE_HOST_WORKSPACE_SIZE : 0;
    tl_captureHostWorkspaceOffset = 0;
    // Track the host workspace as a captured host pointer for lifetime management.
    // On successful capture, this moves to the replay handle (addCapturedHostPtr).
    // On failure, it's freed immediately (line below at tl_capturedHostPtrs cleanup).
    if (captureHostWs != nullptr) {
      tl_capturedHostPtrs.push_back(captureHostWs);
    }

    // Set capture stream so captureSafeStreamOrDefault() routes ops to the correct stream
    cudaStream_t prevCaptureStream = tl_graphCaptureStream;
    tl_graphCaptureStream = cudaStr;
    auto cleanupCaptureTls = [&](bool freeCapturedHostPtrs) {
      tl_graphExecutionActive = false;
      setBatchZeroActive(false);
      tl_captureWorkspace = nullptr;
      tl_captureWorkspaceSize = 0;
      tl_captureWorkspaceOffset = 0;
      if (freeCapturedHostPtrs) {
        for (auto* ptr : tl_capturedHostPtrs) {
          if (ptr != nullptr) {
            cudaFreeHost(ptr);
          }
        }
      }
      tl_captureHostWorkspace = nullptr;
      tl_captureHostWorkspaceSize = 0;
      tl_captureHostWorkspaceOffset = 0;
      tl_capturedHostPtrs.clear();
      tl_captureReplicateCache.clear();
      tl_graphCaptureStream = prevCaptureStream;
    };

    // Pre-allocate cuBLAS workspace to prevent internal cudaMalloc during capture.
    // cuBLAS internally allocates workspace on stream 0 for GEMM operations. During
    // graph capture on a named stream, this cross-stream allocation breaks capture,
    // producing invalid graph nodes that SIGSEGV on cudaGraphLaunch.
    const size_t CUBLAS_WORKSPACE_SIZE = Environment::getInstance().dspCublasWorkspaceMb() * 1024ULL * 1024ULL;
    ensureCublasWorkspace(CUBLAS_WORKSPACE_SIZE);
    // NOTE: setCublasWorkspaceForCapture is deferred to AFTER warmup (see below).
    // Calling it here sets cublasSetStream_v2 to the capture stream, which causes
    // cuBLAS matmuls in gap ops during warmup to run on tritonStr instead of gapStr.
    // This stream mismatch creates data races: cast ops on gapStr write matmul
    // inputs, but cuBLAS on tritonStr starts before gapStr completes.

    // Reset cast cache indices (NOT full clear) before warmup.
    // Previous segments' warmup entries may still be referenced by their captured
    // graphs — clearCastCache() would delete the NDArrays, causing cudaFreeAsync
    // on the GPU buffers. Those addresses are baked into the captured graph nodes
    // (assign + cuBLAS GEMM), so freeing them causes "illegal memory access" (700)
    // on replay. resetCastCacheIndices() preserves the buffers while letting this
    // segment's warmup reuse or append entries as needed.
    //
    // Note: Shape mismatches from speculative decode (draft vs target model) are
    // handled by the mid-execution clearCastCache() call inside MmulHelper::mmul
    // (lines ~761, 1058), which safely skips frees during capture.
    MmulHelper::resetCastCacheIndices();

    // ── Batch-zero preparation (OUTSIDE capture) ─────────────────────────
    // Use the registration-based approach: batchZeroEntries_ was populated
    // by finishBatchZeroRegistration() during the warmup execution (execCount==1).
    // This contains ONLY the buffers that were actually nullified during warmup,
    // avoiding the ~143 extra buffers that collectBatchZeroTargets() would include
    // for slots that don't actually execute (identity ops, fused chains, etc.).
    //
    // If registration didn't happen (e.g., capture retry), fall back to
    // collectBatchZeroTargets for the pre-scan approach.
    if (Environment::getInstance().dspBatchZero()) {
      if (!batchZeroEntries_.empty()) {
        // Registration-based: entries already populated from warmup
        DSP_DIAG(MEMORY, "batch-zero using %d REGISTERED buffers (from warmup observation)",
                  static_cast<int>(batchZeroEntries_.size()));
      } else {
        // Compatibility path: pre-scan approach (may include extra buffers)
        DSP_DIAG(MEMORY, "batch-zero registration empty, falling back to collectBatchZeroTargets");
        std::unordered_set<int> gapSlots;
        if (Environment::getInstance().dspBatchZeroGapOnly()) {
#if HAVE_TRITON && defined(SD_CUDA)
          auto* tritonBE = dynamic_cast<TritonGraphBackend*>(backend);
          if (tritonBE != nullptr) {
            gapSlots = tritonBE->getGapSlots(seg, slots_);
          } else
#endif
          {
            for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) gapSlots.insert(s);
          }
        } else {
          for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) gapSlots.insert(s);
        }
        collectBatchZeroTargets(gapSlots);
      }
      prepareBatchZeroDevice(cudaStr);

      // Save per-segment batch-zero entries so replay uses THIS segment's
      // entries instead of the shared batchZeroEntries_ (which gets overwritten
      // by subsequent segments' warmup/capture cycles).
      seg.exec.segBatchZeroEntries.clear();
      seg.exec.segBatchZeroEntries.reserve(batchZeroEntries_.size());
      for (auto& e : batchZeroEntries_) {
        seg.exec.segBatchZeroEntries.push_back({e.ptr, e.bytes, e.outputSlotIndex});
      }
      DSP_DIAG(MEMORY, "saved %d batch-zero entries to seg[%d-%d]",
                static_cast<int>(seg.exec.segBatchZeroEntries.size()),
                seg.def.startSlot, seg.def.endSlot);
    }

    // Sync external inputs to device before capture — same rationale as non-capture path.
    // Java may have modified host buffers (putScalar + tagLocation(HOST)) between steps.
    // specialBuffer() in arg table population doesn't check for stale device data.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        if (Environment::getInstance().tritonVerifyKernels()) {
          auto* db = externalArrays[ei]->dataBuffer();
          DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(capture) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                    -(ei + 1), ei,
                    db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                    db ? (db->isSpecialActual() ? 1 : 0) : -1,
                    (long long)externalArrays[ei]->lengthOf(),
                    DSP_BUF(externalArrays[ei]));
        }
        externalArrays[ei]->syncToDevice();
      }
    }

    // Synchronize before capture to ensure all prior async work is complete
    cudaStreamSynchronize(cudaStr);
    // Clear any sticky CUDA error before capture — stale errors from prior operations
    // (e.g., cudaFuncGetName on driver-API functions) contaminate capture and launch.
    cudaGetLastError();

    // Diagnostic: dump small variable ext inputs AFTER sync to verify data is current
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] == nullptr) continue;
      if (ei >= static_cast<int>(externalInputIsVariable_.size()) || !externalInputIsVariable_[ei]) continue;
      auto* arr = externalArrays[ei];
      if (arr->lengthOf() > 4) continue;
      arr->syncToHost();
      auto* db = arr->dataBuffer();
      std::string name = (ei < static_cast<int>(externalInputNames_.size())) ? externalInputNames_[ei] : "?";
      if (db != nullptr && db->primary() != nullptr && arr->dataType() == INT64) {
        long long val = 0;
        std::memcpy(&val, static_cast<const char*>(db->primary()) + arr->offset() * 8, 8);
        DSP_DIAG(EXECUTE, "CAPTURE_EXT_INPUT[%d] '%s' INT64 value=%lld host=%p dev=%p execCount=%d",
                 ei, name.c_str(), val, db->primary(), db->special(), seg.exec.executionCount);
      }
    }

    // Configurable: push primary CUDA context during capture.
    // Default OFF — the non-Triton path works without it. Pushing and then popping
    // after capture may cause SIGSEGV on replay (null pointer inside libcuda.so).
    // Enable via ND4J_TRITON_GRAPH_CTX_PUSH=1 for debugging.
    int tritonCaptureDevice = 0;
    cudaGetDevice(&tritonCaptureDevice);
    CUcontext primaryCtx = nullptr;
    CUcontext prevCtx = nullptr;
    bool didPushCtx = false;
    if (Environment::getInstance().tritonGraphCtxPush()) {
      CUdevice cuDev;
      cuDeviceGet(&cuDev, tritonCaptureDevice);
      cuDevicePrimaryCtxRetain(&primaryCtx, cuDev);
      cuCtxGetCurrent(&prevCtx);
      if (prevCtx != primaryCtx) {
        cuCtxPushCurrent(primaryCtx);
        didPushCtx = true;
        DSP_DIAG(EXECUTE, "Triton capture pushed primary ctx %p (was %p) for device %d",
                  (void*)primaryCtx, (void*)prevCtx, tritonCaptureDevice);
      }
    }

    // ── PRE-CAPTURE WARMUP EXECUTION ────────────────────────────────────────
    // During CUDA graph capture, GPU operations are NOT executed — they are only
    // recorded into the graph.  The capture step's output buffers retain whatever
    // values they had BEFORE capture started.  Without a warmup, those values are
    // from the PREVIOUS step's execution, producing a stale/wrong token that
    // corrupts the entire decode sequence.
    //
    // Fix: run a non-capture execution BEFORE capture to produce correct output
    // for this step.  The capture then records the same operations (for replay),
    // but the output buffers already have the correct values from the warmup.
    // This matches the non-Triton CUDA graph path (NativeDynamicShapePlan_cudagraph.cu
    // line 488-490) which runs executeSegmentSlotBySlot() before capture.
    {
      DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton pre-capture warmup for seg[%d-%d] execCount=%d",
                seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);

      // Set cuBLAS workspace during warmup too, so cuBLAS selects the same GEMM
      // algorithms as during capture. Without this, warmup may use different
      // algorithms than capture, causing shape/result divergence.
      setCublasWorkspaceForWarmup();

      // Disable frozen fast path for warmup — same rationale as capture below.
      std::vector<NativeSlot::SlotState> savedSlotStateWarmup(seg.def.endSlot - seg.def.startSlot + 1);
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        savedSlotStateWarmup[s - seg.def.startSlot] = slots_[s].state_;
        if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN)
          slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
      }

      auto warmupStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                   outputSlots_, totalOutputSlots_, stream);
      // Restore frozen state after warmup
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        slots_[s].state_ = savedSlotStateWarmup[s - seg.def.startSlot];
      }

      if (warmupStatus != Status::OK) {
        DSP_DIAG(EXECUTE, "FATAL: Triton pre-capture warmup FAILED for seg[%d-%d] status=%d. "
                  "BLOCKING EXECUTION.",
                  seg.def.startSlot, seg.def.endSlot, static_cast<int>(warmupStatus));
        seg.exec.compilationFailed = true;
        cleanupCaptureTls(true);
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        // Destroy the replay handle created before warmup — it holds workspace
        // memory even though capture never started.
        platformCleanupSegmentForRebuild(seg);
        return warmupStatus;
      }
      // Decrement executionCount — the warmup was an extra execution that should
      // not count toward the capture threshold.
      if (seg.exec.executionCount > 0) seg.exec.executionCount--;

      // Synchronize before capture to ensure warmup results are visible
      cudaStreamSynchronize(cudaStr);
      cudaGetLastError();

      // Reset cast cache indices after warmup so capture starts from index 0.
      // We intentionally preserve the warmup's cast cache entries (NOT full clear).
      // Capture reuses them via assign() — the graph records a cast kernel from the
      // real input to the cached buffer, then cuBLAS reads the cached buffer.
      // clearCastCache() would delete these entries, forcing capture to allocate
      // new ones from the capture workspace. Those workspace sub-allocations
      // cannot be individually freed by cudaFreeAsync (they're interior pointers
      // of the 32MB workspace block), so subsequent clearCastCache() calls
      // corrupt the CUDA memory pool → "illegal memory access" on replay.
      MmulHelper::resetCastCacheIndices();

      DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton pre-capture warmup DONE for seg[%d-%d]",
                seg.def.startSlot, seg.def.endSlot);

      // DIAGNOSTIC: dump warmup's final output argmax for comparison with replay
      {
        int finalOutputSlot = -1;
        if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
          finalOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
        }
        if (finalOutputSlot < 0 || finalOutputSlot >= totalOutputSlots_)
          finalOutputSlot = seg.def.endSlot;
        if (finalOutputSlot >= 0 && finalOutputSlot < totalOutputSlots_ &&
            outputSlots_[finalOutputSlot] != nullptr) {
          auto* warmupOut = outputSlots_[finalOutputSlot];
          if (warmupOut->dataType() == FLOAT32 && warmupOut->lengthOf() > 0) {
            int warmupArgmax = dspArgmax(DSP_BUF(warmupOut), warmupOut->dataType(),
                                          warmupOut->lengthOf());
            std::string warmupVals = dspDumpSlotValues(DSP_BUF(warmupOut), warmupOut->dataType(),
                                                        warmupOut->lengthOf(), 4);
            DSP_DIAG(EXECUTE, "WARMUP ARGMAX: slot=%d argmax=%d len=%lld vals=%s execCount=%d",
                     finalOutputSlot, warmupArgmax, (long long)warmupOut->lengthOf(),
                     warmupVals.c_str(), seg.exec.executionCount);
          }
        }
      }

      // ── RESTORE NULL OUTPUT SLOTS FROM CACHE ─────────────────────────────
      // The warmup execution may clear some outputSlots_ entries (e.g. control
      // flow CF_SWITCH dead outputs, or segment cleanup paths).  The values
      // were captured into outputSlots_ during execution, so restore any
      // Phase 2: outputSlots_ == outputSlots_ (unified).
      // Post-warmup restoration is a no-op — arrays produced during warmup
      // are already in outputSlots_ (which IS outputSlots_).
    }

    // DIAGNOSTIC: warmup-only mode — skip capture, use warmup result directly.
    // Enables bisection: if warmup-only produces correct output but capture+replay
    // does not, the bug is in capture/replay.
    {
      static bool warmupOnly = Environment::getInstance().triton().warmupOnly();
      if (warmupOnly) {
        DSP_DIAG(EXECUTE, "WARMUP_ONLY: skipping capture for seg[%d-%d], using warmup result",
                  seg.def.startSlot, seg.def.endSlot);
        cleanupCaptureTls(true);
        // Don't need the replay handle — fall through to non-capture path next time.
        // Destroy it to free the workspace memory allocated at line 1756.
        seg.exec.compilationFailed = true;
        platformCleanupSegmentForRebuild(seg);
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        return Status::OK;
      }
    }

    // NOW set cuBLAS handle to capture stream — AFTER warmup completed.
    // During warmup, gap ops must use their default stream (gapStr) for cuBLAS.
    // Only during actual capture do we switch cuBLAS to tritonStr so GEMM nodes
    // are recorded into the CUDA graph on the correct stream.
    setCublasWorkspaceForCapture(stream);

    // cuBLAS workspace preservation during capture.
    //
    //  Once shapes are frozen (shapesFrozen_ == true), NEVER zero the cuBLAS workspace.
    // During capture, cuBLAS stores plan/descriptor data in the workspace. Captured CUDA graphs
    // inherit these cached plans and omit H2D re-upload nodes. Zeroing the workspace destroys
    // cached plans, causing GEMM kernels to read zeros and hang on replay.
    //
    // The workspace content must be preserved across ALL captures and replays once frozen.
    // cuBLAS plans are stable for fixed shapes, so preservation is safe.
    //
    // Pre-frozen (shapes not yet frozen): zeroing is acceptable as no graphs are captured yet.
    if (shapesFrozen_ && cublasWorkspaceBuffer_ != nullptr && cublasWorkspaceSize_ > 0) {
      DSP_DIAG(MEMORY, "pre-capture: cuBLAS workspace PRESERVED (%zuMB) — shapes frozen, plans stable",
               cublasWorkspaceSize_ / (1024*1024));
      // Do NOT zero — preserve cuBLAS plan data for captured graph replay
    }
    // Note: Pre-frozen zeroing removed entirely — not needed for correctness and
    // adds unnecessary overhead. cuBLAS handles uninitialized workspace correctly.

    // Disable frozen fast path during capture. Same rationale as non-Triton path:
    // capture may re-create views, and the frozen context has stale input/output pointers
    // from the prior non-capture execution. Using the full (non-frozen) path during capture
    // is a one-time cost — all context pointers are properly reconfigured with capture-time
    // arrays, including correct nullify() calls to zero output buffers.
    // Save and restore frozenContextReady after capture so replay uses frozen fast path.
    std::vector<NativeSlot::SlotState> savedSlotStateTriton(seg.def.endSlot - seg.def.startSlot + 1);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      savedSlotStateTriton[s - seg.def.startSlot] = slots_[s].state_;
      if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN)
        slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
    }
    auto restoreCaptureSlotState = [&]() {
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        slots_[s].state_ = savedSlotStateTriton[s - seg.def.startSlot];
      }
    };

    std::vector<std::pair<int, NDArray*>> savedExtForCapture;
    std::vector<std::pair<int, NDArray*>> savedSlotsForCapture;

    // Pre-capture promotion: view-capable slots should enter FROZEN state BEFORE
    // capture begins. The view installation path reuses the input's DataBuffer
    // directly, which is correct for BOTH constant and variable inputs:
    // - For constant inputs: the view sees the stable warmup value
    // - For variable inputs: the view shares the input's DataBuffer, so replay
    //   automatically sees updated values when the input is refreshed
    //
    // Without this promotion, view-capable slots go through normal execution
    // during capture: they allocate new output buffers and create H2D capture
    // nodes. Later, those H2D nodes can copy stale host data over the output,
    // corrupting the values downstream consumers depend on.
    //
    // isDataDependent is NOT a disqualifier: a reshape's output shape comes from
    // input values, but the view still shares the input's DataBuffer correctly.
    if (shapesFrozen_) {
      int promoted = 0;
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        auto& sl = slots_[s];
        if (!sl.flags.isViewCapableOp || sl.state_ >= NativeSlot::SlotState::FROZEN) continue;
        sl.state_ = NativeSlot::SlotState::FROZEN;
        promoted++;
      }
      if (promoted > 0) {
        DSP_DIAG(EXECUTE, "pre-capture view promotion: %d view-capable slots promoted to FROZEN for seg[%d-%d]",
                  promoted, seg.def.startSlot, seg.def.endSlot);
      }
    }

    // Pre-capture batch-zero: zero all registered buffers BEFORE beginCapture.
    // These cudaMemsetAsync calls execute normally on the stream (not captured).
    // This ensures ops get zeroed outputs during the capture run for correct results.
    // During capture, individual nullify() calls are suppressed (no memset graph nodes).
    // On replay, the same zeroing happens via pre-replay batch-zero above.
    if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty()) {
      for (auto& entry : batchZeroEntries_) {
        if (entry.ptr != nullptr && entry.bytes > 0) {
          cudaMemsetAsync(entry.ptr, 0, entry.bytes, cudaStr);
        }
      }
      DSP_DIAG(MEMORY, "pre-capture batch-zero: %d buffers zeroed via cudaMemsetAsync (fill engines, before beginCapture)",
                static_cast<int>(batchZeroEntries_.size()));
    }

    // ── Save warmup output slot pointers BEFORE capture ─────────────────
    // Gap ops are skipped during capture (they return OK without executing).
    // outputSlots_[] retains warmup values throughout capture — no save/restore needed.
    // Downstream segments see valid warmup data. Triton sub-kernel arg tables
    // reference warmup addresses, which are stable.

    // POST-ALLOCATION MEMORY GATE: workspace + cuBLAS are allocated. Check that
    // enough headroom remains for CUDA driver graph metadata before starting capture.
    // This is tight and accurate — only graph metadata overhead remains.
    {
      size_t gpuFree = 0, gpuTotal = 0;
      cudaMemGetInfo(&gpuFree, &gpuTotal);
      size_t safetyBytes = Environment::getInstance().dspGraphMetadataSafetyMb() * 1024ULL * 1024ULL;
      if (gpuFree < safetyBytes) {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                     "POST-ALLOC GATE FAILED: free=%zuMB < safety=%zuMB for seg[%d-%d]",
                     gpuFree / (1024*1024), safetyBytes / (1024*1024),
                     seg.def.startSlot, seg.def.endSlot);
        platformCleanupSegmentForRebuild(seg);
        return reportOomError(seg, "post_alloc_gate", safetyBytes, deviceId);
      }
    }

    auto* cudaReplay = static_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get());
    auto handle = cudaReplay->getNativeHandle();
    bool captureOk = handle->beginCapture(cudaStr, cudaStreamCaptureModeRelaxed);
    if (captureOk) {
      DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph capture started for seg[%d-%d] execCount=%d",
                seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
      tl_graphExecutionActive = true;

      // Batch-zero during capture: DON'T launch inside the graph — instead,
      // suppress individual nullify() calls so no memset nodes get captured.
      // The actual zeroing happens OUTSIDE the graph before each replay() call
      // using cudaMemsetAsync (fill engines, no SM competition).
      // This removes ~700 memset graph nodes while keeping fill-engine efficiency.
      if (Environment::getInstance().dspBatchZero() && !batchZeroEntries_.empty()) {
        setBatchZeroActive(true);
        DSP_DIAG(MEMORY, "batch-zero CAPTURE-SKIP: suppressing %d individual nullify() calls, "
                  "zeroing will happen outside graph before replay",
                  static_cast<int>(batchZeroEntries_.size()));

        //  Mark ALL output slot DataBuffers as device-actual (sAct=1)
        // after batch-zero.  Batch-zero zeroes device memory directly via a GPU
        // kernel, bypassing NDArray's actuality tracking.  Without this,
        // DataBuffer::syncToSpecial() inside native gap ops sees sAct=0 (stale
        // from a previous step) and generates an H2D memcpy that gets RECORDED
        // in the CUDA graph.  On replay, that H2D copies STALE host data
        // (from capture time) over the freshly batch-zeroed device buffer,
        // corrupting inputs to downstream ops.
        //
        // By marking sAct=1 here, syncToSpecial() during capture becomes a
        // no-op for internal buffers (device is already "actual" — it has
        // zeros, which is the correct initial state). This keeps the direct
        // output buffers consistent with the zeroed device state we capture.
        int markedCount = 0;
        for (int si = seg.def.startSlot; si <= seg.def.endSlot; si++) {
          for (int o = 0; o < slots_[si].wiring.numOutputs; o++) {
            int outIdx = slots_[si].wiring.outputSlotIndices[o];
            if (outIdx >= 0 && outIdx < totalOutputSlots_ && outputSlots_[outIdx]) {
              auto* db = outputSlots_[outIdx]->dataBuffer();
              if (db) {
                db->writeSpecial();
                markedCount++;
              }
            }
          }
        }
        DSP_DIAG(MEMORY, "batch-zero actuality: marked %d output DataBuffers as device-actual",
                  markedCount);
        if (Environment::getInstance().tritonVerifyKernels()) {
          DSP_DIAG(VERIFY, "SLOT_WRITE tag=BATCH_ZERO seg[%d-%d] %d buffers suppressed (nullify skipped), %d marked sAct=1",
                    seg.def.startSlot, seg.def.endSlot, static_cast<int>(batchZeroEntries_.size()), markedCount);
        }
      } else {
        DSP_DIAG(MEMORY, "batch-zero DISABLED (dspBatchZero=%d, entries=%d)",
                  (int)Environment::getInstance().dspBatchZero(), static_cast<int>(batchZeroEntries_.size()));
      }

      // Query node count mid-capture to verify operations are being recorded
      size_t midCaptureNodes = handle->getNumNodesDuringCapture(cudaStr);
      DSP_DIAG(EXECUTE, "Triton capture mid-check: %zu nodes recorded before executeSegment (batchZero=%d entries, outside-graph)",
                midCaptureNodes, static_cast<int>(batchZeroEntries_.size()));

      // Snapshot all buffer addresses at capture entry — compare with replay to detect stale pointers
      {
        std::vector<void*> outAddrs, extAddrs;
        extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
        extractDeviceAddrs(externalArrays, numExt, extAddrs);
        DspDiagnostics::getInstance().clearAddressSnapshots();
        DSP_DIAG_SNAPSHOT_ADDRS("capture-entry", outAddrs.data(), totalOutputSlots_,
                                 extAddrs.data(), numExt);
      }

      auto captureStatus = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                                   outputSlots_, totalOutputSlots_, stream);
      setBatchZeroActive(false);
      tl_graphExecutionActive = false;

      // Snapshot addresses AFTER capture execution to detect pointer changes during capture
      {
        std::vector<void*> outAddrs, extAddrs;
        extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
        extractDeviceAddrs(externalArrays, numExt, extAddrs);
        DSP_DIAG_SNAPSHOT_ADDRS("capture-exit", outAddrs.data(), totalOutputSlots_,
                                 extAddrs.data(), numExt);
        int changed = DSP_DIAG_COMPARE_ADDRS("capture-entry", "capture-exit");
        if (changed > 0) {
          DSP_DIAG(EXECUTE, "WARNING: %d buffer addresses CHANGED during capture execution!", changed);
        }
      }

      // Diagnostic: capture workspace usage
      DSP_DIAG(MEMORY, "capture workspace used: %zu / %zu bytes (%.1f%%)",
               tl_captureWorkspaceOffset, seg.exec.replayHandle->getWorkspaceBytes(),
               seg.exec.replayHandle->getWorkspaceBytes() > 0 ? (100.0 * tl_captureWorkspaceOffset / seg.exec.replayHandle->getWorkspaceBytes()) : 0.0);
      // Check for CUDA errors generated during capture — these become invalid graph nodes.
      // Don't use cudaGetLastError (which clears) — peek first for diagnostics.
      {
        cudaError_t capPhaseErr = cudaPeekAtLastError();
        if (capPhaseErr != cudaSuccess) {
          DSP_DIAG(BACKEND, "WARNING - CUDA error during Triton capture phase: %s (%d)",
                    cudaGetErrorString(capPhaseErr), (int)capPhaseErr);
          // Clear it so endCapture can proceed (the graph may still be partially valid)
          cudaGetLastError();
        }
      }

      // Query node count after execution to see how many ops were captured
      size_t postExecNodes = 0;
      {
        cudaStreamCaptureStatus capStat = cudaStreamCaptureStatusNone;
        cudaGraph_t capGraph = nullptr;
        unsigned long long capId = 0;
        auto capErr = cudaStreamGetCaptureInfo_v2(cudaStr, &capStat, &capId, &capGraph, nullptr, nullptr);
        if (capErr == cudaSuccess && capGraph != nullptr) {
          cudaGraphGetNodes(capGraph, nullptr, &postExecNodes);
        }
      }
      DSP_DIAG(EXECUTE, "Triton capture post-exec: %zu nodes, captureStatus=%d",
                postExecNodes, static_cast<int>(captureStatus));
      fflush(stdout); fflush(stderr);

      bool endOk = false;
      if (captureStatus == Status::OK) {
        endOk = handle->endCapture(cudaStr);
      } else {
        DSP_DIAG(EXECUTE, "FATAL: Triton capture execution FAILED status=%d for seg[%d-%d]. "
                  "BLOCKING EXECUTION.",
                  static_cast<int>(captureStatus), seg.def.startSlot, seg.def.endSlot);
        fflush(stdout); fflush(stderr);
        if (handle->isCapturing()) {
          handle->endCapture(cudaStr);
        }
      }

      if (endOk) {
        size_t numGraphNodes = handle->getNumNodes();
        int segSize = seg.def.endSlot - seg.def.startSlot + 1;
        DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                     "GRAPH CAPTURE COMPLETE: seg[%d-%d] %zu nodes captured from %d slots (%.1f nodes/slot)",
                     seg.def.startSlot, seg.def.endSlot, numGraphNodes, segSize,
                     segSize > 0 ? (double)numGraphNodes / segSize : 0.0);
        DSP_DIAG(EXECUTE, "Triton capture endOk: graph has %zu nodes", numGraphNodes);

        // Empty graphs (0 nodes) have no GPU work — skip replay to avoid
        // spurious fingerprint mismatches when slot addresses change.
        if (numGraphNodes == 0) {
          DSP_DIAG_SEG(COMPILE, seg.def.startSlot,
                       "empty Triton graph for seg[%d-%d] (0 nodes) — marking as non-capturable",
                       seg.def.startSlot, seg.def.endSlot);
          seg.exec.compilationFailed = true;
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          restoreCaptureSlotState();
          platformCleanupSegmentForRebuild(seg);
          seg.exec.executionCount++;
          return Status::OK;
        }

        // Sample final output AFTER endCapture (stream no longer capturing, safe)
        if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
          auto* finalOut = outputSlots_[seg.def.endSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("capture-post-endCapture", seg.def.endSlot,
                               DSP_BUF(finalOut), finalOut->lengthOf());
          }
        }
        // Dump top logit from capture execution via DSP_DIAG
        // Use outputSlotIndices[0] to get the ACTUAL final output slot
        // (matches GRAPH_REPLAY logic for apples-to-apples comparison)
        {
          int captureOutputSlot = -1;
          if (seg.def.endSlot < numSlots_ && slots_[seg.def.endSlot].wiring.numOutputs > 0) {
            captureOutputSlot = slots_[seg.def.endSlot].wiring.outputSlotIndices[0];
          }
          if (captureOutputSlot < 0 || captureOutputSlot >= totalOutputSlots_) {
            captureOutputSlot = seg.def.endSlot;
          }
          if (captureOutputSlot >= 0 && captureOutputSlot < totalOutputSlots_ &&
              outputSlots_[captureOutputSlot] != nullptr) {
            auto* out = outputSlots_[captureOutputSlot];
            if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
              DSP_DIAG_DUMP_SEG_OUTPUT("CAPTURE_EXEC", captureOutputSlot, DSP_BUF(out),
                                       out->lengthOf(), seg.exec.executionCount, stream);
            }
          }
        }
      }

      if (endOk) {
        auto stats = handle->getStatistics();
        DSP_DIAG(EXECUTE, "Triton graph stats: %d kernels, %d memcpys, %d memsets, "
                  "%d memAllocs, %d memFrees, %d hostCallbacks, %d events, %d empty",
                  stats.numKernels, stats.numMemcpyH2D, stats.numMemsets,
                  stats.numMemAllocs, stats.numMemFrees,
                  stats.numHostCallbacks, stats.numEvents, stats.numEmpty);
        fflush(stdout); fflush(stderr);
        if (stats.numMemAllocs > 0 || stats.numMemFrees > 0) {
          DSP_DIAG(EXECUTE, "Triton graph has %d MemAlloc + %d MemFree nodes "
                    "(paired alloc/free from cuBLAS internal workspace - CUDA 12+ handles these on replay).",
                    stats.numMemAllocs, stats.numMemFrees);
        }
        if (stats.numHostCallbacks > 0) {
          DSP_DIAG(BACKEND, "WARNING - Graph has %d host callback nodes!",
                    stats.numHostCallbacks);
        }
      }

      // Skip DOT dump by default for Triton graphs — cudaGraphDebugDotPrint with verbose
      // flags may also call cudaGraphKernelNodeGetParams internally, causing the same
      // cudaErrorInvalidDeviceFunction poisoning as getDetailedNodeInfo().
      if (endOk && Environment::getInstance().tritonDumpGraphDot()) {
        cudaGraphDebugDotPrint(handle->getGraph(), "/tmp/triton_graph_debug.dot", 0);
        DSP_DIAG(EXECUTE, "Triton graph dumped to /tmp/triton_graph_debug.dot");
        fflush(stdout); fflush(stderr);
      }

      // Skip getDetailedNodeInfo() for Triton graphs — it calls cudaFuncGetName on each
      // kernel node, which returns cudaErrorInvalidDeviceFunction (error 98) for Triton
      // kernels loaded via cuModuleLoadDataEx (driver API). The 658+ consecutive errors
      // poison the CUDA error state and cause cudaGraphLaunch to SIGSEGV.
      // Use getNumNodes() for basic stats instead (no per-node introspection).
      bool allKernelsValid = true;
      if (endOk) {
#ifdef SD_CUDA
        size_t totalNodes = handle->getNumNodes();
        DSP_DIAG(EXECUTE, "Triton graph has %zu nodes (skipping per-node inspection to avoid error-98 poisoning)",
                  totalNodes);
        fflush(stdout); fflush(stderr);
        // Ensure no sticky errors before instantiation
        cudaGetLastError();
#endif
      }

      bool instantiateOk = endOk && allKernelsValid && handle->instantiate();
      if (instantiateOk) {
        DSP_DIAG(EXECUTE, "Triton graph instantiated OK (graphExec=%p), about to launch...",
                  handle->getGraphExec());
        fflush(stdout); fflush(stderr);
      }

      if (!instantiateOk) {
        int deviceId = 0;
        cudaGetDevice(&deviceId);

        // Check if instantiation failed due to OOM — retry with eviction if possible.
        auto* cudaReplayForOom = seg.exec.replayHandle
            ? dynamic_cast<CudaGraphReplayHandle*>(seg.exec.replayHandle.get()) : nullptr;
        bool isOom = cudaReplayForOom && cudaReplayForOom->wasLastInstantiateOom();
        if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
          seg.exec.captureOomRetries++;
          seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::retryInterval();
          DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                       "INSTANTIATE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                       seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                       seg.exec.captureRetryAfterExec);

          // Evict LRU graphs to free memory for the next attempt
          evictLruGraphs(segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);

          // Cleanup this failed attempt but do NOT set compilationFailed
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          restoreCaptureSlotState();
          platformCleanupSegmentForRebuild(seg);
          cudaGetLastError();  // Clear sticky error
          // OOM during graph instantiation — throw instead of silently falling back
          // to slot-by-slot. The eviction above freed memory; the next execution
          // attempt (deferred by captureRetryAfterExec) will retry capture.
          // Silently producing output via slot-by-slot masks the OOM and the caller
          // never knows the graph wasn't captured.
          {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "NativeDSP: graph instantiation OOM for seg[%d-%d] on device %d "
                     "(retry %d/%d, retryAfterExec=%d). Evicted LRU graphs. "
                     "Fix memory pressure — do NOT fall back to slot-by-slot.",
                     seg.def.startSlot, seg.def.endSlot, deviceId,
                     seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                     seg.exec.captureRetryAfterExec);
            DSP_DIAG(COMPILE, "%s", buf);
            THROW_EXCEPTION(buf);
          }
        }

        // Not OOM or retries exhausted — permanent failure
        cleanupCaptureTls(true);
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        restoreCaptureSlotState();
        platformCleanupSegmentForRebuild(seg);
        return reportCaptureError(seg, "instantiate", cudaGetLastError(), deviceId);
      }

      // POST-INSTANTIATE MEMORY CHECK removed: the validation launch immediately
      // below will reveal if the graph is usable. No speculative memory gate needed.

      // Graph instantiated — launch to validate the graph is not corrupted.
      // Warmup results are restored from savedWarmupOutputSlots below regardless.
      bool launchOk = false;
      {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        cudaGetLastError();
        bool replayResult = seg.exec.replayHandle->replay(stream);
        if (!replayResult) {
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          restoreCaptureSlotState();
          platformCleanupSegmentForRebuild(seg);
          return reportReplayError(seg, "validation_launch", cudaGetLastError(), deviceId);
        }
        cudaError_t syncErr = cudaStreamSynchronize(cudaStr);
        if (syncErr != cudaSuccess) {
          cleanupCaptureTls(true);
          if (didPushCtx) {
            CUcontext dummy;
            cuCtxPopCurrent(&dummy);
            CUdevice cuDev;
            cuDeviceGet(&cuDev, tritonCaptureDevice);
            cuDevicePrimaryCtxRelease(cuDev);
          }
          restoreCublasWorkspaceAfterCapture(stream);
          restoreCaptureSlotState();
          platformCleanupSegmentForRebuild(seg);
          return reportReplayError(seg, "validation_sync", syncErr, deviceId);
        }
        DSP_DIAG(EXECUTE, "VALIDATION LAUNCH OK: seg[%d-%d] graph launched and synced successfully",
                 seg.def.startSlot, seg.def.endSlot);
        // LRU tracking: record when this segment was last replayed for eviction ordering
        seg.exec.lastReplayExecCount = executeCount_;
        launchOk = true;
      }

      if (launchOk) {
        if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
          auto* finalOut = outputSlots_[seg.def.endSlot];
          if (finalOut->dataType() == FLOAT32) {
            DSP_DIAG_DUMP_SLOT("capture-post-launch", seg.def.endSlot,
                               DSP_BUF(finalOut), finalOut->lengthOf());
          }
        }
        // Dump top logit from first replay (graph launch after capture) via DSP_DIAG
        if (seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
          auto* out = outputSlots_[seg.def.endSlot];
          if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
            DSP_DIAG_DUMP_SEG_OUTPUT("REPLAY_LAUNCH", seg.def.endSlot, DSP_BUF(out),
                                     out->lengthOf(), seg.exec.executionCount, stream);
          }
        }
        // replayHandle already set (created before capture began)
        seg.exec.cachedShapeKey = segShapeKey;
        seg.exec.capturedInputAddrKey = segInputAddrKey;
        seg.exec.capturedCreateValueKey = createValueKey;
        seg.exec.capturedSlotAddrHash = computeSlotAddrHash(
            outputSlots_, seg.def.startSlot, seg.def.endSlot, totalOutputSlots_);
        snapshotExternalAddrs(seg, externalArrays, numExt);

        // Export graph stats and DOT file for diagnostics
        auto stats = handle->getStatistics();
        DSP_DIAG_SEG(EXECUTE, seg.def.startSlot, "Triton graph CAPTURED and launched for seg[%d-%d]: "
                  "%d kernels, %d memcpy, %d memset, %d memAlloc, %d memFree "
                  "(workspace=%zuMB, offset=%zu)",
                  seg.def.startSlot, seg.def.endSlot,
                  stats.numKernels, stats.numMemcpyH2D + stats.numMemcpyD2H + stats.numMemcpyD2D,
                  stats.numMemsets, stats.numMemAllocs, stats.numMemFrees,
                  seg.exec.replayHandle->getWorkspaceBytes() / (1024*1024), tl_captureWorkspaceOffset);
        // Dump H2D memcpy node details to identify the source of baked-in host addresses
        if (stats.numMemcpyH2D > 0 && DSP_DIAG_ENABLED(EXECUTE)) {
          size_t numGraphNodes = 0;
          cudaGraphGetNodes(handle->getGraph(), nullptr, &numGraphNodes);
          if (numGraphNodes > 0) {
            std::vector<cudaGraphNode_t> graphNodes(numGraphNodes);
            cudaGraphGetNodes(handle->getGraph(), graphNodes.data(), &numGraphNodes);
            for (size_t ni = 0; ni < numGraphNodes; ni++) {
              cudaGraphNodeType nodeType;
              cudaGraphNodeGetType(graphNodes[ni], &nodeType);
              if (nodeType == cudaGraphNodeTypeMemcpy) {
                cudaMemcpy3DParms mcpyParams;
                memset(&mcpyParams, 0, sizeof(mcpyParams));
                if (cudaGraphMemcpyNodeGetParams(graphNodes[ni], &mcpyParams) == cudaSuccess) {
                  size_t bytes = mcpyParams.extent.width *
                                 std::max(mcpyParams.extent.height, (size_t)1) *
                                 std::max(mcpyParams.extent.depth, (size_t)1);
                  const char* kindStr = (mcpyParams.kind == cudaMemcpyHostToDevice) ? "H2D" :
                                        (mcpyParams.kind == cudaMemcpyDeviceToDevice) ? "D2D" :
                                        (mcpyParams.kind == cudaMemcpyDeviceToHost) ? "D2H" : "other";
                  DSP_DIAG(EXECUTE, "GRAPH_NODE[%zu] MEMCPY %s: %zu bytes src=%p dst=%p "
                           "seg[%d-%d]",
                           ni, kindStr, bytes,
                           mcpyParams.srcPtr.ptr, mcpyParams.dstPtr.ptr,
                           seg.def.startSlot, seg.def.endSlot);
                }
              }
            }
          }
        }
        // Write DOT file for offline analysis.
        // Default: non-verbose (flag 0). Verbose queries kernel node params via
        // cudaFuncGetName, which returns cudaErrorInvalidDeviceFunction for
        // Triton CUfunction handles and may poison driver state.
        // Enable via ND4J_TRITON_GRAPH_DOT_VERBOSE=1 for debugging.
        {
          std::string dotPath = "/tmp/triton_graph_captured.dot";
          unsigned int dotFlags = Environment::getInstance().tritonGraphDotVerbose()
              ? cudaGraphDebugDotFlagsVerbose : 0;
          auto dotErr = cudaGraphDebugDotPrint(handle->getGraph(), dotPath.c_str(), dotFlags);
          if (dotErr == cudaSuccess) {
            DSP_DIAG(EXECUTE, "Exported Triton graph DOT to %s (verbose=%d)",
                      dotPath.c_str(), dotFlags != 0);
          }
          cudaGetLastError(); // Clear any error from dot print
        }
        // Write stats to a file the test can read
        {
          FILE* f = fopen("/tmp/triton_graph_stats.txt", "w");
          if (f) {
            fprintf(f, "segment=%d-%d\n", seg.def.startSlot, seg.def.endSlot);
            fprintf(f, "kernels=%d\n", stats.numKernels);
            fprintf(f, "memcpyH2D=%d\n", stats.numMemcpyH2D);
            fprintf(f, "memcpyD2H=%d\n", stats.numMemcpyD2H);
            fprintf(f, "memcpyD2D=%d\n", stats.numMemcpyD2D);
            fprintf(f, "memsets=%d\n", stats.numMemsets);
            fprintf(f, "memAllocs=%d\n", stats.numMemAllocs);
            fprintf(f, "memFrees=%d\n", stats.numMemFrees);
            fprintf(f, "hostCallbacks=%d\n", stats.numHostCallbacks);
            fprintf(f, "events=%d\n", stats.numEvents);
            fprintf(f, "childGraphs=%d\n", stats.numChildGraphs);
            fprintf(f, "totalNodes=%zu\n", handle->getNumNodes());
            fclose(f);
          }
        }
        status = Status::OK;
        usedTritonGraphCapture = true;

        // Phase 2: outputSlots_ == outputSlots_ (unified).
        // No need to sync cache ← output — they are the same pointer.

        // FORCE_RECAPTURE: invalidate graph immediately after capture+launch
        // so the NEXT step also re-captures instead of replaying a stale graph.
        // This ensures every single step is a fresh capture+launch with zero replays.
        if (Environment::getInstance().tritonForceRecapture()) {
          platformCleanupSegmentForRebuild(seg);
          seg.exec.argTableStable = false;
          batchD2DCount_ = 0;
          seg.exec.capturedInputAddrKey = 0;
          DSP_DIAG(EXECUTE, "FORCE_RECAPTURE: invalidated after capture+launch execCount=%d", seg.exec.executionCount);
        }
      } else {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        cleanupCaptureTls(true);
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        restoreCaptureSlotState();
        platformCleanupSegmentForRebuild(seg);
        return reportCaptureError(seg, "execute_during_capture", cudaGetLastError(), deviceId);
      }
    } else {
      int deviceId = 0;
      cudaGetDevice(&deviceId);

      // Check if beginCapture failed due to OOM — retry with eviction if possible.
      cudaError_t beginErr = cudaGetLastError();
      bool isOom = (beginErr == cudaErrorMemoryAllocation);
      if (isOom && seg.exec.captureOomRetries < GraphSegment::maxOomRetries()) {
        seg.exec.captureOomRetries++;
        seg.exec.captureRetryAfterExec = seg.exec.executionCount + GraphSegment::retryInterval();
        DSP_DIAG_SEG(MEMORY, seg.def.startSlot,
                     "BEGIN_CAPTURE OOM — retry %d/%d, evicting LRU graphs. retryAfterExec=%d",
                     seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                     seg.exec.captureRetryAfterExec);
        evictLruGraphs(segIdx, TRITON_CAPTURE_WORKSPACE_SIZE, stream);
        cleanupCaptureTls(true);
        if (didPushCtx) {
          CUcontext dummy;
          cuCtxPopCurrent(&dummy);
          CUdevice cuDev;
          cuDeviceGet(&cuDev, tritonCaptureDevice);
          cuDevicePrimaryCtxRelease(cuDev);
        }
        restoreCublasWorkspaceAfterCapture(stream);
        restoreCaptureSlotState();
        platformCleanupSegmentForRebuild(seg);
        // OOM during beginCapture — throw instead of silently falling back to slot-by-slot.
        {
          char buf[256];
          snprintf(buf, sizeof(buf),
                   "NativeDSP: beginCapture OOM for seg[%d-%d] on device %d "
                   "(retry %d/%d, retryAfterExec=%d). Evicted LRU graphs. "
                   "Fix memory pressure — do NOT fall back to slot-by-slot.",
                   seg.def.startSlot, seg.def.endSlot, deviceId,
                   seg.exec.captureOomRetries, GraphSegment::maxOomRetries(),
                   seg.exec.captureRetryAfterExec);
          DSP_DIAG(COMPILE, "%s", buf);
          THROW_EXCEPTION(buf);
        }
      }

      cleanupCaptureTls(true);
      if (didPushCtx) {
        CUcontext dummy;
        cuCtxPopCurrent(&dummy);
        CUdevice cuDev;
        cuDeviceGet(&cuDev, tritonCaptureDevice);
        cuDevicePrimaryCtxRelease(cuDev);
      }
      restoreCublasWorkspaceAfterCapture(stream);
      restoreCaptureSlotState();
      platformCleanupSegmentForRebuild(seg);
      return reportCaptureError(seg, "beginCapture", beginErr, deviceId);
    }

    // ── Restore warmup output slots for downstream segment visibility ──
    // This MUST happen regardless of capture success/failure. During capture
    // execution, ops allocate outputs from capture workspace, overwriting
    // outputSlots_[] with workspace addresses. If capture fails (endCapture
    // error, instantiate error, or execution error), outputSlots_[] still
    // has the stale workspace addresses. Downstream segments reading from
    // these get garbage data → NaN propagation.
    //
    // No WARMUP_RESTORE needed: gap ops were skipped during capture, so
    // outputSlots_[] was never overwritten. Warmup data is intact.

    DSP_DIAG(EXECUTE, "CAPTURE_COMPLETE: seg[%d-%d] hasReplay=%d compilationFailed=%d "
             "numCaptureBuffers=%d",
             seg.def.startSlot, seg.def.endSlot,
             seg.exec.replayHandle != nullptr,
             seg.exec.compilationFailed,
             0);

    // No external/cross-slot rewiring is needed now that replay uses the
    // canonical external and output buffers directly. The restore loops remain
    // harmless no-ops because the saved lists stay empty.
    for (auto& [extIdx, origArr] : savedExtForCapture) {
      externalArrays[extIdx] = origArr;
    }

    // Restore cross-segment output slots to warmup pointers.
    // The producing segment's replay writes fresh data to the warmup array's
    // GPU address (baked during capture). The D2D copy before the consuming
    // segment's replay reads from outputSlots_[] (warmup pointer, with fresh
    // GPU data from the producing segment's replay).
    for (auto& [slotIdx, origArr] : savedSlotsForCapture) {
      if (origArr != nullptr) {
        outputSlots_[slotIdx] = origArr;
      }
    }

    // Restore primary CUDA context if we pushed it
    if (didPushCtx) {
      CUcontext dummy;
      cuCtxPopCurrent(&dummy);
      CUdevice cuDev;
      cuDeviceGet(&cuDev, tritonCaptureDevice);
      cuDevicePrimaryCtxRelease(cuDev);
    }

    // Restore cuBLAS workspace to default (undo setCublasWorkspaceForCapture)
    restoreCublasWorkspaceAfterCapture(stream);

    // Reset thread-local state after capture attempt
    tl_captureWorkspace = nullptr;
    tl_captureWorkspaceSize = 0;
    tl_captureWorkspaceOffset = 0;
    // Reset host workspace thread-locals (ownership moves to tl_capturedHostPtrs → replay handle)
    tl_captureHostWorkspace = nullptr;
    tl_captureHostWorkspaceSize = 0;
    tl_captureHostWorkspaceOffset = 0;
    tl_graphCaptureStream = prevCaptureStream;
    // Pinned host ptrs: graph's H2D memcpy nodes reference these on replay.
    // On success: move to segment so they persist for graph lifetime.
    // On failure: free immediately (no graph to replay).
    if (usedTritonGraphCapture && seg.exec.replayHandle) {
      for (auto* ptr : tl_capturedHostPtrs) {
        seg.exec.replayHandle->addCapturedHostPtr(ptr);
      }
      DSP_DIAG(MEMORY, "preserved %zu pinned host ptrs for Triton graph replay",
                seg.exec.replayHandle->getCapturedHostPtrs().size());
    } else {
      // No graph captured — free pinned host ptrs immediately
      for (auto* ptr : tl_capturedHostPtrs) {
        cudaFreeHost(ptr);
      }
    }
    tl_capturedHostPtrs.clear();
    tl_captureReplicateCache.clear();

    // Arrays persist — no pendingClose_ flush needed after capture

    // Restore frozen context state so subsequent executions (including graph replay
    // steps that fall through to direct execution) use the frozen fast path.
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].state_ = savedSlotStateTriton[s - seg.def.startSlot];
    }
    }  // end else (replayHandle != nullptr — workspace allocation succeeded)
  }
#endif

  if (!usedTritonGraphCapture) {
    // ── Batch-zero registration: learn which buffers actually get nullified ──
    // On the execution right before capture (executionCount == 1 → next call is
    // executionCount == 2 which triggers capture), enable registration mode.
    // Each nullify() site calls registerBatchZeroBuffer() when registering,
    // building the exact set of buffers that need zeroing.
    // This replaces the pre-scan approach (collectBatchZeroTargets) which
    // collected ~143 EXTRA buffers for slots that don't actually execute,
    // including buffers whose GPU addresses alias external KV cache inputs.
    bool batchZeroRegistrationActive = false;
#ifdef SD_CUDA
    {
      // Check the same conditions as shouldCaptureTritonGraph but for executionCount==1
      // (the warmup step right BEFORE capture). We register which buffers get nullified
      // so the batch-zero kernel during capture zeros EXACTLY the right set.
      // Registration doesn't require shapesFrozen_ — shapes may freeze after
      // this execution but before capture. We just need to be the pre-capture
      // warmup step (executionCount == 1) with no existing graph.
      bool wouldCaptureNextStep =
          Environment::getInstance().tritonGraphCapture() &&
          seg.exec.replayHandle == nullptr &&
          !seg.exec.compilationFailed &&
          seg.exec.executionCount == 1;
      if (Environment::getInstance().dspBatchZero() && wouldCaptureNextStep) {
        startBatchZeroRegistration();
        batchZeroRegistrationActive = true;
        DSP_DIAG_SEG(MEMORY, seg.def.startSlot, "batch-zero registration enabled for warmup execution (seg[%d-%d] execCount=%d)",
                  seg.def.startSlot, seg.def.endSlot, seg.exec.executionCount);
      }
    }
#endif

    // ── Sync external inputs to device BEFORE setting tl_graphExecutionActive ──
    // Triton's arg table population uses specialBuffer() to resolve GPU pointers.
    // specialBuffer() only calls syncToDevice() when the device buffer is nullptr
    // or on the wrong device — it does NOT check if the device data is stale.
    // Java modifies external inputs (attention_mask, position_ids, input_ids) on the
    // host via putScalar() + tagLocation(HOST), making the device data stale.
    // Native ops handle this via prepareSpecialUse() which calls syncToDevice()
    // unconditionally, but Triton bypasses native ops and reads device buffers directly.
    // We must sync BEFORE setting tl_graphExecutionActive because that flag changes
    // syncToSpecial() to use an async path that skips cudaStreamSynchronize.
    for (int ei = 0; ei < numExt; ei++) {
      if (externalArrays[ei] != nullptr) {
        if (Environment::getInstance().tritonVerifyKernels()) {
          auto* db = externalArrays[ei]->dataBuffer();
          DSP_DIAG(VERIFY, "SLOT_WRITE slot=%d tag=EXT_SYNC(direct) extIdx=%d pAct=%d sAct=%d len=%lld addr=%p",
                    -(ei + 1), ei,
                    db ? (db->isPrimaryActual() ? 1 : 0) : -1,
                    db ? (db->isSpecialActual() ? 1 : 0) : -1,
                    (long long)externalArrays[ei]->lengthOf(),
                    DSP_BUF(externalArrays[ei]));
        }
        externalArrays[ei]->syncToDevice();
      }
    }

    // NOTE: Do NOT set tl_graphExecutionActive=true here for non-capture Triton execution.
    // That flag suppresses syncToPrimary (D2H transfers), error checking, and
    // PointersManager sync -- behaviors only appropriate during CUDA graph capture.
    // The ordered range executor already handles capture detection independently:
    // it checks cudaStreamIsCapturing() and only sets tl_graphExecutionActive=true
    // when actually capturing. Setting it unconditionally here caused native ordered ops
    // (matmul, gather, etc.) to read stale host data, producing wrong output.

    // Disable frozen fast path for gap ops during Triton segment execution.
    // Same rationale as the capture path (lines 5325-5329): the pre-execution
    // slot restoration at lines 4955-5032 may replace NDArray objects in
    // outputSlots_[], making the frozen context's cached input/output pointers
    // stale. Without clearing frozenContextReady, gap ops write to old arrays
    // while downstream ops read from new arrays, producing wrong output.
    // Save and restore so subsequent executions still benefit from frozen fast path.
    std::vector<NativeSlot::SlotState> savedSlotStateNonCapture(seg.def.endSlot - seg.def.startSlot + 1);
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      savedSlotStateNonCapture[s - seg.def.startSlot] = slots_[s].state_;
      if (slots_[s].state_ >= NativeSlot::SlotState::FROZEN)
        slots_[s].state_ = NativeSlot::SlotState::SHAPE_CACHED;
    }

    // Snapshot addresses for direct execution (baseline for comparison with capture/replay)
    {
      std::vector<void*> outAddrs, extAddrs;
      extractDeviceAddrs(outputSlots_, totalOutputSlots_, outAddrs);
      extractDeviceAddrs(externalArrays, numExt, extAddrs);
      DSP_DIAG_SNAPSHOT_ADDRS("direct-entry", outAddrs.data(), totalOutputSlots_,
                               extAddrs.data(), numExt);
    }

    try {
      status = backend->executeSegment(seg, slots_, externalArrays, numExt,
                                       outputSlots_, totalOutputSlots_, stream);
    } catch (...) {
      // Restore frozenContextReady on exception
      for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
        slots_[s].state_ = savedSlotStateNonCapture[s - seg.def.startSlot];
      }
#ifdef SD_CUDA
      if (batchZeroRegistrationActive) {
        finishBatchZeroRegistration();
      }
#endif
      throw;  // Re-throw after cleanup
    }

    // Restore frozen context state so subsequent calls use the frozen fast path
    // once context pointers are re-established by the normal path above.
    for (int s = seg.def.startSlot; s <= seg.def.endSlot; s++) {
      slots_[s].state_ = savedSlotStateNonCapture[s - seg.def.startSlot];
    }

#ifdef SD_CUDA
    if (batchZeroRegistrationActive) {
      finishBatchZeroRegistration();
    }
#endif
  }

  // Dump final output for direct Triton path (baseline comparison)
  if (status == Status::OK && seg.def.endSlot < totalOutputSlots_ &&
      outputSlots_[seg.def.endSlot] != nullptr) {
    auto* finalOut = outputSlots_[seg.def.endSlot];
    if (finalOut->dataType() == FLOAT32) {
      DSP_DIAG_DUMP_SLOT("direct", seg.def.endSlot,
                         DSP_BUF(finalOut), finalOut->lengthOf());
    }
  }
  // Always-on diagnostic: dump top logit for non-capture Triton execution
  if (!usedTritonGraphCapture && status == Status::OK &&
      seg.def.endSlot < totalOutputSlots_ && outputSlots_[seg.def.endSlot] != nullptr) {
    auto* out = outputSlots_[seg.def.endSlot];
    if (out->dataType() == FLOAT32 && out->lengthOf() > 0) {
      DSP_DIAG_DUMP_SEG_OUTPUT("DIRECT_TRITON", seg.def.endSlot, DSP_BUF(out),
                               out->lengthOf(), seg.exec.executionCount, stream);
    }
  }

  DSP_DIAG(EXECUTE, "executeSegmentWithGpuGraph: exec%d seg[%d-%d]: backend=%s %s status=%d(%s) "
            "executionCount=%d compilationFailed=%d usedCapture=%d",
            seg.exec.executionCount, seg.def.startSlot, seg.def.endSlot,
            backendName, status == Status::OK ? "OK" : "FAILED",
            static_cast<int>(status), statusName_gpu(status),
            seg.exec.executionCount,
            seg.exec.compilationFailed ? 1 : 0, usedTritonGraphCapture ? 1 : 0);

  if (status == Status::OK) {
    seg.exec.executionCount++;
    totalGraphReplays_++;
    if (seg.exec.compiledByBackend.empty()) {
      seg.exec.compiledByBackend = backendName;
    }
  }

#ifdef SD_CUDA
  if (Environment::getInstance().tritonVerifyKernels()) {
    DSP_DIAG(VERIFY, "SEG_EXIT seg[%d-%d] status=%s execCount=%d",
              seg.def.startSlot, seg.def.endSlot, statusName_gpu(status), seg.exec.executionCount);
  }
#endif

  return status;
}

}  // namespace graph
}  // namespace sd
