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

#if HAVE_TRITON

#include <graph/gpu/FusionScoring.h>
#include <graph/gpu/SectionTypeConfig.h>
#include <graph/gpu/TritonIRBuilder.h>
#include <graph/NativeDynamicShapePlan.h>
#include <array/DataType.h>
#include <graph/DspDiagnostics.h>

#include <unordered_set>
#include <algorithm>

namespace sd {
namespace graph {

// Grid type classification used by the fusion cost model. Mixed grid types are
// generally discouraged, with an explicit allowance for attention neighborhoods.
static int gridTypeForSection(const KernelSection& sec) {
  return static_cast<int>(getSectionTypeConfig(sec.type).gridType);
}

// Estimate shared memory usage per section (heuristic based on type + ops).
// Returns bytes.
static size_t estimateSharedMemBytes(const KernelSection& sec) {
  switch (sec.type) {
    case KernelSectionType::REDUCTION:
    case KernelSectionType::NORMALIZATION:
      // Reduction typically uses one warp-reduction buffer per output
      return static_cast<size_t>(sec.numOps) * 256 * sizeof(float);
    case KernelSectionType::FUSED_ATTENTION:
      return 48 * 1024;  // Flash attention uses ~48KB shared mem
    case KernelSectionType::MATMUL:
      return 32 * 1024;  // Tiled matmul shared mem
    default:
      return 0;  // Elementwise/gather/etc use no shared mem
  }
}

static bool isAttentionAdjacentType(KernelSectionType type) {
  switch (type) {
    case KernelSectionType::GATHER:
    case KernelSectionType::GATHER_ND:
    case KernelSectionType::CONCAT:
    case KernelSectionType::STACK:
      return true;
    default:
      return false;
  }
}

// Estimate bytes of a DataType
static size_t dtypeBytes(DataType dt) {
  switch (dt) {
    case DataType::HALF: case DataType::BFLOAT16: return 2;
    case DataType::FLOAT32: case DataType::INT32: return 4;
    case DataType::DOUBLE: case DataType::INT64: return 8;
    case DataType::INT8: case DataType::UINT8: case DataType::BOOL: return 1;
    case DataType::INT16: case DataType::UINT16: return 2;
    default: return 4;
  }
}

float scoreSectionFusionRange(
    const std::vector<KernelSection>& sections,
    int rangeStartSectionIdx,
    int rangeEndSectionIdx,
    int nextSectionIdx,
    NativeSlot* slots,
    int segmentStartSlot,
    int segmentEndSlot,
    NDArray** outputSlots,
    int totalOutputSlots) {
  if (rangeStartSectionIdx < 0 || rangeEndSectionIdx < rangeStartSectionIdx ||
      nextSectionIdx <= rangeEndSectionIdx ||
      nextSectionIdx >= static_cast<int>(sections.size())) {
    return -1.0f;
  }

  const auto& nextSection = sections[nextSectionIdx];
  int rangeStartSlot = sections[rangeStartSectionIdx].startSlot;
  int rangeEndSlot = sections[rangeEndSectionIdx].endSlot;
  int combinedEndSlot = nextSection.endSlot;

  // 1. Grid compatibility — incompatible grid types cannot merge
  std::unordered_set<int> rangeGridTypes;
  std::unordered_set<int> rangeOutputSlots;
  size_t maxShared = 0;
  int combinedOps = 0;
  bool rangeHasAttention = false;
  bool rangeHasAttnAdj = false;

  for (int secIdx = rangeStartSectionIdx; secIdx <= rangeEndSectionIdx; secIdx++) {
    const auto& sec = sections[secIdx];
    rangeGridTypes.insert(gridTypeForSection(sec));
    maxShared = std::max(maxShared, estimateSharedMemBytes(sec));
    combinedOps += sec.numOps;
    rangeHasAttention = rangeHasAttention || sec.type == KernelSectionType::FUSED_ATTENTION;
    rangeHasAttnAdj = rangeHasAttnAdj || isAttentionAdjacentType(sec.type);
    for (int s = sec.startSlot; s <= sec.endSlot; s++) {
      NativeSlot& slot = slots[s];
      for (int i = 0; i < slot.numOutputs; i++) {
        rangeOutputSlots.insert(slot.outputSlotIndices[i]);
      }
    }
  }

  int nextGrid = gridTypeForSection(nextSection);
  bool nextIsAttention = nextSection.type == KernelSectionType::FUSED_ATTENTION;
  bool nextIsAttnAdj = isAttentionAdjacentType(nextSection.type);
  bool allowAttentionNeighborhoodMismatch =
      sd::Environment::getInstance().tritonFuseAttentionNeighborhoods() &&
      ((rangeHasAttention && nextIsAttnAdj) || (nextIsAttention && rangeHasAttnAdj));
  bool gridCompatible =
      rangeGridTypes.empty() || (rangeGridTypes.size() == 1 && rangeGridTypes.count(nextGrid) == 1);
  if (!gridCompatible && !allowAttentionNeighborhoodMismatch) {
    DSP_DIAG(FUSION, "FusionScoring: range [%d-%d] + [%d-%d] grid mismatch -> -1.0",
             rangeStartSlot, rangeEndSlot, nextSection.startSlot, nextSection.endSlot);
    return -1.0f;
  }

  // 2. Shared memory check — sectioned kernels reserve the max section requirement.
  maxShared = std::max(maxShared, estimateSharedMemBytes(nextSection));
  static constexpr size_t SM_SHARED_MEM_LIMIT = 96 * 1024;  // Conservative 96KB
  if (maxShared > SM_SHARED_MEM_LIMIT) {
    DSP_DIAG(FUSION, "FusionScoring: range [%d-%d] + [%d-%d] max shared mem %zuKB > limit %zuKB -> -1.0",
             rangeStartSlot, rangeEndSlot, nextSection.startSlot, nextSection.endSlot,
             maxShared / 1024, SM_SHARED_MEM_LIMIT / 1024);
    return -1.0f;
  }

  // 3. Memory traffic savings — count outputs from the current range that become
  // internal when the next section is fused, but only if they are not consumed
  // elsewhere in the surrounding segment.
  size_t intermediateBytes = 0;
  for (int outSlotIdx : rangeOutputSlots) {
    if (outSlotIdx < 0 || outSlotIdx >= totalOutputSlots) continue;
    bool consumedOutside = false;
    NDArray* arr = (outSlotIdx < totalOutputSlots) ? outputSlots[outSlotIdx] : nullptr;
    if (arr == nullptr) continue;

    bool consumedByNext = false;
    for (int s = nextSection.startSlot; s <= nextSection.endSlot && !consumedByNext; s++) {
      NativeSlot& bSlot = slots[s];
      for (int i = 0; i < bSlot.numInputs; i++) {
        int srcIdx = bSlot.inputSourceIndices[i];
        if (srcIdx == outSlotIdx) {
          consumedByNext = true;
          break;
        }
      }
    }
    if (!consumedByNext) continue;

    for (int s = segmentStartSlot; s <= segmentEndSlot && !consumedOutside; s++) {
      if (s >= rangeStartSlot && s <= combinedEndSlot) continue;
      NativeSlot& otherSlot = slots[s];
      for (int i = 0; i < otherSlot.numInputs; i++) {
        if (otherSlot.inputSourceIndices[i] == outSlotIdx) {
          consumedOutside = true;
          break;
        }
      }
    }

    if (!consumedOutside) {
      size_t elemBytes = dtypeBytes(arr->dataType());
      intermediateBytes += arr->lengthOf() * elemBytes;
    }
  }

  float memScore = static_cast<float>(intermediateBytes) / (1024.0f * 1024.0f);  // in MB

  // 4. Kernel launch overhead saved (~15μs per eliminated launch)
  float launchScore = 15.0f;

  // 5. Register pressure penalty — large merged kernels risk spills
  combinedOps += nextSection.numOps;
  float regPenalty = std::max(0.0f, (combinedOps - 256) * 0.1f);

  // 6. Attention neighborhood bonus — only when the candidate extension is
  // actually adjacent to an attention section in the current range.
  float attnBonus = allowAttentionNeighborhoodMismatch ? 50.0f : 0.0f;

  float totalScore = memScore * 10.0f + launchScore - regPenalty + attnBonus;

  DSP_DIAG(FUSION, "FusionScoring: range [%d-%d]+[%d-%d] memMB=%.2f launch=%.1f regPen=%.1f attnBonus=%.1f -> score=%.2f",
           rangeStartSlot, rangeEndSlot, nextSection.startSlot, nextSection.endSlot,
           memScore, launchScore, regPenalty, attnBonus, totalScore);

  return totalScore;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
