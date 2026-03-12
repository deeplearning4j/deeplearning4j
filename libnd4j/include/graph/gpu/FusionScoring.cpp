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
#include <graph/gpu/TritonIRBuilder.h>
#include <graph/NativeDynamicShapePlan.h>
#include <array/DataType.h>
#include <graph/DspDiagnostics.h>

#include <unordered_set>
#include <algorithm>

namespace sd {
namespace graph {

// Grid type classification — sections with different grid types cannot be fused
// because they require different launch configurations.
static int gridTypeForSection(const KernelSection& sec) {
  switch (sec.type) {
    case KernelSectionType::ELEMENTWISE:
    case KernelSectionType::IDENTITY:
      return 0;  // 1D linear grid
    case KernelSectionType::REDUCTION:
    case KernelSectionType::NORMALIZATION:
      return 1;  // Reduction grid (row-major, per-row threads)
    case KernelSectionType::GATHER:
    case KernelSectionType::GATHER_ND:
    case KernelSectionType::SCATTER_ND:
    case KernelSectionType::SCATTER_ND_UPDATE:
      return 2;  // Indexed access grid
    case KernelSectionType::CONCAT:
    case KernelSectionType::SPLIT:
    case KernelSectionType::SPLIT_V:
    case KernelSectionType::STACK:
      return 3;  // Data movement grid
    case KernelSectionType::MATMUL:
      return 4;  // 2D tiled grid
    case KernelSectionType::FUSED_ATTENTION:
      return 5;  // Multi-dim attention grid
    case KernelSectionType::STRIDED_SLICE:
    case KernelSectionType::TILE:
      return 6;  // Strided access grid
    case KernelSectionType::SHAPE_MANIPULATION:
    case KernelSectionType::CONSTANT_GENERATION:
      return 7;  // Host/metadata ops
    case KernelSectionType::CONVOLUTION:
      return 8;  // Convolution grid
    default:
      return -1;
  }
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

float scoreSectionFusion(
    const KernelSection& sectionA,
    const KernelSection& sectionB,
    NativeSlot* slots,
    NDArray** outputSlots,
    int totalOutputSlots) {

  // 1. Grid compatibility — incompatible grid types cannot merge
  int gridA = gridTypeForSection(sectionA);
  int gridB = gridTypeForSection(sectionB);
  if (gridA != gridB) {
    DSP_DIAG(FUSION, "FusionScoring: sections [%d-%d] and [%d-%d] grid mismatch (%d vs %d) -> -1.0",
             sectionA.startSlot, sectionA.endSlot, sectionB.startSlot, sectionB.endSlot, gridA, gridB);
    return -1.0f;
  }

  // 2. Shared memory check — combined must fit SM limit (typically 96KB-164KB)
  size_t combinedShared = estimateSharedMemBytes(sectionA) + estimateSharedMemBytes(sectionB);
  static constexpr size_t SM_SHARED_MEM_LIMIT = 96 * 1024;  // Conservative 96KB
  if (combinedShared > SM_SHARED_MEM_LIMIT) {
    DSP_DIAG(FUSION, "FusionScoring: sections [%d-%d]+[%d-%d] combined shared mem %zuKB > limit %zuKB -> -1.0",
             sectionA.startSlot, sectionA.endSlot, sectionB.startSlot, sectionB.endSlot,
             combinedShared / 1024, SM_SHARED_MEM_LIMIT / 1024);
    return -1.0f;
  }

  // 3. Memory traffic savings — count intermediate bytes eliminated
  //    Find outputs of A consumed only by B (not by other sections or final outputs)
  std::unordered_set<int> sectionAOutputSlots;
  for (int s = sectionA.startSlot; s <= sectionA.endSlot; s++) {
    NativeSlot& slot = slots[s];
    for (int i = 0; i < slot.numOutputs; i++) {
      sectionAOutputSlots.insert(slot.outputSlotIndices[i]);
    }
  }

  // Check which A outputs are consumed ONLY by B
  size_t intermediateBytes = 0;
  for (int outSlotIdx : sectionAOutputSlots) {
    if (outSlotIdx >= totalOutputSlots) continue;

    // Check if this output is a final output (consumed outside any section)
    // For simplicity, check if any slot outside A+B range consumes it
    bool consumedOutside = false;

    // Simple heuristic: if the output array exists, assume it's intermediate
    // between A and B. Count it as saved memory traffic.
    NDArray* arr = (outSlotIdx < totalOutputSlots) ? outputSlots[outSlotIdx] : nullptr;
    if (arr != nullptr) {
      // Check if B consumes this slot
      bool consumedByB = false;
      for (int s = sectionB.startSlot; s <= sectionB.endSlot && !consumedByB; s++) {
        NativeSlot& bSlot = slots[s];
        for (int i = 0; i < bSlot.numInputs; i++) {
          int srcIdx = bSlot.inputSourceIndices[i];
          if (srcIdx == outSlotIdx) {
            consumedByB = true;
            break;
          }
        }
      }

      if (consumedByB) {
        size_t elemBytes = dtypeBytes(arr->dataType());
        intermediateBytes += arr->lengthOf() * elemBytes;
      }
    }
  }

  float memScore = static_cast<float>(intermediateBytes) / (1024.0f * 1024.0f);  // in MB

  // 4. Kernel launch overhead saved (~15μs per eliminated launch)
  float launchScore = 15.0f;

  // 5. Register pressure penalty — large merged kernels risk spills
  int combinedOps = sectionA.numOps + sectionB.numOps;
  float regPenalty = std::max(0.0f, (combinedOps - 256) * 0.1f);

  float totalScore = memScore * 10.0f + launchScore - regPenalty;

  DSP_DIAG(FUSION, "FusionScoring: [%d-%d]+[%d-%d] memMB=%.2f launch=%.1f regPen=%.1f -> score=%.2f",
           sectionA.startSlot, sectionA.endSlot, sectionB.startSlot, sectionB.endSlot,
           memScore, launchScore, regPenalty, totalScore);

  return totalScore;
}

}  // namespace graph
}  // namespace sd

#endif  // HAVE_TRITON
