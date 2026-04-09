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
// ViewRecipe — captures view-producing ops (reshape, permute, expand_dims,
// squeeze, strided_slice) so they can be installed during REPLAYING without
// launching a kernel or executing a fallback gap.
//
// Contract:
//   SHAPES_FROZEN:  Recipe is captured and validated (source shape matches).
//   POINTERS_STABLE: Source buffer address is verified stable.
//   REPLAYING:      View is installed before consumer replay — no gap execution.
//
// If any view op would materialize instead of remaining a view, hard error.
//

#ifndef LIBND4J_VIEW_RECIPE_H
#define LIBND4J_VIEW_RECIPE_H

#include <config.h>

#include <array/DataType.h>
#include <system/common.h>

#include <cstdint>
#include <vector>

namespace sd {
namespace graph {

/**
 * Type of view-producing operation.
 * Only ops that can produce a zero-copy view are listed here.
 */
enum class ViewRecipeType : uint8_t {
  RESHAPE,           // Reshape with same element count
  RESHAPE_NO_COPY,   // Explicit no-copy reshape
  PERMUTE,           // Transpose/permute — stride reordering only
  EXPAND_DIMS,       // Insert singleton dimension
  SQUEEZE,           // Remove singleton dimension
  FLATTEN,           // Flatten to 1D or 2D
  STRIDED_SLICE,     // Slice with strides (only when result is a contiguous view)
};

/**
 * Captured parameters for a single view-producing op.
 * Small fixed-size buffers to avoid heap allocation during capture.
 */
struct ViewRecipe {
  static constexpr int MAX_SHAPE_RANK = 8;

  ViewRecipeType type;
  int outputSlotIndex;       // Which output slot this recipe produces
  int sourceSlotIndex;       // Which input/source slot the view is derived from
  int rank;                  // Rank of the output shape
  char outputOrder;          // Frozen output order from warmup/capture
  LongType outputEws;        // Frozen output EWS from warmup/capture
  LongType outputOffset;     // Frozen output offset into the shared data buffer
  LongType outputExtra;      // Frozen output extras/flags (dtype is rewritten at install)

  LongType outputShape[MAX_SHAPE_RANK];   // Frozen output shape
  LongType outputStrides[MAX_SHAPE_RANK]; // Computed output strides (for view)

  // For permute: the permutation vector
  int perm[MAX_SHAPE_RANK];

  // For strided_slice: begin/end/stride per dim
  LongType sliceBegin[MAX_SHAPE_RANK];
  LongType sliceEnd[MAX_SHAPE_RANK];
  LongType sliceStride[MAX_SHAPE_RANK];
  int sliceRank;  // Rank of the slice parameters

  // Validation state
  bool validated;              // True after POINTERS_STABLE address check
  void* capturedSourceAddr;    // Source buffer address at capture time
  LongType capturedSourceBytes;// Source buffer size at capture time

  ViewRecipe()
      : type(ViewRecipeType::RESHAPE),
        outputSlotIndex(-1), sourceSlotIndex(-1), rank(0),
        outputOrder('c'), outputEws(1), outputOffset(0), outputExtra(0),
        perm{}, sliceBegin{}, sliceEnd{}, sliceStride{},
        sliceRank(0), validated(false),
        capturedSourceAddr(nullptr), capturedSourceBytes(0) {
    for (int i = 0; i < MAX_SHAPE_RANK; i++) {
      outputShape[i] = 0;
      outputStrides[i] = 0;
    }
  }

  // Check if this recipe is semantically a no-op (input shape == output shape).
  // No-op views don't need to be installed — the output slot can just alias
  // the source slot directly.
  bool isNoop() const {
    if (type != ViewRecipeType::RESHAPE &&
        type != ViewRecipeType::RESHAPE_NO_COPY &&
        type != ViewRecipeType::FLATTEN) {
      return false;
    }
    if (rank != sliceRank) return false;
    for (int i = 0; i < rank; i++) {
      if (outputShape[i] != sliceBegin[i]) return false; // sliceBegin reused for input shape
    }
    return true;
  }
};

/**
 * A sequence of view recipes that form a view chain within a segment.
 * Multiple view ops can chain together (e.g., reshape -> permute -> expand_dims).
 * Each recipe's output may be the next recipe's source.
 */
struct ViewRecipeChain {
  std::vector<ViewRecipe> recipes;
  int segmentStartSlot;
  int segmentEndSlot;

  ViewRecipeChain() : segmentStartSlot(-1), segmentEndSlot(-1) {}
};

}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_VIEW_RECIPE_H
