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
// CPU gather_bp helper.
//
// For each element j in gradOut (same shape as gather's forward output):
//   1. Decompose j into gradOut coords.
//   2. The gather-axis "slot" in gradOut coords holds indices sub-coords;
//      look up the actual axis index from the indices tensor.
//   3. Map to the corresponding dInput element and add.
//
// Single-threaded accumulation → no atomics needed on CPU.
//

#include <execution/Threads.h>
#include <helpers/shape.h>
#include <ops/declarable/helpers/gather_bp.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#if NOT_EXCLUDED(OP_gather_bp)

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
static void gatherBpImpl(NDArray* indices, NDArray* gradOut,
                         NDArray* dInput, LongType axis) {
  const LongType inputRank  = dInput->rankOf();
  const LongType indicesRank = indices->rankOf();
  const LongType zRank      = gradOut->rankOf();  // = inputRank + indicesRank - 1

  const X* gradBuf  = gradOut->bufferAsT<X>();
  X*       dinBuf   = dInput->bufferAsT<X>();
  const Y* idxBuf   = indices->bufferAsT<Y>();

  const auto zShapeInfo = gradOut->shapeInfo();
  const auto xShapeInfo = dInput->shapeInfo();
  const auto yShapeInfo = indices->shapeInfo();

  const LongType* zShape  = shape::shapeOf(zShapeInfo);
  const LongType* zStride = shape::stride(zShapeInfo);
  const LongType* xShape  = shape::shapeOf(xShapeInfo);
  const LongType* xStride = shape::stride(xShapeInfo);
  const LongType* yShape  = shape::shapeOf(yShapeInfo);
  const LongType* yStride = shape::stride(yShapeInfo);

  const LongType gradLen = gradOut->lengthOf();
  const LongType dimSize = dInput->sizeAt(axis);

  for (LongType j = 0; j < gradLen; ++j) {
    // Decompose flat index j into gradOut coordinates
    LongType zCoords[SD_MAX_RANK] = {};
    INDEX2COORDS(j, zRank, zShape, zCoords);
    LongType zOffset;
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    // Extract indices sub-coordinates from zCoords[axis .. axis+indicesRank-1]
    LongType yCoords[SD_MAX_RANK] = {};
    for (LongType r = 0; r < indicesRank; ++r) {
      yCoords[r] = zCoords[axis + r];
    }
    LongType yOffset;
    COORDS2INDEX(indicesRank, yStride, yCoords, yOffset);
    LongType gatherIdx = static_cast<LongType>(idxBuf[yOffset]);

    // Bounds check: skip out-of-range indices (mirrors gather forward clamping)
    if (gatherIdx < 0 || gatherIdx >= dimSize) continue;

    // Build dInput coordinates
    // pre-axis: zCoords[0..axis-1]  → xCoords[0..axis-1]
    // at axis:  gatherIdx           → xCoords[axis]
    // post-axis: zCoords[axis+indicesRank..zRank-1] → xCoords[axis+1..inputRank-1]
    LongType xCoords[SD_MAX_RANK] = {};
    for (LongType r = 0;       r < axis;       ++r) xCoords[r] = zCoords[r];
    xCoords[axis] = gatherIdx;
    for (LongType r = axis + 1; r < inputRank; ++r) xCoords[r] = zCoords[r + indicesRank - 1];

    LongType xOffset;
    COORDS2INDEX(inputRank, xStride, xCoords, xOffset);

    // Scatter-accumulate: dInput[xCoords] += gradOut[zCoords]
    dinBuf[xOffset] += gradBuf[zOffset];
  }
}

////////////////////////////////////////////////////////////////////////
void gatherBp(LaunchContext* context, NDArray* indices,
              NDArray* gradOut, NDArray* dInput, LongType axis) {
  // Zero dInput before scatter-accumulation.
  // Do NOT use dInput->assign(0): the literal 0 is a null-pointer constant, and the
  // scalar assign(T&) overload needs an lvalue, so assign(0) binds to
  // assign(NDArray* = nullptr) -> validateAssign -> nullptr->isEmpty() -> SIGSEGV.
  dInput->nullify();

  BUILD_DOUBLE_SELECTOR(gradOut->dataType(), indices->dataType(), gatherBpImpl,
                        (indices, gradOut, dInput, axis),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_gather_bp)
