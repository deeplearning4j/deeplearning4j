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
// CUDA gather_bp helper.
//
// Each CUDA thread processes one element of gradOut.  After locating the
// corresponding dInput element via the gather-axis mapping, it accumulates
// using sd::math::atomics::sd_atomicAdd so that duplicate indices are handled
// correctly (their contributions are summed, not overwritten).
//

#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/gather_bp.h>
#include <system/op_boilerplate.h>
#include <types/types.h>

#include "execution/cuda/LaunchDims.h"

#if NOT_EXCLUDED(OP_gather_bp)

namespace sd {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////
// gatherBpCudaKernel
// One CUDA thread per element of gradOut.
// Computes the destination in dInput and does an atomic-add.
//
// Template params:
//   X  – float element type (gradOut / dInput)
//   Y  – integer index type (indices)
//
template <typename X, typename Y>
SD_KERNEL static void gatherBpCudaKernel(
    const void*    vGradOut,     const LongType* gradOutShapeInfo,
    const void*    vIndices,     const LongType* indicesShapeInfo,
          void*    vDInput,      const LongType* dInputShapeInfo,
    LongType       axis,
    LongType       gradLen,
    LongType       dimSize)      // valid range of gather index along axis
{
  const X* gradBuf = reinterpret_cast<const X*>(vGradOut);
  const Y* idxBuf  = reinterpret_cast<const Y*>(vIndices);
        X* dinBuf  = reinterpret_cast<X*>(vDInput);

  // Cache shape/stride pointers in shared memory to reduce repeated global reads
  __shared__ LongType       zRank, xRank, yRank;
  __shared__ const LongType *zShape, *zStride;
  __shared__ const LongType *xShape, *xStride;
  __shared__ const LongType *yShape, *yStride;

  if (threadIdx.x == 0) {
    zRank  = shape::rank(gradOutShapeInfo);
    xRank  = shape::rank(dInputShapeInfo);
    yRank  = shape::rank(indicesShapeInfo);

    zShape  = shape::shapeOf(gradOutShapeInfo);
    zStride = shape::stride(gradOutShapeInfo);
    xShape  = shape::shapeOf(dInputShapeInfo);
    xStride = shape::stride(dInputShapeInfo);
    yShape  = shape::shapeOf(indicesShapeInfo);
    yStride = shape::stride(indicesShapeInfo);
  }
  __syncthreads();

  const LongType indicesRank = yRank;

  for (LongType j = blockIdx.x * blockDim.x + threadIdx.x; j < gradLen;
       j += blockDim.x * gridDim.x) {
    // ---- 1. Decompose flat index j into gradOut coordinates ----
    LongType zCoords[SD_MAX_RANK] = {};
    INDEX2COORDS(j, zRank, zShape, zCoords);
    LongType zOffset;
    COORDS2INDEX(zRank, zStride, zCoords, zOffset);

    // ---- 2. Extract indices sub-coordinates: zCoords[axis .. axis+indicesRank-1] ----
    LongType yCoords[SD_MAX_RANK] = {};
    for (LongType r = 0; r < indicesRank; ++r) {
      yCoords[r] = zCoords[axis + r];
    }
    LongType yOffset;
    COORDS2INDEX(yRank, yStride, yCoords, yOffset);
    LongType gatherIdx = static_cast<LongType>(idxBuf[yOffset]);

    // Bounds check: skip out-of-range indices
    if (gatherIdx < 0 || gatherIdx >= dimSize) continue;

    // ---- 3. Build dInput coordinates ----
    // pre-axis:  zCoords[0..axis-1]                 → xCoords[0..axis-1]
    // at axis:   gatherIdx                           → xCoords[axis]
    // post-axis: zCoords[axis+indicesRank..zRank-1]  → xCoords[axis+1..xRank-1]
    LongType xCoords[SD_MAX_RANK] = {};
    for (LongType r = 0;        r < axis;   ++r) xCoords[r] = zCoords[r];
    xCoords[axis] = gatherIdx;
    for (LongType r = axis + 1; r < xRank; ++r) xCoords[r] = zCoords[r + indicesRank - 1];

    LongType xOffset;
    COORDS2INDEX(xRank, xStride, xCoords, xOffset);

    // ---- 4. Atomic scatter-add: handles duplicate indices correctly ----
    sd::math::atomics::sd_atomicAdd<X>(&dinBuf[xOffset], gradBuf[zOffset]);
  }
}

////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST static void gatherBpCudaLauncher(const cudaStream_t* stream,
                                         const void*    vGradOut,     const LongType* gradOutShapeInfo,
                                         const void*    vIndices,     const LongType* indicesShapeInfo,
                                               void*    vDInput,      const LongType* dInputShapeInfo,
                                         LongType axis, LongType gradLen, LongType dimSize) {
  dim3 launchDims = getLaunchDims("gather_linear");
  gatherBpCudaKernel<X, Y><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
      vGradOut, gradOutShapeInfo, vIndices, indicesShapeInfo, vDInput, dInputShapeInfo,
      axis, gradLen, dimSize);
  DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "gatherBpCudaKernel failed");
}

////////////////////////////////////////////////////////////////////////
void gatherBp(LaunchContext* context, NDArray* indices,
              NDArray* gradOut, NDArray* dInput, LongType axis) {
  const LongType gradLen = gradOut->lengthOf();
  const LongType dimSize = dInput->sizeAt(axis);

  NDArray::prepareSpecialUse({dInput}, {indices, gradOut});

  // Zero the device buffer before atomicAdd accumulation
  cudaMemsetAsync(dInput->specialBuffer(), 0,
                  static_cast<size_t>(dInput->lengthOf()) * dInput->sizeOfT(),
                  *context->getCudaStream());

  // specialShapeInfo() is already device-resident — pass it straight to the kernel
  // (do NOT replicatePointer it: that API expects a HOST source pointer).
  BUILD_DOUBLE_SELECTOR(
      gradOut->dataType(), indices->dataType(), gatherBpCudaLauncher,
      (context->getCudaStream(),
       gradOut->specialBuffer(), gradOut->specialShapeInfo(),
       indices->specialBuffer(), indices->specialShapeInfo(),
       dInput->specialBuffer(),  dInput->specialShapeInfo(),
       axis, gradLen, dimSize),
      SD_COMMON_TYPES, SD_INDEXING_TYPES);

  // registerSpecialUse: marks device buffer as the authoritative copy for dInput
  NDArray::registerSpecialUse({dInput}, {indices, gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_gather_bp)
