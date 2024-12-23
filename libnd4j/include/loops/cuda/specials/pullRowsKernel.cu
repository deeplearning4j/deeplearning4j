````````````cpp
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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
 * the License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>

    namespace sd {

  template <typename T>
  SD_DEVICE void pullRowsKernel(
      void* vx,
      void* vz,
      LongType len,
      LongType* indexes,
      LongType* tadShapeInfo,
      LongType* tadOffsets,
      LongType* zTadShapeInfo,
      LongType* zTadOffsets) {

    auto x = reinterpret_cast<T*>(vx);
    auto z = reinterpret_cast<T*>(vz);

    __shared__ int tadRank;
    __shared__ const sd::LongType* tadShapePtr;
    __shared__ const sd::LongType* tadStridePtr;
    __shared__ int zTadRank;
    __shared__ const sd::LongType* zTadShapePtr;
    __shared__ const sd::LongType* zTadStridePtr;
    __shared__ sd::LongType tadLen;

    if (threadIdx.x == 0) {
      tadRank       = shape::rank(tadShapeInfo);
      tadShapePtr   = shape::shapeOf(tadShapeInfo);
      tadStridePtr  = shape::stride(tadShapeInfo);

      zTadRank      = shape::rank(zTadShapeInfo);
      zTadShapePtr  = shape::shapeOf(zTadShapeInfo);
      zTadStridePtr = shape::stride(zTadShapeInfo);

      tadLen        = shape::length(tadShapeInfo);
    }
    __syncthreads();

    // Each block handles some subset of the total 'len' (the number of TAD pulls)
    for (sd::LongType idx = blockIdx.x; idx < len; idx += gridDim.x) {
      const auto xTadOffset = tadOffsets[indexes[idx]];
      const auto zTadOffset = zTadOffsets[idx];

      auto rX = x + xTadOffset;
      auto rZ = z + zTadOffset;

      for (sd::LongType i = threadIdx.x; i < tadLen; i += blockDim.x) {
        sd::LongType xCoords[SD_MAX_RANK];
        sd::LongType zCoords[SD_MAX_RANK];
        sd::LongType xOffset;
        sd::LongType zOffset;

        INDEX2COORDS(i, tadRank, tadShapePtr, xCoords);
        COORDS2INDEX(tadRank, tadStridePtr, xCoords, xOffset);

        INDEX2COORDS(i, zTadRank, zTadShapePtr, zCoords);
        COORDS2INDEX(zTadRank, zTadStridePtr, zCoords, zOffset);

        rZ[zOffset] = rX[xOffset];
      }
    }
  }

  template <typename T>
  SD_KERNEL void execPullRowsKernel(
      void* vx,
      void* vz,
      LongType len,
      LongType* indexes,
      LongType* tadShapeInfo,
      LongType* tadOffsets,
      LongType* zTadShapeInfo,
      LongType* zTadOffsets) {

    pullRowsKernel<T>(
        vx, vz, len, indexes, tadShapeInfo, tadOffsets,
        zTadShapeInfo, zTadOffsets);
  }

  template <typename T>
  SD_HOST void pullRowsKernelGeneric(
      dim3 &launchDims,
      cudaStream_t* stream,
      void* vx,
      void* vz,
      LongType len,
      LongType* indexes,
      LongType* tadShapeInfo,
      LongType* tadOffsets,
      LongType* zTadShapeInfo,
      LongType* zTadOffsets) {

    execPullRowsKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
        vx, vz, len, indexes, tadShapeInfo, tadOffsets,
        zTadShapeInfo, zTadOffsets);

    DebugHelper::checkErrorCode(stream, "pullRows(...) failed");
  }

  BUILD_SINGLE_TEMPLATE(
      template void pullRowsKernelGeneric,
      (dim3 &launchDims,
       cudaStream_t* stream,
       void* vx,
       void* vz,
       sd::LongType len,
       sd::LongType* indexes,
       sd::LongType* tadShapeInfo,
       sd::LongType* tadOffsets,
       sd::LongType* zTadShapeInfo,
       sd::LongType* zTadOffsets),
      SD_COMMON_TYPES);

}  // namespace sd
