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
  SD_DEVICE void fillDimensionalIsMax(
      const void* vdX,
      void* vdZ,
      const LongType* zShapeInfo,
      const LongType* tadOnlyShapeInfo,
      LongType* dimension,
      LongType dimensionLength,
      const LongType* tadOffsets) {

    const auto dX = reinterpret_cast<const LongType*>(vdX);
    auto dZ       = reinterpret_cast<T*>(vdZ);

    __shared__ int tadLen;
    __shared__ int numTads;
    __shared__ int tadRank;
    __shared__ const sd::LongType* tadShapePtr;
    __shared__ const sd::LongType* tadStridePtr;
    __shared__ int zRank;
    __shared__ const sd::LongType* zShapePtr;
    __shared__ const sd::LongType* zStridePtr;

    if (threadIdx.x == 0) {
      tadLen      = static_cast<int>(shape::length(tadOnlyShapeInfo));
      numTads     = static_cast<int>(shape::length(zShapeInfo) / tadLen);

      tadRank     = shape::rank(tadOnlyShapeInfo);
      tadShapePtr = shape::shapeOf(tadOnlyShapeInfo);
      tadStridePtr= shape::stride(tadOnlyShapeInfo);

      zRank       = shape::rank(zShapeInfo);
      zShapePtr   = shape::shapeOf(zShapeInfo);
      zStridePtr  = shape::stride(zShapeInfo);
    }
    __syncthreads();

    // each block handles some portion of the TADs
    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
      const auto tadOffsetForBlock = tadOffsets[r];
      const auto highestElement    = dX[r];  // this is presumably the index in [0..tadLen)

      // Each thread does part of the tad's length
      // dimensionLength determines if we have multiple dims, but code is the same
      for (LongType e = threadIdx.x; e < tadLen; e += blockDim.x) {
        sd::LongType coords[SD_MAX_RANK];
        sd::LongType offset;

        INDEX2COORDS(e, tadRank, tadShapePtr, coords);
        COORDS2INDEX(tadRank, tadStridePtr, coords, offset);

        const auto finalOffset = tadOffsetForBlock + offset;
        dZ[finalOffset] = (e == highestElement ? static_cast<T>(1) : static_cast<T>(0));
      }
    }
  }

  template <typename T>
  SD_KERNEL void execfillDimensionalIsMax(
      const void* dX,
      void* dZ,
      const LongType* zShapeInfo,
      const LongType* tadOnlyShapeInfo,
      LongType* dimension,
      LongType dimensionLength,
      const LongType* tadOffsets) {

    fillDimensionalIsMax<T>(
        dX, dZ, zShapeInfo, tadOnlyShapeInfo,
        dimension, dimensionLength, tadOffsets);
  }

  template <typename T>
  SD_HOST void fillDimensionalIsMaxGeneric(
      dim3& launchDims,
      cudaStream_t* stream,
      const void* dX,
      void* dZ,
      const LongType* zShapeInfo,
      const LongType* tadOnlyShapeInfo,
      LongType* dimension,
      LongType dimensionLength,
      const LongType* tadOffsets) {

    execfillDimensionalIsMax<T>
        <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
            dX, dZ, zShapeInfo, tadOnlyShapeInfo,
            dimension, dimensionLength, tadOffsets);

    DebugHelper::checkErrorCode(stream, "fillDimensionalIsMax(...) failed");
  }

  BUILD_SINGLE_TEMPLATE(
      template void fillDimensionalIsMaxGeneric,
      (dim3 & launchDims,
       cudaStream_t *stream,
       const void* dX,
       void* dZ,
       const sd::LongType* zShapeInfo,
       const sd::LongType* tadOnlyShapeInfo,
       sd::LongType* dimension,
       sd::LongType dimensionLength,
       const sd::LongType* tadOffsets),
      SD_COMMON_TYPES);

}  // namespace sd
