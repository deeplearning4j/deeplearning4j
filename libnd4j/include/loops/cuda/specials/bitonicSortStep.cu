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
// @author Yurii Shyrma, created on 28.11.2018
//
#include <ops/specials_cuda.h>

    //////////////////////////////////////////////////////////////////////////
    template <typename X, typename Y>
    SD_KERNEL void bitonicSortStepKernelKey(
        void* vx,
        const sd::LongType* xShapeInfo,
        void* vy,
        const sd::LongType* yShapeInfo,
        int j,
        int k,
        int length,
        bool descending) {

  auto x           = static_cast<X*>(vx);
  auto y           = static_cast<Y*>(vy);
  const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ sd::LongType yRank;
  __shared__ const sd::LongType* yShapePtr;
  __shared__ const sd::LongType* yStridePtr;

  __shared__ sd::LongType xLength;

  if (threadIdx.x == 0) {
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);

    yRank      = shape::rank(yShapeInfo);
    yShapePtr  = shape::shapeOf(yShapeInfo);
    yStridePtr = shape::stride(yShapeInfo);

    xLength    = shape::length(xShapeInfo);
  }
  __syncthreads();

  if (i >= static_cast<unsigned int>(length)) return;

  const unsigned int ixj = i ^ j;
  if (ixj <= i) return;

  sd::LongType iCoords[SD_MAX_RANK];
  sd::LongType ixjCoords[SD_MAX_RANK];
  sd::LongType iOffset;
  sd::LongType ixjOffset;

  INDEX2COORDS(i, xRank, xShapePtr, iCoords);
  COORDS2INDEX(xRank, xStridePtr, iCoords, iOffset);

  INDEX2COORDS(ixj, xRank, xShapePtr, ixjCoords);
  COORDS2INDEX(xRank, xStridePtr, ixjCoords, ixjOffset);

  const bool ascending = ((i & k) == 0);
  X xi = x[iOffset];
  X xixj = x[ixjOffset];

  if (ascending) {
    // Sort ascending
    if (!descending == (xi > xixj)) {
      x[iOffset]      = xixj;
      x[ixjOffset]    = xi;

      sd::LongType iCoordsY[SD_MAX_RANK];
      sd::LongType ixjCoordsY[SD_MAX_RANK];
      sd::LongType iOffsetY;
      sd::LongType ixjOffsetY;

      INDEX2COORDS(i, yRank, yShapePtr, iCoordsY);
      COORDS2INDEX(yRank, yStridePtr, iCoordsY, iOffsetY);

      INDEX2COORDS(ixj, yRank, yShapePtr, ixjCoordsY);
      COORDS2INDEX(yRank, yStridePtr, ixjCoordsY, ixjOffsetY);

      Y yi   = y[iOffsetY];
      Y yixj = y[ixjOffsetY];
      y[iOffsetY]   = yixj;
      y[ixjOffsetY] = yi;
    }
  }
  else {
    // Sort descending
    if (!descending == (xi < xixj)) {
      x[iOffset]      = xixj;
      x[ixjOffset]    = xi;

      sd::LongType iCoordsY[SD_MAX_RANK];
      sd::LongType ixjCoordsY[SD_MAX_RANK];
      sd::LongType iOffsetY;
      sd::LongType ixjOffsetY;

      INDEX2COORDS(i, yRank, yShapePtr, iCoordsY);
      COORDS2INDEX(yRank, yStridePtr, iCoordsY, iOffsetY);

      INDEX2COORDS(ixj, yRank, yShapePtr, ixjCoordsY);
      COORDS2INDEX(yRank, yStridePtr, ixjCoordsY, ixjOffsetY);

      Y yi   = y[iOffsetY];
      Y yixj = y[ixjOffsetY];
      y[iOffsetY]   = yixj;
      y[ixjOffsetY] = yi;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void bitonicSortStepKernel(
    void* vx,
    const sd::LongType* xShapeInfo,
    int j,
    int k,
    int length,
    bool descending) {

  auto x           = static_cast<T*>(vx);
  const unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;
  __shared__ sd::LongType xLength;

  if (threadIdx.x == 0) {
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);

    xLength    = shape::length(xShapeInfo);
  }
  __syncthreads();

  if (i >= static_cast<unsigned int>(length)) return;

  const unsigned int ixj = i ^ j;
  if (ixj <= i) return;

  sd::LongType iCoords[SD_MAX_RANK];
  sd::LongType ixjCoords[SD_MAX_RANK];
  sd::LongType iOffset;
  sd::LongType ixjOffset;

  INDEX2COORDS(i, xRank, xShapePtr, iCoords);
  COORDS2INDEX(xRank, xStridePtr, iCoords, iOffset);

  INDEX2COORDS(ixj, xRank, xShapePtr, ixjCoords);
  COORDS2INDEX(xRank, xStridePtr, ixjCoords, ixjOffset);

  const bool ascending = ((i & k) == 0);
  T xi   = x[iOffset];
  T xixj = x[ixjOffset];

  if (ascending) {
    // Sort ascending
    if (!descending == (xi > xixj)) {
      x[iOffset]    = xixj;
      x[ixjOffset]  = xi;
    }
  }
  else {
    // Sort descending
    if (!descending == (xi < xixj)) {
      x[iOffset]    = xixj;
      x[ixjOffset]  = xi;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void bitonicSortStepGeneric(
    dim3 &launchDims,
    cudaStream_t *stream,
    void* vx,
    const sd::LongType* xShapeInfo,
    int j,
    int k,
    int length,
    bool descending) {

  bitonicSortStepKernel<T>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          vx,
          xShapeInfo,
          j,
          k,
          length,
          descending);

  sd::DebugHelper::checkErrorCode(stream, "bitonicSortStepGeneric failed");
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicSortStepGenericKey(
    dim3 &launchDims,
    cudaStream_t *stream,
    void* vx,
    const sd::LongType* xShapeInfo,
    void* vy,
    const sd::LongType* yShapeInfo,
    int j,
    int k,
    int length,
    bool descending) {

  bitonicSortStepKernelKey<X, Y>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          vx,
          xShapeInfo,
          vy,
          yShapeInfo,
          j,
          k,
          length,
          descending);

  sd::DebugHelper::checkErrorCode(stream, "bitonicSortStepGenericKey failed");
}

BUILD_SINGLE_TEMPLATE(
    template void bitonicSortStepGeneric,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
     int j, int k, int length, bool descending),
    SD_COMMON_TYPES);

BUILD_DOUBLE_TEMPLATE(
    template void bitonicSortStepGenericKey,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
     void *vy, sd::LongType const *yShapeInfo, int j, int k, int length, bool descending),
    SD_COMMON_TYPES, SD_COMMON_TYPES);
