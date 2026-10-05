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

// One compare-exchange step (j, k) of the bitonic sorting network over the length logical elements of a
// power-of-two array: element i is compared with element i ^ j and the pair is ordered by the direction of the
// k-sized block that holds i. The pairs of one step are disjoint (the lower index of each decides), so the grid
// walks the indices with a grid-stride loop: any launch covers every element however long the array is.
// Every element is addressed through the array's own shape and strides.

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL SD_INLINE void bitonicSortStepKernelKey(
    void* vx,
    const sd::LongType* xShapeInfo,
    void* vy,
    const sd::LongType* yShapeInfo,
    int j,
    int k,
    int length,
    bool descending) {

  auto x = static_cast<X*>(vx);
  auto y = static_cast<Y*>(vy);

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ sd::LongType yRank;
  __shared__ const sd::LongType* yShapePtr;
  __shared__ const sd::LongType* yStridePtr;

  if (threadIdx.x == 0) {
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);

    yRank      = shape::rank(yShapeInfo);
    yShapePtr  = shape::shapeOf(yShapeInfo);
    yStridePtr = shape::stride(yShapeInfo);
  }
  __syncthreads();

  const sd::LongType step = static_cast<sd::LongType>(gridDim.x) * blockDim.x;
  for (sd::LongType i = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length; i += step) {
    const sd::LongType ixj = i ^ j;
    if (ixj <= i || ixj >= length) continue;

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

    // ascending blocks put the smaller key first (the larger when descending), descending blocks the reverse
    const bool exchange = ascending ? (!descending == (xi > xixj)) : (!descending == (xi < xixj));
    if (exchange) {
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
SD_KERNEL SD_INLINE void bitonicSortStepKernel(
    void* vx,
    const sd::LongType* xShapeInfo,
    int j,
    int k,
    int length,
    bool descending) {

  auto x = static_cast<T*>(vx);

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  if (threadIdx.x == 0) {
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);
  }
  __syncthreads();

  const sd::LongType step = static_cast<sd::LongType>(gridDim.x) * blockDim.x;
  for (sd::LongType i = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length; i += step) {
    const sd::LongType ixj = i ^ j;
    if (ixj <= i || ixj >= length) continue;

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

    const bool exchange = ascending ? (!descending == (xi > xixj)) : (!descending == (xi < xixj));
    if (exchange) {
      x[iOffset]    = xixj;
      x[ixjOffset]  = xi;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// The launchers enqueue one step on the caller's stream. The step kernels use no dynamic shared memory, so none is
// requested (launchDims.z is the size getSortFullDims reserves for them), and the launch is checked without a
// stream synchronization: a sort runs hundreds of steps and the caller waits for the result once.
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
      <<<launchDims.x, launchDims.y, 0, *stream>>>(
          vx,
          xShapeInfo,
          j,
          k,
          length,
          descending);

  if (!sd::DebugHelper::inGraphCapture(stream)) sd::DebugHelper::checkGlobalErrorCode("bitonicSortStepGeneric failed");
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
      <<<launchDims.x, launchDims.y, 0, *stream>>>(
          vx,
          xShapeInfo,
          vy,
          yShapeInfo,
          j,
          k,
          length,
          descending);

  if (!sd::DebugHelper::inGraphCapture(stream)) sd::DebugHelper::checkGlobalErrorCode("bitonicSortStepGenericKey failed");
}

//////////////////////////////////////////////////////////////////////////
// Value version: compares by Y values, swaps both X and Y
template <typename X, typename Y>
SD_KERNEL SD_INLINE void bitonicSortStepKernelValue(
    void* vx,
    const sd::LongType* xShapeInfo,
    void* vy,
    const sd::LongType* yShapeInfo,
    int j,
    int k,
    int length,
    bool descending) {

  auto x = static_cast<X*>(vx);
  auto y = static_cast<Y*>(vy);

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ sd::LongType yRank;
  __shared__ const sd::LongType* yShapePtr;
  __shared__ const sd::LongType* yStridePtr;

  if (threadIdx.x == 0) {
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);

    yRank      = shape::rank(yShapeInfo);
    yShapePtr  = shape::shapeOf(yShapeInfo);
    yStridePtr = shape::stride(yShapeInfo);
  }
  __syncthreads();

  const sd::LongType step = static_cast<sd::LongType>(gridDim.x) * blockDim.x;
  for (sd::LongType i = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x; i < length; i += step) {
    const sd::LongType ixj = i ^ j;
    if (ixj <= i || ixj >= length) continue;

    sd::LongType iCoordsY[SD_MAX_RANK];
    sd::LongType ixjCoordsY[SD_MAX_RANK];
    sd::LongType iOffsetY;
    sd::LongType ixjOffsetY;

    INDEX2COORDS(i, yRank, yShapePtr, iCoordsY);
    COORDS2INDEX(yRank, yStridePtr, iCoordsY, iOffsetY);

    INDEX2COORDS(ixj, yRank, yShapePtr, ixjCoordsY);
    COORDS2INDEX(yRank, yStridePtr, ixjCoordsY, ixjOffsetY);

    const bool ascending = ((i & k) == 0);
    Y yi = y[iOffsetY];
    Y yixj = y[ixjOffsetY];

    // the values decide: ascending blocks put the smaller value first (the larger when descending), descending
    // blocks the reverse
    const bool exchange = ascending ? (!descending == (yi > yixj)) : (!descending == (yi < yixj));
    if (exchange) {
      y[iOffsetY]      = yixj;
      y[ixjOffsetY]    = yi;

      sd::LongType iCoordsX[SD_MAX_RANK];
      sd::LongType ixjCoordsX[SD_MAX_RANK];
      sd::LongType iOffsetX;
      sd::LongType ixjOffsetX;

      INDEX2COORDS(i, xRank, xShapePtr, iCoordsX);
      COORDS2INDEX(xRank, xStridePtr, iCoordsX, iOffsetX);

      INDEX2COORDS(ixj, xRank, xShapePtr, ixjCoordsX);
      COORDS2INDEX(xRank, xStridePtr, ixjCoordsX, ixjOffsetX);

      X xi   = x[iOffsetX];
      X xixj = x[ixjOffsetX];
      x[iOffsetX]   = xixj;
      x[ixjOffsetX] = xi;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicSortStepGenericValue(
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

  bitonicSortStepKernelValue<X, Y>
      <<<launchDims.x, launchDims.y, 0, *stream>>>(
          vx,
          xShapeInfo,
          vy,
          yShapeInfo,
          j,
          k,
          length,
          descending);

  if (!sd::DebugHelper::inGraphCapture(stream)) sd::DebugHelper::checkGlobalErrorCode("bitonicSortStepGenericValue failed");
}

#ifdef SD_SPLIT_TYPE_INDEX
#define SD_COMMON_TYPES_FIRST SD_SPLIT_TYPE_LIST
#else
#define SD_COMMON_TYPES_FIRST SD_COMMON_TYPES
#endif
#if !defined(SD_SPLIT_TYPE_INDEX) || (COUNT_NARG(SD_COMMON_TYPES) > SD_SPLIT_TYPE_INDEX)
BUILD_SINGLE_TEMPLATE( void bitonicSortStepGeneric,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
     int j, int k, int length, bool descending),
    SD_COMMON_TYPES_FIRST);

BUILD_DOUBLE_TEMPLATE( void bitonicSortStepGenericKey,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
     void *vy, sd::LongType const *yShapeInfo, int j, int k, int length, bool descending),
    SD_COMMON_TYPES_FIRST, SD_COMMON_TYPES);

BUILD_DOUBLE_TEMPLATE( void bitonicSortStepGenericValue,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
     void *vy, sd::LongType const *yShapeInfo, int j, int k, int length, bool descending),
    SD_COMMON_TYPES_FIRST, SD_COMMON_TYPES);
#endif
#ifdef SD_COMMON_TYPES_FIRST
#undef SD_COMMON_TYPES_FIRST
#endif
