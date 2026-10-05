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

// One step of the bitonic merge network for a length that need not be a power of two. The length logical elements
// are cut into windows of `window` (a power of two, at least 2); element j of a window's first half is compared with
// the mirrored element window - 1 - j of the window (reverse == 0: the first step of a merge) or with element
// j + half (reverse != 0). A partner past the end of the array is left alone, which is what a pad that sorts after
// every element would do. The pairs of one step are disjoint, so the slots (window, j) are walked in one grid-stride
// loop with 64-bit indices: any launch covers every pair however long the array is.
// Every element is addressed through the array's own shape and strides.

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL SD_INLINE void bitonicArbitraryStepKernelKey(
    void* vx,
    const sd::LongType* xShapeInfo,
    void* vy,
    const sd::LongType* yShapeInfo,
    int window,
    int length,
    int reverse,
    bool descending) {

  auto x         = static_cast<X*>(vx);
  auto y         = static_cast<Y*>(vy);

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

  const sd::LongType half    = window >> 1;
  if (half < 1) return;  // a window holds a pair at least
  const sd::LongType windows = (static_cast<sd::LongType>(length) + window - 1) / window;
  const sd::LongType slots   = windows * half;
  const sd::LongType step    = static_cast<sd::LongType>(gridDim.x) * blockDim.x;

  for (sd::LongType slot = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x; slot < slots;
       slot += step) {
    const sd::LongType first = (slot / half) * window;
    const sd::LongType j     = slot % half;
    const sd::LongType ij    = first + j;
    const sd::LongType it    = (reverse) ? ij + half : first + window - 1 - j;
    if (it >= length || ij >= length) continue;

    sd::LongType itCoords[SD_MAX_RANK];
    sd::LongType ijCoords[SD_MAX_RANK];
    sd::LongType itOffset;
    sd::LongType ijOffset;

    INDEX2COORDS(it, xRank, xShapePtr, itCoords);
    COORDS2INDEX(xRank, xStridePtr, itCoords, itOffset);

    INDEX2COORDS(ij, xRank, xShapePtr, ijCoords);
    COORDS2INDEX(xRank, xStridePtr, ijCoords, ijOffset);

    X v0 = x[ijOffset];
    X v1 = x[itOffset];

    const bool condition = (!descending == (v0 > v1));
    if (condition) {
      x[ijOffset] = v1;
      x[itOffset] = v0;

      sd::LongType itCoordsY[SD_MAX_RANK];
      sd::LongType ijCoordsY[SD_MAX_RANK];
      sd::LongType itOffsetY;
      sd::LongType ijOffsetY;

      INDEX2COORDS(it, yRank, yShapePtr, itCoordsY);
      COORDS2INDEX(yRank, yStridePtr, itCoordsY, itOffsetY);

      INDEX2COORDS(ij, yRank, yShapePtr, ijCoordsY);
      COORDS2INDEX(yRank, yStridePtr, ijCoordsY, ijOffsetY);

      Y ytemp        = y[ijOffsetY];
      y[ijOffsetY]   = y[itOffsetY];
      y[itOffsetY]   = ytemp;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL SD_INLINE void execBitonicArbitraryStepKernel(
    void* vx,
    const sd::LongType* xShapeInfo,
    int window,
    int length,
    int reverse,
    bool descending) {

  auto x         = static_cast<T*>(vx);

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  if (threadIdx.x == 0) {
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);
  }
  __syncthreads();

  const sd::LongType half    = window >> 1;
  if (half < 1) return;  // a window holds a pair at least
  const sd::LongType windows = (static_cast<sd::LongType>(length) + window - 1) / window;
  const sd::LongType slots   = windows * half;
  const sd::LongType step    = static_cast<sd::LongType>(gridDim.x) * blockDim.x;

  for (sd::LongType slot = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x; slot < slots;
       slot += step) {
    const sd::LongType first = (slot / half) * window;
    const sd::LongType j     = slot % half;
    const sd::LongType ij    = first + j;
    const sd::LongType it    = (reverse) ? ij + half : first + window - 1 - j;
    if (it >= length || ij >= length) continue;

    sd::LongType itCoords[SD_MAX_RANK];
    sd::LongType ijCoords[SD_MAX_RANK];
    sd::LongType itOffset;
    sd::LongType ijOffset;

    INDEX2COORDS(it, xRank, xShapePtr, itCoords);
    COORDS2INDEX(xRank, xStridePtr, itCoords, itOffset);

    INDEX2COORDS(ij, xRank, xShapePtr, ijCoords);
    COORDS2INDEX(xRank, xStridePtr, ijCoords, ijOffset);

    T v0 = x[ijOffset];
    T v1 = x[itOffset];

    const bool condition = (!descending == (v0 > v1));
    if (condition) {
      x[ijOffset] = v1;
      x[itOffset] = v0;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// The launchers enqueue one step on the caller's stream. The step kernels use no dynamic shared memory, so none is
// requested (launchDims.z is the size getSortFullDims reserves for them), and the launch is checked without a
// stream synchronization: a sort runs hundreds of steps and the caller waits for the result once.
template <typename T>
SD_HOST void bitonicArbitraryStepGeneric(
    dim3 &launchDims,
    cudaStream_t *stream,
    void* vx,
    const sd::LongType* xShapeInfo,
    int window,
    int length,
    int reverse,
    bool descending) {

  execBitonicArbitraryStepKernel<T>
      <<<launchDims.x, launchDims.y, 0, *stream>>>(
          vx,
          xShapeInfo,
          window,
          length,
          reverse,
          descending);

  if (!sd::DebugHelper::inGraphCapture(stream)) sd::DebugHelper::checkGlobalErrorCode("execBitonicArbitraryStepKernel  failed");
}

template <typename X, typename Y>
SD_HOST void bitonicArbitraryStepGenericKey(
    dim3 &launchDims,
    cudaStream_t *stream,
    void* vx,
    const sd::LongType* xShapeInfo,
    void* vy,
    const sd::LongType* yShapeInfo,
    int window,
    int length,
    int reverse,
    bool descending) {

  bitonicArbitraryStepKernelKey<X, Y>
      <<<launchDims.x, launchDims.y, 0, *stream>>>(
          vx,
          xShapeInfo,
          vy,
          yShapeInfo,
          window,
          length,
          reverse,
          descending);

  if (!sd::DebugHelper::inGraphCapture(stream)) sd::DebugHelper::checkGlobalErrorCode("bitonicArbitraryStepKernelKey failed");
}

//////////////////////////////////////////////////////////////////////////
// Value version: compares by Y values, swaps both X and Y
template <typename X, typename Y>
SD_KERNEL SD_INLINE void bitonicArbitraryStepKernelValue(
    void* vx,
    const sd::LongType* xShapeInfo,
    void* vy,
    const sd::LongType* yShapeInfo,
    int window,
    int length,
    int reverse,
    bool descending) {

  auto x         = static_cast<X*>(vx);
  auto y         = static_cast<Y*>(vy);

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

  const sd::LongType half    = window >> 1;
  if (half < 1) return;  // a window holds a pair at least
  const sd::LongType windows = (static_cast<sd::LongType>(length) + window - 1) / window;
  const sd::LongType slots   = windows * half;
  const sd::LongType step    = static_cast<sd::LongType>(gridDim.x) * blockDim.x;

  for (sd::LongType slot = static_cast<sd::LongType>(blockIdx.x) * blockDim.x + threadIdx.x; slot < slots;
       slot += step) {
    const sd::LongType first = (slot / half) * window;
    const sd::LongType j     = slot % half;
    const sd::LongType ij    = first + j;
    const sd::LongType it    = (reverse) ? ij + half : first + window - 1 - j;
    if (it >= length || ij >= length) continue;

    sd::LongType itCoordsY[SD_MAX_RANK];
    sd::LongType ijCoordsY[SD_MAX_RANK];
    sd::LongType itOffsetY;
    sd::LongType ijOffsetY;

    INDEX2COORDS(it, yRank, yShapePtr, itCoordsY);
    COORDS2INDEX(yRank, yStridePtr, itCoordsY, itOffsetY);

    INDEX2COORDS(ij, yRank, yShapePtr, ijCoordsY);
    COORDS2INDEX(yRank, yStridePtr, ijCoordsY, ijOffsetY);

    Y v0 = y[ijOffsetY];
    Y v1 = y[itOffsetY];

    const bool condition = (!descending == (v0 > v1));
    if (condition) {
      y[ijOffsetY] = v1;
      y[itOffsetY] = v0;

      sd::LongType itCoordsX[SD_MAX_RANK];
      sd::LongType ijCoordsX[SD_MAX_RANK];
      sd::LongType itOffsetX;
      sd::LongType ijOffsetX;

      INDEX2COORDS(it, xRank, xShapePtr, itCoordsX);
      COORDS2INDEX(xRank, xStridePtr, itCoordsX, itOffsetX);

      INDEX2COORDS(ij, xRank, xShapePtr, ijCoordsX);
      COORDS2INDEX(xRank, xStridePtr, ijCoordsX, ijOffsetX);

      X xtemp        = x[ijOffsetX];
      x[ijOffsetX]   = x[itOffsetX];
      x[itOffsetX]   = xtemp;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicArbitraryStepGenericValue(
    dim3 &launchDims,
    cudaStream_t *stream,
    void* vx,
    const sd::LongType* xShapeInfo,
    void* vy,
    const sd::LongType* yShapeInfo,
    int window,
    int length,
    int reverse,
    bool descending) {

  bitonicArbitraryStepKernelValue<X, Y>
      <<<launchDims.x, launchDims.y, 0, *stream>>>(
          vx,
          xShapeInfo,
          vy,
          yShapeInfo,
          window,
          length,
          reverse,
          descending);

  if (!sd::DebugHelper::inGraphCapture(stream)) sd::DebugHelper::checkGlobalErrorCode("bitonicArbitraryStepKernelValue failed");
}

#ifdef SD_SPLIT_TYPE_INDEX
#define SD_COMMON_TYPES_FIRST SD_SPLIT_TYPE_LIST
#else
#define SD_COMMON_TYPES_FIRST SD_COMMON_TYPES
#endif
#if !defined(SD_SPLIT_TYPE_INDEX) || (COUNT_NARG(SD_COMMON_TYPES) > SD_SPLIT_TYPE_INDEX)
BUILD_SINGLE_TEMPLATE(
     void bitonicArbitraryStepGeneric,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, int window,
     int length, int reverse, bool descending),
    SD_COMMON_TYPES_FIRST);

BUILD_DOUBLE_TEMPLATE(
     void bitonicArbitraryStepGenericKey,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, void *vy,
     sd::LongType const *yShapeInfo, int window, int length, int reverse, bool descending),
    SD_COMMON_TYPES_FIRST, SD_COMMON_TYPES);

BUILD_DOUBLE_TEMPLATE(
     void bitonicArbitraryStepGenericValue,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, void *vy,
     sd::LongType const *yShapeInfo, int window, int length, int reverse, bool descending),
    SD_COMMON_TYPES_FIRST, SD_COMMON_TYPES);
#endif
#ifdef SD_COMMON_TYPES_FIRST
#undef SD_COMMON_TYPES_FIRST
#endif
