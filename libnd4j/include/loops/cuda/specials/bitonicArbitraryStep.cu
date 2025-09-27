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
    SD_KERNEL void bitonicArbitraryStepKernelKey(
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
  const int tid  = threadIdx.x + blockDim.x * blockIdx.x;
  const int half = window >> 1;

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ sd::LongType yRank;    // Potentially unused for direct indexing, but let's keep the pattern consistent
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

  const int WARP_SIZE = 32;
  const int numWarps  = (gridDim.x * blockDim.x) / WARP_SIZE;
  const int warpId    = tid / WARP_SIZE;
  const int warpIdx   = tid % WARP_SIZE;

  int firstPosition;
  int firstStep;
  int secondPosition;
  int secondStep;

  if (half >= 128) {
    firstPosition = blockIdx.x * window;
    firstStep     = gridDim.x * window;

    secondPosition = threadIdx.x;
    secondStep     = blockDim.x;
  }
  else if (half >= 32) {
    firstPosition = warpId * window;
    firstStep     = numWarps * window;

    secondPosition = warpIdx;
    secondStep     = WARP_SIZE;
  }
  else {
    firstPosition = tid * window;
    firstStep     = blockDim.x * gridDim.x * window;

    secondPosition = 0;
    secondStep     = 1;
  }

  for (int i = firstPosition; i < length; i += firstStep) {
    for (int j = secondPosition; j < half; j += secondStep) {
      const int it = (reverse) ? i + j + half : i + window - j - 1;
      const int ij = i + j;
      if (it < length && ij < length) {
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
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void execBitonicArbitraryStepKernel(
    void* vx,
    const sd::LongType* xShapeInfo,
    int window,
    int length,
    int reverse,
    bool descending) {

  auto x         = static_cast<T*>(vx);
  const int tid  = threadIdx.x + blockDim.x * blockIdx.x;
  const int half = window >> 1;

  __shared__ sd::LongType xRank;
  __shared__ const sd::LongType* xShapePtr;
  __shared__ const sd::LongType* xStridePtr;

  __shared__ sd::LongType xLength;

  // We'll omit using shared memory for x data except for small merges,
  // but keep the pattern of caching shape info
  if (threadIdx.x == 0) {
    xRank      = shape::rank(xShapeInfo);
    xShapePtr  = shape::shapeOf(xShapeInfo);
    xStridePtr = shape::stride(xShapeInfo);

    xLength    = shape::length(xShapeInfo);
  }
  __syncthreads();

  const int WARP_SIZE = 32;
  const int numWarps  = (gridDim.x * blockDim.x) / WARP_SIZE;
  const int warpId    = tid / WARP_SIZE;
  const int warpIdx   = tid % WARP_SIZE;

  int firstPosition;
  int firstStep;
  int secondPosition;
  int secondStep;

  if (half >= 128) {
    firstPosition  = blockIdx.x * window;
    firstStep      = gridDim.x * window;

    secondPosition = threadIdx.x;
    secondStep     = blockDim.x;
  }
  else if (half >= 32) {
    firstPosition  = warpId * window;
    firstStep      = numWarps * window;

    secondPosition = warpIdx;
    secondStep     = WARP_SIZE;
  }
  else {
    firstPosition  = tid * window;
    firstStep      = blockDim.x * gridDim.x * window;

    secondPosition = 0;
    secondStep     = 1;
  }

  for (int i = firstPosition; i < length; i += firstStep) {
    for (int j = secondPosition; j < half; j += secondStep) {
      const int it = (reverse) ? i + j + half : i + window - j - 1;
      const int ij = i + j;
      if (it < length && ij < length) {
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
  }
}

//////////////////////////////////////////////////////////////////////////
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
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          vx,
          xShapeInfo,
          window,
          length,
          reverse,
          descending);

  sd::DebugHelper::checkErrorCode(stream, "execBitonicArbitraryStepKernel  failed");
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
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
          vx,
          xShapeInfo,
          vy,
          yShapeInfo,
          window,
          length,
          reverse,
          descending);

  sd::DebugHelper::checkErrorCode(stream, "bitonicArbitraryStepKernelKey failed");
}

BUILD_SINGLE_TEMPLATE(
     void bitonicArbitraryStepGeneric,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, int window,
     int length, int reverse, bool descending),
    SD_COMMON_TYPES);

BUILD_DOUBLE_TEMPLATE(
     void bitonicArbitraryStepGenericKey,
    (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, void *vy,
     sd::LongType const *yShapeInfo, int window, int length, int reverse, bool descending),
    SD_COMMON_TYPES, SD_COMMON_TYPES);
