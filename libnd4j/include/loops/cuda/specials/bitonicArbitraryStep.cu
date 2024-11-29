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
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 28.11.2018
//
#include <ops/specials_cuda.h>


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_KERNEL void bitonicArbitraryStepKernelKey(void *vx, sd::LongType const *xShapeInfo, void *vy,
                                             sd::LongType const *yShapeInfo, int window, int length, int reverse,
                                             bool descending) {
  auto x = static_cast<X *>(vx);
  auto y = static_cast<Y *>(vy);

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int half = window >> 1;

  __shared__ sd::LongType xLength;
  if (threadIdx.x == 0) {
    xLength = shape::length(xShapeInfo);
  }
  __syncthreads();

  int firstPosition;
  int firstStep;
  int secondPosition;
  int secondStep;

  int WARP_SIZE = 32;
  int numWarps = (gridDim.x * blockDim.x) / 32;
  int warpId = tid / WARP_SIZE;
  int warpIdx = tid % WARP_SIZE;

  if (half >= 128) {
    firstPosition = blockIdx.x * window;
    firstStep = gridDim.x * window;

    secondPosition = threadIdx.x;
    secondStep = blockDim.x;
  } else if (half >= 32) {
    firstPosition = warpId * window;
    firstStep = numWarps * window;

    secondPosition = warpIdx;
    secondStep = WARP_SIZE;
  } else {
    firstPosition = tid * window;
    firstStep = blockDim.x * gridDim.x * window;

    secondPosition = 0;
    secondStep = 1;
  }

  for (int i = firstPosition; i < length; i += firstStep) {
    for (int j = secondPosition; j < half; j += secondStep) {
      int it = (reverse) ? i + j + half : i + window - j - 1;
      int ij = i + j;
      if (it < length && ij < length) {
        sd::LongType itCoords[SD_MAX_RANK];
        sd::LongType ijCoords[SD_MAX_RANK];
        sd::LongType itOffset;
        sd::LongType ijOffset;

        INDEX2COORDS(it, shape::rank(xShapeInfo), xShapeInfo, itCoords);
        COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), itCoords, itOffset);
        INDEX2COORDS(ij, shape::rank(xShapeInfo), xShapeInfo, ijCoords);
        COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), ijCoords, ijOffset);

        X v0 = x[ijOffset];
        X v1 = x[itOffset];

        if (!descending == (v0 > v1)) {
          x[ijOffset] = v1;
          x[itOffset] = v0;

          Y ytemp = y[ijOffset];
          y[ijOffset] = y[itOffset];
          y[itOffset] = ytemp;
        }
      }
    }
  }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void execBitonicArbitraryStepKernel(void *vx, sd::LongType const *xShapeInfo, int window, int length,
                                              int reverse, bool descending) {
  auto x = static_cast<T *>(vx);

  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int half = window >> 1;

  __shared__ T *shmem;
  __shared__ sd::LongType xLength;
  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shrd[];
    shmem = (T *)shrd;
    xLength = shape::length(xShapeInfo);
  }
  __syncthreads();

  int firstPosition;
  int firstStep;
  int secondPosition;
  int secondStep;

  int WARP_SIZE = 32;
  int numWarps = (gridDim.x * blockDim.x) / 32;
  int warpId = tid / WARP_SIZE;
  int warpIdx = tid % WARP_SIZE;

  if (half >= 128) {
    firstPosition = blockIdx.x * window;
    firstStep = gridDim.x * window;

    secondPosition = threadIdx.x;
    secondStep = blockDim.x;
  } else if (half >= 32) {
    firstPosition = warpId * window;
    firstStep = numWarps * window;

    secondPosition = warpIdx;
    secondStep = WARP_SIZE;
  } else {
    firstPosition = tid * window;
    firstStep = blockDim.x * gridDim.x * window;

    secondPosition = 0;
    secondStep = 1;
  }

  for (int i = firstPosition; i < length; i += firstStep) {
    for (int j = secondPosition; j < half; j += secondStep) {
      int it = (reverse) ? i + j + half : i + window - j - 1;
      int ij = i + j;
      if (it < length && ij < length) {
        sd::LongType itCoords[SD_MAX_RANK];
        sd::LongType ijCoords[SD_MAX_RANK];
        sd::LongType itOffset;
        sd::LongType ijOffset;

        INDEX2COORDS(it, shape::rank(xShapeInfo), xShapeInfo, itCoords);
        COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), itCoords, itOffset);
        INDEX2COORDS(ij, shape::rank(xShapeInfo), xShapeInfo, ijCoords);
        COORDS2INDEX(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), ijCoords, ijOffset);

        shmem[threadIdx.x] = x[ijOffset];
        shmem[threadIdx.x + blockDim.x] = x[itOffset];

        if (!descending == (shmem[threadIdx.x] > shmem[threadIdx.x + blockDim.x])) {
          x[ijOffset] = shmem[threadIdx.x + blockDim.x];
          x[itOffset] = shmem[threadIdx.x];
        }
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void bitonicArbitraryStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx,
                                         sd::LongType const *xShapeInfo, int window, int length, int reverse,
                                         bool descending) {
  execBitonicArbitraryStepKernel<T>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, window, length, reverse, descending);
  sd::DebugHelper::checkErrorCode(stream, "execBitonicArbitraryStepKernel  failed");

}

template <typename X, typename Y>
SD_HOST void bitonicArbitraryStepGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx,
                                            sd::LongType const *xShapeInfo, void *vy, sd::LongType const *yShapeInfo,
                                            int window, int length, int reverse, bool descending) {
  bitonicArbitraryStepKernelKey<X, Y><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      vx, xShapeInfo, vy, yShapeInfo, window, length, reverse, descending);
  sd::DebugHelper::checkErrorCode(stream, "bitonicArbitraryStepKernelKey  failed");

}

BUILD_SINGLE_TEMPLATE(template void bitonicArbitraryStepGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, int window,
                       int length, int reverse, bool descending),
                      SD_COMMON_TYPES);
BUILD_DOUBLE_TEMPLATE(template void bitonicArbitraryStepGenericKey,
                      (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, void *vy,
                       sd::LongType const *yShapeInfo, int window, int length, int reverse, bool descending),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);
