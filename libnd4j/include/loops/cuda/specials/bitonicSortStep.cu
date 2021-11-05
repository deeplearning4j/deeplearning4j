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
SD_KERNEL void bitonicSortStepKernelKey(void *vx, sd::LongType const *xShapeInfo, void *vy,
                                        sd::LongType const *yShapeInfo, int j, int k, int length, bool descending) {
  auto x = static_cast<X *>(vx);
  auto y = static_cast<Y *>(vy);

  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ sd::LongType xLength;
  if (threadIdx.x == 0) xLength = shape::length(xShapeInfo);

  __syncthreads();

  if (i >= length) return;

  ixj = i ^ j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj) > i) {
    int posI = shape::getIndexOffset(i, xShapeInfo);
    int posIXJ = shape::getIndexOffset(ixj, xShapeInfo);

    if ((i & k) == 0) {
      /* Sort ascending */
      if (!descending == (x[posI] > x[posIXJ])) {
        /* exchange(i,ixj); */
        X temp = x[posI];
        x[posI] = x[posIXJ];
        x[posIXJ] = temp;

        Y ytemp = y[posI];
        y[posI] = y[posIXJ];
        y[posIXJ] = ytemp;
      }
    } else if ((i & k) != 0) {
      /* Sort descending */
      if (!descending == (x[posI] < x[posIXJ])) {
        /* exchange(i,ixj); */
        X temp = x[posI];
        x[posI] = x[posIXJ];
        x[posIXJ] = temp;

        Y ytemp = y[posI];
        y[posI] = y[posIXJ];
        y[posIXJ] = ytemp;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void bitonicSortStepKernel(void *vx, sd::LongType const *xShapeInfo, int j, int k, int length,
                                     bool descending) {
  auto x = static_cast<T *>(vx);

  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ sd::LongType xLength;
  if (threadIdx.x == 0) xLength = shape::length(xShapeInfo);

  __syncthreads();

  if (i >= length) return;

  ixj = i ^ j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj) > i) {
    int posI = shape::getIndexOffset(i, xShapeInfo);
    int posIXJ = shape::getIndexOffset(ixj, xShapeInfo);

    if ((i & k) == 0) {
      /* Sort ascending */
      if (!descending == (x[posI] > x[posIXJ])) {
        /* exchange(i,ixj); */
        T temp = x[posI];
        x[posI] = x[posIXJ];
        x[posIXJ] = temp;
      }
    } else if ((i & k) != 0) {
      /* Sort descending */
      if (!descending == (x[posI] < x[posIXJ])) {
        /* exchange(i,ixj); */
        T temp = x[posI];
        x[posI] = x[posIXJ];
        x[posIXJ] = temp;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void bitonicSortStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                                    int j, int k, int length, bool descending) {
  bitonicSortStepKernel<T>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, j, k, length, descending);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
SD_HOST void bitonicSortStepGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo,
                                       void *vy, sd::LongType const *yShapeInfo, int j, int k, int length,
                                       bool descending) {
  bitonicSortStepKernelKey<X, Y>
      <<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, j, k, length, descending);
}

BUILD_SINGLE_TEMPLATE(template void bitonicSortStepGeneric,
                      (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, int j, int k,
                       int length, bool descending),
                      SD_COMMON_TYPES);
BUILD_DOUBLE_TEMPLATE(template void bitonicSortStepGenericKey,
                      (dim3 & launchDims, cudaStream_t *stream, void *vx, sd::LongType const *xShapeInfo, void *vy,
                       sd::LongType const *yShapeInfo, int j, int k, int length, bool descending),
                      SD_COMMON_TYPES, SD_COMMON_TYPES);
