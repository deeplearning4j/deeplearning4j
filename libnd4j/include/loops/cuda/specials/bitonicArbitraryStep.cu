/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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
__global__ void bitonicArbitraryStepKernelKey(void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, int window, int length,  int reverse, bool descending) {
    auto x = static_cast<X*>(vx);
    auto y = static_cast<Y*>(vy);

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int half = window>>1;

    __shared__ Nd4jLong xLength;
    if (threadIdx.x == 0) {
        xLength = shape::length(xShapeInfo);
    }
    __syncthreads();

    //for (int i = 0; i < length; i+= window)
    /*
        if window == 4;
        iterations will be: 0; 4; 8; 12; 16; 20
        if gridDim = 3;
        on first iteration we'll have: 0; 4; 8;
        on second iteration we'll have: 0 + (3 * 4) = 12;  4 + (3 * 4) = 16; 8 + (3 * 4) = 20
    */
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
            int ij = i+j;
            if (it < length && ij < length ) {
                int posIT = shape::getIndexOffset(it, xShapeInfo);
                int posIJ = shape::getIndexOffset(ij, xShapeInfo);

                X v0 = x[posIJ];
                X v1 = x[posIT];

                if(!descending == (v0 > v1)) {
                    x[posIJ] = v1;
                    x[posIT] = v0;

                    Y ytemp = y[posIJ];
                    y[posIJ] = y[posIT];
                    y[posIT] = ytemp;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__global__ void execBitonicArbitraryStepKernel(void *vx, Nd4jLong *xShapeInfo, int window, int length,  int reverse, bool descending) {
    auto x = static_cast<T*>(vx);

    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int half = window>>1;

    __shared__ T *shmem;
    __shared__ Nd4jLong xLength;
    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shrd[];
        shmem = (T *) shrd;
        xLength = shape::length(xShapeInfo);
    }
    __syncthreads();

    //for (int i = 0; i < length; i+= window)
    /*
        if window == 4;
        iterations will be: 0; 4; 8; 12; 16; 20
        if gridDim = 3;
        on first iteration we'll have: 0; 4; 8;
        on second iteration we'll have: 0 + (3 * 4) = 12;  4 + (3 * 4) = 16; 8 + (3 * 4) = 20
    */
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
            int ij = i+j;
            if (it < length && ij < length ) {
                int posIT = shape::getIndexOffset(it, xShapeInfo);
                int posIJ = shape::getIndexOffset(ij, xShapeInfo);

                shmem[threadIdx.x] = x[posIJ];
                shmem[threadIdx.x + blockDim.x] = x[posIT];

                if(!descending == (shmem[threadIdx.x] > shmem[threadIdx.x + blockDim.x])) {
                    x[posIJ] = shmem[threadIdx.x + blockDim.x];
                    x[posIT] = shmem[threadIdx.x];
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
__host__ void bitonicArbitraryStepGeneric(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, int window, int length,  int reverse, bool descending) {
    execBitonicArbitraryStepKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, window, length, reverse, descending);
}

template <typename X, typename Y>
__host__ void bitonicArbitraryStepGenericKey(dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, int window, int length,  int reverse, bool descending) {
    bitonicArbitraryStepKernelKey<X,Y><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, xShapeInfo, vy, yShapeInfo, window, length, reverse, descending);
}

BUILD_SINGLE_TEMPLATE(template void ND4J_EXPORT bitonicArbitraryStepGeneric, (dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, int window, int length,  int reverse, bool descending), LIBND4J_TYPES);
BUILD_DOUBLE_TEMPLATE(template void ND4J_EXPORT bitonicArbitraryStepGenericKey, (dim3 &launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, void *vy, Nd4jLong *yShapeInfo, int window, int length,  int reverse, bool descending), LIBND4J_TYPES, LIBND4J_TYPES);
