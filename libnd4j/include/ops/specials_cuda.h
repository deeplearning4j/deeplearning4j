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
//

#ifndef PROJECT_SPECIALS_CUDA_H
#define PROJECT_SPECIALS_CUDA_H

#ifdef __CUDACC__

__device__ inline int getDevicePosition(Nd4jLong *xShapeInfo, int index) {
    int xEWS = shape::elementWiseStride(xShapeInfo);

    if (xEWS == 1) {
        return index;
    } else if (xEWS > 1) {
        return index * xEWS;
    } else {
        Nd4jLong xCoord[MAX_RANK];
        int xRank = shape::rank(xShapeInfo);
        auto xShape = shape::shapeOf(xShapeInfo);
        auto xStride = shape::stride(xShapeInfo);

        shape::ind2subC(xRank, xShape, index, xCoord);
        auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);

        return xOffset;
    }
}

template<typename T>
__device__
 void bitonic_sort_step(T *x, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending) {

    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= length)
        return;

    ixj = i^j;

    /* The threads with the lowest ids sort the array. */
    if ((ixj)>i) {
        int posI = getDevicePosition(xShapeInfo, i);
        int posIXJ = getDevicePosition(xShapeInfo, ixj);

        if ((i&k)==0) {
            /* Sort ascending */
            if (!descending == (x[posI]>x[posIXJ])) {
                /* exchange(i,ixj); */
                T temp = x[posI];
                x[posI] = x[posIXJ];
                x[posIXJ] = temp;
            }
        } else if ((i&k)!=0) {
            /* Sort descending */
            if (!descending == (x[posI]<x[posIXJ])) {
                /* exchange(i,ixj); */
                T temp = x[posI];
                x[posI] = x[posIXJ];
                x[posIXJ] = temp;
            }
        }
    }
}

template<typename T>
__device__
 void bitonic_arbitrary_step(T *x, Nd4jLong *xShapeInfo, int window, int length,  int reverse, bool descending) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int half = window>>1;

    __shared__ T *shmem;
    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shrd[];
        shmem = (T *) shrd;
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
                int posIT = getDevicePosition(xShapeInfo,it);
                int posIJ = getDevicePosition(xShapeInfo, ij);

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


template<typename T>
__device__
void oes_tad(T *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    __shared__ int xLength;
    __shared__ int xTadLength;
    __shared__ int numTads;
    __shared__ T *shmem;
    __shared__ T *dx;
    if (threadIdx.x == 0) {
        xLength = shape::length(xShapeInfo);
        xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        numTads = xLength / xTadLength;

        extern __shared__ unsigned char shrd[];
        shmem = (T *) shrd;
    }
    __syncthreads();

    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
        if (threadIdx.x == 0) {
            dx = x + tadOffsets[r];
        }
        __syncthreads();

        // this is general loop, we go uncached

        int rem = xTadLength % 2;
        if (xTadLength > 1024) {

            for (int i = 0; i < (xTadLength / 2) + rem; i++) {

                // since we can have TAD larger then blockDim, we'll have this loop here
                for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x ) {
                    if((!(tid & 1)) && tid < xTadLength - 1) {
                        int t0 = getDevicePosition(tadShapeInfo, tid);
                        int t1 = getDevicePosition(tadShapeInfo, tid+1);

                        if(!descending == (dx[t0] > dx[t1])) {
                            T temp = dx[t1];
                            dx[t1] = dx[t0];
                            dx[t0] = temp;
                        }
                    }
                }
                __syncthreads();

                for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x ) {
                    if((tid & 1) && tid < xTadLength - 1) {
                        int t0 = getDevicePosition(tadShapeInfo, tid);
                        int t1 = getDevicePosition(tadShapeInfo, tid+1);

                        if(!descending == (dx[t0] > dx[t1])) {
                            T temp = dx[t1];
                            dx[t1] = dx[t0];
                            dx[t0] = temp;
                        }
                    }
                }
                __syncthreads();
            }

        } else {
            // we just load up to 1024 elements into shared memory, and sort will be applied there
            for (int e = threadIdx.x; e < xTadLength; e += blockDim.x)
                shmem[e] = dx[getDevicePosition(tadShapeInfo, e)];

            __syncthreads();


            for(int i=0; i < (xTadLength / 2) + rem; i++) {

                for (int tid = threadIdx.x; tid < xTadLength; tid += blockDim.x ) {
                    if((!(tid & 1)) && tid < xTadLength - 1) {
                        int t0 = tid;
                        int t1 = tid+1;

                        if(!descending == (shmem[t0] > shmem[t1])) {
                            T temp = shmem[t1];
                            shmem[t1] = shmem[t0];
                            shmem[t0] = temp;
                        }
                    }
                    __syncthreads();

                    if((tid & 1) && tid < xTadLength - 1) {
                        int t0 = tid;
                        int t1 = tid+1;

                        if(!descending == (shmem[t0] > shmem[t1])) {
                            T temp = shmem[t1];
                            shmem[t1] = shmem[t0];
                            shmem[t0] = temp;
                        }
                    }
                    __syncthreads();
                }
            }

            // we're dumping our shared memory back to device memory
            for (int e = threadIdx.x; e < xTadLength; e += blockDim.x)
                dx[getDevicePosition(tadShapeInfo, e)] = shmem[e];

            __syncthreads();
        }

        __syncthreads();
    }
}



extern "C" __global__ void cudaBitonicSortFloat(float *x, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending) {
    bitonic_sort_step<float>(x, xShapeInfo, j, k, length, descending);
}

extern "C" __global__ void cudaBitonicSortDouble(double *x, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending) {
    bitonic_sort_step<double>(x, xShapeInfo, j, k, length, descending);
}

extern "C" __global__ void cudaBitonicSortHalf(float16 *x, Nd4jLong *xShapeInfo, int j, int k, int length, bool descending) {
    bitonic_sort_step<float16>(x, xShapeInfo, j, k, length, descending);
}


extern "C" __global__ void cudaSortFloat(float *x, Nd4jLong *xShapeInfo, int window, int length,  int reverse, bool descending) {
    bitonic_arbitrary_step<float>(x, xShapeInfo, window, length, reverse, descending);
}

extern "C" __global__ void cudaSortDouble(double *x, Nd4jLong *xShapeInfo, int window, int length, int reverse, bool descending) {
    bitonic_arbitrary_step<double>(x, xShapeInfo, window, length, reverse, descending);
}

extern "C" __global__ void cudaSortHalf(float16 *x, Nd4jLong *xShapeInfo, int window, int length, int reverse, bool descending) {
    bitonic_arbitrary_step<float16>(x, xShapeInfo, window, length, reverse, descending);
}

extern "C" __global__ void cudaSortTadFloat(float *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    //bitonic_sort_step<float>(x, xShapeInfo, j, k, descending);
    oes_tad<float>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets,  descending);
}

extern "C" __global__ void cudaSortTadDouble(double *x,Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    //bitonic_sort_step<float>(x, xShapeInfo, j, k, descending);
    oes_tad<double>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

extern "C" __global__ void cudaSortTadHalf(float16 *x, Nd4jLong *xShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool descending) {
    //bitonic_sort_step<float>(x, xShapeInfo, j, k, descending);
    oes_tad<float16>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

#endif

#endif //PROJECT_SPECIALS_CUDA_H
