//
// @author raver119@gmail.com
//

#ifndef PROJECT_SPECIALS_CUDA_H
#define PROJECT_SPECIALS_CUDA_H

#ifdef __CUDACC__

__device__ inline int getDevicePosition(int *xShapeInfo, int index) {
    int xEWS = shape::elementWiseStride(xShapeInfo);

    if (xEWS == 1) {
        return index;
    } else if (xEWS > 1) {
        return index * xEWS;
    } else {
        int xCoord[MAX_RANK];
        int xRank = shape::rank(xShapeInfo);
        int *xShape = shape::shapeOf(xShapeInfo);
        int *xStride = shape::stride(xShapeInfo);

        shape::ind2subC(xRank, xShape, index, xCoord);
        int xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);

        return xOffset;
    }
}

template<typename T>
__device__
 void bitonic_sort_step(T *x, int *xShapeInfo, int j, int k, int length, bool descending) {

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
void oes_tad(T *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, bool descending) {
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



extern "C" __global__ void cudaBitonicSortFloat(float *x, int *xShapeInfo, int j, int k, int length, bool descending) {
    bitonic_sort_step<float>(x, xShapeInfo, j, k, length, descending);
}

extern "C" __global__ void cudaBitonicSortDouble(double *x, int *xShapeInfo, int j, int k, int length, bool descending) {
    bitonic_sort_step<double>(x, xShapeInfo, j, k, length, descending);
}

extern "C" __global__ void cudaBitonicSortHalf(float16 *x, int *xShapeInfo, int j, int k, int length, bool descending) {
    bitonic_sort_step<float16>(x, xShapeInfo, j, k, length, descending);
}


extern "C" __global__ void cudaSortFloat(float *x, int *xShapeInfo, int j, int k, bool descending) {
    //bitonic_sort_step<float>(x, xShapeInfo, j, k, descending);
    //odd_even_sort<float>(x, xShapeInfo, descending);
}

extern "C" __global__ void cudaSortDouble(double *x, int *xShapeInfo, int j, int k, bool descending) {
    //bitonic_sort_step<double>(x, xShapeInfo, j, k, descending);
}

extern "C" __global__ void cudaSortHalf(float16 *x, int *xShapeInfo, int j, int k, bool descending) {
    //bitonic_sort_step<float16>(x, xShapeInfo, j, k, descending);
}

extern "C" __global__ void cudaSortTadFloat(float *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, bool descending) {
    //bitonic_sort_step<float>(x, xShapeInfo, j, k, descending);
    oes_tad<float>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets,  descending);
}

extern "C" __global__ void cudaSortTadDouble(double *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, bool descending) {
    //bitonic_sort_step<float>(x, xShapeInfo, j, k, descending);
    oes_tad<double>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

extern "C" __global__ void cudaSortTadHalf(float16 *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, bool descending) {
    //bitonic_sort_step<float>(x, xShapeInfo, j, k, descending);
    oes_tad<float16>(x, xShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets, descending);
}

#endif

#endif //PROJECT_SPECIALS_CUDA_H
