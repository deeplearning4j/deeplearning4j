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
 void bitonic_sort_step(T *x, int *xShapeInfo, int j, int k, bool descending) {
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    int posI = getDevicePosition(xShapeInfo, i);
    int posIXJ = getDevicePosition(xShapeInfo, ixj);

    if ((i&k)==0) {
      /* Sort ascending */
      if (x[posI]>x[posIXJ]) {
        /* exchange(i,ixj); */
        T temp = x[posI];
        x[posI] = x[posIXJ];
        x[posIXJ] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (x[posI]<x[posIXJ]) {
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
void odd_even_sort(T *x, int *xShapeInfo, int length, bool descending) {
    int tid = threadIdx.x;

    int xLength = length;

    int rem = xLength % 2;
    for(int i=0;i < (xLength / 2) + rem; i++) {

            if((!(tid & 1)) && tid < xLength - 1) {
                if(x[tid] > x[tid+1]) {
                    T temp = x[tid+1];
                    x[tid+1] = x[tid];
                    x[tid] = temp;
                }
            }
            __syncthreads();

            if((tid & 1) && tid < xLength - 1) {
                if(x[tid] > x[tid+1]) {
                    T temp = x[tid+1];
                    x[tid+1] = x[tid];
                    x[tid] = temp;
                }
            }
            __syncthreads();
    }
}


template<typename T>
__device__
void oes_tad(T *x, int *xShapeInfo, int *dimension, int dimensionLength, int *tadShapeInfo, int *tadOffsets, bool descending) {
    __shared__ int xLength;
    __shared__ int xTadLength;
    __shared__ int numTads;
    __shared__ T *dx;
    if (threadIdx.x == 0) {
        xLength = shape::length(xShapeInfo);
        xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        numTads = xLength / xTadLength;
    }
    __syncthreads();

    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
        if (threadIdx.x == 0) {
            dx = x + tadOffsets[r];
        }
        __syncthreads();

        odd_even_sort<T>(dx, tadShapeInfo, xTadLength, descending);
        __syncthreads();
    }
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
