//
// @author raver119@gmail.com
//

#ifndef LIBND4J_TYPE_CONVERSIONS_H
#define LIBND4J_TYPE_CONVERSIONS_H

#define ND4J_FLOAT8 0
#define ND4J_INT8 1
#define ND4J_UINT8 2
#define ND4J_FLOAT16 3
#define ND4J_INT16 4
#define ND4J_UINT16 5
#define ND4J_FLOAT32 6
#define ND4J_DOUBLE 7
#define ND4J_FLOAT24 119

#include <ops.h>
#include <types/float16.h>
#include <types/float8.h>
#include <types/uint8.h>
#include <types/int8.h>
#include "types/int16.h"
#include "types/uint16.h"


#ifdef __CUDACC__
template<typename S, typename T>
__device__ inline void convertKernelGeneric(void *dx, long N, void *dz) {
    S *x = reinterpret_cast<S *> (dx);
    T *z = reinterpret_cast<T *> (dz);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jIndex i = tid; i < N; i+= blockDim.x * gridDim.x) {
        z[i] = (T) ((float) x[i]);
    }
};
#endif

template<typename S, typename T>
void convertGeneric(void *dx,const long N, void *dz) {
    S *x = reinterpret_cast<S *> (dx);
    T *z = reinterpret_cast<T *> (dz);

    if (N < 8000) {

#pragma omp simd
        for (int i = 0; i < N; i++) {
            z[i] = (T) ((float) x[i]);
        }
    } else {

#pragma omp simd
        for (int i = 0; i < N; i++) {
            z[i] = (T) ((float) x[i]);
        }
    }
};

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z);
 */
void NativeOps::convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z) {
    void *dx = reinterpret_cast<void *> (x);
    void *dz = reinterpret_cast<void *> (z);

    if (srcType == ND4J_FLOAT8) {
        if (dstType == ND4J_FLOAT8) {
           // convertGeneric<double, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            convertGeneric<nd4j::float8, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            convertGeneric<nd4j::float8, nd4j::uint8>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            convertGeneric<nd4j::float8, float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            convertGeneric<nd4j::float8, nd4j::int16>(dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            convertGeneric<nd4j::float8, nd4j::uint16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            convertGeneric<nd4j::float8, float>(dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            convertGeneric<nd4j::float8, double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT8) {
        if (dstType == ND4J_FLOAT8) {
            convertGeneric<nd4j::int8, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            //convertGeneric<nd4j::int8, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            convertGeneric<nd4j::int8, nd4j::uint8>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            convertGeneric<nd4j::int8, float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            convertGeneric<nd4j::int8, nd4j::int16>(dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            convertGeneric<nd4j::int8, nd4j::uint16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            convertGeneric<nd4j::int8, float>(dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            convertGeneric<nd4j::int8, double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_UINT8) {
        if (dstType == ND4J_FLOAT8) {
            convertGeneric<nd4j::uint8, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            convertGeneric<nd4j::uint8, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            convertGeneric<nd4j::uint8, nd4j::uint8>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            convertGeneric<nd4j::uint8, float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            convertGeneric<nd4j::uint8, nd4j::int16>(dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            convertGeneric<nd4j::uint8, nd4j::uint16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            convertGeneric<nd4j::uint8, float>(dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            convertGeneric<nd4j::uint8, double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT16) {
        if (dstType == ND4J_FLOAT8) {
            convertGeneric<float16, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            convertGeneric<float16, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            convertGeneric<float16, nd4j::uint8>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            convertGeneric<float16, float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            convertGeneric<float16, nd4j::int16>(dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            convertGeneric<float16, nd4j::uint16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            convertGeneric<float16, float>(dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            convertGeneric<float16, double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_INT16) {
        if (dstType == ND4J_FLOAT8) {
            convertGeneric<nd4j::int16, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            convertGeneric<nd4j::int16, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            convertGeneric<nd4j::int16, nd4j::uint8>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            convertGeneric<nd4j::int16, float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            convertGeneric<nd4j::int16, nd4j::int16>(dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            convertGeneric<nd4j::int16, nd4j::uint16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            convertGeneric<nd4j::int16, float>(dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            convertGeneric<nd4j::int16, double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_FLOAT24) {

    } else if (srcType == ND4J_FLOAT32) {
        if (dstType == ND4J_FLOAT8) {
            convertGeneric<float, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            convertGeneric<float, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            convertGeneric<float, nd4j::uint8>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            convertGeneric<float, float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            convertGeneric<float, nd4j::int16>(dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            convertGeneric<float, nd4j::uint16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_DOUBLE) {
            convertGeneric<float, double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_DOUBLE) {
        if (dstType == ND4J_FLOAT8) {
            convertGeneric<double, nd4j::float8>(dx, N, dz);
        } else if (dstType == ND4J_INT8) {
            convertGeneric<double, nd4j::int8>(dx, N, dz);
        } else if (dstType == ND4J_UINT8) {
            convertGeneric<double, nd4j::uint8>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT16) {
            convertGeneric<double, float16>(dx, N, dz);
        } else if (dstType == ND4J_INT16) {
            convertGeneric<double, nd4j::int16>(dx, N, dz);
        } else if (dstType == ND4J_UINT16) {
            convertGeneric<double, nd4j::uint16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT24) {

        } else if (dstType == ND4J_FLOAT32) {
            convertGeneric<double, float>(dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            //
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else {
        printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
}


#endif //LIBND4J_TYPE_CONVERSIONS_H
