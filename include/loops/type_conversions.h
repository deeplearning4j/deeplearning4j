/*
 * This set of methods provides dataType conversions in all possible directions supported:
 *  FP8, FP16, FLOAT, DOUBLE, INT8, UINT8, UINT16,
 *
 * @author raver119@gmail.com
 */

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
#define ND4J_THRESHOLD 8
#define ND4J_FLOAT24 119 // not supported after all. might want to add support later.

#include <ops/ops.h>
#include <atomic>
#include <types/float16.h>
#include <types/float8.h>
#include <types/uint8.h>
#include <types/int8.h>
#include <types/int16.h>
#include <types/uint16.h>

typedef union
{
    float f_;
    int   i_;
} FloatBits;


#ifdef __CUDACC__
template<typename S, typename T>
__device__ inline void convertKernelGeneric(void *dx, Nd4jIndex N, void *dz) {
    S *x = reinterpret_cast<S *> (dx);
    T *z = reinterpret_cast<T *> (dz);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jIndex i = tid; i < N; i+= blockDim.x * gridDim.x) {
        z[i] = (T) ((float) x[i]);
    }
};
#endif

template<typename S, typename T>
void convertGeneric(void *dx, Nd4jIndex N, void *dz) {
    S *x = reinterpret_cast<S *> (dx);
    T *z = reinterpret_cast<T *> (dz);

    if (N < 8000) {

#pragma omp simd
        for (int i = 0; i < N; i++) {
            z[i] = (T) ((float) x[i]);
        }
    } else {

#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            z[i] = (T) ((float) x[i]);
        }
    }
};


template <typename T>
void convertToThreshold(void *dx, Nd4jIndex N, void *dz) {
    // we suppose that first 4 bytes are integer, second 4 bytes are float
    // integer: enc length
    // integer: dec length
    // float: threshold
    FloatBits fb;
    T *x = (T *) dx;
    int *z = (int *) dz;
    int limit = z[0];
    fb.i_ = z[2];
    float threshold = fb.f_;

    // FIXME: int limit is sad thing here, 2B elements limitation
    z[1] = (int) N;

    // we use 3 as offset, since first 12 bytes are occupied with header
    int flimit = limit + 3;
    volatile std::atomic<int> cnt;
    cnt.store(3);
    volatile  std::atomic<bool> flag;
    flag.store(false);
#pragma omp parallel for schedule(guided) default(shared)
    for (int e = 0; e < N;  e++) {
        if (flag.load())
            continue;

        T cUpd = x[e];
        if (cUpd >= (T) threshold) {
            int idx = cnt++;

            if (idx >= flimit) {
                flag.store(true);
                continue;
            }

            z[idx] = e + 1;
            x[e] -= (T) threshold;
        } else if (cUpd <= (T) -threshold) {
            int idx = cnt++;

            if (idx >= flimit) {
                flag.store(true);
                continue;
            }

            z[idx] = -e - 1;
            x[e] += (T) threshold;
        }
    }
}

template <typename T>
void convertFromThreshold(void *dx, Nd4jIndex N, void *dz) {
    FloatBits fb;
    T *z = (T *) dz;
    int *x = (int *) dx;
    int limit = x[0];
    int size = x[1];
    fb.i_ = x[2];
    float threshold = fb.f_;

    // everything is set to 0 now
    memset(z, 0, sizeof(T) * size);

    // we use 3 as offset, since first 12 bytes are occupied with header
    int flimit = limit + 3;

#pragma omp parallel for schedule(guided)
    for (int e = 3; e < flimit; e++) {
        int el = x[e];
        int ael = nd4j::math::nd4j_abs<int>(el) - 1;
        z[ael] = el > 0 ? threshold : -threshold;
    }
}

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z);
 */
void NativeOps::convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, Nd4jIndex N, int dstType, Nd4jPointer z) {
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
        } else if (dstType == ND4J_THRESHOLD) {
            convertToThreshold<float16>(dx, N, dz);
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
        } else if (dstType == ND4J_THRESHOLD) {
            convertToThreshold<float>(dx, N, dz);
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
        } else if (dstType == ND4J_THRESHOLD) {
            convertToThreshold<double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else if (srcType == ND4J_THRESHOLD) {
        if (dstType == ND4J_FLOAT16) {
            convertFromThreshold<float16>(dx, N, dz);
        } else if (dstType == ND4J_FLOAT32) {
            convertFromThreshold<float>(dx, N, dz);
        } else if (dstType == ND4J_DOUBLE) {
            convertFromThreshold<double>(dx, N, dz);
        } else {
            printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
        }
    } else {
        printf("Unsupported types conversion: [%i] -> [%i]\n", srcType, dstType);
    }
}


#endif //LIBND4J_TYPE_CONVERSIONS_H
