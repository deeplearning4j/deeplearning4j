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
#include <templatemath.h>
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

/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 */
template<typename T>
__device__ inline void encoderKernelP1Generic(void *dx, Nd4jIndex N, void *dz, float threshold) {
    T *x = reinterpret_cast<T *> (dx);
    int *z = reinterpret_cast<int *> (dz);

    //basically, for phase One we want do calculation: how many eligible values we have, and which blocks will be holding data
    Nd4jIndex tid = blockIdx.x * blockDim.x + threadIdx.x;

    int pass = tid < N && nd4j::math::nd4j_abs<T>(x[tid]) >= (T) threshold ? 1 : 0;
    int bp=__syncthreads_count(pass);

    if (threadIdx.x == 0) {
        // saving out per-block passes
        z[blockIdx.x+1] = bp;

        // saving out sum
        atomicAdd(&z[0], bp);
    }
}

__device__ __inline__ int pow2i (int e){
	return 1<<e;
}

/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 *
 * Based on: https://github.com/knotman90/cuStreamComp <-- efficient CUDA stream compaction algorithm
 */
template<typename T>
__device__ inline void encoderKernelP2Generic(void *dx, int *offsets, Nd4jIndex N, void *dz) {
    T *x = reinterpret_cast<T *> (dx);
    int *z = reinterpret_cast<int *> (dz);

    Nd4jIndex tid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int warpTotals[];

    // fetch block offset only once
    __shared__ float threshold;
    __shared__ FloatBits fb;
	__shared__ int bo;
	if (threadIdx.x == 0) {
        fb.i_ = z[2];
        threshold = fb.f_;
	    bo = offsets[blockIdx.x+1];
	}
	__syncthreads();

	if (tid < N) {
	    T value = x[tid];
        int pred = nd4j::math::nd4j_abs<T>(value) >= (T) threshold ? 1 : 0;
		int w_i = threadIdx.x/warpSize; //warp index
		int w_l = tid % warpSize;//thread index within a warp
		int t_m = INT_MAX >> (warpSize-w_l-1); //thread mask (ERROR IN THE PAPER minus one is required)

		int b	= __ballot(pred) & t_m; //balres = number whose ith bit isone if the ith's thread pred is true masked up to the current index in warp
		int t_u	= __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

		if(w_l==warpSize-1){
			warpTotals[w_i]=t_u+pred;
		}
		__syncthreads();

		if(w_i==0 && w_l<blockDim.x/warpSize){
			int w_i_u=0;
			for(int j=0;j<=5;j++){
				int b_j =__ballot( warpTotals[w_l] & pow2i(j) ); //# of the ones in the j'th digit of the warp offsets
				w_i_u += (__popc(b_j & t_m)  ) << j;
				//printf("indice %i t_m=%i,j=%i,b_j=%i,w_i_u=%i\n",w_l,t_m,j,b_j,w_i_u);
			}
			warpTotals[w_l]=w_i_u;
		}

		__syncthreads();

		if(pred){
			z[t_u+warpTotals[w_i]+ bo]= value > (T) 0.0f ? tid+1 : -(tid + 1);
		}
	}
}

/*
*   This kernel handles decode from sparse threshold array, to dense array
 *
 *   PLEASE NOTE: Z is expected to be memset to 0
*/
template<typename T>
__device__ inline void decoderKernelGeneric(void *dx, Nd4jIndex N, void *dz) {
    int *x = reinterpret_cast<int *> (dx);
    T *z = reinterpret_cast<T *> (dz);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float threshold;
    __shared__ int limit;

    __shared__ FloatBits fb;
    if (threadIdx.x == 0) {
        limit = z[0];
        fb.i_ = z[2];
        threshold = fb.f_;
    }
    __syncthreads();

    for (int e = tid; e < limit; e += blockDim.x * gridDim.x) {
        int el = x[e+3];
        int ael = nd4j::math::nd4j_abs<int>(el) - 1;

        // TODO: investigate, if += would work better here, as in "decoded accumulation"
        z[ael] = el > 0 ? threshold : -threshold;
    }
}


extern "C" __global__ void encoderKernelP1Float(void *dx, Nd4jIndex N, void *dz, float threshold) {
    encoderKernelP1Generic<float>(dx, N, dz, threshold);
}

extern "C" __global__ void encoderKernelP1Double(void *dx, Nd4jIndex N, void *dz, float threshold) {
    encoderKernelP1Generic<double>(dx, N, dz, threshold);
}

extern "C" __global__ void encoderKernelP1Half(void *dx, Nd4jIndex N, void *dz, float threshold) {
    encoderKernelP1Generic<float16>(dx, N, dz, threshold);
}

extern "C" __global__ void encoderKernelP2Float(void *dx, int *offsets, Nd4jIndex N, void *dz) {
    encoderKernelP2Generic<float>(dx, offsets, N, dz);
}

extern "C" __global__ void encoderKernelP2Double(void *dx, int *offsets, Nd4jIndex N, void *dz) {
    encoderKernelP2Generic<double>(dx, offsets, N, dz);
}

extern "C" __global__ void encoderKernelP2Half(void *dx, int *offsets, Nd4jIndex N, void *dz) {
    encoderKernelP2Generic<float16>(dx, offsets, N, dz);
}

extern "C" __global__ void decoderKernelFloat(void *dx, Nd4jIndex N, void *dz) {
    decoderKernelGeneric<float>(dx, N, dz);
}
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
