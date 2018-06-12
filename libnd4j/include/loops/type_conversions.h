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
__device__ inline void convertKernelGeneric(void *dx, Nd4jLong N, void *dz) {
    auto x = reinterpret_cast<S *>(dx);
    auto z = reinterpret_cast<T *>(dz);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (Nd4jLong i = tid; i < N; i+= blockDim.x * gridDim.x) {
        // despite it's stupid, it simplifies conversion to bottom dtypes
        z[i] = static_cast<T>(static_cast<float>(x[i]));
    }
};

/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 */
template<typename T>
__device__ inline void encoderKernelP1Generic(void *dx, Nd4jLong N, void *dz, float threshold) {
    auto x = reinterpret_cast<T *> (dx);
    auto z = reinterpret_cast<int *> (dz);

    //basically, for phase One we want do calculation: how many eligible values we have, and which blocks will be holding data
    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;

    int pass = tid < N && nd4j::math::nd4j_abs<T>(x[tid]) >= static_cast<T>(threshold) ? 1 : 0;
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

#define NUM_BANKS 32
#define LOG_NUM_BANKS 4

// Define this to more rigorously avoid bank conflicts, even at the lower (root) levels of the tree
//#define ZERO_BANK_CONFLICTS

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index)   temp[index]
#endif


inline bool
isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int
floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127);
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}


template <bool isNP2>
__device__ void loadSharedChunkFromMem(int *s_data, const int *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB) {
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;

    // compute spacing to avoid bank conflicts
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    // pad values beyond n with zeros
    s_data[ai + bankOffsetA] = g_idata[mem_ai];

    if (isNP2) { // compile-time decision
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0;
    } else {
        s_data[bi + bankOffsetB] = g_idata[mem_bi];
    }
}

template <bool isNP2>
__device__ void storeSharedChunkToMem(int* g_odata, int* s_data, int n, int ai, int bi, int mem_ai, int mem_bi, int bankOffsetA, int bankOffsetB) {
    __syncthreads();

    // write results to global memory
    g_odata[mem_ai] = s_data[ai + bankOffsetA];
    if (isNP2) { // compile-time decision
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB];
    } else {
        g_odata[mem_bi] = s_data[bi + bankOffsetB];
    }
}

template <bool storeSum>
__device__ void clearLastElement(int* s_data, int *g_blockSums, int blockIndex) {
    if (threadIdx.x == 0)
    {
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);

        if (storeSum) { // compile-time decision
            // write this block's total sum to the corresponding index in the blockSums array
            g_blockSums[blockIndex] = s_data[index];
        }

        // zero the last element in the scan so it will propagate back to the front
        s_data[index] = 0;
    }
}



__device__ unsigned int buildSum(int *s_data) {
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;

    // build the sum in place up the tree
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

__device__ void scanRootToLeaves(int *s_data, unsigned int stride) {
     unsigned int thid = threadIdx.x;

    // traverse down the tree building the scan in place
    for (int d = 1; d <= blockDim.x; d *= 2) {
        stride >>= 1;

        __syncthreads();

        if (thid < d) {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum>
__device__ void prescanBlock(int *data, int blockIndex, int *blockSums) {
    int stride = buildSum(data);               // build the sum in place up the tree
    clearLastElement<storeSum>(data, blockSums,
                               (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanRootToLeaves(data, stride);            // traverse down tree to build the scan
}


template <bool storeSum, bool isNP2>
__global__ void prescan(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    extern __shared__ int s_data[];

    // load data into shared memory
    loadSharedChunkFromMem<isNP2>(reinterpret_cast<int *>(s_data), g_idata, n, (baseIndex == 0) ? __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);

    // scan the data in each block
    prescanBlock<storeSum>(s_data, blockIndex, g_blockSums);

    // write results to device memory
    storeSharedChunkToMem<isNP2>(g_odata, s_data, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
}


__global__ void uniformAdd(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex) {
    __shared__ float uni;
    if (threadIdx.x == 0)
        uni = uniforms[blockIdx.x + blockOffset];

    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x;

    __syncthreads();

    // note two adds per thread
    g_data[address] += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

/*
 * This kernel does prefix sum in parallel, to calculate offsets for each block
 */
template<typename T>
__device__ inline void encoderKernelP2Generic(void *dx, Nd4jLong n, void *dz) {
 // TODO: to be remove
}

/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 *
 * Based on: https://github.com/knotman90/cuStreamComp <-- efficient CUDA stream compaction algorithm
 */
template<typename T>
__device__ inline void encoderKernelP3Generic(void *dx, int *offsets, Nd4jLong N, void *dz) {
    T *x = reinterpret_cast<T *> (dx);
    int *z = reinterpret_cast<int *> (dz);

    Nd4jLong tid = blockIdx.x * blockDim.x + threadIdx.x;
	extern __shared__ int warpTotals[];

    // fetch block offset only once
    __shared__ float threshold;
    __shared__ FloatBits fb;
	__shared__ int bo;
	__shared__ int limit;
	if (threadIdx.x == 0) {
	    limit = z[0];
        fb.i_ = z[2];
        threshold = fb.f_;
	    bo = offsets[blockIdx.x];
	}
	__syncthreads();

	if (tid < N) {
	    T value = x[tid];
        int pred = nd4j::math::nd4j_abs<T>(value) >= static_cast<T>(threshold) ? 1 : 0;
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
		    int idx = t_u + warpTotals[w_i] + bo + 4;
		    if (idx < limit + 4) {
			    z[idx]= value > static_cast<T>(0.0f) ? tid+1 : -(tid + 1);
			    x[tid] = value > static_cast<T>(0.0f) ? x[tid] - threshold : x[tid] + threshold;
			}
		}
	}
}

/*
*   This kernel handles decode from sparse threshold array, to dense array
 *
 *   PLEASE NOTE: Z is expected to be memset to 0
*/
template<typename T>
__device__ inline void decoderKernelGeneric(void *dx, Nd4jLong N, void *dz) {
    auto x = reinterpret_cast<int *> (dx);
    auto z = reinterpret_cast<T *> (dz);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float threshold;
    __shared__ int limit;

    __shared__ FloatBits fb;
    if (threadIdx.x == 0) {
        limit = x[0];
        fb.i_ = x[2];
        threshold = fb.f_;
    }
    __syncthreads();

    for (int e = tid; e < limit; e += blockDim.x * gridDim.x) {
        int el = x[e+4];
        int ael = nd4j::math::nd4j_abs<int>(el) - 1;

        // TODO: investigate, if += would work better here, as in "decoded accumulation"
        z[ael] += el > 0 ? threshold : -threshold;
    }
}

template<typename T>
__device__ inline void cudaDecodeBitmapGeneric(void *dx, Nd4jLong N, T *dz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T *shmem;
    __shared__ FloatBits fb;
    __shared__ float threshold;
    __shared__ int *x;
    if (threadIdx.x == 0){
        extern __shared__ char mem[];
        shmem = reinterpret_cast<T*>(mem);
        x = reinterpret_cast<int *>(dx);
        fb.i_ = x[2];
        threshold = fb.f_;
    }
    __syncthreads();

    int lim = N / 16 + 5;
    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        int byteId = i / 16 + 4;
//        printf("I: [%i]; byteId: [%i]\n", i, byteId);

        shmem[threadIdx.x] = dz[i];
        __syncthreads();

        if (threadIdx.x % 16 == 0) {
            int byte = x[byteId];

            for (int e = 0; e < 16; e++) {
                if (i + e >= N)
                    continue;

                int bitId = (i + e) % 16;

                bool hasBit = (byte & 1 << (bitId) ) != 0;
                bool hasSign = (byte & 1 << (bitId + 16) ) != 0;

                if (hasBit) {
                    if (hasSign)
                        shmem[threadIdx.x + bitId] -= threshold;
                    else
                        shmem[threadIdx.x + bitId] += threshold;
                } else if (hasSign) {
                    shmem[threadIdx.x + bitId] -= threshold / 2;
                }
            }
        }
        __syncthreads();

        dz[i] = shmem[threadIdx.x];
    }
}


template<typename T>
__device__ inline void cudaEncodeBitmapGeneric(T *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int counter;
    __shared__ int *shmem;
    __shared__ T *vals;
    if (threadIdx.x == 0){
        extern __shared__ char mem[];
        shmem = reinterpret_cast<int*>(mem);
        vals = reinterpret_cast<T *>(shmem + blockDim.x);
        counter = 0;
    }
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x * gridDim.x) {
        // all threads in block reading stuff
        T val = dx[i];
        T abs = nd4j::math::nd4j_abs<T>(val);

        int byteId = i / 16 + 4;
        int bitId = i % 16;

        shmem[threadIdx.x] = 0;
        vals[threadIdx.x] = val;

        if (abs >= static_cast<T>(threshold)) {
            shmem[threadIdx.x] = 1 << (bitId);
            atomicAdd(&counter, 1);
            if (val < static_cast<T>(0.0f)) {
                shmem[threadIdx.x] |= 1 << (bitId + 16);
                vals[threadIdx.x] += static_cast<T>(threshold);
            } else {
                vals[threadIdx.x] -= static_cast<T>(threshold);
            }
        } else if (abs >= static_cast<T>(threshold) / static_cast<T>(2.0f) && val < static_cat<T>(0.0f)) {
            atomicAdd(&counter, 1);
            shmem[threadIdx.x] = 1 << (bitId + 16);

            vals[threadIdx.x] += static_cast<T>(threshold) / static_cast<T>(2.0f);
        }
        __syncthreads();

        if (threadIdx.x % 16 == 0) {
            int byte = 0;
            for (int e = 0; e < 16; e++) {
                if (i + e >= N)
                    continue;

                byte |= shmem[threadIdx.x + e];
            }
            dz[byteId] = byte;
        }
        __syncthreads();

        dx[i] = vals[threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(scalar, counter);
    }
}

extern "C" __global__ void cudaEncodeBitmapFloat(float *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
    cudaEncodeBitmapGeneric<float>(dx, N, dz, scalar, reductionBuffer, threshold);
}

extern "C" __global__ void cudaEncodeBitmapDouble(double *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
    cudaEncodeBitmapGeneric<double>(dx, N, dz, scalar, reductionBuffer, threshold);
}

extern "C" __global__ void cudaEncodeBitmapHalf(float16 *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
    cudaEncodeBitmapGeneric<float16>(dx, N, dz, scalar, reductionBuffer, threshold);
}

extern "C" __global__ void cudaDecodeBitmapFloat(void *dx, Nd4jLong N, float *dz) {
    cudaDecodeBitmapGeneric<float>(dx, N, dz);
}

extern "C" __global__ void cudaDecodeBitmapDouble(void *dx, Nd4jLong N, double *dz) {
    cudaDecodeBitmapGeneric<double>(dx, N, dz);
}

extern "C" __global__ void cudaDecodeBitmapHalf(void *dx, Nd4jLong N, float16 *dz) {
    cudaDecodeBitmapGeneric<float16>(dx, N, dz);
}


extern "C" __global__ void encoderKernelP1Float(void *dx, Nd4jLong N, void *dz, float threshold) {
    encoderKernelP1Generic<float>(dx, N, dz, threshold);
}

extern "C" __global__ void encoderKernelP1Double(void *dx, Nd4jLong N, void *dz, float threshold) {
    encoderKernelP1Generic<double>(dx, N, dz, threshold);
}

extern "C" __global__ void encoderKernelP1Half(void *dx, Nd4jLong N, void *dz, float threshold) {
    encoderKernelP1Generic<float16>(dx, N, dz, threshold);
}

extern "C" __global__ void encoderKernelP2Float(int *dx, Nd4jLong N, int *dz) {
    encoderKernelP2Generic<float>(dx, N, dz);
}

extern "C" __global__ void encoderKernelP3Float(void *dx, int *offsets, Nd4jLong N, void *dz) {
    encoderKernelP3Generic<float>(dx, offsets, N, dz);
}

extern "C" __global__ void encoderKernelP3Double(void *dx, int *offsets, Nd4jLong N, void *dz) {
    encoderKernelP3Generic<double>(dx, offsets, N, dz);
}

extern "C" __global__ void encoderKernelP3Half(void *dx, int *offsets, Nd4jLong N, void *dz) {
    encoderKernelP3Generic<float16>(dx, offsets, N, dz);
}

extern "C" __global__ void decoderKernelFloat(void *dx, Nd4jLong N, void *dz) {
    decoderKernelGeneric<float>(dx, N, dz);
}

extern "C" __global__ void decoderKernelDouble(void *dx, Nd4jLong N, void *dz) {
    decoderKernelGeneric<double>(dx, N, dz);
}

extern "C" __global__ void decoderKernelHalf(void *dx, Nd4jLong N, void *dz) {
    decoderKernelGeneric<float16>(dx, N, dz);
}
#endif

template<typename S, typename T>
void convertGeneric(void *dx, Nd4jLong N, void *dz) {
    auto x = reinterpret_cast<S *>(dx);
    auto z = reinterpret_cast<T *>(dz);

    if (N < 8000) {
#pragma omp simd
        for (int i = 0; i < N; i++) {
            z[i] = static_cast<T>(static_cast<float>(x[i]));
        }
    } else {

#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            z[i] = static_cast<T>(static_cast<float>(x[i]));
        }
    }
};


template <typename T>
void convertToThreshold(void *dx, Nd4jLong N, void *dz) {
    // we suppose that first 4 bytes are integer, second 4 bytes are float
    // integer: enc length
    // integer: dec length
    // float: threshold
    FloatBits fb;
    auto x = reinterpret_cast<T *>(dx);
    auto z = reinterpret_cast<int *>(dz);
    int limit = z[0];
    fb.i_ = z[2];
    float threshold = fb.f_;

    // FIXME: int limit is sad thing here, 2B elements limitation
    z[1] = (int) N;

    // we use 3 as offset, since first 12 bytes are occupied with header
    int flimit = limit + 4;
    volatile int cnt = 4;
    volatile bool flag = false;
#pragma omp parallel for schedule(guided) default(shared)
    for (int e = 0; e < N;  e++) {
        bool flag_load;
#pragma omp atomic read
        flag_load = flag;
        if (flag_load)
            continue;

        T cUpd = x[e];
        if (cUpd >= static_cast<T>(threshold)) {
            int idx;
#pragma omp atomic capture
            idx = cnt++;

            if (idx >= flimit) {
#pragma omp atomic write
                flag = true;
                continue;
            }

            z[idx] = e + 1;
            x[e] -= static_cast<T>(threshold);
        } else if (cUpd <= static_cast<T>(-threshold)) {
            int idx;
#pragma omp atomic capture
            idx = cnt++;

            if (idx >= flimit) {
#pragma omp atomic write
                flag = true;
                continue;
            }

            z[idx] = -e - 1;
            x[e] += static_cast<T>(threshold);
        }
    }
}

template <typename T>
void convertFromThreshold(void *dx, Nd4jLong N, void *dz) {
    FloatBits fb;
    auto z = reinterpret_cast<T *>(dz);
    auto x = reinterpret_cast<int *>(dx);
    int limit = x[0];
    fb.i_ = x[2];
    float threshold = fb.f_;

    // we use 3 as offset, since first 12 bytes are occupied with header
    int flimit = limit + 4;

#pragma omp parallel for schedule(guided)
    for (int e = 4; e < flimit; e++) {
        int el = x[e];
        int ael = nd4j::math::nd4j_abs<int>(el) - 1;
        z[ael] += el > 0 ? threshold : -threshold;
    }
}

/*
 * TypeDef:
 *     void convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, long N, int dstType, Nd4jPointer z);
 */
void NativeOps::convertTypes(Nd4jPointer *extras, int srcType, Nd4jPointer x, Nd4jLong N, int dstType, Nd4jPointer z) {
    auto dx = reinterpret_cast<void *>(x);
    auto dz = reinterpret_cast<void *>(z);

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
