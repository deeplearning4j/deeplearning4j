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
#include <Environment.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 4


namespace nd4j {

    typedef union {
        float f_;
        int i_;
    } FloatBits;


    class TypeCast {

    public:
        template<typename S, typename T>
        static _CUDA_H void convertGeneric(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);

        template <typename T>
        static _CUDA_H void convertToThreshold(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);

        template <typename T>
        static _CUDA_H void convertFromThreshold(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);

        static _CUDA_H Nd4jLong estimateQuantizedSize(Nd4jLong rawSize);

        template <typename T>
        static _CUDA_H void convertToQuantized(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

        template <typename T>
        static _CUDA_H void convertFromQuantized(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz);

        #ifdef __CUDACC__
        template<typename S, typename T>
        static _CUDA_H void convertGenericCuda(Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz);        
        #endif
    };


    FORCEINLINE _CUDA_HD bool isPowerOfTwo(int n) {
        return ((n&(n-1))==0) ;
    }

    FORCEINLINE _CUDA_HD int floorPow2(int n) {
#ifdef WIN32
        // method 2
    return 1 << static_cast<int>(logb(static_cast<float>(n)));
#else
        // method 1
        // float nf = (float)n;
        // return 1 << (((*(int*)&nf) >> 23) - 127);
        int exp;
        frexp(static_cast<float>(n), &exp);
        return 1 << (exp - 1);
#endif
    }

#ifdef __CUDACC__
    /*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 */
    template<typename T>
    __device__ inline void encoderKernelP1(void *dx, Nd4jLong N, void *dz, float threshold) {
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

    template<typename T>
    __host__ void encoderKernelP1Generic(dim3 &launchDims, cudaStream_t *stream, void *dx, Nd4jLong N, void *dz, float threshold);

/*
 * PLEASE NOTE: This kernel doesn't allow loop for data. Basically: grid will be huge.
 *
 * Based on: https://github.com/knotman90/cuStreamComp <-- efficient CUDA stream compaction algorithm
 */
    template<typename T>
    __device__ inline void encoderKernelP3(void *dx, int *offsets, Nd4jLong N, void *dz) {
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

            int b   = __ballot_sync(t_m, pred); //balres = number whose ith bit isone if the ith's thread pred is true masked up to the current index in warp
            int t_u = __popc(b); // popc count the number of bit one. simply count the number predicated true BEFORE MY INDEX

            if(w_l==warpSize-1){
                warpTotals[w_i]=t_u+pred;
            }
            __syncthreads();

            if(w_i==0 && w_l<blockDim.x/warpSize){
                int w_i_u=0;
                for(int j=0;j<=5;j++){
                    int b_j =__ballot_sync( t_m, warpTotals[w_l] & pow2i(j) ); //# of the ones in the j'th digit of the warp offsets
                    w_i_u += (__popc(b_j)  << j);
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

    template<typename T>
    __host__ void encoderKernelP3Generic(dim3 &launchDims, cudaStream_t *stream, void *dx, int *offsets, Nd4jLong N, void *dz);


    /*
*   This kernel handles decode from sparse threshold array, to dense array
 *
 *   PLEASE NOTE: Z is expected to be memset to 0
*/
    template<typename T>
    __device__ inline void decoderKernel(void *dx, Nd4jLong N, void *dz) {
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
    __host__ void decoderKernelGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, Nd4jLong N, void *dz);


//////////////////////////////////////////////////////////////////////////    
    template<typename T>
    __device__ inline void cudaEncodeBitmapKernel(void *vdx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {

        auto dx = reinterpret_cast<T *>(vdx);
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
            } else if (abs >= static_cast<T>(threshold) / static_cast<T>(2.0f) && val < static_cast<T>(0.0f)) {
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

    template<typename T>
    __host__ void cudaEncodeBitmapGeneric(dim3 &launchDims, cudaStream_t *stream, void *vdx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold);

//////////////////////////////////////////////////////////////////////////
    template<typename T>
    __device__ inline void cudaDecodeBitmapKernel(void *dx, Nd4jLong N, void *vdz) {

        auto dz = static_cast<T*>(vdz);
        
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
    __host__ void cudaDecodeBitmapGeneric(dim3 &launchDims, cudaStream_t *stream, void *dx, Nd4jLong N, void *vdz);

    // __global__ void cudaEncodeBitmapFloat(float *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold);

    // __global__ void cudaEncodeBitmapDouble(double *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold);

    // __global__ void cudaEncodeBitmapHalf(float16 *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold);

    // __global__ void cudaDecodeBitmapFloat(void *dx, Nd4jLong N, float *dz);

    // __global__ void cudaDecodeBitmapDouble(void *dx, Nd4jLong N, double *dz);

    // __global__ void cudaDecodeBitmapHalf(void *dx, Nd4jLong N, float16 *dz);

    // __global__ void encoderKernelP1Float(void *dx, Nd4jLong N, void *dz, float threshold);

    // __global__ void encoderKernelP1Double(void *dx, Nd4jLong N, void *dz, float threshold);

    // __global__ void encoderKernelP1Half(void *dx, Nd4jLong N, void *dz, float threshold);

    // __global__ void encoderKernelP2Float(int *dx, Nd4jLong N, int *dz);

    // __global__ void encoderKernelP3Float(void *dx, int *offsets, Nd4jLong N, void *dz);

    // __global__ void encoderKernelP3Double(void *dx, int *offsets, Nd4jLong N, void *dz);

    // __global__ void encoderKernelP3Half(void *dx, int *offsets, Nd4jLong N, void *dz);

    // __global__ void decoderKernelFloat(void *dx, Nd4jLong N, void *dz);

    // __global__ void decoderKernelDouble(void *dx, Nd4jLong N, void *dz);

    // __global__ void decoderKernelHalf(void *dx, Nd4jLong N, void *dz);

    __global__ void uniformAdd(int *g_data, int *uniforms, int n, int blockOffset, int baseIndex);

    template <bool storeSum, bool isNP2>
    __global__ void prescan(int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);


    template <bool storeSum, bool isNP2>
    __host__ void prescanLauncher(dim3 &blocks, dim3 &threads, int shmem, cudaStream_t *stream, int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex);

    template <typename S, typename T>
    __global__ void convertKernel(void *dx, Nd4jLong N, void *dz);
#endif

}

#endif //LIBND4J_TYPE_CONVERSIONS_H
