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
//
//

#include <loops/type_conversions.h>
#include <types/types.h>

namespace nd4j {
    template<typename S, typename T>
    void TypeCast::convertGenericCuda(Nd4jPointer *extras, void *dx, Nd4jLong N, void *dz) {
        auto stream = reinterpret_cast<cudaStream_t *>(&extras[1]);

        nd4j::convertKernel<S, T><<<256, 1024, 1024, *stream>>>(dx, N, dz);
    };


    template<typename S, typename T>
    __device__ void convertKernelGeneric(S *x, Nd4jLong N, T *z) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;

        for (Nd4jLong i = tid; i < N; i+= blockDim.x * gridDim.x) {
            // despite it's stupid, it simplifies conversion to bottom dtypes
            // FIXME: get rid of through-float though
            z[i] = static_cast<T>(static_cast<float>(x[i]));
        }
    };


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






    __global__ void cudaEncodeBitmapFloat(float *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
        nd4j::cudaEncodeBitmapGeneric<float>(dx, N, dz, scalar, reductionBuffer, threshold);
    }

    __global__ void cudaEncodeBitmapDouble(double *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
        nd4j::cudaEncodeBitmapGeneric<double>(dx, N, dz, scalar, reductionBuffer, threshold);
    }

    __global__ void cudaEncodeBitmapHalf(float16 *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold) {
        nd4j::cudaEncodeBitmapGeneric<float16>(dx, N, dz, scalar, reductionBuffer, threshold);
    }

    __global__ void cudaDecodeBitmapFloat(void *dx, Nd4jLong N, float *dz) {
        nd4j::cudaDecodeBitmapGeneric<float>(dx, N, dz);
    }

    __global__ void cudaDecodeBitmapDouble(void *dx, Nd4jLong N, double *dz) {
        nd4j::cudaDecodeBitmapGeneric<double>(dx, N, dz);
    }

    __global__ void cudaDecodeBitmapHalf(void *dx, Nd4jLong N, float16 *dz) {
        nd4j::cudaDecodeBitmapGeneric<float16>(dx, N, dz);
    }


    __global__ void encoderKernelP1Float(void *dx, Nd4jLong N, void *dz, float threshold) {
        nd4j::encoderKernelP1Generic<float>(dx, N, dz, threshold);
    }

    __global__ void encoderKernelP1Double(void *dx, Nd4jLong N, void *dz, float threshold) {
        nd4j::encoderKernelP1Generic<double>(dx, N, dz, threshold);
    }

    __global__ void encoderKernelP1Half(void *dx, Nd4jLong N, void *dz, float threshold) {
        nd4j::encoderKernelP1Generic<float16>(dx, N, dz, threshold);
    }

    __global__ void encoderKernelP2Float(int *dx, Nd4jLong N, int *dz) {
        nd4j::encoderKernelP2Generic<float>(dx, N, dz);
    }

    __global__ void encoderKernelP3Float(void *dx, int *offsets, Nd4jLong N, void *dz) {
        nd4j::encoderKernelP3Generic<float>(dx, offsets, N, dz);
    }

    __global__ void encoderKernelP3Double(void *dx, int *offsets, Nd4jLong N, void *dz) {
        nd4j::encoderKernelP3Generic<double>(dx, offsets, N, dz);
    }

    __global__ void encoderKernelP3Half(void *dx, int *offsets, Nd4jLong N, void *dz) {
        nd4j::encoderKernelP3Generic<float16>(dx, offsets, N, dz);
    }

    __global__ void decoderKernelFloat(void *dx, Nd4jLong N, void *dz) {
        nd4j::decoderKernelGeneric<float>(dx, N, dz);
    }

    __global__ void decoderKernelDouble(void *dx, Nd4jLong N, void *dz) {
        nd4j::decoderKernelGeneric<double>(dx, N, dz);
    }

    __global__ void decoderKernelHalf(void *dx, Nd4jLong N, void *dz) {
        nd4j::decoderKernelGeneric<float16>(dx, N, dz);
    }

    template <bool storeSum, bool isNP2>
    __host__ void prescanLauncher(dim3 &blocks, dim3 &threads, int shmem, cudaStream_t *stream, int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex) {
        prescan<storeSum, isNP2><<<blocks, threads, shmem, *stream>>>(g_odata, g_idata, g_blockSums, n, blockIndex, baseIndex);
    };

    template <typename S, typename T>
    __global__ void convertKernel(void *dx, Nd4jLong N, void *dz) {
        auto x = reinterpret_cast<S *>(dx);
        auto z = reinterpret_cast<T *>(dz);

        nd4j::convertKernelGeneric(x, N, z);
    }


#define LIBND4J_BOOLS \
    (randomName0, 0), \
    (randomName1, 1)

    BUILD_DOUBLE_TEMPLATE(template void TypeCast::convertGenericCuda, (Nd4jPointer * extras, void *dx, Nd4jLong N, void *dz), LIBND4J_TYPES, LIBND4J_TYPES);
    BUILD_DOUBLE_TEMPLATE(template void prescanLauncher, (dim3 &blocks, dim3 &threads, int shmem, cudaStream_t *stream, int *g_odata, const int *g_idata, int *g_blockSums, int n, int blockIndex, int baseIndex), LIBND4J_BOOLS, LIBND4J_BOOLS);

#undef LIBND4J_BOOLS
}