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
    __global__ void cudaEncodeBitmapFloat(float *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold);

    __global__ void cudaEncodeBitmapDouble(double *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold);

    __global__ void cudaEncodeBitmapHalf(float16 *dx, Nd4jLong N, int *dz, int *scalar, int *reductionBuffer, float threshold);

    __global__ void cudaDecodeBitmapFloat(void *dx, Nd4jLong N, float *dz);

    __global__ void cudaDecodeBitmapDouble(void *dx, Nd4jLong N, double *dz);

    __global__ void cudaDecodeBitmapHalf(void *dx, Nd4jLong N, float16 *dz);

    __global__ void encoderKernelP1Float(void *dx, Nd4jLong N, void *dz, float threshold);

    __global__ void encoderKernelP1Double(void *dx, Nd4jLong N, void *dz, float threshold);

    __global__ void encoderKernelP1Half(void *dx, Nd4jLong N, void *dz, float threshold);

    __global__ void encoderKernelP2Float(int *dx, Nd4jLong N, int *dz);

    __global__ void encoderKernelP3Float(void *dx, int *offsets, Nd4jLong N, void *dz);

    __global__ void encoderKernelP3Double(void *dx, int *offsets, Nd4jLong N, void *dz);

    __global__ void encoderKernelP3Half(void *dx, int *offsets, Nd4jLong N, void *dz);

    __global__ void decoderKernelFloat(void *dx, Nd4jLong N, void *dz);

    __global__ void decoderKernelDouble(void *dx, Nd4jLong N, void *dz);

    __global__ void decoderKernelHalf(void *dx, Nd4jLong N, void *dz);

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
