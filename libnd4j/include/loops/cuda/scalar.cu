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

#ifndef SCALAR_CU
#define SCALAR_CU

#include "../scalar.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <op_boilerplate.h>
#include <helpers/TAD.h>
#include <types/float16.h>

using namespace simdOps;

template<typename X, typename Y, typename Z, typename OpType>
__device__ void scalarAlongDimensionGeneric(void *x,
                                            Nd4jLong *xShapeInfo,
                                            void *extraParams,
                                            void *z,
                                            Nd4jLong *zShapeInfo,
                                            void *scalars,
                                            int *dimension,
                                            int dimensionLength,
                                            Nd4jLong *tadShapeInfo,
                                            Nd4jLong *tadOffsets,
                                            Nd4jLong *tadShapeInfoZ,
                                            Nd4jLong *tadOffsetsZ) {

    functions::scalar::ScalarTransform<X,Y,Z>::template transformCuda<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

template<typename X, typename Y, typename Z, typename OpType>
__device__ void scalarSimpleGeneric(
        Nd4jLong n,
        void* x,
        void *y,
        Nd4jLong yEWS, void *params,
        void *z, Nd4jLong zEWS, int *allocationBuffer) {

    functions::scalar::ScalarTransform<X,Y,Z>::template transformCuda<OpType>(
            n,
            x,
            y,
            yEWS,
            params,
            z,
            zEWS,
            allocationBuffer,
            NULL);
}

/*
// LEGACY KERNELS,
template <typename T>
__device__ void scalarGenericIndexes(
        int opNum,
        Nd4jLong n,
        T x,
        T *y,
        T *params,
        T *z,int *indexes, int *allocationBuffer) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::scalar::ScalarTransform<T>), sizeof(shape::TAD), 0);
    }
    __syncthreads();

    functions::scalar::ScalarTransform<T>::transform(
            opNum,
            n,
            x,
            y,
            params,
            z,
            indexes,
            allocationBuffer,
            manager);
}

__global__ void scalarDoubleIndexes(
        int opNum,
        Nd4jLong n,
        double x,
        double *y,
        double *params,
        double *z,int *indexes, int *allocationBuffer) {
    scalarGenericIndexes<double>(opNum,
                                 n,
                                 x,
                                 y,
                                 params,
                                 z,
                                 indexes, allocationBuffer);
}

__global__ void scalarFloatIndexes(
        int opNum,
        Nd4jLong n,
        float x,
        float *y,
        float *params,
        float *z,
        int *indexes, int *allocationBuffer) {
    scalarGenericIndexes<float>(opNum,
                                n,
                                x,
                                y,
                                params,
                                z,
                                indexes, allocationBuffer);
}
*/

template<typename X, typename Y, typename Z, typename OpType>
__device__ void scalarSimpleGeneric(
        void* x,
        void *y,
        Nd4jLong *xShapeInfo,
        void *params,
        void *z,
        Nd4jLong *zShapeInfo,
        int *allocationBuffer) {

    functions::scalar::ScalarTransform<X,Y,Z>::template transformCuda<OpType>(
            x,
            y,
            xShapeInfo,
            params,
            z,
            zShapeInfo,
            allocationBuffer,
            nullptr);
}



// // ScalarOp Along Dimension kernels
// DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, float, INPUT(float *x, Nd4jLong *xShapeInfo, float *extraParams, float *z, Nd4jLong *zShapeInfo, float *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, double, INPUT(double *x, Nd4jLong *xShapeInfo, double *extraParams, double *z, Nd4jLong *zShapeInfo, double *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, float16, INPUT(float16 *x, Nd4jLong *xShapeInfo, float16 *extraParams, float16 *z, Nd4jLong *zShapeInfo, float16 *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

// // scalar shape
// DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, float, INPUT(float x, float *y, Nd4jLong *xShapeInfo, float *params, float *z, Nd4jLong *zShapeInfo, int *allocationBuffer), PARAMS(x, y, xShapeInfo, params, z, zShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, double, INPUT(double x, double *y, Nd4jLong *xShapeInfo, double *params, double *z, Nd4jLong *zShapeInfo, int *allocationBuffer), PARAMS(x, y, xShapeInfo, params, z, zShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, float16, INPUT(float16 x, float16 *y, Nd4jLong *xShapeInfo, float16 *params, float16 *z, Nd4jLong *zShapeInfo, int *allocationBuffer), PARAMS(x, y, xShapeInfo, params, z, zShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))

// // scalar strided
// DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, float, INPUT(Nd4jLong n, float x, float *y, Nd4jLong yEWS, float *params, float *z,Nd4jLong zEWS, int *allocationBuffer), PARAMS(n, x, y, yEWS, params, z, zEWS, allocationBuffer), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, double, INPUT(Nd4jLong n, double x, double *y, Nd4jLong yEWS, double *params, double *z,Nd4jLong zEWS, int *allocationBuffer), PARAMS(n, x, y, yEWS, params, z, zEWS, allocationBuffer), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, float16, INPUT(Nd4jLong n, float16 x, float16 *y, Nd4jLong yEWS, float16 *params, float16 *z,Nd4jLong zEWS, int *allocationBuffer), PARAMS(n, x, y, yEWS, params, z, zEWS, allocationBuffer), OPS_A(SCALAR_OPS))


namespace functions {
    namespace scalar {

    template<typename X, typename Y, typename Z>
    void ScalarTransform<X,Y,Z>::executeCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, void *vx, Nd4jLong xEWS, void *vz, Nd4jLong zEWS, void* vscalar, void *vextraParams, Nd4jLong n) {

        auto x = static_cast<X *>(vx);
        auto z = static_cast<Z *>(vz);
        auto scalar = static_cast<Y *>(vscalar)[0];
        auto extraParams = static_cast<Z *>(vextraParams);

        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("F13 opNum:[%i]\n", opNum);

		int *allocPointer = static_cast<int *>(extraPointers[3]);

	    // this macro builds bunch of IF/ELSE selectors for kernel launch
        DISPATCH_SIMPLE(scalarSimpleStrided, float, PARAMS(n, scalar, x, xEWS, extraParams, z, zEWS, allocPointer), OPS_A(SCALAR_OPS))
    }


  //   template<>
  //   void ScalarTransform<float16>::executeCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *x, Nd4jLong xStride, float16 *z, Nd4jLong zEWS, float16 scalar, float16 *extraParams, Nd4jLong n) {
  //       cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	 //    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		//     printf("H13 opNum:[%i]\n", opNum);

		// int *allocPointer = static_cast<int *>(extraPointers[3]);

	 //    // this macro builds bunch of IF/ELSE selectors for kernel launch
  //       DISPATCH_SIMPLE(scalarSimpleStrided, float16, PARAMS(n, scalar, x, xStride, extraParams, z, zEWS, allocPointer), OPS_A(SCALAR_OPS))
  //   }


  //   template<>
  //   void ScalarTransform<double>::executeCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, Nd4jLong xStride, double *z, Nd4jLong zEWS, double scalar, double *extraParams, Nd4jLong n) {
  //       cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

	 //    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		//     printf("D13 opNum:[%i]\n", opNum);

		// int *allocPointer = static_cast<int *>(extraPointers[3]);

	 //    // this macro builds bunch of IF/ELSE selectors for kernel launch
  //       DISPATCH_SIMPLE(scalarSimpleStrided, double, PARAMS(n, scalar, x, xStride, extraParams, z, zEWS, allocPointer), OPS_A(SCALAR_OPS))
  //   }


    template<typename X, typename Y, typename Z>
    void ScalarTransform<X,Y,Z>::executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, void* vscalar, void *vextraParams) {
        
        auto x = static_cast<X *>(vx);
        auto z = static_cast<Z *>(vz);
        auto scalar = static_cast<Y *>(vscalar)[0];
        auto extraParams = static_cast<Z *>(vextraParams);

        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("H14 opNum:[%i]\n", opNum);

		int *allocPointer = static_cast<int *>(extraPointers[3]);

        DISPATCH_SIMPLE(scalarSimpleShaped, float16, PARAMS(scalar, x, xShapeInfo, extraParams, z, zShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
    }

  //   template<>
  //   void ScalarTransform<float>::executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *x, Nd4jLong *xShapeInfo, float *z, Nd4jLong *zShapeInfo, float scalar, float *extraParams) {
  //       cudaStream_t *stream = static_cast<cudaStream_t *>(&extraPointers[1]);

  //       if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		//     printf("F14 opNum:[%i]\n", opNum);

  //       int *allocPointer = static_cast<int *>(extraPointers[3]);

  //       DISPATCH_SIMPLE(scalarSimpleShaped, float, PARAMS(scalar, x, xShapeInfo, extraParams, z, zShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
  //   }

  //   template<>
  //   void ScalarTransform<double>::executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, Nd4jLong *xShapeInfo, double *z, Nd4jLong *zShapeInfo, double scalar, double *extraParams) {
  //       cudaStream_t *stream = static_cast<cudaStream_t *>(&extraPointers[1]);

  //       if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		//     printf("D14 opNum:[%i]\n", opNum);

		// int *allocPointer = static_cast<int *>(extraPointers[3]);

  //       DISPATCH_SIMPLE(scalarSimpleShaped, double, PARAMS(scalar, x, xShapeInfo, extraParams, z, zShapeInfo, allocPointer), OPS_A(SCALAR_OPS))
  //   }

    template<typename X, typename Y, typename Z>
    void ScalarTransform<X,Y,Z>::executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, void *vscalars, void *vextraParams, int *dimension, int dimensionLength) {

        auto x = static_cast<X *>(vx);
        auto z = static_cast<Z *>(vz);
        auto scalars = static_cast<Y*>(vscalars);
        auto extraParams = static_cast<Z *>(vextraParams);

        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        auto tadShapeInfo = static_cast<Nd4jLong *>(extraPointers[10]);
        auto tadOffsets = static_cast<Nd4jLong *>(extraPointers[11]);
        auto tadShapeInfoZ = static_cast<Nd4jLong *>(extraPointers[12]);
        auto tadOffsetsZ = static_cast<Nd4jLong *>(extraPointers[13]);

        DISPATCH_SIMPLE(scalarAlongDimension, double, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
    }

    // template<>
    // void ScalarTransform<float>::executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers,int opNum, float *x, Nd4jLong *xShapeInfo, float *z, Nd4jLong *zShapeInfo, float *scalars, float *extraParams, int *dimension, int dimensionLength) {
    //     cudaStream_t *stream = static_cast<cudaStream_t *>(&extraPointers[1]);

    //     auto tadShapeInfo = static_cast<Nd4jLong *>(extraPointers[10]);
    //     auto tadOffsets = static_cast<Nd4jLong *>(extraPointers[11]);
    //     auto tadShapeInfoZ = static_cast<Nd4jLong *>(extraPointers[12]);
    //     auto tadOffsetsZ = static_cast<Nd4jLong *>(extraPointers[13]);

    //     DISPATCH_SIMPLE(scalarAlongDimension, float, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
    // }

    // template<>
    // void ScalarTransform<float16>::executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers,int opNum, float16 *x, Nd4jLong *xShapeInfo, float16 *z, Nd4jLong *zShapeInfo, float16 *scalars, float16 *extraParams, int *dimension, int dimensionLength) {
    //     cudaStream_t *stream = static_cast<cudaStream_t *>(&extraPointers[1]);

    //     auto tadShapeInfo = static_cast<Nd4jLong *>(extraPointers[10]);
    //     auto tadOffsets = static_cast<Nd4jLong *>(extraPointers[11]);
    //     auto tadShapeInfoZ = static_cast<Nd4jLong *>(extraPointers[12]);
    //     auto tadOffsetsZ = static_cast<Nd4jLong *>(extraPointers[13]);

    //     DISPATCH_SIMPLE(scalarAlongDimension, float16, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
    // }

////////////////////////////////////////////////////////////////////////////////
/**
* Cuda implementation of transform
* @param x
* @param xShapeInfo
* @param z
* @param zShapeInfo
* @param extraParams
* @param n
*/
template<typename X, typename Y, typename Z>
template<typename OpType>
__device__ void ScalarTransform<X,Y,Z>::transform( Nd4jLong n,
                                                void* vscalar,
                                                void *vy,
                                                void *vparams,
                                                void *vz,
                                                Nd4jLong *indexes,
                                                int *allocationBuffer,
                                                UnifiedSharedMemory *manager) {

    auto scalar = static_cast<X*>(vscalar)[0];
    auto y = static_cast<Y*>(vy);
    auto params = static_cast<Z*>(vparams);
    auto z = static_cast<Z*>(vz);

    int totalThreads = gridDim.x * blockDim.x;    
    Nd4jLong i = blockIdx.x * blockDim.x + threadIdx.x;

    /* equal, positive, non-unit increments. */
    for (; i < n; i+= totalThreads) 
        z[indexes[i]] = OpType::op(y[indexes[i]], scalar, params);
}

////////////////////////////////////////////////////////////////////////////////
/**
* Cuda implementation of transform
* @param x
* @param xShapeInfo
* @param z
* @param zShapeInfo
* @param extraParams
* @param n
*/
template<typename X, typename Y, typename Z>
template<typename OpType>
__device__ void ScalarTransform<X,Y,Z>::transformCuda(void* vscalar,
                                                    void *vy, Nd4jLong *yShapeInfo,
                                                    void *vparams,
                                                    void *vz, Nd4jLong *zShapeInfo,
                                                    int *allocationBuffer,
                                                    UnifiedSharedMemory *manager) {

    auto scalar = static_cast<X*>(vscalar)[0];
    auto y      = static_cast<Y*>(vy);
    auto params = static_cast<Z*>(vparams);
    auto z = static_cast<Z*>(vz);

    auto yRank   = shape::rank(yShapeInfo);
    auto yEWS    = shape::elementWiseStride(yShapeInfo);
    auto yShape  = shape::shapeOf(yShapeInfo);
    auto yStride = shape::stride(yShapeInfo);        
    
    auto zRank   = shape::rank(zShapeInfo);
    auto zEWS    = shape::elementWiseStride(zShapeInfo);
    auto zShape  = shape::shapeOf(zShapeInfo);
    auto zStride = shape::stride(zShapeInfo);

    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int length;
    if(threadIdx.x == 0)
        length = shape::length(yShapeInfo);
    __syncthreads();

    if(yEWS >= 1 && zEWS >= 1 && shape::order(yShapeInfo) == shape::order(zShapeInfo))
            transformCuda<OpType>(length, scalar, y, yEWS, params, z, zEWS, allocationBuffer, manager);
    else {
        Nd4jLong xIdx[MAX_RANK];

        for (Nd4jLong i = tid; i < length; i+= totalThreads) {
            shape::ind2sub(yRank, yShape, i, length, xIdx);
            auto yOffset = shape::getOffset(0, yShape, yStride, xIdx, yRank);
            auto zOffset = shape::getOffset(0, zShape, zStride, xIdx, zRank);
            z[zOffset] = OpType::op(y[yOffset], scalar, params);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// ScalarOp along dimension
template<typename X, typename Y, typename Z>
template<typename OpType>
void __device__ ScalarTransform<X,Y,Z>::transformCuda(void *vx, Nd4jLong *xShapeInfo,
                                                      void *vextraParams,
                                                      void *vz, Nd4jLong *zShapeInfo,
                                                      void *vscalars,
                                                      int *dimension, int dimensionLength,
                                                      Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets,
                                                      Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {

    auto x = static_cast<X*>(vx);
    auto extraParams = static_cast<Z*>(vextraParams);
    auto z = static_cast<Z*>(vz);
    auto scalars = static_cast<Y*>(vscalars);

    if (tadShapeInfoZ == nullptr) {
        tadShapeInfoZ = tadShapeInfo;
        tadOffsetsZ = tadOffsets;
    }

    // tad preparation
    auto tadEWS = shape::elementWiseStride(tadShapeInfo);
    auto zEWS = shape::elementWiseStride(tadShapeInfo);
    auto tadRank = shape::rank(tadShapeInfo);
    auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    auto numTads =shape::length(xShapeInfo) / tadLength;

    // main loop, rolling over tads
    for (int r = blockIdx.x; r < numTads; r+=gridDim.x) {
        auto offset = tadOffsets[r];
        auto offsetZ = tadOffsetsZ[r];
        Y scalar = scalars[r];

        if (tadEWS >= 1 && zEWS >= 1) {
            Z *oZ = z + offsetZ;
            X *oX = x + offset;

            for (int f = threadIdx.x; f < tadLength; f+= blockDim.x)
                oZ[f] = OpType::op(oX[f], scalar, extraParams);
        } 
        else        
            printf("Super-bad loop visited. Shouldn't ever happen\n");
    }
}

////////////////////////////////////////////////////////////////////////////////
/**
*
* @param n
* @param idx
* @param x
* @param y
* @param yEWS
* @param params
* @param z
* @param blockSize
*/
template<typename X, typename Y, typename Z>
template<typename OpType>
__device__ void ScalarTransform<X,Y,Z>::transformCuda( Nd4jLong n,
                                                    void* vx,
                                                    void *vy, Nd4jLong yEWS,
                                                    void *vparams,
                                                    void *vz, Nd4jLong zEWS,
                                                    int *allocationBuffer, UnifiedSharedMemory *manager) {

    auto x = static_cast<X*>(vx);
    auto y = static_cast<Y*>(vy);
    auto z = static_cast<Z*>(vz);
    auto params = static_cast<Z*>(vparams);
            
    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Nd4jLong i = tid;
    if(yEWS == 1 && zEWS == 1) {
        for (; i < n; i += totalThreads)
            z[i] = OpType::op(y[i], x, params);
    }
    else {
        for (; i < n; i += totalThreads) 
            z[i * zEWS] = OpType::op(y[i * yEWS], x, params);

    }
}

/*
        static inline __device__ void transformCuda(
            const int opNum,
            T scalar,
            T *y,
            int *shapeInfo,
            T *params,
            T *z,
            int *zShapeInfo,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(scalar, y, shapeInfo, params, z, zShapeInfo, allocationBuffer, manager), SCALAR_OPS);
                    }


        static inline __device__ void transform(
            const int opNum,
            Nd4jLong n,
            T scalar,
            T *y,
            T *params,
            T *z,
            int *indexes,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transform, PARAMS(n, scalar, y, params, z, indexes, allocationBuffer, manager), SCALAR_OPS);
        }


        static inline __device__ void transformCuda(
            const int opNum,
            Nd4jLong n,
            T x,
            T *y,
            int yEWS,
            T *params,
            T *z,
            int zEWS,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, x, y, yEWS, params, z, zEWS, allocationBuffer, manager), SCALAR_OPS);
        }
        */

        // BUILD_CALL_1(template __device__ void ScalarTransform<float>::transformCuda, float, (float, float*, Nd4jLong *, float*, float*, Nd4jLong*, int*, UnifiedSharedMemory *), SCALAR_OPS)
        // BUILD_CALL_1(template __device__ void ScalarTransform<float16>::transformCuda, float16, (float16, float16*, Nd4jLong *, float16*, float16*, Nd4jLong*, int*, UnifiedSharedMemory *), SCALAR_OPS)
        // BUILD_CALL_1(template __device__ void ScalarTransform<double>::transformCuda, double, (double, double*, Nd4jLong *, double*, double*, Nd4jLong*, int*, UnifiedSharedMemory *), SCALAR_OPS)
    }
}



#endif // SCALAR_CU