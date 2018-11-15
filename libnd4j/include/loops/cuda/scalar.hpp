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

#include "loops/scalar.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <op_boilerplate.h>
#include <helpers/TAD.h>
#include <types/types.h>

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
        void *z, Nd4jLong zEws, int *allocationBuffer) {

    functions::scalar::ScalarTransform<X,Y,Z>::template transformCuda<OpType>(
            n,
            x,
            y,
            yEWS,
            params,
            z,
            zEws,
            allocationBuffer,
            NULL);
}


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
// DISPATCH_KERNEL_SIMPLE(scalarAlongDimension, scalarAlongDimensionGeneric, float, INPUT(float *x, Nd4jLong *xShapeInfo, float *extraParams, float *z, Nd4jLong *zShapeInfo, float *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarAlongDimension, scalarAlongDimensionGeneric, double, INPUT(double *x, Nd4jLong *xShapeInfo, double *extraParams, double *z, Nd4jLong *zShapeInfo, double *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
// DISPATCH_KERNEL_SIMPLE(scalarAlongDimension, scalarAlongDimensionGeneric, float16, INPUT(float16 *x, Nd4jLong *xShapeInfo, float16 *extraParams, float16 *z, Nd4jLong *zShapeInfo, float16 *scalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))


    template <typename X, typename Y, typename Z, typename OpType>
    __global__ void scalarSimpleShaped(void* x, void *y, Nd4jLong *xShapeInfo, void *params, void *z, Nd4jLong *zShapeInfo, int *allocationBuffer) {
        scalarSimpleGeneric<X, Y, Z, OpType>(x, y, xShapeInfo, params, z, zShapeInfo, allocationBuffer);
    }

    template <typename X, typename Y, typename Z, typename OpType>
    __global__ void scalarSimpleStrided(Nd4jLong length, void* x, void *y, Nd4jLong xEws, void *params, void *z, Nd4jLong zEws, int *allocationBuffer) {
        scalarSimpleGeneric<X, Y, Z, OpType>(length, x, y, xEws, params, z, zEws, allocationBuffer);
    }

    template <typename X, typename Y, typename Z, typename OpType>
    __global__ void scalarAlongDimension(void *x,
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
        scalarAlongDimensionGeneric<X, Y, Z, OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
    }


namespace functions {
    namespace scalar {


    template<typename X, typename Y, typename Z>
    template<typename OpType>
    void _CUDA_H ScalarTransform<X,Y,Z>::intermediateShaped(dim3& launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void* vscalar, void *vextraParams, int *allocPointer){
        auto xEws = shape::elementWiseStride(hxShapeInfo);
        auto xOrder = shape::order(hxShapeInfo);

        auto zEws = shape::elementWiseStride(hzShapeInfo);
        auto zOrder = shape::order(hzShapeInfo);

        auto length = shape::length(hxShapeInfo);

        if (xEws >= 0 && zEws >= 0 && xOrder == zOrder) {
            scalarSimpleStrided<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(length, vx, vscalar, xEws, vextraParams, vz, zEws, allocPointer);
        } else {
            scalarSimpleShaped<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, vscalar, xShapeInfo, vextraParams, vz, zShapeInfo, allocPointer);
        }
    }

    template<typename X, typename Y, typename Z>
    template<typename OpType>
    void _CUDA_H ScalarTransform<X,Y,Z>::intermediateAlongDimension(dim3& launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *scalars, void *extraParams, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
        scalarAlongDimension<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z>>>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
    }


    template<typename X, typename Y, typename Z>
    void ScalarTransform<X,Y,Z>::executeCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void* vscalar, void *vextraParams) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        if (nd4j::Environment::getInstance()->isDebugAndVerbose())
		    printf("H14 opNum:[%i]\n", opNum);

		auto allocPointer = reinterpret_cast<int *>(extraPointers[3]);

        auto xType = nd4j::DataTypeUtils::fromT<X>();
        auto yType = nd4j::DataTypeUtils::fromT<Y>();
        auto zType = nd4j::DataTypeUtils::fromT<Z>();

        DISPATCH_BY_OPNUM_TTT(intermediateShaped, PARAMS(launchDims, stream, vx, xShapeInfo, hxShapeInfo, vz, zShapeInfo, hzShapeInfo, vscalar, vextraParams, allocPointer), SCALAR_OPS);
    }


    template<typename X, typename Y, typename Z>
    void ScalarTransform<X,Y,Z>::executeCudaAlongDimension(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, void *vscalars, void *vextraParams, int *dimension, int dimensionLength) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        auto tadShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
        auto tadOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);
        auto tadShapeInfoZ = reinterpret_cast<Nd4jLong *>(extraPointers[12]);
        auto tadOffsetsZ = reinterpret_cast<Nd4jLong *>(extraPointers[13]);

        DISPATCH_BY_OPNUM_TTT(intermediateAlongDimension, PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo, vscalars, vextraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_OPS);
    }


    template<typename X, typename Y, typename Z>
    __device__ void ScalarTransform<X,Y,Z>::transformCudaLegacy(int opNum, void* vscalar,
            void *vy, Nd4jLong *yShapeInfo,
            void *vparams,
            void *vz, Nd4jLong *zShapeInfo,
            int *allocationBuffer,
            UnifiedSharedMemory *manager) {
        DISPATCH_BY_OPNUM_TTT(transformCuda, PARAMS(vscalar, vy, yShapeInfo, vparams, vz, zShapeInfo, allocationBuffer, manager), SCALAR_OPS);
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

    auto scalar = reinterpret_cast<X*>(vscalar)[0];
    auto y      = reinterpret_cast<Y*>(vy);
    auto params = reinterpret_cast<Z*>(vparams);
    auto z = reinterpret_cast<Z*>(vz);

    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ Nd4jLong length;
    if(threadIdx.x == 0)
        length = shape::length(yShapeInfo);
    __syncthreads();


    for (Nd4jLong i = tid; i < length; i+= totalThreads) {
        z[shape::getIndexOffset(i, zShapeInfo, length)] = OpType::op(y[shape::getIndexOffset(i, yShapeInfo, length)], scalar, params);
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

    auto x = reinterpret_cast<X*>(vx);
    auto extraParams = reinterpret_cast<Z*>(vextraParams);
    auto z = reinterpret_cast<Z*>(vz);
    auto scalars = reinterpret_cast<Y*>(vscalars);

    if (tadShapeInfoZ == nullptr) {
        tadShapeInfoZ = tadShapeInfo;
        tadOffsetsZ = tadOffsets;
    }

    // tad preparation
    auto tadEws = shape::elementWiseStride(tadShapeInfo);
    auto zEws = shape::elementWiseStride(tadShapeInfo);
    auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    auto numTads =shape::length(xShapeInfo) / tadLength;

    if(tadEws < 1 || zEws < 1) {
        printf("ScalarTransform<X,Y,Z>::transformCuda: super-bad loop visited. Shouldn't ever happen\n");
        return;
    }

    // main loop, rolling over tads
    for (int r = blockIdx.x; r < numTads; r+=gridDim.x) {
        
        Z *oZ = z + tadOffsetsZ[r];
        X *oX = x + tadOffsets[r];

        for (int f = threadIdx.x; f < tadLength; f+= blockDim.x)
            oZ[f] = OpType::op(oX[f], scalars[r], extraParams);         
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
                                                    void *vz, Nd4jLong zEws,
                                                    int *allocationBuffer, UnifiedSharedMemory *manager) {

    auto x = reinterpret_cast<X*>(vx)[0];
    auto y = reinterpret_cast<Y*>(vy);
    auto z = reinterpret_cast<Z*>(vz);
    auto params = reinterpret_cast<Z*>(vparams);

    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;            
            
    for (Nd4jLong i = tid; i < n; i += totalThreads)
        z[i * zEws] = OpType::op(y[i * yEWS], x, params);
            
}


}
}



#endif // SCALAR_CU