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

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
__global__ static void scalarSimpleShaped(void* vx, void *vscalar, Nd4jLong *xShapeInfo, void *vparams, void *vz, Nd4jLong *zShapeInfo, int *allocationBuffer) {
    
    auto scalar = reinterpret_cast<Y*>(vscalar)[0];
    auto x      = reinterpret_cast<X*>(vx);
    auto params = reinterpret_cast<Z*>(vparams);
    auto z = reinterpret_cast<Z*>(vz);

    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ Nd4jLong length;
    if(threadIdx.x == 0) {
        length = shape::length(xShapeInfo);
    }
    __syncthreads();

    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto zEws = shape::elementWiseStride(zShapeInfo);

    auto xOrder = shape::order(xShapeInfo);
    auto zOrder = shape::order(zShapeInfo);


    if (xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
        for (Nd4jLong i = tid; i < length; i += totalThreads) {
            z[i * zEws] = OpType::op(x[i * xEws], scalar, params);
        }
    } else {
        for (Nd4jLong i = tid; i < length; i += totalThreads) {
            z[shape::getIndexOffset(i, zShapeInfo, length)] = OpType::op(x[shape::getIndexOffset(i, xShapeInfo, length)], scalar, params);
        }
    }
    
}

////////////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z, typename OpType>
__global__ static void scalarAlongDimension(void *vx, Nd4jLong *xShapeInfo,
                                          void *vextraParams,
                                          void *vz, Nd4jLong *zShapeInfo,
                                          void *vscalars,
                                          int *dimension, int dimensionLength,
                                          Nd4jLong *tadShapeInfo,  Nd4jLong *tadOffsets,
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
    auto zEws = shape::elementWiseStride(tadShapeInfoZ);
    auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    auto numTads =shape::length(xShapeInfo) / tadLength;

    if (tadEws > 0 && zEws > 0 && shape::order(tadShapeInfo) == shape::order(zShapeInfo)) {

        // main loop, rolling over tads
        for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
            Z *oZ = z + tadOffsetsZ[r];
            X *oX = x + tadOffsets[r];

            auto s = scalars[r];

            for (int f = threadIdx.x; f < tadLength; f += blockDim.x)
                oZ[f * zEws] = OpType::op(oX[f * tadEws], s, extraParams);
        }
    } else {
        // main loop, rolling over tads
        for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
            Z *oZ = z + tadOffsetsZ[r];
            X *oX = x + tadOffsets[r];

            auto s = scalars[r];

            for (int f = threadIdx.x; f < tadLength; f += blockDim.x)
                oZ[shape::getIndexOffset(f, tadShapeInfoZ, tadLength)] = OpType::op(oX[shape::getIndexOffset(f, tadShapeInfo, tadLength)], s, extraParams);
        }
    }
}


namespace functions {
namespace scalar    {

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void _CUDA_H ScalarTransform<X,Y,Z>::intermediateShaped(dim3& launchDims, cudaStream_t *stream, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void* vscalar, void *vextraParams, int *allocPointer){
    
    auto xEws = shape::elementWiseStride(hxShapeInfo);
    auto xOrder = shape::order(hxShapeInfo);

    auto zEws = shape::elementWiseStride(hzShapeInfo);
    auto zOrder = shape::order(hzShapeInfo);

    auto length = shape::length(hxShapeInfo);

    scalarSimpleShaped<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, vscalar, xShapeInfo, vextraParams, vz, zShapeInfo, allocPointer);
}

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void _CUDA_H ScalarTransform<X,Y,Z>::intermediateAlongDimension(dim3& launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *z, Nd4jLong *zShapeInfo, void *scalars, void *extraParams, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    scalarAlongDimension<X, Y, Z, OpType><<<launchDims.x, launchDims.y, launchDims.z>>>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X,Y,Z>::executeCudaShaped(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, void* vscalar, void *vextraParams) {

    if (nd4j::Environment::getInstance()->isDebugAndVerbose())
	   printf("H14 opNum:[%i]\n", opNum);

    DISPATCH_BY_OPNUM_TTT(intermediateShaped, PARAMS(launchDims, stream, vx, xShapeInfo, hxShapeInfo, vz, zShapeInfo, hzShapeInfo, vscalar, vextraParams, nullptr), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X,Y,Z>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, void *vz, Nd4jLong *zShapeInfo, void *vscalars, void *vextraParams, int *dimension, int dimensionLength,  Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    DISPATCH_BY_OPNUM_TTT(intermediateAlongDimension, PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo, vscalars, vextraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_OPS);
}



BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_0);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_1);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_2);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_3);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_4);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_5);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_6);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_7);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_8);
BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_9);

}
}



#endif // SCALAR_CU