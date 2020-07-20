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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 08.11.2018
// @author raver119@gmail.com
//

#include "../scalar_int.h"
#include <system/op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"

using namespace simdOps;

////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
__global__ void scalarAlongDimension(void const* x, Nd4jLong const* xShapeInfo,
                                    void *extraParams,
                                    void *z, Nd4jLong const* zShapeInfo,
                                    void const* scalars,
                                    int *dimension, int dimensionLength,
                                    Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets,
                                    Nd4jLong const* tadShapeInfoZ, Nd4jLong const* tadOffsetsZ) {

    functions::scalar::ScalarIntTransform<X>::template transformCuda<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}


////////////////////////////////////////////////////////////////////////
template <typename X, typename OpType>
__global__ void scalarSimpleShaped(void const* x, void const* y, Nd4jLong const* xShapeInfo, void *params, void *z, Nd4jLong const* zShapeInfo, int *allocationBuffer) {

    functions::scalar::ScalarIntTransform<X>::template transformCuda<OpType>(y, x, xShapeInfo, params, z, zShapeInfo, allocationBuffer);
}





// *********************************************************************//
// *********************************************************************//
namespace functions {
namespace scalar    {

////////////////////////////////////////////////////////////////////////
template<typename X>
template<typename OpType>
__device__ void  ScalarIntTransform<X>::transformCuda(void const* vscalar,
                                                        void const* vy, Nd4jLong const* yShapeInfo,
                                                        void *vparams,
                                                        void *vz, Nd4jLong const* zShapeInfo,
                                                        int *allocationBuffer) {
    auto scalar = reinterpret_cast<X const*>(vscalar)[0];
    auto y      = reinterpret_cast<X const*>(vy);
    auto params = reinterpret_cast<X*>(vparams);
    auto z      = reinterpret_cast<X*>(vz);

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

    __shared__ int len;
    if(threadIdx.x == 0)
        len = shape::length(yShapeInfo);
    __syncthreads();

    if(yEWS >= 1 && zEWS >= 1 && shape::order(yShapeInfo) == shape::order(zShapeInfo)) {
            transformCuda<OpType>(len, vscalar, vy, yEWS, vparams, vz, zEWS, allocationBuffer);
    }
    else {
        for (Nd4jLong i = tid; i < len; i+= totalThreads)
            z[shape::getIndexOffset(i, zShapeInfo)] = OpType::op(y[shape::getIndexOffset(i, yShapeInfo)], scalar, params);
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X>
template<typename OpType>
__device__ void  ScalarIntTransform<X>::transformCuda(Nd4jLong len,
                                                          void const* vx,
                                                          void const* vy, Nd4jLong yEWS,
                                                          void *vparams,
                                                          void *vz, Nd4jLong zEWS,
                                                          int *allocationBuffer) {

    auto x = reinterpret_cast<X const*>(vx)[0];
    auto y = reinterpret_cast<X const*>(vy);
    auto z = reinterpret_cast<X*>(vz);
    auto params = reinterpret_cast<X*>(vparams);

    int totalThreads = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    Nd4jLong i = tid;
    if(yEWS == 1 && zEWS == 1) {
        for (; i < len; i += totalThreads)
            z[i] = OpType::op(y[i], x, params);
    }
    else {
        for (; i < len; i += totalThreads)
            z[i * zEWS] = OpType::op(y[i * yEWS], x, params);
    }
}


////////////////////////////////////////////////////////////////////////
template<typename X>
template<typename OpType>
__device__ void  ScalarIntTransform<X>::transformCuda(void const* vx, Nd4jLong const* xShapeInfo,
                                                        void *vextraParams,
                                                        void *vz, Nd4jLong const* zShapeInfo,
                                                        void const* vscalars,
                                                        int *dimension, int dimensionLength,
                                                        Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets,
                                                        Nd4jLong const* tadShapeInfoZ, Nd4jLong const* tadOffsetsZ) {
    auto x = reinterpret_cast<X const*>(vx);
    auto scalars = reinterpret_cast<X const*>(vscalars);
    auto z = reinterpret_cast<X*>(vz);
    auto extraParams = reinterpret_cast<X*>(vextraParams);

    if (tadShapeInfoZ == nullptr) {
        tadShapeInfoZ = tadShapeInfo;
        tadOffsetsZ = tadOffsets;
    }

    // tad preparation
    auto tadEws = shape::elementWiseStride(tadShapeInfo);
    auto zEws = shape::elementWiseStride(tadShapeInfoZ);
    auto tadLength = shape::length(tadShapeInfo);//shape::tadLength(xShapeInfo, dimension, dimensionLength);
    auto numTads =shape::length(xShapeInfo) / tadLength;

    if (tadEws > 0 && zEws > 0 && shape::order(tadShapeInfo) == shape::order(zShapeInfo)) {

        // main loop, rolling over tads
        for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
            X *oZ = z + tadOffsetsZ[r];
            auto oX = x + tadOffsets[r];

            auto s = scalars[r];

            for (int f = threadIdx.x; f < tadLength; f += blockDim.x)
                oZ[f * zEws] = OpType::op(oX[f * tadEws], s, extraParams);
        }
    } else {
        // main loop, rolling over tads
        for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
            X *oZ = z + tadOffsetsZ[r];
            auto oX = x + tadOffsets[r];

            auto s = scalars[r];

            for (int f = threadIdx.x; f < tadLength; f += blockDim.x)
                oZ[shape::getIndexOffset(f, tadShapeInfoZ)] = OpType::op(oX[shape::getIndexOffset(f, tadShapeInfo)], s, extraParams);
        }
    }
}


////////////////////////////////////////////////////////////////////////
template<typename X>
template <typename OpType>
_CUDA_H void ScalarIntTransform<X>::intermediateAlongDimension(dim3& launchDims, cudaStream_t *stream,
                                                                void const* x, Nd4jLong const* xShapeInfo,
                                                                void *z, Nd4jLong const* zShapeInfo,
                                                                void const* scalars,
                                                                void *extraParams,
                                                                int *dimension, int dimensionLength,
                                                                Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets,
                                                                Nd4jLong const* tadShapeInfoZ, Nd4jLong const* tadOffsetsZ) {

    scalarAlongDimension<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

////////////////////////////////////////////////////////////////////////
template<typename X>
template<typename OpType>
void _CUDA_H ScalarIntTransform<X>::intermediateShaped(dim3& launchDims, cudaStream_t *stream,
                                                            void const* vx, Nd4jLong const* xShapeInfo,
                                                            void *vz, Nd4jLong const* zShapeInfo,
                                                            void const* vscalar,
                                                            void *vextraParams, int *allocPointer){

    scalarSimpleShaped<X, OpType><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, vscalar, xShapeInfo, vextraParams, vz, zShapeInfo, allocPointer);
}

////////////////////////////////////////////////////////////////////////
template<typename X>
void ScalarIntTransform<X>::executeCudaShaped(dim3& launchDims, cudaStream_t *stream,
                                                int opNum,
                                                void const* vx, Nd4jLong const* xShapeInfo,
                                                void *vz, Nd4jLong const* zShapeInfo,
                                                void const* vscalar,
                                                void* vextraParams) {

    if (sd::Environment::getInstance().isDebugAndVerbose())
        printf("H14 opNum:[%i]\n", opNum);

    DISPATCH_BY_OPNUM_T(intermediateShaped, PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo, vscalar, vextraParams, nullptr), SCALAR_INT_OPS);
}

////////////////////////////////////////////////////////////////////////
template<typename X>
void ScalarIntTransform<X>::executeCudaAlongDimension(dim3& launchDims, cudaStream_t *stream, int opNum, void const* vx, Nd4jLong const* xShapeInfo, void *vz, Nd4jLong const* zShapeInfo, void const* vscalars, void *vextraParams, int *dimension, int dimensionLength, Nd4jLong const* tadShapeInfo, Nd4jLong const* tadOffsets, Nd4jLong const* tadShapeInfoZ, Nd4jLong const* tadOffsetsZ) {
    DISPATCH_BY_OPNUM_T(intermediateAlongDimension, PARAMS(launchDims, stream, vx, xShapeInfo, vz, zShapeInfo, vscalars, vextraParams, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_INT_OPS);
}

    BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT ScalarIntTransform, , INTEGER_TYPES);

}
}

