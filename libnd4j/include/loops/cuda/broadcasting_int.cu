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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/broadcasting_int.h>
#include <loops/legacy_ops.h>
#include <types/types.h>
#include <Environment.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <StringUtils.h>

using namespace simdOps;

//////////////////////////////////////////////////////////////////////////
template<typename X, typename OpClass>
static __global__ void broadcastIntSimple(
        void *x,
        Nd4jLong *xShapeInfo,
        void *y,
        Nd4jLong *yShapeInfo,
        void *z,
        Nd4jLong *zShapeInfo,
        int *dimension,
        int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
    
    functions::broadcast::BroadcastInt<X>::template transformCuda<OpClass>(x,xShapeInfo,y,yShapeInfo,z,zShapeInfo,dimension,dimensionLength,tadOnlyShapeInfo,tadOffsets,tadOnlyShapeInfoZ,tadOffsetsZ);
}

//////////////////////////////////////////////////////////////////////////
template<typename X, typename OpClass>
static __global__ void broadcastBoolInverseSimple(
        void *x,
        Nd4jLong *xShapeInfo,
        void *y,
        Nd4jLong *yShapeInfo,
        void *z,
        Nd4jLong *zShapeInfo,
        int *dimension,
        int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {

    functions::broadcast::BroadcastInt<X>::template transformInverseCuda<OpClass>(x,xShapeInfo,y,yShapeInfo,z,zShapeInfo,dimension,dimensionLength,tadOnlyShapeInfo,tadOffsets,tadOnlyShapeInfoZ,tadOffsetsZ);
}

namespace functions {
    namespace broadcast {
//////////////////////////////////////////////////////////////////////////
        template<typename X>
        template <typename OpClass>
        __host__ void BroadcastInt<X>::intermediateBroadcast(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            broadcastIntSimple<X, OpClass><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
        }

//////////////////////////////////////////////////////////////////////////
        template<typename X>
        __host__ void BroadcastInt<X>::execBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            DISPATCH_BY_OPNUM_T(intermediateBroadcast,  PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_INT_OPS))
        }

//////////////////////////////////////////////////////////////////////////
        template<typename X>
        template <typename OpClass>
        __host__ void BroadcastInt<X>::intermediateInverseBroadcast(dim3 launchDims, cudaStream_t *stream, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            broadcastBoolInverseSimple<X, OpClass><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ);
        }

//////////////////////////////////////////////////////////////////////////
        template<typename X>
        __host__ void BroadcastInt<X>::execInverseBroadcast(dim3 launchDims, cudaStream_t *stream, int opNum, void *x, Nd4jLong *xShapeInfo, void *y, Nd4jLong *yShapeInfo, void *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            DISPATCH_BY_OPNUM_T(intermediateInverseBroadcast,  PARAMS(launchDims, stream, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dimension, dimensionLength, tadOnlyShapeInfo, tadOffsets, tadOnlyShapeInfoZ, tadOffsetsZ), OPS_A(BROADCAST_INT_OPS))
        }

//////////////////////////////////////////////////////////////////////////
        template<typename X>
        template <typename OpType>
        __device__ void BroadcastInt<X>::transformInverseCuda(
                void *vx, Nd4jLong *xShapeInfo,
                void *vy, Nd4jLong *yShapeInfo,
                void *vz, Nd4jLong *zShapeInfo,
                int *dimension, int dimensionLength,
                Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {

            if (tadOnlyShapeInfoZ == nullptr) {
                tadOnlyShapeInfoZ = tadOnlyShapeInfo;
                tadOffsetsZ = tadOffsets;
            }

            auto x = reinterpret_cast<X*>(vx);
            auto y = reinterpret_cast<X*>(vy);
            auto z = reinterpret_cast<X*>(vz);

            //decompose in to several sub tads after
            //moving all dimensions (in sorted order)
            //to the back.
            //permuted version of the x shape info for setting up the tad problem
            __shared__ Nd4jLong tadLength;
            __shared__ Nd4jLong tadEWS;
            __shared__ int numTads;
            __shared__ Nd4jLong xEWS;
            __shared__ Nd4jLong zEWS;

            if (threadIdx.x == 0) {
                tadLength = shape::length(tadOnlyShapeInfo);//shape::tadLength(xShapeInfo, dimension, dimensionLength);
                tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
                numTads = shape::length(yShapeInfo) / tadLength;
                xEWS = shape::elementWiseStride(xShapeInfo);
                zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);
            }
            __syncthreads();

            for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
                auto rZ = z + tadOffsetsZ[r];
                auto rY = y + tadOffsets[r];

                if(tadEWS > 0 && zEWS > 0 && xEWS > 0 && dimensionLength == 1) {

                    for (int i = threadIdx.x; i < tadLength; i+= blockDim.x)
                        rZ[i * zEWS] = OpType::op(x[i * xEWS], rY[i * tadEWS]);
                }
                else {
                    // it is expected that x and z tads and y array all have the same length
                    for (Nd4jLong i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                        auto xOffset = shape::getIndexOffset(i, xShapeInfo,  tadLength);
                        auto yOffset = shape::getIndexOffset(i, tadOnlyShapeInfo, tadLength);
                        auto zOffset = shape::getIndexOffset(i, tadOnlyShapeInfoZ, tadLength);

                        rZ[zOffset] = OpType::op(x[xOffset], rY[yOffset]);
                    }
                }
            }
        }

//////////////////////////////////////////////////////////////////////////
        template<typename X>
        template <typename OpType>
		__device__ void BroadcastInt<X>::transformCuda(
		                              void *vx, Nd4jLong *xShapeInfo,
		                              void *vy, Nd4jLong *yShapeInfo,
		                              void *vz, Nd4jLong *zShapeInfo,
		                              int *dimension, int dimensionLength,
                                      Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadOnlyShapeInfoZ, Nd4jLong *tadOffsetsZ) {

            if (tadOnlyShapeInfoZ == nullptr) {
                tadOnlyShapeInfoZ = tadOnlyShapeInfo;
                tadOffsetsZ = tadOffsets;
            }

            auto x = reinterpret_cast<X*>(vx);
            auto y = reinterpret_cast<X*>(vy);
            auto z = reinterpret_cast<X*>(vz);

            //decompose in to several sub tads after
            //moving all dimensions (in sorted order)
            //to the back.
            //permuted version of the x shape info for setting up the tad problem
            __shared__ Nd4jLong tadLength;
            __shared__ Nd4jLong tadEWS;
            __shared__ int numTads;
            __shared__ Nd4jLong yEWS;
            __shared__ Nd4jLong zEWS;
      
            if (threadIdx.x == 0) {
   	            tadLength = shape::length(tadOnlyShapeInfo);//shape::tadLength(xShapeInfo, dimension, dimensionLength);
                tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
                numTads = shape::length(xShapeInfo) / tadLength;
                yEWS = shape::elementWiseStride(yShapeInfo);
                zEWS = shape::elementWiseStride(tadOnlyShapeInfoZ);    
            }
            __syncthreads();

            __shared__ X *rZ;
            __shared__ X *rX;

		for (int r = blockIdx.x; r < numTads; r += gridDim.x) {

            if (threadIdx.x == 0) {
                rZ = z + tadOffsetsZ[r];
                rX = x + tadOffsets[r];
            }
            __syncthreads();


            if(tadEWS > 0 && zEWS > 0 && yEWS > 0 && dimensionLength == 1) {

                for (int i = threadIdx.x; i < tadLength; i+= blockDim.x)
                    rZ[i * zEWS] = OpType::op(rX[i * tadEWS], y[i * yEWS]);
            }
            else {
                // it is expected that x and z tads and y array all have the same length
                for (Nd4jLong i = threadIdx.x; i < tadLength; i+= blockDim.x) {
                    auto xOffset = shape::getIndexOffset(i, tadOnlyShapeInfo,  tadLength);
                    auto yOffset = shape::getIndexOffset(i, yShapeInfo, tadLength);
                    auto zOffset = shape::getIndexOffset(i, tadOnlyShapeInfoZ, tadLength);

                    rZ[zOffset] = OpType::op(rX[xOffset], y[yOffset]);
                }
            }
		}
	}


        template<typename X>
        void BroadcastInt<X>::exec(int opNum,
                         void *x,
                         Nd4jLong *xShapeInfo,
                         void *y,
                         Nd4jLong *yShapeInfo,
                         void *result,
                         Nd4jLong *resultShapeInfo,
                         int *dimension,
                         int dimensionLength,
                         Nd4jLong *tadShapeInfo,
                         Nd4jLong *tadOffset,
                         Nd4jLong *tadShapeInfoZ,
                         Nd4jLong *tadOffsetZ) {

        }

        template<typename X>
        void BroadcastInt<X>::execInverse(int opNum,
                                void *x,
                                Nd4jLong *xShapeInfo,
                                void *y,
                                Nd4jLong *yShapeInfo,
                                void *result,
                                Nd4jLong *resultShapeInfo,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffset,
                                Nd4jLong *tadShapeInfoZ,
                                Nd4jLong *tadOffsetZ) {

        }

        template<typename X>
        template<typename OpType>
        void BroadcastInt<X>::exec(void *x,
                         Nd4jLong *xShapeInfo,
                         void *y,
                         Nd4jLong *yShapeInfo,
                         void *result,
                         Nd4jLong *resultShapeInfo,
                         int *dimension,
                         int dimensionLength,
                         Nd4jLong *tadShapeInfo,
                         Nd4jLong *tadOffset,
                         Nd4jLong *tadShapeInfoZ,
                         Nd4jLong *tadOffsetZ) {

        }

        template<typename X>
        template<typename OpType>
        void BroadcastInt<X>::execInverse(void *x,
                                Nd4jLong *xShapeInfo,
                                void *y,
                                Nd4jLong *yShapeInfo,
                                void *result,
                                Nd4jLong *resultShapeInfo,
                                int *dimension,
                                int dimensionLength,
                                Nd4jLong *tadShapeInfo,
                                Nd4jLong *tadOffset,
                                Nd4jLong *tadShapeInfoZ,
                                Nd4jLong *tadOffsetZ) {

        }

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT BroadcastInt, , INTEGER_TYPES);
    }
}