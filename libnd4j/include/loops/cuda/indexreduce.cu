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
// Created by raver on 4/9/2018.
//

#include <Environment.h>
#include "../indexreduce.h"
#include <op_boilerplate.h>
#include <helpers/DebugHelper.h>
#include <types/types.h>

#include "../legacy_ops.h"

using namespace simdOps;


template <typename X, typename Z>
static __global__ void simpleIndexReduceGeneric(const int op,
                                           void *dx,
                                           Nd4jLong *xShapeInfo, int xRank,
                                           void *extraParams,
                                           void *result,
                                           Nd4jLong *resultShapeInfo, int zRank,
                                           int *dimension,
                                           int dimensionLength,
                                           int postProcessOrNot, int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

     functions::indexreduce::IndexReduce<X, Z>::transform(op,dx,xShapeInfo,extraParams,result,resultShapeInfo,dimension,dimensionLength,postProcessOrNot,allocationBuffer,reductionBuffer,tadOnlyShapeInfo,tadOffsets);
}

namespace functions {
    namespace indexreduce {

        template <typename X, typename Z>
        _CUDA_H void IndexReduce<X,Z>::executeIndexReduceScalar(dim3 launchDims, cudaStream_t *stream,
                                                                const int opNum,
                                                                void *dx, Nd4jLong *xShapeInfo,
                                                                int xRank,
                                                                void *extraParams,
                                                                void *result, Nd4jLong *resultShapeInfo,
                                                                int zRank,
                                                                int *dimension, int dimensionLength,
                                                                int postProcessOrNot,
                                                                int *allocationBuffer, void *reductionBuffer,
                                                                Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

            simpleIndexReduceGeneric<X, Z><<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(opNum,
                                                                                            dx, xShapeInfo, xRank,
                                                                                            extraParams,
                                                                                            result, resultShapeInfo, 0,
                                                                                            nullptr, 0,
                                                                                            1,
                                                                                            allocationBuffer, reductionBuffer,
                                                                                            tadOnlyShapeInfo, tadOffsets);
        }

        template <typename X, typename Z>
        _CUDA_H void IndexReduce<X, Z>::executeIndexReduce(dim3 launchDims, cudaStream_t *stream, const int opNum, void *dx, Nd4jLong *xShapeInfo, int xRank, void *extraParams, void *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            simpleIndexReduceGeneric<X, Z><<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
			 opNum,
			 dx,
			 xShapeInfo, xRank,
			 extraParams,
			 result,
			 resultShapeInfo, zRank,
			 dimension,
			 dimensionLength,
			 1, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);
        }

        // This is the un-specialized struct.  Note that we prevent instantiation of this
        // struct by putting an undefined symbol in the function body so it won't compile.
        template<typename T>
        struct SharedIndexValue {
            // Ensure that we won't compile any un-specialized types
            __device__ T * getPointer() {
                extern __device__ void error(void);
                error();
                return 0;
            }
        };

// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

        template<>
        struct SharedIndexValue<float> {
            __device__ IndexValue<float> * getPointer() {
                extern __shared__ IndexValue<float> s_int2[];
                return s_int2;
            }
        };
// Following are the specializations for the following types.
// int, uint, char, uchar, short, ushort, long long, ulong long, bool, float, and double
// One could also specialize it for user-defined types.

        template<>
        struct SharedIndexValue<double> {
            __device__ IndexValue<double> * getPointer() {
                extern __shared__ IndexValue<double> s_int6[];
                return s_int6;
            }
        };

        template <typename X, typename Z>
        template <typename OpType>
        __device__ void IndexReduce<X, Z>::aggregatePartials(IndexValue<X> **sPartialsRef, Nd4jLong tid, Nd4jLong numElements, void *vextraParams) {
            // start the shared memory loop on the next power of 2 less
            // than the block size.  If block size is not a power of 2,
            // accumulate the intermediate sums in the remainder range.
            auto extraParams = static_cast<X*>(vextraParams);
            IndexValue<X> *sPartials = *sPartialsRef;
            Nd4jLong floorPow2 = blockDim.x;

            if (floorPow2 & (floorPow2 - 1)) {
                while ( floorPow2 & (floorPow2 - 1) ) {
                    floorPow2 &= floorPow2 - 1;
                }

                if (tid >= floorPow2) {
                    IndexValue<X> prev = sPartials[tid - floorPow2];
                    IndexValue<X> curr = sPartials[tid];
                    sPartials[tid - floorPow2] = OpType::update(prev,curr,extraParams);
                }
                __syncthreads();
            }

            for (int activeThreads = floorPow2 >> 1;activeThreads; activeThreads >>= 1) {
                if (tid < activeThreads && tid + activeThreads < numElements) {
                    IndexValue<X> curr = sPartials[tid];
                    IndexValue<X> next = sPartials[tid + activeThreads];
                    sPartials[tid] = OpType::update(curr,next,extraParams);
                }
                __syncthreads();
            }
        }

        template <typename X, typename Y>
        __device__ void IndexReduce<X, Y>::transform(
                const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams,
                void *result,
                Nd4jLong *resultShapeInfo,
                int *dimension,
                int dimensionLength,
                int postProcessOrNot,
                int *allocationBuffer,
                void *reductionBuffer,
                Nd4jLong *tadShapeInfo,
                Nd4jLong *tadOffset) {
             DISPATCH_BY_OPNUM_TT(transform, PARAMS(x, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationBuffer, reductionBuffer, tadShapeInfo, tadOffset), INDEX_REDUCE_OPS);
        }


        template <typename X, typename Z>
        template <typename OpType>
        __device__ void IndexReduce<X, Z>::transform(void *vdx, Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vresult, Nd4jLong *resultShapeInfo,
                                                int *dimension, int dimensionLength,
                                                int postProcessOrNot,
                                                int *allocationBuffer, void *vreductionBuffer,
                                                Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets){
            /**int
             * Gpu information for the problem
             */
            auto dx = reinterpret_cast<X*>(vdx);
            auto result = reinterpret_cast<Z*>(vresult);
            auto extraParams = static_cast<X*>(vextraParams);
            auto reductionBuffer = static_cast<X*>(vreductionBuffer);
            auto order = shape::order(xShapeInfo);
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ volatile int resultScalar;

            //shared memory space for storing intermediate results
            __shared__ IndexValue<X>* sPartials;
            if(threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                sPartials = reinterpret_cast<IndexValue<X>*>(shmem);
            }
            __syncthreads();

            sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

            //length for the tad
            __shared__ volatile Nd4jLong xLength;

            __shared__ volatile Nd4jLong resultLength;


            //only compute the tad indexes once
            IndexValue<X> reduction = OpType::startingIndexValue(dx);

            if (threadIdx.x == 0) {
                if (resultShapeInfo != nullptr)
                    resultLength = shape::length(resultShapeInfo);
                else resultLength = 1;

                if (dimensionLength == 1) {
                    if (resultLength == 1 && (dimension == nullptr || dimension[0] == MAX_DIMENSION))
                        resultScalar = 1;
                    else
                        resultScalar = 0;
                }
                else
                    resultScalar = 0;

                if (resultLength == 1)
                    resultScalar = 1;

                xLength = shape::length(xShapeInfo);
            }
            __syncthreads();

            if (!resultScalar) {

                __shared__ Nd4jLong tadLength;
                __shared__ int tadEWS;
                __shared__ int numTads;

                if (threadIdx.x == 0) {
                    tadLength = shape::length(tadOnlyShapeInfo);
                    tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
                    numTads = shape::length(xShapeInfo) / tadLength;
                }
                __syncthreads();

                if (dimensionLength > 1 || tadEWS < 1) {

                    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {

                        auto tadOffsetForBlock = tadOffsets[r];
                        sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

                        for(int i = threadIdx.x;i < tadLength; i += blockDim.x) {
                            auto xOffset = tadOffsetForBlock + shape::getIndexOffset(i, tadOnlyShapeInfo);
                            IndexValue<X> comp {dx[xOffset], i};
                            sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], comp, extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength),extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            result[r] = (Z) sPartials[threadIdx.x].index;
                        }
                        __syncthreads();
                    }
                } else {

                    for(int i = blockIdx.x; i < numTads; i+= gridDim.x) {
                        Nd4jLong tadOffsetForBlock = tadOffsets[i];

                        sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

                        for (int x = threadIdx.x; x < tadLength; x+= blockDim.x) {
                            IndexValue<X> comp {dx[tadOffsetForBlock + x * tadEWS], x};
                            sPartials[threadIdx.x] =  OpType::update(sPartials[threadIdx.x], comp, extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength),extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            result[i] = (Z) sPartials[threadIdx.x].index; //postProcess(sPartials[0],tadLength ,extraParams);
                        }
                        __syncthreads();
                    }
                }
            } else {
                auto n = shape::length(xShapeInfo);
                auto xElementWiseStride = shape::elementWiseStride(xShapeInfo);

                if(xElementWiseStride >= 1 && order == 'c') {
                    for(Nd4jLong i = tid;i < n; i += (blockDim.x * gridDim.x)) {
                        IndexValue<X> indexVal = {dx[i * xElementWiseStride], i};
                        reduction = OpType::update(reduction, indexVal, extraParams);
                    }
                } else {

                    for(Nd4jLong i = tid;i < n; i += blockDim.x * gridDim.x) {
                        auto offset = shape::getIndexOffset(i, xShapeInfo);
                        IndexValue<X> indexVal = {dx[offset], i};
                        reduction = OpType::update(reduction, indexVal, extraParams);
                    }
                }


                sPartials[threadIdx.x] = reduction;
                __syncthreads();

                aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, (int) n),extraParams);
                __syncthreads();

                if (gridDim.x > 1) {
                    __shared__ bool amLast;
                    unsigned int *tc = (unsigned int *) reductionBuffer;
                    tid = threadIdx.x;
                    if (threadIdx.x == 0) {
                        auto pBuffer = reinterpret_cast<IndexValue<X> *>(reductionBuffer);
                        pBuffer[blockIdx.x] = {sPartials[0].value, sPartials[0].index};
                    }
                    __threadfence();
                    __syncthreads();

                    if (tid==0) {
                        unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
                        amLast = (ticket == gridDim.x-1);
                    }

                    __syncthreads();

                    if (amLast) {
                        tc[16384] = 0;
                        IndexValue<X> *pBuffer = (IndexValue<X> *) reductionBuffer;

                        sPartials[threadIdx.x] = OpType::startingIndexValue(dx);

                        for (Nd4jLong i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
                            sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], pBuffer[i], extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x),extraParams);

                        __syncthreads();
                        if (tid == 0) {
                            result[0] = (Z) sPartials[0].index;
                        }
                    }
                } else {
                    if (tid == 0) {
                        auto tc = reinterpret_cast<unsigned int *>(reductionBuffer);
                        tc[16384] = 0;
                        result[0] = (Z) sPartials[0].index;
                    }
                }

            }
        }

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT IndexReduce, , LIBND4J_TYPES, INDEXING_TYPES);
    }
}



