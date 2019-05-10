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


#include <pointercast.h>
#include <types/types.h>
#include <types/float16.h>
#include <op_boilerplate.h>
#include <loops/summarystatsreduce.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <dll.h>
#include <Environment.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <specials_cuda.h>

using namespace simdOps;

namespace functions {
    namespace summarystats {

template <typename X, typename Z>
void _CUDA_G summaryStatsReduceT(int op, void *dx, Nd4jLong *xShapeInfo, int xRank, void *extraParams, void *z, Nd4jLong *zShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected,int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            
    functions::summarystats::SummaryStatsReduce<X,Z>::transform(op,dx,xShapeInfo,extraParams,z,zShapeInfo,dimension,dimensionLength,biasCorrected,allocationBuffer,reductionBuffer,tadOnlyShapeInfo,tadOffsets);
}

        /**
		 *
		 * @param sPartialsRef
		 * @param tid
		 * @param extraParams
		 */
        template<typename X, typename Z>
        template<typename OpType>
        _CUDA_D void SummaryStatsReduce<X,Z>::aggregatePartials(SummaryStatsData<X> **sPartialsRef, Nd4jLong tid, Nd4jLong numElements, void *vextraParams) {
            // start the shared memory loop on the next power of 2 less
            // than the block size.  If block size is not a power of 2,
            // accumulate the intermediate sums in the remainder range.
            auto extraParams = static_cast<Z*>(vextraParams);
            SummaryStatsData<X> *sPartials = *sPartialsRef;
            Nd4jLong floorPow2 = blockDim.x;

            if (floorPow2 & (floorPow2 - 1)) {
                while (floorPow2 & (floorPow2 - 1)) {
                    floorPow2 &= floorPow2 - 1;
                }

                if (tid >= floorPow2) {
                    SummaryStatsData<X> prev = sPartials[tid - floorPow2];
                    SummaryStatsData<X> curr = sPartials[tid];
                    sPartials[tid - floorPow2] = update(prev, curr, extraParams);
                }
                __syncthreads();
            }

            for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
                if (tid < activeThreads && tid + activeThreads < numElements) {
                    SummaryStatsData<X> curr = sPartials[tid];
                    SummaryStatsData<X> next = sPartials[tid + activeThreads];
                    sPartials[tid] = update(curr, next, extraParams);
                }
                __syncthreads();
            }
        };

        /**
			 * @param n n is the number of
			 *        elements to loop through
			 * @param dx the data to operate on
			 * @param xVectorInfo the meta data for the vector:
			 *                              0 is the offset
			 *                              1 is the increment/stride
			 *                              2 is the real length of the buffer (n and dx.length won't always be the same)
			 *                              3 is the element wise stride for the buffer
			 *                              4 is the number of elements it takes to get to the next row/column/tensor
			 * @param gpuInformation
			 *                              0 is the block size
			 *                              1 is the grid size
			 *                              2 is the shared memory size
			 * @param problemDefinition
			 *                          0 is the number of elements per vector
			 *                          1 is the number of vectors
			 */
        template<typename X, typename Z>
        template<typename OpType>
        _CUDA_D void SummaryStatsReduce<X,Z>::transform(void *vx, Nd4jLong *xShapeInfo, 
                                                        void *vextraParams, 
                                                        void *vz, Nd4jLong *zShapeInfo, 
                                                        int *dimension, int dimensionLength, 
                                                        int postProcessOrNot, 
                                                        int *allocationBuffer, void *vreductionBuffer, 
                                                        Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

            auto dx = static_cast<X*>(vx);
            auto z = static_cast<Z*>(vz);
            auto extraParams = static_cast<Z*>(vextraParams);
            auto reductionBuffer = static_cast<Z*>(vreductionBuffer);

            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ volatile int resultScalar;

            __shared__ int xElementWiseStride;

            int numElements = blockDim.x;
            //shared memory space for storing intermediate results
            __shared__ SummaryStatsData<X> *sPartials;
            if(threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                sPartials = reinterpret_cast<SummaryStatsData<X>*>(shmem);
            }
            __syncthreads();

            Z startingVal = startingValue(dx);

            SummaryStatsData<X> val;
            val.initWithValue(startingVal);
            val.n = 0;
            sPartials[threadIdx.x] = val;


            //length for the tad
            __shared__ volatile int xLength;

            __shared__ volatile int resultLength;


            SummaryStatsData<X> reduction;
            reduction.initWithValue(0.0);
            reduction.n = 0;
            if (threadIdx.x == 0) {
                if (zShapeInfo != nullptr)
                    resultLength = shape::length(zShapeInfo);
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

                auto xStride = shape::stride(xShapeInfo);
                auto xOrder = shape::order(xShapeInfo);

                if (dimension != nullptr && (dimension[0] != MAX_DIMENSION && dimensionLength == 1)) {
                    xElementWiseStride = xStride[dimension[0]];
                }
                else {
                    xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                }


                xLength = shape::length(xShapeInfo);


            }
            __syncthreads();
            if (!resultScalar) {

                __shared__ int tadLength;
                __shared__ int tadEWS;
                __shared__ int numTads;

                if (threadIdx.x == 0) {
                    tadLength = shape::length(tadOnlyShapeInfo);//shape::tadLength(xShapeInfo, dimension, dimensionLength);
                    tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
                    numTads = shape::length(xShapeInfo) / tadLength;
                }
                __syncthreads();

                if (tadEWS == 0) {

                    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
                        auto tadOffsetForBlock = tadOffsets[r];

                        val.initWithValue(startingVal);
                        val.n = 0;
                        sPartials[threadIdx.x] = val;

                        for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                            auto xOffset = tadOffsetForBlock + shape::getIndexOffset(i, tadOnlyShapeInfo, tadLength);
                            SummaryStatsData<X> indexVal2;
                            indexVal2.initWithValue(dx[xOffset]);

                            sPartials[threadIdx.x] = update(sPartials[threadIdx.x], OpType::op(indexVal2, extraParams), extraParams);
                        }
                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            z[r] = OpType::getValue(postProcessOrNot, sPartials[threadIdx.x]);
                        }

                    }
                }
                else {

                    for (int i = blockIdx.x; i < numTads; i += gridDim.x) {
                        auto tadOffsetForBlock = tadOffsets[i];

                        val.initWithValue(startingVal);
                        val.n = 0;
                        sPartials[threadIdx.x] = val;

                        for (int x = threadIdx.x; x < tadLength; x += blockDim.x) {
                            indexX = tadOffsetForBlock + x * tadEWS;
                            SummaryStatsData<X> indexVal2;
                            indexVal2.initWithValue(dx[indexX]);
                            sPartials[threadIdx.x] = update(sPartials[threadIdx.x], OpType::op(indexVal2, extraParams), extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            z[i] = OpType::getValue(postProcessOrNot, sPartials[threadIdx.x]); //postProcess(sPartials[0],tadLength ,extraParams);
                        }
                    }
                }
            }
            else if (resultScalar) {
                __shared__ int n;
                if (threadIdx.x == 0) {
                    xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                    n = shape::length(xShapeInfo);
                }
                __syncthreads();

                if (xElementWiseStride >= 1) {
                    for (Nd4jLong i = tid; i < n; i += (blockDim.x * gridDim.x)) {
                        SummaryStatsData<X> indexVal2;
                        indexVal2.initWithValue(dx[i * xElementWiseStride]);
                        reduction = update(reduction, indexVal2, extraParams);
                    }
                }
                else {

                    for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x) {
                        
                        auto offset = shape::getIndexOffset(i, xShapeInfo, n);                        
                        SummaryStatsData<X> indexVal2;
                        indexVal2.initWithValue(dx[offset]);
                        reduction = update(reduction, indexVal2, extraParams);
                    }
                }
                sPartials[threadIdx.x] = reduction;

                __syncthreads();
                aggregatePartials<OpType>(&sPartials, threadIdx.x, blockDim.x, extraParams);
                __syncthreads();

                if (gridDim.x > 1) {
                    __shared__ bool amLast;
                    unsigned int *tc = (unsigned int *)reductionBuffer;                    
                    tid = threadIdx.x;
                    if (threadIdx.x == 0) {
                        SummaryStatsData<X> *pBuffer = (SummaryStatsData<X>*) reductionBuffer;
                        pBuffer[blockIdx.x] = sPartials[0];
                    }
                    __syncthreads();
                    __threadfence();

                    if (tid == 0) {
                        unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
                        amLast = (ticket == gridDim.x - 1);
                    }

                    __syncthreads();

                    if (amLast) {
                        tc[16384] = 0;
                        SummaryStatsData<X>* pBuffer = (SummaryStatsData<X>*) reductionBuffer;

                        Z startingVal = startingValue(dx);

                        SummaryStatsData<X> val;
                        val.initWithValue(startingVal);
                        val.n = 0;
                        sPartials[threadIdx.x] = val;

                        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
                            sPartials[threadIdx.x] = update(sPartials[threadIdx.x], pBuffer[i], extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, gridDim.x, extraParams);
                        __syncthreads();

                        if (tid == 0) {
                            z[0] = OpType::getValue(postProcessOrNot, sPartials[0]);
                        }
                    }
                }
                else {
                    if (tid == 0) {
                        unsigned int *tc = (unsigned *)reductionBuffer;
                        tc[16384] = 0;
                        z[0] = z[0] = OpType::getValue(postProcessOrNot, sPartials[0]);
                    }
                }
            }
        };


        template <typename X, typename Y>
        _CUDA_D void SummaryStatsReduce<X,Y>::transform(const int opNum, void *dx, Nd4jLong *xShapeInfo, void *extraParams, void *z, Nd4jLong *zShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, void *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            DISPATCH_BY_OPNUM_TT(transform, PARAMS(dx, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets), SUMMARY_STATS_OPS);
        };


        template <typename X, typename Z>
        _CUDA_H void SummaryStatsReduce<X,Z>::execSummaryStatsReduceScalar(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool biasCorrected, void *reductionBuffer) {
            
            auto x = static_cast<X*>(vx);
            auto extraParams = static_cast<Z*>(vextraParams);                                        
            auto z = reinterpret_cast<Z*>(vz);
            auto reductionPointerA = reinterpret_cast<Z*>(reductionBuffer);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("D16 opNum:[%i]\n", opNum);

            summaryStatsReduceT<X,Z><<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                            opNum,
                            x,
                            xShapeInfo, shape::rank(hxShapeInfo),
                            extraParams,
                            z,
                            zShapeInfo, shape::rank(hzShapeInfo),
                            nullptr,
                            1,
                            1,biasCorrected, nullptr, reductionPointerA, tadShapeInfo, tadOffsets);

            // this is blocking method since method should return scalar
            nd4j::DebugHelper::checkErrorCode(stream, "execSSReduceScalar(...) failed");
        }

        template <typename X, typename Z>
        _CUDA_H void SummaryStatsReduce<X,Z>::execSummaryStatsReduce(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool biasCorrected, void *reductionBuffer) {

            auto x = static_cast<X*>(vx);
            auto z = static_cast<Z*>(vz);
            auto extraParams = static_cast<Z*>(vextraParams);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("F17 opNum:[%i]\n", opNum);

            auto reductionPointerA = reinterpret_cast<Z*>(reductionBuffer);

            summaryStatsReduceT<X,Z><<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                            x,
                            xShapeInfo, shape::rank(hxShapeInfo),
                            extraParams,
                            z,
                            zShapeInfo, shape::rank(hzShapeInfo),
                            nullptr,
                            1,
                            1,biasCorrected, nullptr, reductionPointerA, tadShapeInfo, tadOffsets);

            DEBUG_KERNEL(stream, opNum);
        }


        template<typename X, typename Z>
        _CUDA_H void SummaryStatsReduce<X,Z>::execSummaryStatsReduce(dim3& launchDims, cudaStream_t *stream, int opNum, void *vx, Nd4jLong *xShapeInfo, Nd4jLong *hxShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, Nd4jLong *hzShapeInfo, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, bool biasCorrected, void *reductionBuffer) {

            auto x = static_cast<X*>(vx);
            auto z = static_cast<Z*>(vz);
            auto extraParams = static_cast<Z*>(vextraParams);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("D18 opNum:[%i]\n", opNum);

            summaryStatsReduceT<X, Z><<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                            x,
                            xShapeInfo, shape::rank(hxShapeInfo),
                            extraParams,
                            z,
                            zShapeInfo, shape::rank(hzShapeInfo),
                            dimension,
                            dimensionLength,
                            1, biasCorrected, nullptr, reinterpret_cast<Z*>(reductionBuffer), tadShapeInfo, tadOffsets);

            DEBUG_KERNEL(stream, opNum);
        }

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT SummaryStatsReduce, , LIBND4J_TYPES, FLOAT_TYPES);
    }
}