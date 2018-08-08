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
#include <types/float16.h>
#include <op_boilerplate.h>
#include <loops/summarystatsreduce.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>
#include <dll.h>
#include <Environment.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_launch_config.h>
#include <helpers/DebugHelper.h>



namespace functions {
    namespace summarystats {

        /**
 * The driver interface for summary stats
 * @param op the op number
 * @param n the length
 * @param dx the input
 * @param xShapeInfo the shape information for x
 * @param extraParams the extra parameters
 * @param result the result buffer
 * @param resultShapeInfo the shape information for the result
 * @param gpuInformation the gpu information such as block dim, grid dim and shared memory
 * @param dimension the dimension to execute along long
 * @param dimensionLength the length of the dimension
 * @param postProcessOrNot whether to post process or not
 */
        template <typename T>
        _CUDA_D void SummaryStatsReduce<T>::summaryStatsReduceGeneric(const int op, T *dx, Nd4jLong *xShapeInfo, int xRank, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected, int *allocationBuffer, T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

            __shared__ UnifiedSharedMemory *manager;

            if (threadIdx.x == 0) {
                extern __shared__ unsigned char shmem[];
                manager = new(shmem) UnifiedSharedMemory((int *) shmem);
                manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::summarystats::SummaryStatsReduce<T>), sizeof(shape::TAD), xRank);
            }
            __syncthreads();

            functions::summarystats::SummaryStatsReduce<T>::transform(
                    op,
                    dx,
                    xShapeInfo,
                    extraParams,
                    result,
                    resultShapeInfo,
                    dimension,
                    dimensionLength,
                    biasCorrected,
                    allocationBuffer,
                    reductionBuffer,
                    manager,
                    tadOnlyShapeInfo,
                    tadOffsets);
        }

        _CUDA_G void summaryStatsReduceDouble(int op, double *dx, Nd4jLong *xShapeInfo, int xRank, double *extraParams, double *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot, bool biasCorrected, int *allocationBuffer, double *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            SummaryStatsReduce<double>::summaryStatsReduceGeneric(
                    op,
                    dx,
                    xShapeInfo, xRank,
                    extraParams,
                    result,
                    resultShapeInfo, zRank,
                    dimension,
                    dimensionLength,
                    postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

        }

        _CUDA_G void summaryStatsReduceFloat(int op, float *dx, Nd4jLong *xShapeInfo, int xRank, float *extraParams, float *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected,int *allocationBuffer, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            SummaryStatsReduce<float>::summaryStatsReduceGeneric(
                    op,
                    dx,
                    xShapeInfo, xRank,
                    extraParams,
                    result,
                    resultShapeInfo, zRank,
                    dimension,
                    dimensionLength,
                    postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

        }

        _CUDA_G void summaryStatsReduceHalf(int op, float16 *dx, Nd4jLong *xShapeInfo, int xRank, float16 *extraParams, float16 *result, Nd4jLong *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected,int *allocationBuffer, float16 *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            SummaryStatsReduce<float16>::summaryStatsReduceGeneric(
                    op,
                    dx,
                    xShapeInfo, xRank,
                    extraParams,
                    result,
                    resultShapeInfo, zRank,
                    dimension,
                    dimensionLength,
                    postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

        }

        /*
        template <typename T>
        void __global__ SummaryStatsReduce<T>::summaryStatsReduceT(int op, T *dx, int *xShapeInfo, int xRank, T *extraParams, T *result, int *resultShapeInfo, int zRank, int *dimension, int dimensionLength, int postProcessOrNot,bool biasCorrected,int *allocationBuffer, T *reductionBuffer, int *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            summaryStatsReduceGeneric<T>(
                    op,
                    dx,
                    xShapeInfo, xRank,
                    extraParams,
                    result,
                    resultShapeInfo, zRank,
                    dimension,
                    dimensionLength,
                    postProcessOrNot,biasCorrected, allocationBuffer, reductionBuffer, tadOnlyShapeInfo, tadOffsets);

        }
        */


        /**
		 *
		 * @param sPartialsRef
		 * @param tid
		 * @param extraParams
		 */
        template<typename T>
        template<typename OpType>
        _CUDA_D void SummaryStatsReduce<T>::aggregatePartials(SummaryStatsData<T> **sPartialsRef, Nd4jLong tid, Nd4jLong numElements, T *extraParams) {
            // start the shared memory loop on the next power of 2 less
            // than the block size.  If block size is not a power of 2,
            // accumulate the intermediate sums in the remainder range.
            SummaryStatsData<T> *sPartials = *sPartialsRef;
            Nd4jLong floorPow2 = blockDim.x;

            if (floorPow2 & (floorPow2 - 1)) {
                while (floorPow2 & (floorPow2 - 1)) {
                    floorPow2 &= floorPow2 - 1;
                }

                if (tid >= floorPow2) {
                    SummaryStatsData<T> prev = sPartials[tid - floorPow2];
                    SummaryStatsData<T> curr = sPartials[tid];
                    sPartials[tid - floorPow2] = update(prev, curr, extraParams);
                }
                __syncthreads();
            }

            for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
                if (tid < activeThreads && tid + activeThreads < numElements) {
                    SummaryStatsData<T> curr = sPartials[tid];
                    SummaryStatsData<T> next = sPartials[tid + activeThreads];
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
        template<typename T>
        template<typename OpType>
        _CUDA_D void SummaryStatsReduce<T>::transform(T *dx, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {

            /**
             * Gpu information for the problem
             */
            int tid = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ volatile int resultScalar;

            __shared__ int xElementWiseStride;

            int numElements = blockDim.x;
            //shared memory space for storing intermediate results
            SummaryStatsData<T> *sPartials;
            //functions::summarystats::SharedSummaryStatsData<T> holder;

            sPartials = (SummaryStatsData<T> *) manager->getSharedReductionBuffer(); //holder.getPointer();
            T startingVal = startingValue(dx);

            SummaryStatsData<T> val;
            val.initWithValue(startingVal);
            val.n = 0;
            sPartials[threadIdx.x] = val;


            //length for the tad
            __shared__ volatile int xLength;

            __shared__ volatile int resultLength;


            SummaryStatsData <T> reduction;
            reduction.initWithValue(0.0);
            reduction.n = 0;
            if (threadIdx.x == 0) {
                if (resultShapeInfo != nullptr)
                    resultLength = shape::length(resultShapeInfo);
                else resultLength = 1;

                if (dimensionLength == 1) {
                    if (dimension == nullptr || dimension[0] == MAX_DIMENSION)
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
                __shared__ int tadRank;
                __shared__ int numTads;
                __shared__ Nd4jLong *tadShape;
                __shared__ Nd4jLong *tadStride;

                if (threadIdx.x == 0) {
                    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                    tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
                    tadRank = shape::rank(tadOnlyShapeInfo);
                    numTads = shape::length(xShapeInfo) / tadLength;

                    tadShape = shape::shapeOf(tadOnlyShapeInfo);
                    tadStride = shape::stride(tadOnlyShapeInfo);
                }
                __syncthreads();

                if (dimensionLength > 1) {
                    Nd4jLong xCoord[MAX_RANK];

                    for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
                        auto tadOffsetForBlock = tadOffsets[r];

                        val.initWithValue(startingVal);
                        val.n = 0;
                        sPartials[threadIdx.x] = val;

                        for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                            shape::ind2subC(tadRank, tadShape, i, tadLength, xCoord);
                            Nd4jLong xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

                            SummaryStatsData <T> indexVal2;
                            indexVal2.initWithValue(dx[xOffset]);

                            sPartials[threadIdx.x] = update(sPartials[threadIdx.x], OpType::op(indexVal2, extraParams), extraParams);
                        }
                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            result[r] = OpType::getValue(postProcessOrNot, sPartials[threadIdx.x]);
                        }

                    }
                }
                else {


                    for (int i = blockIdx.x; i < numTads; i += gridDim.x) {
                        auto tadOffsetForBlock = tadOffsets[i];

                        val.initWithValue(startingVal);
                        val.n = 0;
                        sPartials[threadIdx.x] = val;

                        auto indexX = tadOffsetForBlock + (xElementWiseStride * threadIdx.x);

                        if (threadIdx.x < tadLength) {
                            SummaryStatsData <T> indexVal;
                            indexVal.initWithValue(dx[indexX]);
                            sPartials[threadIdx.x] = OpType::op(indexVal, extraParams);
                        }

                        for (int x = threadIdx.x + blockDim.x; x < tadLength; x += blockDim.x) {
                            indexX = tadOffsetForBlock + x * tadEWS;
                            SummaryStatsData <T> indexVal2;
                            indexVal2.initWithValue(dx[indexX]);
                            sPartials[threadIdx.x] = update(sPartials[threadIdx.x], OpType::op(indexVal2, extraParams), extraParams);
                        }

                        __syncthreads();
                        aggregatePartials<OpType>(&sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

                        __syncthreads();
                        if (threadIdx.x == 0) {
                            result[i] = OpType::getValue(postProcessOrNot, sPartials[threadIdx.x]); //postProcess(sPartials[0],tadLength ,extraParams);
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
                        SummaryStatsData <T> indexVal2;
                        indexVal2.initWithValue(dx[i * xElementWiseStride]);
                        reduction = update(reduction, indexVal2, extraParams);
                    }
                }
                else {
                    __shared__ int rank;
                    __shared__ Nd4jLong *xShape;
                    __shared__ Nd4jLong *xStride;
                    if (threadIdx.x == 0) {
                        rank = shape::rank(xShapeInfo);
                        xShape = shape::shapeOf(xShapeInfo);
                        xStride = shape::stride(xShapeInfo);
                    }
                    __syncthreads();

                    Nd4jLong ind2sub[MAX_RANK];

                    for (Nd4jLong i = tid; i < n; i += blockDim.x * gridDim.x) {
                        shape::ind2sub(rank, shape::shapeOf(xShapeInfo), i, n, ind2sub);
                        auto offset = shape::getOffset(0, xShape, xStride, ind2sub, rank);

                        SummaryStatsData <T> indexVal2;
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
                    int rank = shape::rank(xShapeInfo);
                    tid = threadIdx.x;
                    if (threadIdx.x == 0) {
                        SummaryStatsData<T> *pBuffer = (SummaryStatsData<T> *) reductionBuffer;
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
                        SummaryStatsData<T> *pBuffer = (SummaryStatsData<T> *) reductionBuffer;

                        T startingVal = startingValue(dx);

                        SummaryStatsData<T> val;
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
                            result[0] = OpType::getValue(postProcessOrNot, sPartials[0]);
                        }
                    }
                }
                else {
                    if (tid == 0) {
                        unsigned int *tc = (unsigned *)reductionBuffer;
                        tc[16384] = 0;
                        result[0] = result[0] = OpType::getValue(postProcessOrNot, sPartials[0]);
                    }
                }
            }
        };


        template <typename T>
        _CUDA_D void SummaryStatsReduce<T>::transform(const int opNum, T *dx, Nd4jLong *xShapeInfo, T *extraParams, T *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength, int postProcessOrNot, int *allocationBuffer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets) {
            DISPATCH_BY_OPNUM(transform, PARAMS(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationBuffer, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets), SUMMARY_STATS_OPS);
        };


        template <>
        _CUDA_H double SummaryStatsReduce<double>::execSummaryStatsReduceScalar(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, Nd4jLong *xShapeInfo, double *extraParams, bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("D16 opNum:[%i]\n", opNum);

            double *resultPointer = reinterpret_cast<double *>(extraPointers[5]);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);


            functions::summarystats::summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                            x,
                            xShapeInfo, shape::rank(hostXShapeInfo),
                            extraParams,
                            resultPointer,
                            nullptr, 0,
                            nullptr,
                            1,
                            1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            // this is blocking method since method should return scalar
            nd4j::DebugHelper::checkErrorCode(stream, "execSSReduceScalarDouble(...) failed");

            double result = resultPointer[0];
            return result;
        }

        template <>
        _CUDA_H float SummaryStatsReduce<float>::execSummaryStatsReduceScalar(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *x, Nd4jLong *xShapeInfo, float *extraParams, bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("F16 opNum:[%i]\n", opNum);

            float *resultPointer = reinterpret_cast<float *>(extraPointers[5]);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

            functions::summarystats::summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z * 2, *stream>>>(
                    opNum,
                            x,
                            xShapeInfo, shape::rank(hostXShapeInfo),
                            extraParams,
                            resultPointer,
                            nullptr, 0,
                            nullptr,
                            1,
                            1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            // this is blocking method since method should return scalar
            nd4j::DebugHelper::checkErrorCode(stream, "execSSReduceScalarFloat(...) failed");

            double result = resultPointer[0];
            return result;
        }


        template <>
        _CUDA_H float16 SummaryStatsReduce<float16>::execSummaryStatsReduceScalar(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *x, Nd4jLong *xShapeInfo, float16 *extraParams, bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("H16 opNum:[%i]\n", opNum);

            float16 *resultPointer = reinterpret_cast<float16 *>(extraPointers[5]);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);


            functions::summarystats::summaryStatsReduceHalf<<<launchDims.x,launchDims.y,launchDims.z * 4, *stream>>>(
                    opNum,
                            x,
                            xShapeInfo, shape::rank(hostXShapeInfo),
                            extraParams,
                            resultPointer,
                            nullptr, 0,
                            nullptr,
                            1,
                            1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            // this is blocking method since method should return scalar
            nd4j::DebugHelper::checkErrorCode(stream, "execSSReduceScalarHalf(...) failed");

            double result = resultPointer[0];
            return result;
        }


        template <>
        _CUDA_H void SummaryStatsReduce<float>::execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *x, Nd4jLong *xShapeInfo, float *extraParams, float *result, Nd4jLong *resultShapeInfo,bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("F17 opNum:[%i]\n", opNum);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

            if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
                printf("AF17 opNum:[%i]\n", opNum);

            functions::summarystats::summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                            x,
                            xShapeInfo, shape::rank(hostXShapeInfo),
                            extraParams,
                            result,
                            resultShapeInfo, shape::rank(hostZShapeInfo),
                            nullptr,
                            1,
                            1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            DEBUG_KERNEL(stream, opNum);
        }


    template <>
    _CUDA_H void SummaryStatsReduce<float16>::execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *x, Nd4jLong *xShapeInfo, float16 *extraParams, float16 *result, Nd4jLong *resultShapeInfo,bool biasCorrected) {
        cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

        auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
        auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

        auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
        auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
        auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

        if (nd4j::Environment::getInstance()->isDebugAndVerbose())
            printf("H17 opNum:[%i]\n", opNum);

        int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
        float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

        if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
            printf("AH17 opNum:[%i]\n", opNum);

        functions::summarystats::summaryStatsReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                opNum,
                        x,
                        xShapeInfo, shape::rank(hostXShapeInfo),
                        extraParams,
                        result,
                        resultShapeInfo, shape::rank(hostZShapeInfo),
                        nullptr,
                        1,
                        1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

        DEBUG_KERNEL(stream, opNum);
    }

        template <>
        _CUDA_H void SummaryStatsReduce<double>::execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, Nd4jLong *xShapeInfo, double *extraParams, double *result, Nd4jLong *resultShapeInfo,bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);
            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("D17 opNum:[%i]\n", opNum);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

            if (nd4j::Environment::getInstance()->isVerbose() && launchDims.x == 1)
                printf("AD17 opNum:[%i]\n", opNum);

            functions::summarystats::summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                opNum,
                        x,
                        xShapeInfo, shape::rank(hostXShapeInfo),
                        extraParams,
                        result,
                        resultShapeInfo, shape::rank(hostZShapeInfo),
                        nullptr,
                        1,
                        1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            DEBUG_KERNEL(stream, opNum);
        }




        template <>
        _CUDA_H void SummaryStatsReduce<double>::execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, double *x, Nd4jLong *xShapeInfo, double *extraParams, double *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength,bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("D18 opNum:[%i]\n", opNum);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            double *reductionPointer = reinterpret_cast<double *>(extraPointers[4]);

            functions::summarystats::summaryStatsReduceDouble<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                            x,
                            xShapeInfo, shape::rank(hostXShapeInfo),
                            extraParams,
                            result,
                            resultShapeInfo, shape::rank(hostZShapeInfo),
                            dimension,
                            dimensionLength,
                            1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            DEBUG_KERNEL(stream, opNum);
        }


        template <>
        _CUDA_H void SummaryStatsReduce<float>::execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float *x, Nd4jLong *xShapeInfo, float *extraParams, float *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength,bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("F18 opNum:[%i]\n", opNum);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            float *reductionPointer = reinterpret_cast<float *>(extraPointers[4]);

            // we need shmem buffer big enough to hold double values
            launchDims.z *= 2;

            functions::summarystats::summaryStatsReduceFloat<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                        x,
                        xShapeInfo, shape::rank(hostXShapeInfo),
                        extraParams,
                        result,
                        resultShapeInfo, shape::rank(hostZShapeInfo),
                        dimension,
                        dimensionLength,
                        1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            DEBUG_KERNEL(stream, opNum);
        }


        template <>
        _CUDA_H void SummaryStatsReduce<float16>::execSummaryStatsReduce(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, float16 *x, Nd4jLong *xShapeInfo, float16 *extraParams, float16 *result, Nd4jLong *resultShapeInfo, int *dimension, int dimensionLength,bool biasCorrected) {
            cudaStream_t *stream = reinterpret_cast<cudaStream_t *>(&extraPointers[1]);

            auto hostXShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[0]);
            auto hostZShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[8]);

            auto hostTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[9]);
            auto deviceTADShapeInfo = reinterpret_cast<Nd4jLong *>(extraPointers[10]);

            auto deviceTADOffsets = reinterpret_cast<Nd4jLong *>(extraPointers[11]);

            if (nd4j::Environment::getInstance()->isDebugAndVerbose())
                printf("H18 opNum:[%i]\n", opNum);

            int *allocationPointer = reinterpret_cast<int *>(extraPointers[3]);
            float16 *reductionPointer = reinterpret_cast<float16 *>(extraPointers[4]);

            // we need shmem buffer big enough to hold double values
            launchDims.z *= 4;

            functions::summarystats::summaryStatsReduceHalf<<<launchDims.x,launchDims.y,launchDims.z, *stream>>>(
                    opNum,
                        x,
                        xShapeInfo, shape::rank(hostXShapeInfo),
                        extraParams,
                        result,
                        resultShapeInfo, shape::rank(hostZShapeInfo),
                        dimension,
                        dimensionLength,
                        1,biasCorrected, allocationPointer, reductionPointer, deviceTADShapeInfo, deviceTADOffsets);

            DEBUG_KERNEL(stream, opNum);
        }


        template class ND4J_EXPORT SummaryStatsReduce<float>;
        template class ND4J_EXPORT SummaryStatsReduce<float16>;
        template class ND4J_EXPORT SummaryStatsReduce<double>;
        template class ND4J_EXPORT SummaryStatsReduce<int>;
        template class ND4J_EXPORT SummaryStatsReduce<Nd4jLong>;
    }
}