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
 * reduce3.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef REDUCE3_H_
#define REDUCE3_H_

#define EXTRA_PARAMS_LENGTH 10

#include <templatemath.h>
#include <helper_cuda.h>
#include <helpers/sharedmem.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pairwise_util.h>
#include <dll.h>
#include <helpers/shape.h>
#include <ops/ops.h>
#include <op_boilerplate.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include "legacy_ops.h"

namespace functions {
    namespace reduce3 {

/**
 * Reduce involving
 * 2 arrays
 */
        template<typename X, typename Y>
        class Reduce3 {

        public:
#ifdef __CUDACC__
            virtual __device__

			inline T opAtomic(T d1, T d2, T *extraParamsRef) = 0;
#endif

#ifdef __CUDACC__
            /**
     * Aggregate shared memory
     * @param sPartialsRef
     * @param tid
     * @param extraParams
     */
template<typename OpType>
			static __inline__ __device__ void aggregatePartials(T **sPartialsRef, Nd4jLong tid, Nd4jLong numItems, T *extraParamsRef) {
				// start the shared memory loop on the next power of 2 less
				// than the block size.  If block size is not a power of 2,
				// accumulate the intermediate sums in the remainder range.
				T *sPartials = *sPartialsRef;
				Nd4jLong floorPow2 = numItems;

				if (floorPow2 & (floorPow2 - 1)) {
					while (floorPow2 & (floorPow2 - 1)) {
						floorPow2 &= floorPow2 - 1;
					}
					if (tid >= floorPow2) {
						sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParamsRef);
					}
					__syncthreads();
				}

				for (Nd4jLong activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
					if (tid < activeThreads) {
						sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParamsRef);
					}
					__syncthreads();
				}
			}
			/**

                     Perform a reduction
                     @param n the number of elements
                     @param xOffset the starting offset
                     @param dx the data to perform the reduction on
                     @param incx the increment on which to perform the reduction
                     @param extraParams extra parameters used for calculations
                     @param result where to store the result of the reduction
             */
			virtual __inline__ __device__ void transformNoElementWiseStride(
					T *dx,
					Nd4jLong *xShapeInfo,
					T *dy,
					Nd4jLong *yShapeInfo,
					T *extraParams,
					T *result,
					Nd4jLong *resultShapeInfo,
					int postProcessOrNot, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo) {
				Nd4jLong n = shape::length(xShapeInfo);
				int rank = shape::rank(xShapeInfo);

				T *sPartials = (T *) manager->getSharedReductionBuffer(); //val.getPointer();
				T startingVal = this->startingValue(dx);

				// FIXME: this ugly fast fix.
				__shared__ T extraZ[2];
				if (threadIdx.x == 0) {
					extraZ[0] = (T) 0.0;
					extraZ[1] = (T) 0.0;
				}
				sPartials[threadIdx.x] = startingVal;
				__syncthreads();

                Nd4jLong idx[MAX_RANK];

				for(Nd4jLong i = blockIdx.x * gridDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
					shape::ind2subC(rank,shape::shapeOf(xShapeInfo),i, idx);
					auto offset = shape::getOffset(0,shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),idx,rank);
					auto yOffset = shape::getOffset(0,shape::shapeOf(yShapeInfo),shape::stride(yShapeInfo),idx,rank);
					
					sPartials[threadIdx.x] = update(sPartials[threadIdx.x], this->opAtomic(dx[offset], dy[yOffset], extraZ), extraZ);
				}

				T **sPartialsRef = (T **) &sPartials;
				aggregatePartials(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, n), extraZ);
				/**
                 * Look at something that uses the extra params
                 * and aggregates the extra values propelry.
                 *This will be used in summary stats too.
                 */
				// write result for this block to global mem
				if (threadIdx.x == 0) {
					if (postProcessOrNot) {
						result[blockIdx.x] = postProcess(sPartials[0], n, extraZ);
					}
					else {
						result[blockIdx.x] = sPartials[0];
					}


				}
			}


			/**
             *
             */
template<typename OpType>
			static inline __device__ void execScalarCuda(
					T *dx,
					Nd4jLong *xShapeInfo,
					T *dy,
					Nd4jLong *yShapeInfo,
					T *extraParams,
					T *result,
					Nd4jLong *resultShapeInfo, int *allocationPointer, T *reductionBuffer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo) {


//		SharedMemory <T> val;
				T *sPartials = (T *) manager->getSharedReductionBuffer(); // val.getPointer();

				// FIXME: this ugly fast fix.
				__shared__ T extraZ[3];
				if (threadIdx.x == 0) {
					extraZ[0] = (T) 0.0f;
					extraZ[1] = (T) 0.0f;

					if (extraParams != NULL) {
                        extraZ[2] = extraParams[0];
                    } else extraZ[2] = (T) 0.0f;
				}

				__syncthreads();

				T startingVal = OpType::startingValue(dx);
				Nd4jLong length = shape::length(xShapeInfo);
				int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
				int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
				int tid = blockIdx.x * blockDim.x + threadIdx.x;
				char xOrder = shape::order(xShapeInfo);
				char yOrder = shape::order(yShapeInfo);

				if(xOrder == yOrder && (xElementWiseStride > 0 && yElementWiseStride > 0) && shape::strideDescendingCAscendingF(xShapeInfo) && shape::strideDescendingCAscendingF(yShapeInfo)) {
					if (xElementWiseStride == 1 && yElementWiseStride == 1) {
						for(Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
							startingVal = OpType::update(startingVal, OpType::opAtomic(dx[i], dy[i], extraZ), extraZ);
						}
					}
					else {
						for(Nd4jLong i = tid; i < length; i+= gridDim.x * blockDim.x) {
							startingVal = OpType::update(startingVal, OpType::opAtomic(dx[i * xElementWiseStride], dy[i * yElementWiseStride], extraZ), extraZ);
						}
					}

					sPartials[threadIdx.x] = startingVal;
				} else {
				    __shared__ Nd4jLong *xShape;
				    __shared__ Nd4jLong *yShape;
				    __shared__ Nd4jLong *xStride;
				    __shared__ Nd4jLong *yStride;
				    __shared__ int rank;
				    if (threadIdx.x == 0) {

					    xShape = shape::shapeOf(xShapeInfo);
					    yShape = shape::shapeOf(yShapeInfo);
					    xStride = shape::stride(xShapeInfo);
					    yStride = shape::stride(yShapeInfo);
					    rank = shape::rank(xShapeInfo);
					}
					__syncthreads();
					T startingVal = OpType::startingValue(dx);

					T *sPartials = (T *) manager->getSharedReductionBuffer();

					Nd4jLong xCoords[MAX_RANK];
					Nd4jLong yCoords[MAX_RANK];

					sPartials[threadIdx.x] = startingVal;

					for(Nd4jLong i = tid ;i < length; i += gridDim.x * blockDim.x) {
						shape::ind2subC(rank,xShape,i,xCoords);
						shape::ind2subC(rank,yShape,i,yCoords);

						auto offset = shape::getOffset(0, xShape, xStride, xCoords,rank);
						auto yOffset = shape::getOffset(0,yShape, yStride, yCoords,rank);

						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[offset], dy[yOffset], extraZ), extraZ);
					}
				}

				__syncthreads();

				T **sPartialsRef = (T **) &sPartials;
				aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, length), extraZ);

				__syncthreads();
				if (gridDim.x > 1) {
                    unsigned int *tc = (unsigned int *)reductionBuffer;
					__shared__ bool amLast;
					int rank = shape::rank(xShapeInfo);
					tid = threadIdx.x;
					T *extraBuffer = (T *) allocationPointer;
					if (threadIdx.x == 0) {
						reductionBuffer[blockIdx.x] = sPartials[0];
						extraBuffer[blockIdx.x] = extraZ[0];
						extraBuffer[gridDim.x + blockIdx.x] = extraZ[1];
					}
					__threadfence();
					__syncthreads();

					if (threadIdx.x == 0) {
						unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
						amLast = (ticket == gridDim.x - 1);
					}

                    sPartials[tid] = startingVal;
					__syncthreads();

					if (amLast) {
						tc[16384] = 0;

						sPartials[threadIdx.x] = OpType::startingValue(dx);

                        // TODO: later probably replace this. Right now we need extraZ sync for CosineSimilarity ONLY
						if (tid == 0 && extraZ[0] != (T) 0.0 && extraZ[1] != (T) 0.0) {
						    extraZ[0] = 0.0;
						    extraZ[1] = 0.0;
                            for (int i = 0; i < gridDim.x; i++) {
                                extraZ[0] += extraBuffer[i];
                                extraZ[1] += extraBuffer[gridDim.x + i];
                            }
						}

						for (Nd4jLong i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
							sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], reductionBuffer[i], extraZ);
						}
						__syncthreads();

						aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraZ);

						__syncthreads();
						if (threadIdx.x == 0) {
							result[0] = OpType::postProcess(sPartials[0], length, extraZ);
						}
					}
				} else {
					if (tid == 0) {
					    unsigned int *tc = (unsigned *)reductionBuffer;
					    tc[16384] = 0;

						result[0] = OpType::postProcess(sPartials[0], length, extraZ);
					}
				}


			}



			template<typename OpType>
			__device__
			static inline void transformAll(
					T *dx,
					Nd4jLong *xShapeInfo,
					T *dy,
					Nd4jLong *yShapeInfo,
					T *extraParams,
					T *result,
					Nd4jLong *resultShapeInfo,
					int *dimension,
					int dimensionLength,
					int postProcessOrNot,
					int *allocationPointer,
					UnifiedSharedMemory *manager,
					Nd4jLong *xTadShapeInfo,
					Nd4jLong *xOffsets,
					Nd4jLong *yTadShapeInfo,
					Nd4jLong *yOffsets) {

                        // initialize partials first
                        T *sPartials = (T *) manager->getSharedReductionBuffer();
                        T startingVal = OpType::startingValue(dx);
				        sPartials[threadIdx.x] = startingVal;
				        T *tempX = sPartials + blockDim.x;

                        const int maxBlock = blockDim.x;

				        __shared__ T extraZ[OpType::extraParamsLen > 0 ? OpType::extraParamsLen : 1];

                        __shared__ int xTadLength;
                        __shared__ int yTadLength;

                        __shared__ int xTads;
                        __shared__ int yTads;

                        __shared__ Nd4jLong *xShape;
                        __shared__ Nd4jLong *xStride;
                        __shared__ int xRank;

                        __shared__ Nd4jLong *yShape;
                        __shared__ Nd4jLong *yStride;
                        __shared__ int yRank;

                        //reading initial data
                        if (threadIdx.x == 0) {
				            xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                            yTadLength = shape::tadLength(yShapeInfo, dimension, dimensionLength);

                            xTads = shape::length(xShapeInfo) / xTadLength;
                            yTads = shape::length(yShapeInfo) / yTadLength;

                            xShape = shape::shapeOf(xTadShapeInfo);
                            xStride = shape::stride(xTadShapeInfo);
                            xRank = shape::rank(xTadShapeInfo);

                            yShape = shape::shapeOf(yTadShapeInfo);
                            yStride = shape::stride(yTadShapeInfo);
                            yRank = shape::rank(yTadShapeInfo);
                        }
                        __syncthreads();


                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong yCoord[MAX_RANK];


                        int limit = xTadLength / maxBlock;
				        if (xTadLength % maxBlock > 0)
				            limit++;


                        for (int r = blockIdx.x; r < xTads; r += blockDim.x * gridDim.x) {
                            T *x = dx + xOffsets[r];

                            if (threadIdx.x < xTadLength && threadIdx.x < maxBlock) {
                                if (shape::order(xTadShapeInfo) == 'c') {
                                    shape::ind2subC(xRank, xShape, threadIdx.x, xCoord);
                                } else {
                                    shape::ind2sub(xRank, xShape, threadIdx.x, xCoord);
                                }

                                auto xO = shape::getOffset(0, xShape, xStride, xCoord, xRank);

                                tempX[threadIdx.x] = x[xO];
                            }

                            for (int g = 0; g < yTads; g++) {
                                T *y = dy + yOffsets[g];

                                int ri = (r * yTads) + g;

                                sPartials[threadIdx.x] = startingVal;
                                if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
					                extraZ[threadIdx.x] = (T) startingVal;
				                }
				                __syncthreads();

                                // we might have data too large for single cache block, rendering cache useless though :(
                                for (int t = 0; t < limit; t++) {

                                    // we reset tempX IF we have >1 tiles
                                    if (t >= 1 || (limit > 1 && g > 0))
                                        if (threadIdx.x + (t * maxBlock) < xTadLength) {
                                            if (shape::order(xTadShapeInfo) == 'c') {
                                                shape::ind2subC(xRank, xShape, threadIdx.x + (t * maxBlock), xCoord);
                                            } else {
                                                shape::ind2sub(xRank, xShape, threadIdx.x + (t * maxBlock), xCoord);
                                            }

                                            Nd4jLong xO = shape::getOffset(0, xShape, xStride, xCoord, xRank);

                                            tempX[threadIdx.x] = x[xO];
            //                                tempX[threadIdx.x] = x[threadIdx.x + (t * maxBlock)];
                                        }

                                    for (int f = threadIdx.x + (t * maxBlock); f < xTadLength && f < threadIdx.x + ((t + 1) * maxBlock); f += blockDim.x * gridDim.x) {
                                        if (shape::order(yTadShapeInfo) == 'c') {
                                            shape::ind2subC(yRank, yShape, f, yCoord);
                                        } else {
                                            shape::ind2sub(yRank, yShape, f, yCoord);
                                        }

                                        Nd4jLong yO = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                                        sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::opAtomic(tempX[threadIdx.x], y[yO], extraZ), extraZ);
                                    }

                                    // we MUST step through this block altogether
							        __syncthreads();
                                }

                                T **sPartialsRef = (T **) &sPartials;
				                aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, xTadLength), extraZ);

                                __syncthreads();

                                if (threadIdx.x == 0) {
							        result[ri] = OpType::postProcess(sPartials[threadIdx.x],xTadLength, extraZ);
							    }

							    __syncthreads();
                            }
                        }
					}

			/**
                     Perform a reduction
                     @param n the number of elements
                     @param xOffset the starting offset
                     @param dx the data to perform the reduction on
                     @param incx the increment on which to perform the reduction
                     @param extraParams extra parameters used for calculations
                     @param result where to store the result of the reduction
             */
template<typename OpType>
			__device__
			static inline void transform(
					T *dx,
					Nd4jLong *xShapeInfo,
					T *dy,
					Nd4jLong *yShapeInfo,
					T *extraParams,
					T *result,
					Nd4jLong *resultShapeInfo,
					int *dimension,
					int dimensionLength,
					int postProcessOrNot,
					int *allocationPointer,
					UnifiedSharedMemory *manager,
					Nd4jLong *tadOnlyShapeInfo,
					Nd4jLong *tadOffsets,
					Nd4jLong *yTadOnlyShapeInfo,
					Nd4jLong *yTadOffsets) {
				/**
                 * Gpu information for the problem
                 */
				int tid = threadIdx.x + blockIdx.x * blockDim.x;

				__shared__ int resultScalar;

				__shared__ int xElementWiseStride;
				__shared__ int yElementWiseStride;
				//shared memory space for storing intermediate results
				//SharedMemory <T> val;
				T *sPartials = (T *) manager->getSharedReductionBuffer(); //val.getPointer();
				T init = OpType::startingValue(dx);
				sPartials[threadIdx.x] = init;

				__shared__ T extraZ[OpType::extraParamsLen > 0 ? OpType::extraParamsLen : 1];

				//length for the tad

				__shared__ Nd4jLong resultLength;
				__shared__ int tadLength;
				__shared__ int yLength;
				__shared__ int tadElementWiseStride;
				__shared__ int yTadElementWiseStride;

			    T startingVal = OpType::startingValue(dx);

				T reduction = OpType::startingValue(dx);
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
					char xOrder = shape::order(xShapeInfo);

					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadElementWiseStride = shape::elementWiseStride(tadOnlyShapeInfo);
					yLength = shape::length(yShapeInfo);

					if (yTadOnlyShapeInfo != nullptr)
					    yTadElementWiseStride = shape::elementWiseStride(yTadOnlyShapeInfo);
				}
				__syncthreads();

                // code branch for TAD vs full array
                if (tadLength == yLength) {
                    Nd4jLong xCoord[MAX_RANK];
                    Nd4jLong yCoord[MAX_RANK];

                    auto yShape = shape::shapeOf(yShapeInfo);
					auto yStride = shape::stride(yShapeInfo);
					auto xShape = shape::shapeOf(tadOnlyShapeInfo);
					auto xStride = shape::stride(tadOnlyShapeInfo);
					int yRank = shape::rank(yShapeInfo);
					int xRank = shape::rank(tadOnlyShapeInfo);


					for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
					    int xOffsetForTad = tadOffsets[i];

					    if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
					            extraZ[threadIdx.x] = (T) startingVal;
				        }
				        __syncthreads();

						for(int j = threadIdx.x; j < tadLength; j += blockDim.x) {
                            shape::ind2subC(xRank,xShape, j, xCoord);
                            shape::ind2subC(yRank,yShape, j, yCoord);

                            Nd4jLong xOffset = shape::getOffset(xOffsetForTad, xShape, xStride, xCoord, xRank);
                            Nd4jLong yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);

							sPartials[threadIdx.x] =  j < blockDim.x ? OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ) : OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ), extraZ);
						}
						__syncthreads();

                        T **sPartialsRef = (T **) &sPartials;
				        aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                        __syncthreads();
                        if (threadIdx.x == 0)
						    result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);

						__syncthreads();
					}
                } else  if (!resultScalar) {
					if(tadElementWiseStride >= 1 && yTadElementWiseStride) {
    					for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
							int xOffsetForTad = tadOffsets[i];
							int yOffsetForTad = yTadOffsets[i];

							if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
					            extraZ[threadIdx.x] = (T) startingVal;
				            }
				            __syncthreads();

                            if (threadIdx.x < tadLength)
							    sPartials[threadIdx.x] =  OpType::op(dx[xOffsetForTad + tadElementWiseStride * threadIdx.x],dy[yOffsetForTad + yTadElementWiseStride * threadIdx.x], extraZ);

							for(int j = threadIdx.x + blockDim.x; j < tadLength; j += blockDim.x) {
								sPartials[threadIdx.x] =  OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffsetForTad + tadElementWiseStride * j],dy[yOffsetForTad + yTadElementWiseStride * j], extraZ), extraZ);
							}
							__syncthreads();

                            T **sPartialsRef = (T **) &sPartials;
				            aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                            __syncthreads();
                            if (threadIdx.x == 0)
							    result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);

							__syncthreads();
						}
					}
					else {
/*
						// DO NOT REMOVE THIS COMMENTED BLOCK PLEASE

						for (int r = blockIdx.x; r < tad->numTads; r += gridDim.x) {
                            if (threadIdx.x == 0)
                                tad->createOffsetForBlock(r);
                            __syncthreads();

                            int tadOffsetForBlock = tad->tadOffsetForBlock;
                            T *xVal = dx + tadOffsetForBlock;


                            sPartials[threadIdx.x] = this->startingValue(xVal);
                            for(int i = threadIdx.x; i < tad->tadLength; i+= blockDim.x) {
                    			int xOffsetForTad = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength, nullptr);
								int yOffsetForTad = shape::tadOffset(i, yShapeInfo, dimension, dimensionLength, nullptr);

                                sPartials[threadIdx.x] = this->update(sPartials[threadIdx.x],dx[tadOffsetForBlock + i *  tad->tadElementWiseStride], extraParams);
                            }
                            __syncthreads();

                            // aggregate. do NOT reduce for elements > tadLength
                            T **sPartialsRef = (T **) &sPartials;
                            aggregatePartials(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tad->tadLength), extraParams);


                            __syncthreads();
                            if (threadIdx.x == 0)
                                result[r] = this->postProcess(sPartials[threadIdx.x], tad->tadLength, extraParams);
                        }

*/

                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong yCoord[MAX_RANK];

                        auto yShape = shape::shapeOf(yTadOnlyShapeInfo);
						auto yStride = shape::stride(yTadOnlyShapeInfo);
						auto xShape = shape::shapeOf(tadOnlyShapeInfo);
						auto xStride = shape::stride(tadOnlyShapeInfo);
						int yRank = shape::rank(yTadOnlyShapeInfo);
						int xRank = shape::rank(tadOnlyShapeInfo);


						for(int i = blockIdx.x; i < resultLength; i+= gridDim.x) {
							auto xOffsetForTad = tadOffsets[i];
							auto yOffsetForTad = yTadOffsets[i];

							if (OpType::extraParamsLen > 0 && threadIdx.x < OpType::extraParamsLen) {
					            extraZ[threadIdx.x] = (T) startingVal;
				            }
				            __syncthreads();

							for(int j = threadIdx.x; j < tadLength; j += blockDim.x) {
                                shape::ind2subC(xRank,xShape, j, xCoord);
                                shape::ind2subC(yRank,yShape, j, yCoord);

                                auto xOffset = shape::getOffset(xOffsetForTad, xShape, xStride, xCoord, xRank);
                                auto yOffset = shape::getOffset(yOffsetForTad, yShape, yStride, yCoord, yRank);

								sPartials[threadIdx.x] =  j < blockDim.x ? OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ) : OpType::update(sPartials[threadIdx.x], OpType::opAtomic(dx[xOffset],dy[yOffset], extraZ), extraZ);
							}
							__syncthreads();

                            T **sPartialsRef = (T **) &sPartials;
				            aggregatePartials<OpType>(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraZ);

                            __syncthreads();
                            if (threadIdx.x == 0)
							    result[i] = OpType::postProcess(sPartials[threadIdx.x],tadLength, extraZ);

							__syncthreads();
						}

					}
				}
			}





#endif

#ifdef __CUDACC__
            __device__
			static inline void exec(
				const int opNum,
				T *dx,
				Nd4jLong *xShapeInfo,
				T *dy,
				Nd4jLong *yShapeInfo,
				T *extraParams,
				T *result,
				Nd4jLong *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				int postProcessOrNot,
				int *allocationPointer,
				UnifiedSharedMemory *manager,
				Nd4jLong *tadOnlyShapeInfo,
				Nd4jLong *tadOffsets,
				Nd4jLong *yTadOnlyShapeInfo,
				Nd4jLong *yTadOffsets) {
                            DISPATCH_BY_OPNUM_T(transform, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets), REDUCE3_OPS);
			}

            __device__
			static inline void execAllCuda(
				const int opNum,
				T *dx,
				Nd4jLong *xShapeInfo,
				T *dy,
				Nd4jLong *yShapeInfo,
				T *extraParams,
				T *result,
				Nd4jLong *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				int postProcessOrNot,
				int *allocationPointer,
				UnifiedSharedMemory *manager,
				Nd4jLong *tadOnlyShapeInfo,
				Nd4jLong *tadOffsets,
				Nd4jLong *yTadOnlyShapeInfo,
				Nd4jLong *yTadOffsets) {
                            DISPATCH_BY_OPNUM_T(transformAll, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets), REDUCE3_OPS);
			}


			__device__
			static inline void execScalarCuda(
				const int opNum,
				T *dx,
				Nd4jLong *xShapeInfo,
				T *dy,
				Nd4jLong *yShapeInfo,
				T *extraParams,
				T *result,
				Nd4jLong *resultShapeInfo,
				int * allocationPointer,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				Nd4jLong *tadOnlyShapeInfo) {
                            DISPATCH_BY_OPNUM_T(execScalarCuda, PARAMS(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, allocationPointer, reductionBuffer, manager, tadOnlyShapeInfo), REDUCE3_OPS);
			}
#endif


#ifdef __CUDACC__
            __host__
#endif

            static double execScalar(
                    const int opNum,
                    void *x,
                    Nd4jLong *xShapeInfo,
                    void *extraParamsVals,
                    void *y,
                    Nd4jLong *yShapeInfo) {
                RETURNING_DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(x,
                                                               xShapeInfo,
                                                               extraParamsVals,
                                                               y,
                                                               yShapeInfo), REDUCE3_OPS);
            }

            static void exec( const int opNum,
                              void *x,
                              Nd4jLong *xShapeInfo,
                              void *extraParamsVals,
                              void *y,
                              Nd4jLong *yShapeInfo,
                              void *result,
                              Nd4jLong *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength) {
                DISPATCH_BY_OPNUM_TT(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParamsVals,
                                               y, yShapeInfo,
                                               result,
                                               resultShapeInfoBuffer,
                                               dimension,
                                               dimensionLength), REDUCE3_OPS);
            }


            static void exec( const int opNum,
                              void *x,
                              Nd4jLong *xShapeInfo,
                              void *extraParamsVals,
                              void *y,
                              Nd4jLong *yShapeInfo,
                              void *result,
                              Nd4jLong *resultShapeInfoBuffer,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *tadShapeInfo,
                              Nd4jLong *tadOffsets) {
                DISPATCH_BY_OPNUM_TT(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParamsVals,
                                               y, yShapeInfo,
                                               result,
                                               resultShapeInfoBuffer,
                                               dimension,
                                               dimensionLength, tadShapeInfo, tadOffsets), REDUCE3_OPS);
            }

            static void execAll( const int opNum,
                                 void *x,
                                 Nd4jLong *xShapeInfo,
                                 void *extraParamsVals,
                                 void *y,
                                 Nd4jLong *yShapeInfo,
                                 void *result,
                                 Nd4jLong *resultShapeInfoBuffer,
                                 int *dimension,
                                 int dimensionLength,
                                 Nd4jLong *xTadShapeInfo,
                                 Nd4jLong *xOffsets,
                                 Nd4jLong *yTadShapeInfo,
                                 Nd4jLong *yOffsets) {
                DISPATCH_BY_OPNUM_TT(execAll, PARAMS(x,
                                                  xShapeInfo,
                                                  extraParamsVals,
                                                  y, yShapeInfo,
                                                  result,
                                                  resultShapeInfoBuffer,
                                                  dimension,
                                                  dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets), REDUCE3_OPS);
            }



            template<typename OpType>
#ifdef __CUDACC__
            __host__
#endif
            static double execScalar(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vextraParams,
                    void *vy,
                    Nd4jLong *yShapeInfo) {
                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<Y *>(vy);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                auto startingVal = OpType::startingValue(x);
                auto length = shape::length(xShapeInfo);
                auto xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                auto yElementWiseStride = shape::elementWiseStride(yShapeInfo);

                X extraParamsVals[3] = {(X) 0.0f, (X) 0.0f, (X) 0.0f};
                // it's possible case for EqualsWithEps op
                if (extraParams != nullptr) {
                    extraParamsVals[2] = extraParams[0];
                }


                auto xOrder = shape::order(xShapeInfo);
                auto yOrder = shape::order(yShapeInfo);
                if(xOrder == yOrder && (xElementWiseStride  >=1 && yElementWiseStride >= 1) && shape::strideDescendingCAscendingF(xShapeInfo) && shape::strideDescendingCAscendingF(yShapeInfo)) {
                    if (xElementWiseStride == 1 && yElementWiseStride == 1) {

// TODO:: proper reduction required here
                        for(int i = 0; i < length; i++) {
                            startingVal = OpType::update(startingVal,
                                                         OpType::op(x[i],y[i],
                                                                    extraParamsVals),
                                                         extraParamsVals);
                        }

                        return  OpType::postProcess(startingVal, length, extraParamsVals);

                    }

                    else {
// TODO:: proper reduction required here
                        for(Nd4jLong i = 0; i < length; i++) {
                            startingVal = OpType::update(startingVal, OpType::op(x[i * xElementWiseStride],y[i * yElementWiseStride], extraParamsVals), extraParamsVals);
                        }

                        return  OpType::postProcess(startingVal, length, extraParamsVals);
                    }

                }


                else {
                    Nd4jLong xCoords[MAX_RANK];
                    Nd4jLong yCoords[MAX_RANK];

                    int xRank = shape::rank(xShapeInfo);
                    int yRank = shape::rank(yShapeInfo);

                    auto xShape = shape::shapeOf(xShapeInfo);
                    auto xStride = shape::stride(xShapeInfo);
                    auto yShape = shape::shapeOf(yShapeInfo);
                    auto yStride = shape::stride(yShapeInfo);

                    for(unsigned int i = 0 ;i < length; i++) {
                        shape::ind2subC(xRank, xShape, i, xCoords);
                        shape::ind2subC(yRank, yShape, i, yCoords);

                        auto offset = shape::getOffset(0, xShape, xStride, xCoords, xRank);
                        auto yOffset = shape::getOffset(0, yShape, yStride, yCoords, yRank);

                        startingVal = OpType::update(startingVal, OpType::op(x[offset], y[yOffset], extraParamsVals), extraParamsVals);
                    }
                }

                return OpType::postProcess(startingVal, length, extraParamsVals);;


            }


            template<typename OpType>
            static void execAll(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vextraParams,
                    void *vy,
                    Nd4jLong *yShapeInfo,
                    void *vresult,
                    Nd4jLong *resultShapeInfoBuffer,
                    int *dimension,
                    int dimensionLength, Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<Y *>(vy);
                auto result = reinterpret_cast<X *>(vresult);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                auto xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                auto yTadLength = shape::tadLength(yShapeInfo, dimension, dimensionLength);

                auto xTads = shape::length(xShapeInfo) / xTadLength;
                auto yTads = shape::length(yShapeInfo) / yTadLength;

                auto xShape = shape::shapeOf(xTadShapeInfo);
                auto xStride = shape::stride(xTadShapeInfo);
                int xRank = shape::rank(xTadShapeInfo);

                auto yShape = shape::shapeOf(yTadShapeInfo);
                auto yStride = shape::stride(yTadShapeInfo);
                int yRank = shape::rank(yTadShapeInfo);


                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

                auto startingVal = OpType::startingValue(x);

#pragma  omp parallel for proc_bind(AFFINITY) default(shared) private(xCoord, yCoord)
                for (Nd4jLong r = 0; r < xTads; r++) {
                    Nd4jLong xOffset = xOffsets[r];

                    auto lX = x + xOffset;

                    for (Nd4jLong g = 0; g < yTads; g++) {
                        auto yOffset = yOffsets[g];
                        auto lY = y + yOffset;

                        auto ri = (r * yTads) + g;

                        X *localExtraParams = nullptr;
                        if (OpType::extraParamsLen > 0)
                            localExtraParams = new X[OpType::extraParamsLen];
                        for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) {
                            localExtraParams[extraParamsIdx] = startingVal;
                        }

                        for (int f = 0; f < xTadLength; f++) {
                            if (shape::order(yTadShapeInfo) == 'c') {
                                shape::ind2subC(yRank, yShape, f, yCoord);
                            } else {
                                shape::ind2sub(yRank, yShape, f, yCoord);
                            }

                            if (shape::order(xTadShapeInfo) == 'c') {
                                shape::ind2subC(xRank, xShape, f, xCoord);
                            } else {
                                shape::ind2sub(xRank, xShape, f, xCoord);
                            }

                            auto xO = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto yO = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                            result[ri] = OpType::update(result[ri], OpType::op(lX[xO], lY[yO], localExtraParams), localExtraParams);
                        }

                        result[ri] = OpType::postProcess(result[ri], xTadLength, localExtraParams);

                        if (localExtraParams != nullptr)
                            delete[] localExtraParams;
                    }
                }

            }


            template<typename OpType>
            static void exec(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vextraParams,
                    void *vy,
                    Nd4jLong *yShapeInfo,
                    void *vresult,
                    Nd4jLong *resultShapeInfoBuffer,
                    int *dimension,
                    int dimensionLength,
                    Nd4jLong *tadShapeInfo,
                    Nd4jLong *tadOffsets) {

                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<Y *>(vy);
                auto result = reinterpret_cast<X *>(vresult);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

/*
                nd4j_printf("Xp: [%p]; Yp: [%p]; Zp: [%p];\n", (void *) x, (void *) y, (void *) result);
                nd4j_printf("XSp: [%p]; YSp: [%p]; ZSp: [%p];\n", (void *) xShapeInfo, (void *) yShapeInfo, (void *) resultShapeInfoBuffer);
                nd4j_printf("Ep: [%p]; Dp: [%p]\n", (void *) extraParams, (void *) dimension);
                nd4j_printf("TSp: [%p]; TOp: [%p]\n", (void *) tadShapeInfo, (void *) tadOffsets);

                nd4j_printf("X[0]: %f\n", x[0]);
                nd4j_printf("Y[0]: %f\n", y[0]);
                nd4j_printf("Z[0]: %f\n", result[0]);

                nd4j_printf("XS[0]: %i\n", xShapeInfo[0]);
                nd4j_printf("YS[0]: %i\n", yShapeInfo[0]);
                nd4j_printf("ZS[0]: %i\n", resultShapeInfoBuffer[0]);

                nd4j_printf("E[0]: %f\n", extraParams[0]);
                nd4j_printf("D[0]: %i\n", dimension[0]);
                nd4j_printf("TS[0]: %i\n", tadShapeInfo[0]);
                nd4j_printf("TO[0]: %lld\n", tadOffsets[0]);
                nd4j_printf("dimLength: %i\n", dimensionLength);
*/
                auto startingVal = OpType::startingValue(x);

                auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                auto tads = shape::length(xShapeInfo) / tadLength;

                auto *xShape = shape::shapeOf(tadShapeInfo);
                auto *xStride = shape::stride(tadShapeInfo);
                int xRank = shape::rank(tadShapeInfo);

                auto *yShape = shape::shapeOf(yShapeInfo);
                auto *yStride = shape::stride(yShapeInfo);
                int yRank = shape::rank(yShapeInfo);

                //shape::printShapeInfoLinear(xShapeInfo);
                //shape::printShapeInfoLinear(yShapeInfo);
                //shape::printShapeInfoLinear(resultShapeInfoBuffer);
                //shape::printShapeInfoLinear(tadShapeInfo);

                Nd4jLong xCoord[MAX_RANK];
                Nd4jLong yCoord[MAX_RANK];

//#pragma  omp parallel for proc_bind(AFFINITY) default(shared)
                for (Nd4jLong r = 0; r < tads; r++) {
                    Nd4jLong offset = tadOffsets[r];

                    X *localExtraParams = nullptr;
                    if (OpType::extraParamsLen > 0)
                        localExtraParams = new X[OpType::extraParamsLen];
                    for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) {
                        localExtraParams[extraParamsIdx] = startingVal;
                    }

                    for (Nd4jLong f = 0; f < tadLength; f++) {
                        if (shape::order(tadShapeInfo) == 'c') {
                            shape::ind2subC(xRank, xShape, f, xCoord);
                            shape::ind2subC(yRank, yShape, f, yCoord);
                        } else {
                            shape::ind2sub(xRank, xShape, f, xCoord);
                            shape::ind2sub(yRank, yShape, f, yCoord);
                        }

                        auto xOffset = shape::getOffset(offset, xShape, xStride, xCoord, xRank);
                        auto yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);

                        result[r] = OpType::update(result[r], OpType::op(x[xOffset], y[yOffset], localExtraParams), localExtraParams);
                    }

                    result[r] = OpType::postProcess(result[r], tadLength, localExtraParams);

                    if (localExtraParams != nullptr)
                        delete[] localExtraParams;
                }
            }

            template<typename OpType>
            static void exec(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vextraParams,
                    void *vy,
                    Nd4jLong *yShapeInfo,
                    void *vresult,
                    Nd4jLong *resultShapeInfoBuffer,
                    int *dimension,
                    int dimensionLength) {

                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<Y *>(vy);
                auto result = reinterpret_cast<X *>(vresult);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

/*
                nd4j_printf("Xp: [%p]; Yp: [%p]; Zp: [%p];\n", (void *) x, (void *) y, (void *) result);
                nd4j_printf("XSp: [%p]; YSp: [%p]; ZSp: [%p];\n", (void *) xShapeInfo, (void *) yShapeInfo, (void *) resultShapeInfoBuffer);
                nd4j_printf("Ep: [%p]; Dp: [%p]\n", (void *) extraParams, (void *) dimension);

                nd4j_printf("X[0]: %f\n", x[0]);
                nd4j_printf("Y[0]: %f\n", y[0]);
                nd4j_printf("Z[0]: %f\n", result[0]);

                nd4j_printf("XS[0]: %i\n", xShapeInfo[0]);
                nd4j_printf("YS[0]: %i\n", yShapeInfo[0]);
                nd4j_printf("ZS[0]: %i\n", resultShapeInfoBuffer[0]);

                nd4j_printf("E[0]: %f\n", extraParams[0]);
                nd4j_printf("D[0]: %i\n", dimension[0]);
                nd4j_printf("dimLength: %i\n", dimensionLength);
*/

                X extraParamsVals[3] = {(X) 0.0, (X) 0.0, (X) 0.0};


                if(shape::isScalar(resultShapeInfoBuffer)) {
                    result[0] = execScalar<OpType>(
                            x,
                            xShapeInfo,
                            extraParamsVals,
                            y,
                            yShapeInfo);
                    return;
                }



                char xOrder = shape::order(xShapeInfo);
                char yOrder = shape::order(yShapeInfo);
                if(xOrder != yOrder) {

                    Nd4jLong shapeIter[MAX_RANK];
                    Nd4jLong coord[MAX_RANK];
                    int dim;
                    Nd4jLong xStridesIter[MAX_RANK];
                    Nd4jLong yStridesIter[MAX_RANK];

                    auto xShape = shape::shapeOf(xShapeInfo);

                    auto xStride = shape::stride(xShapeInfo);
                    auto yStride = shape::stride(yShapeInfo);

                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<X, Y>(rank,
                                                 xShape,
                                                 x,
                                                 xStride,
                                                 y,
                                                 yStride,
                                                 &rank,
                                                 shapeIter,
                                                 &x,
                                                 xStridesIter,
                                                 &y,
                                                 yStridesIter) >= 0) {

                        auto resultLength = shape::length(resultShapeInfoBuffer);
                        auto tadLength = shape::tadLength(xShapeInfo,dimension,dimensionLength);

                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                Nd4jLong xOffset = shape::getOffset(0,xShape,xStride,coord,rank);
                                auto reductionIndex = xOffset / resultLength;
                                result[reductionIndex] = OpType::update(result[reductionIndex], OpType::op(x[0],y[0], extraParamsVals), extraParamsVals);
                            } ND4J_RAW_ITER_TWO_NEXT(dim,
                                                     rank,
                                                     coord,
                                                     shapeIter,
                                                     x,
                                                     xStridesIter,
                                                     y,
                                                     yStridesIter);


//#pragma  omp parallel for proc_bind(AFFINITY) default(shared)
                        for(Nd4jLong i = 0; i < resultLength ;i++) {
                            result[i] = OpType::postProcess(result[i],tadLength, extraParamsVals);
                        }
                    }

                    else {
                        printf("Unable to prepare array\n");
                    }
                }
                else {
                    auto startingVal = OpType::startingValue(x);

                    Nd4jLong resultLength = shape::length(resultShapeInfoBuffer);
                    shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
                    xTad.createTadOnlyShapeInfo();
                    xTad.createOffsets();


                    shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
                    yTad.createTadOnlyShapeInfo();
                    yTad.createOffsets();

                    /**
                     * The element wise stride belong longs to a reduction index.
                     * When used out of order, we can get rid of the data
                     * dependencies and rely on using the max dimension
                     * specified for stride instead.
                     * Say we take the sum(0,1) along long arr
                     * we can use arr.stride(1) as a representation
                     * along long which to iterate.
                     */
                    int largerElementWiseStride;
                    int smallerElementWiseStride;
                    auto xElementWiseStride = shape::elementWiseStride(xTad.tadOnlyShapeInfo);
                    auto yElementWiseStride = shape::elementWiseStride(yTad.tadOnlyShapeInfo);
                    int tadLength;
                    Nd4jLong xModLength;
                    Nd4jLong yModLength;
                    Nd4jLong *iterationTadInfo;
                    bool xTadBigger;
                    if(shape::length(xShapeInfo) > shape::length(yShapeInfo)) {
                        tadLength = shape::length(xTad.tadOnlyShapeInfo);
                        iterationTadInfo = xTad.tadOnlyShapeInfo;
                        largerElementWiseStride = shape::elementWiseStride(xShapeInfo);
                        smallerElementWiseStride = shape::elementWiseStride(yShapeInfo);
                        xModLength = 1;
                        yModLength = tadLength;
                        xTadBigger = true;

                    }
                    else {
                        tadLength = shape::length(yTad.tadOnlyShapeInfo);
                        iterationTadInfo = yTad.tadOnlyShapeInfo;
                        largerElementWiseStride = shape::elementWiseStride(yShapeInfo);
                        smallerElementWiseStride = shape::elementWiseStride(xShapeInfo);
                        xModLength = tadLength;
                        yModLength = 1;
                        xTadBigger = false;
                    }




                    if (largerElementWiseStride >= 1 && smallerElementWiseStride >= 1 && xElementWiseStride >= 1 && yElementWiseStride >= 1) {
                        if(shape::length(xShapeInfo) == shape::length(yShapeInfo)) {
                            //#pragma omp parallel for proc_bind(AFFINITY) default(shared)
                            for (Nd4jLong i = 0; i < resultLength; i++) {
                                X *localExtraParams = nullptr;
                                if (OpType::extraParamsLen > 0)
                                    localExtraParams = new X[OpType::extraParamsLen];
                                for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) {
                                    localExtraParams[extraParamsIdx] = startingVal;
                                }

                                Nd4jLong offset = xTad.tadOffsets[i];
                                Nd4jLong yOffset = yTad.tadOffsets[i];
                                result[i] = OpType::op(x[offset], y[yOffset], localExtraParams);
                                for (int j = 1; j < tadLength; j++) {
                                    int xIdx = (offset + xElementWiseStride * j);
                                    int yIdx = (yOffset + yElementWiseStride * j);
                                    result[i] = OpType::update(result[i], OpType::op(x[xIdx],
                                                                                     y[yIdx],
                                                                                     localExtraParams), localExtraParams);
                                }

                                result[i] = OpType::postProcess(result[i], tadLength, localExtraParams);

                                if (localExtraParams != nullptr)
                                    delete[] localExtraParams;
                            }
                        }
                        else {
                            int tadsPerThread = resultLength / TAD_THRESHOLD;
                            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());


//#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                            for (int i = 0; i < resultLength; i++) {
                                Nd4jLong xOffset = xTadBigger ? xTad.tadOffsets[i] : 0;
                                Nd4jLong yOffset = !xTadBigger ? yTad.tadOffsets[i] : 0;
                                auto xShape = xTadBigger ? xTad.tadShape : shape::shapeOf(xShapeInfo);
                                auto yShape = !xTadBigger ? yTad.tadShape : shape::shapeOf(yShapeInfo);
                                auto xStride = xTadBigger ? xTad.tadStride : shape::stride(xShapeInfo);
                                auto yStride = !xTadBigger ? yTad.tadStride : shape::stride(yShapeInfo);
                                int xRank = xTadBigger ? shape::rank(xTad.tadOnlyShapeInfo) : shape::rank(xShapeInfo);
                                int yRank = !xTadBigger ? shape::rank(yTad.tadOnlyShapeInfo) : shape::rank(yShapeInfo);
                                Nd4jLong coord[MAX_RANK];
                                Nd4jLong yCoord[MAX_RANK];
                                auto start = OpType::startingValue(x);

                                for (int j = 0; j < tadLength; j++) {
                                    if(xTadBigger) {
                                        shape::ind2subC(shape::rank(xTad.tadOnlyShapeInfo),
                                                        xTad.tadStride, j, coord);
                                        shape::ind2subC(shape::rank(yShapeInfo),
                                                        shape::shapeOf(yShapeInfo), j, yCoord);
                                    }
                                    else {
                                        shape::ind2subC(shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), j, coord);
                                        shape::ind2subC(shape::rank(yTad.tadOnlyShapeInfo),
                                                        yTad.tadShape, j, yCoord);
                                    }



                                    int xOffset2 =  shape::getOffset(xOffset,xShape,xStride,coord,xRank);
                                    int yOffset2 =  shape::getOffset(yOffset,yShape,yStride,yCoord,yRank);
                                    start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2],extraParams), extraParamsVals);
                                }

                                result[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                            }
                        }

                    } else {
                        shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
                        xTad.createTadOnlyShapeInfo();
                        xTad.createOffsets();


                        shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
                        yTad.createTadOnlyShapeInfo();
                        yTad.createOffsets();
                        int tadsPerThread = resultLength / TAD_THRESHOLD;
                        int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                        num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                        Nd4jLong coord[MAX_RANK];

//#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(coord)
                        for (int i = 0; i < resultLength; i++) {
                            Nd4jLong xOffset = xTad.tadOffsets[i];
                            Nd4jLong yOffset = yTad.tadOffsets[i];


                            auto start = OpType::startingValue(x + xOffset);

                            for (int j = 0; j < tadLength; j++) {
                                shape::ind2subC(shape::rank(iterationTadInfo), shape::shapeOf(iterationTadInfo), j, coord);
                                Nd4jLong xOffset2 = shape::getOffset(xOffset,shape::shapeOf(xTad.tadOnlyShapeInfo),shape::stride(xTad.tadOnlyShapeInfo),coord,shape::rank(xTad.tadOnlyShapeInfo));
                                Nd4jLong yOffset2 = shape::getOffset(yOffset,shape::shapeOf(yTad.tadOnlyShapeInfo),shape::stride(yTad.tadOnlyShapeInfo),coord,shape::rank(yTad.tadOnlyShapeInfo));
                                start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2],extraParamsVals), extraParamsVals);
                            }

                            result[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                        }
                    }


                }
            }


        };
    }
}

#ifdef __CUDACC__
/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post
 */
template <typename T>
__device__ void reduce3Generic(
		const int opNum,
		T *dx,
		Nd4jLong *xShapeInfo,
		T *dy,
		Nd4jLong *yShapeInfo,
		T *extraParams,
		T *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));

	}
	__syncthreads();

	functions::reduce3::Reduce3<T>::exec(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationPointer,
			manager,
			tadOnlyShapeInfo,
			tadOffsets,
			yTadOnlyShapeInfo,
			yTadOffsets);
}

template <typename T>
__device__ void reduce3AllGeneric(
		const int opNum,
		T *dx,
		Nd4jLong *xShapeInfo,
		T *dy,
		Nd4jLong *yShapeInfo,
		T *extraParams,
		T *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot,
		int *allocationPointer,
		Nd4jLong *tadOnlyShapeInfo,
		Nd4jLong *tadOffsets,
		Nd4jLong *yTadOnlyShapeInfo,
		Nd4jLong *yTadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));

	}
	__syncthreads();

	functions::reduce3::Reduce3<T>::execAllCuda(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot,
			allocationPointer,
			manager,
			tadOnlyShapeInfo,
			tadOffsets,
			yTadOnlyShapeInfo,
			yTadOffsets);
}

template <typename T>
__device__ void reduce3ScalarGeneric(
		int opNum,
		T *dx,
		Nd4jLong *xShapeInfo,
		T *dy,
		Nd4jLong *yShapeInfo,
		T *extraParams,
		T *result,
		Nd4jLong *resultShapeInfo, int *allocationPointer,
		T *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));
	}
	__syncthreads();

	functions::reduce3::Reduce3<T>::execScalarCuda(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			allocationPointer,
			reductionBuffer,
			manager,
			tadOnlyShapeInfo);
}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */
extern "C"
__global__ void reduce3Double(
		int opNum,
		double *dx,
		Nd4jLong *xShapeInfo,
		double *dy,
		Nd4jLong *yShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3Generic<double>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3AllDouble(
		int opNum,
		double *dx,
		Nd4jLong *xShapeInfo,
		double *dy,
		Nd4jLong *yShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3AllGeneric<double>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

/**
 * The driver api
 * @param opNum the number
 * @param n the length of the reduce
 * @param dx the input data
 * @param xShapeInfo the shape information
 * @param dy the pair wise reduce
 * @param yShapeInfo the shape information for y
 * @param extraParams the extra parameters in the operation
 * @param result where to store the result
 * @param resultShapeInfo the shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to reduce along long
 * @param dimensionLength the dimension length
 * @param postProcessOrNot whether to post [
 */
extern "C"
__global__ void reduce3Float(
		int opNum,
		float *dx,
		Nd4jLong *xShapeInfo,
		float *dy,
		Nd4jLong *yShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3Generic<float>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3AllFloat(
		int opNum,
		float *dx,
		Nd4jLong *xShapeInfo,
		float *dy,
		Nd4jLong *yShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3AllGeneric<float>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3Half(
		int opNum,
		float16 *dx,
		Nd4jLong *xShapeInfo,
		float16 *dy,
		Nd4jLong *yShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3Generic<float16>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3AllHalf(
		int opNum,
		float16 *dx,
		Nd4jLong *xShapeInfo,
		float16 *dy,
		Nd4jLong *yShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3AllGeneric<float16>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			dimension,
			dimensionLength,
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3ScalarFloat(
		int opNum,
		float *dx,
		Nd4jLong *xShapeInfo,
		float *dy,
		Nd4jLong *yShapeInfo,
		float *extraParams,
		float *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, float *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3ScalarGeneric<float>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo, allocationPointer,
			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C" __global__ void reduce3ScalarHalf(
		int opNum,
		float16 *dx,
		Nd4jLong *xShapeInfo,
		float16 *dy,
		Nd4jLong *yShapeInfo,
		float16 *extraParams,
		float16 *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, float16 *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3ScalarGeneric<float16>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo, allocationPointer,
			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

extern "C"
__global__ void reduce3ScalarDouble(
		int opNum,
		double *dx,
		Nd4jLong *xShapeInfo,
		double *dy,
		Nd4jLong *yShapeInfo,
		double *extraParams,
		double *result,
		Nd4jLong *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, double *reductionBuffer, Nd4jLong *tadOnlyShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *yTadOnlyShapeInfo, Nd4jLong *yTadOffsets) {
	reduce3ScalarGeneric<double>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo, allocationPointer,
			reductionBuffer, tadOnlyShapeInfo, tadOffsets, yTadOnlyShapeInfo, yTadOffsets);

}

#endif



#endif /* REDUCE3_H_ */
