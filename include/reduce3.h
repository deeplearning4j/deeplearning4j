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
#include <sharedmem.h>
#include <omp.h>
#include <pairwise_util.h>
#include <dll.h>
#include <shape.h>
#include <ops.h>

#ifdef __JNI__
#include <jni.h>
#endif

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace functions {
	namespace reduce3 {

/**
 * Reduce involving
 * 2 arrays
 */
		template<typename T>
		class Reduce3 {

		public:
#ifdef __CUDACC__
			virtual __device__

			inline T opAtomic(T d1, T d2, T **extraParamsRef) = 0;
#endif

#ifdef __CUDACC__
			/**
     * Aggregate shared memory
     * @param sPartialsRef
     * @param tid
     * @param extraParams
     */
			virtual __inline__ __device__ void aggregatePartials(T **sPartialsRef, int tid, int numItems, T **extraParamsRef) {
				// start the shared memory loop on the next power of 2 less
				// than the block size.  If block size is not a power of 2,
				// accumulate the intermediate sums in the remainder range.
				T *sPartials = *sPartialsRef;
				int floorPow2 = numItems;

				if (floorPow2 & (floorPow2 - 1)) {
					while (floorPow2 & (floorPow2 - 1)) {
						floorPow2 &= floorPow2 - 1;
					}
					if (tid >= floorPow2) {
						sPartials[tid - floorPow2] = update(sPartials[tid - floorPow2], sPartials[tid], extraParamsRef);
					}
					__syncthreads();
				}

				for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
					if (tid < activeThreads) {
						sPartials[tid] = update(sPartials[tid], sPartials[tid + activeThreads], extraParamsRef);
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
					int *xShapeInfo,
					T *dy,
					int *yShapeInfo,
					T *extraParams,
					T *result,
					int *resultShapeInfo,
					int postProcessOrNot, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {
				Nd4jIndex n = shape::length(xShapeInfo);
				int rank = shape::rank(xShapeInfo);
				//shared memory space for storing intermediate results
				//SharedMemory <T> val;
				T *sPartials = (T *) manager->getSharedReductionBuffer(); //val.getPointer();
				T startingVal = this->startingValue(dx);


				sPartials[threadIdx.x] = startingVal;

                int idx[MAX_RANK];
#pragma unroll
				for(Nd4jIndex i = blockIdx.x * gridDim.x + threadIdx.x;i < n; i += gridDim.x * blockDim.x) {
					shape::ind2subC(rank,shape::shapeOf(xShapeInfo),i, idx);
					Nd4jIndex offset = shape::getOffset(0,shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),idx,rank);
					Nd4jIndex yOffset = shape::getOffset(0,shape::shapeOf(yShapeInfo),shape::stride(yShapeInfo),idx,rank);
					sPartials[threadIdx.x] = update(sPartials[threadIdx.x], this->opAtomic(dx[offset], dy[yOffset], &extraParams),&extraParams);

				}

				T **sPartialsRef = (T **) &sPartials;
				aggregatePartials(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, n), &extraParams);
				/**
                 * Look at something that uses the extra params
                 * and aggregates the extra values propelry.
                 *This will be used in summary stats too.
                 */
				// write result for this block to global mem
				if (threadIdx.x == 0) {
					if (postProcessOrNot) {
						result[blockIdx.x] = postProcess(sPartials[0], n,&extraParams);
					}
					else {
						result[blockIdx.x] = sPartials[0];
					}


				}


				if(threadIdx.x == 0 && this->extraParamsLength() > 0)
					this->finalizeExtraParams(&extraParams);



			}


			/**
             *
             */
			virtual __inline__ __device__ void execScalarCuda(
					T *dx,
					int *xShapeInfo,
					T *dy,
					int *yShapeInfo,
					T *extraParams,
					T *result,
					int *resultShapeInfo, int *allocationBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {


//		SharedMemory <T> val;
				T *sPartials = (T *) manager->getSharedReductionBuffer(); // val.getPointer();

				T startingVal = this->startingValue(dx);
				Nd4jIndex length = shape::length(xShapeInfo);
				int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
				int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
				int tid = blockIdx.x * blockDim.x + threadIdx.x;
				char xOrder = shape::order(xShapeInfo);
				char yOrder = shape::order(yShapeInfo);
				if(xOrder == yOrder) {
					if (xElementWiseStride == 1 && yElementWiseStride == 1) {
						for(Nd4jIndex i = threadIdx.x; i < length; i+= gridDim.x * blockDim.x) {
							startingVal = update(startingVal, this->opAtomic(dx[i], dy[i], &extraParams), &extraParams);
						}
					}
					else {
						for(int i = threadIdx.x; i < length; i+= gridDim.x * blockDim.x) {
							startingVal = update(startingVal, this->opAtomic(dx[i * xElementWiseStride], dy[i * yElementWiseStride], &extraParams), &extraParams);
						}
					}

					sPartials[tid] = startingVal;
					__syncthreads();


					T **sPartialsRef = (T **) &sPartials;
					aggregatePartials(sPartialsRef, tid, nd4j::math::nd4j_min<int>(blockDim.x, length), &extraParams);

					/**
                     * Look at something that uses the extra params
                     * and aggregates the extra values properly.
                     *This will be used in summary stats too.
                     */
					// write result for this block to global mem
					__syncthreads();
					if (tid == 0) {
						result[0] = postProcess(sPartials[0], length,&extraParams);
					}
				}

				else {
					int *xShape = shape::shapeOf(xShapeInfo);
					int *xStride = shape::stride(xShapeInfo);
					int *yStride = shape::stride(yShapeInfo);
					T startingVal = this->startingValue(dx);
					int n = shape::length(xShapeInfo);

					//SharedMemory <T> val;
					T *sPartials = (T *) manager->getSharedReductionBuffer(); //val.getPointer();


					Nd4jIndex length = shape::length(xShapeInfo);
					int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
					int yElementWiseStride = shape::elementWiseStride(yShapeInfo);
					char xOrder = shape::order(xShapeInfo);
					char yOrder = shape::order(yShapeInfo);


					//int *idx = (int *) malloc(sizeof(int) * shape::rank(xShapeInfo));
					int rank = shape::rank(xShapeInfo);
					/*
					long allocSize = sizeof(int) * rank;
					int *idx = shape::cuMalloc(allocationBuffer, allocSize, manager);
					*/
					int idx[MAX_RANK];

					//shared memory space for storing intermediate results
					sPartials[threadIdx.x] = startingVal;


#pragma unroll
					for(unsigned int i = tid ;i < n; i += gridDim.x * blockDim.x) {
						shape::ind2sub(rank,shape::shapeOf(xShapeInfo),i,idx);
						Nd4jIndex offset = shape::getOffset(0,shape::shapeOf(xShapeInfo),shape::stride(xShapeInfo),idx,rank);
						Nd4jIndex yOffset = shape::getOffset(0,shape::shapeOf(yShapeInfo),shape::stride(yShapeInfo),idx,rank);
						sPartials[threadIdx.x] = update(sPartials[threadIdx.x], this->opAtomic(dx[offset], dy[yOffset], &extraParams),&extraParams);
					}

/*
					if (rank > MAX_COORD && tid * allocSize > PREALLOC_SIZE - allocSize) {
						free(idx);
					}
*/

					T **sPartialsRef = (T **) &sPartials;
					aggregatePartials(sPartialsRef, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, length), &extraParams);
					/**
                     * Look at something that uses the extra params
                     * and aggregates the extra values propelry.
                     *This will be used in summary stats too.
                     */
					// write result for this block to global mem
					__syncthreads();
					if (tid == 0) {
						result[tid] = postProcess(sPartials[0], n,&extraParams);
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
            template<template <typename> typename OpType>
			static inline __device__ void transform(
					T *dx,
					int *xShapeInfo,
					T *dy,
					int *yShapeInfo,
					T *extraParams,
					T *result,
					int *resultShapeInfo,
					int *dimension,
					int dimensionLength,
					int postProcessOrNot,
					int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, int *tadOffsets) {
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
				T init = OpType<T>::startingValue(dx);
				sPartials[threadIdx.x] = init;


				//length for the tad

				__shared__ Nd4jIndex resultLength;


				T reduction = OpType<T>::startingValue(dx);
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

					int *xStride = shape::stride(xShapeInfo);
					char xOrder = shape::order(xShapeInfo);

					xElementWiseStride = shape::elementWiseStride(xShapeInfo);
					yElementWiseStride = shape::elementWiseStride(yShapeInfo);
				}
				__syncthreads();


				if (!resultScalar) {
					__shared__ int tadLength;
        	        __shared__ int tadEWS;
    	            __shared__ int tadRank;
	                __shared__ int numTads;

                	if (threadIdx.x == 0) {
            		    tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
        	            tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
    	                tadRank = shape::rank(tadOnlyShapeInfo);
	                    numTads = shape::length(xShapeInfo) / tadLength;
                	}
                	__syncthreads();

					if(dimensionLength > 1) {
						int *xShape = shape::shapeOf(tadOnlyShapeInfo);
						int *xStride = shape::stride(tadOnlyShapeInfo);
						int *yStride = shape::stride(yShapeInfo);

                        int xStridesIter[MAX_RANK];
						int yStridesIter[MAX_RANK];
						int shapeIter[MAX_RANK];
						int coord[MAX_RANK];
						Nd4jIndex n = shape::length(xShapeInfo);
						int rank = shape::rank(xShapeInfo);
#pragma unroll
						for(Nd4jIndex i = tid; i < resultLength; i+= gridDim.x * blockDim.x) {
							int offset = tadOffsets[i];
							int dim;
							T *xPointer = dx + offset;
							T start = this->startingValue(xPointer);
							T startingVal = this->startingValue(dx);
							if(PrepareTwoRawArrayIter<T>(rank,
														 xShape,
														 dx,
														 xStride,
														 dy,
														 yStride,
														 &rank,
														 shapeIter,
														 &dx,
														 xStridesIter,
														 &dy,
														 yStridesIter) >= 0) {
								ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
										/* Process the innermost dimension */
										T *xIter = dx;
										T *yIter = dy;
										startingVal = OpType<T>::update(startingVal, OpType<T>::op(xIter[0],yIter[0],&extraParams),&extraParams);
									} ND4J_RAW_ITER_TWO_NEXT(dim,
															 rank,
															 coord,
															 shapeIter,
															 dx,
															 xStridesIter,
															 dy,
															 yStridesIter);

								result[i] = postProcess(startingVal,n,&extraParams);
							}
							else {
								printf("Unable to prepare array\n");
							}

						}

						__syncthreads();
					}
					else {
				//		Nd4jIndex xLength = shape::length(xShapeInfo);

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

						for(int i = tid; i < resultLength; i+= blockDim.x * gridDim.x) {
							int xOffsetForTad = tadOffsets[i];
							int yOffsetForTad = xOffsetForTad;//tad->tadOffset(i);
							//int xOffsetForTad = shape::tadOffset(i, xShapeInfo, dimension, dimensionLength, nullptr);
							//int yOffsetForTad = shape::tadOffset(i, yShapeInfo, dimension, dimensionLength, nullptr);

							sPartials[threadIdx.x] = op(dx[xOffsetForTad],dy[yOffsetForTad], &extraParams);
							for(int j = 1; j < tadLength; j++) {
								sPartials[threadIdx.x] =  OpType<T>::update(sPartials[threadIdx.x], OpType<T>::op(dx[xOffsetForTad + xElementWiseStride * j],dy[yOffsetForTad + yElementWiseStride * j], &extraParams), &extraParams);
							}

							result[i] = postProcess(sPartials[threadIdx.x],tadLength,&extraParams);
						}

					}
				}
			}





#endif


#ifdef __CUDACC__
	__host__ __device__


#endif

			static T execScalar(
				const int op,
				T *x,
				int *xShapeInfo,
				T *extraParamsVals,
				T *y,
				int *yShapeInfo) {
				if (op == 0)
					return execScalar<simdOps::ManhattanDistance>(x, xShapeInfo, extraParamsVals, y, yShapeInfo);
				else if (op == 1)
					return execScalar<simdOps::EuclideanDistance>(x, xShapeInfo, extraParamsVals, y, yShapeInfo);
				else if (op == 2)
					return execScalar<simdOps::CosineSimilarity>(x, xShapeInfo, extraParamsVals, y, yShapeInfo);
				else if (op == 3)
					return execScalar<simdOps::Dot>(x, xShapeInfo, extraParamsVals, y, yShapeInfo);
				else
					printf("[ERROR] Unknown opNum=%d for reduce3!\n", op);
				return 0;
			}

#ifdef __CUDACC__
			__host__ __device__

			static void transform(
				const int op,
				T *dx,
				int *xShapeInfo,
				T *dy,
				int *yShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				int postProcessOrNot,
				int *allocationPointer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo,
				int *tadOffsets) {

				if (op == 0)
					transform<simdOps::ManhattanDistance>(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets);
				else if (op == 1)
					transform<simdOps::EuclideanDistance>(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets);
				else if (op == 2)
					transform<simdOps::CosineSimilarity>(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets);
				else if (op == 3)
					transform<simdOps::Dot>(dx, xShapeInfo, dy, yShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, postProcessOrNot, allocationPointer, manager, tadOnlyShapeInfo, tadOffsets);
				else
					printf("[ERROR] Unknown opNum=%d for reduce3!\n", op);
			}
#endif

			static void exec( const int op,
				T *x, int *xShapeInfo,
				T *extraParamsVals,
				T *y,
				int *yShapeInfo,
				T *result,
				int *resultShapeInfoBuffer,
				int *dimension,
				int dimensionLength) {
				if (op == 0)
					exec<simdOps::ManhattanDistance>(x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength);
				else if (op == 1)
					exec<simdOps::EuclideanDistance>(x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength);
				else if (op == 2)
					exec<simdOps::CosineSimilarity>(x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength);
				else if (op == 3)
					exec<simdOps::Dot>(x, xShapeInfo, extraParamsVals, y, yShapeInfo, result, resultShapeInfoBuffer, dimension, dimensionLength);
				else
					printf("[ERROR] Unknown opNum=%d for reduce3!\n", op);
			}
			

			template<template <typename> typename OpType>
#ifdef __CUDACC__
			__host__
#endif
			static T execScalar(
					T *x,
					int *xShapeInfo,
					T *extraParamsVals,
					T *y,
					int *yShapeInfo) {
				T startingVal = OpType<T>::startingValue(x);
				Nd4jIndex length = shape::length(xShapeInfo);
				int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
				int yElementWiseStride = shape::elementWiseStride(yShapeInfo);

				for(int i = 0; i < OpType<T>::extraParamsLen;i++) {
					extraParamsVals[i] = startingVal;
				}

				char xOrder = shape::order(xShapeInfo);
				char yOrder = shape::order(yShapeInfo);
				if(xOrder == yOrder) {
					if (xElementWiseStride == 1 && yElementWiseStride == 1) {
#pragma omp simd
						for(int i = 0; i < length; i++) {
							startingVal = OpType<T>::update(startingVal, OpType<T>::op(x[i],y[i],&extraParamsVals),&extraParamsVals);
						}

						return  OpType<T>::postProcess(startingVal, length,&(extraParamsVals));

					}

					else {
#pragma omp simd
						for(Nd4jIndex i = 0; i < length; i++) {
							startingVal = OpType<T>::update(startingVal, OpType<T>::op(x[i * xElementWiseStride],y[i * yElementWiseStride],&extraParamsVals),&extraParamsVals);
						}

						return   OpType<T>::postProcess(startingVal, length,&(extraParamsVals));
					}

				}


				else {
					int *xShape = shape::shapeOf(xShapeInfo);
					int *xStride = shape::stride(xShapeInfo);
					int *yStride = shape::stride(yShapeInfo);
					T startingVal = OpType<T>::startingValue(x);
					Nd4jIndex n = shape::length(xShapeInfo);
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int xStridesIter[MAX_RANK];
					int yStridesIter[MAX_RANK];
					int rank = shape::rank(xShapeInfo);
					if(PrepareTwoRawArrayIter<T>(rank,
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
						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
								/* Process the innermost dimension */
								T *xIter = x;
								T *yIter = y;
								startingVal = OpType<T>::update(startingVal, OpType<T>::op(xIter[0],yIter[0],&extraParamsVals),&extraParamsVals);
							} ND4J_RAW_ITER_TWO_NEXT(dim,
													 rank,
													 coord,
													 shapeIter,
													 x,
													 xStridesIter,
													 y,
													 yStridesIter);

						return OpType<T>::postProcess(startingVal,n,&extraParamsVals);
					}
					else {
						printf("Unable to prepare array\n");
					}

				}

				return startingVal;


			}


			template<template <typename> typename OpType>
			static void exec(T *x, int *xShapeInfo,
					  T *extraParamsVals,
					  T *y,
					  int *yShapeInfo,
					  T *result,
					  int *resultShapeInfoBuffer,
					  int *dimension,
					  int dimensionLength) {
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
					int shapeIter[MAX_RANK];
					int coord[MAX_RANK];
					int dim;
					int xStridesIter[MAX_RANK];
					int yStridesIter[MAX_RANK];

					int *xShape = shape::shapeOf(xShapeInfo);

					int *xStride = shape::stride(xShapeInfo);
					int *yStride = shape::stride(yShapeInfo);

					int rank = shape::rank(xShapeInfo);
					if(PrepareTwoRawArrayIter<T>(rank,
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

						Nd4jIndex resultLength = shape::length(resultShapeInfoBuffer);
						Nd4jIndex tadLength = shape::tadLength(xShapeInfo,dimension,dimensionLength);
						ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
								/* Process the innermost dimension */
								T *xIter = x;
								T *yIter = y;
								Nd4jIndex xOffset = shape::getOffset(0,xShape,xStride,coord,rank);
								int reductionIndex = xOffset / resultLength;
								result[reductionIndex] = OpType<T>::update(result[reductionIndex], OpType<T>::op(xIter[0],yIter[0],&extraParamsVals),&extraParamsVals);
							} ND4J_RAW_ITER_TWO_NEXT(dim,
													 rank,
													 coord,
													 shapeIter,
													 x,
													 xStridesIter,
													 y,
													 yStridesIter);


#pragma  omp parallel for
						for(Nd4jIndex i = 0; i < resultLength ;i++) {
							result[i] = OpType<T>::postProcess(result[i],tadLength,&extraParamsVals);
						}
					}

					else {
						printf("Unable to prepare array\n");
					}
				}
				else {
					T startingVal = OpType<T>::startingValue(x);

					Nd4jIndex resultLength = shape::length(resultShapeInfoBuffer);
					shape::TAD xTad(yShapeInfo,dimension,dimensionLength);
					xTad.createTadOnlyShapeInfo();
					xTad.createOffsets();

					/**
                     * The element wise stride belong longs to a reduction index.
                     * When used out of order, we can get rid of the data
                     * dependencies and rely on using the max dimension
                     * specified for stride instead.
                     * Say we take the sum(0,1) along long arr
                     * we can use arr.stride(1) as a representation
                     * along long which to iterate.
                     */
					int tadElementWiseStride = shape::elementWiseStride(xTad.tadOnlyShapeInfo);
					int tadLength = shape::length(xTad.tadOnlyShapeInfo);
#pragma omp parallel for
					for(Nd4jIndex i = 0; i < resultLength; i++) {
						T *localExtraParams = nullptr;
						if(OpType<T>::extraParamsLen > 0)
							localExtraParams = new T[OpType<T>::extraParamsLen];
						for(int extraParamsIdx = 0; extraParamsIdx <  OpType<T>::extraParamsLen; extraParamsIdx++) {
							localExtraParams[extraParamsIdx] = startingVal;
						}

						Nd4jIndex offset = xTad.tadOffsets[i];
						result[i] = OpType<T>::op(x[offset], y[offset],&localExtraParams);
						for(int j = 1; j < tadLength; j++) {
							result[i] = OpType<T>::update(result[i], OpType<T>::op(x[offset + tadElementWiseStride * j],y[offset + tadElementWiseStride * j], &localExtraParams), &localExtraParams);
						}

						result[i] = OpType<T>::postProcess(result[i],tadLength,&localExtraParams);

						if(localExtraParams != nullptr)
							delete[] localExtraParams;
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
		int *xShapeInfo,
		T *dy,
		int *yShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets) {

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));

	}
	__syncthreads();

	functions::reduce3::Reduce3<T>::transform(
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
			tadOffsets);
}

template <typename T>
__device__ void reduce3ScalarGeneric(
		int opNum,
		T *dx,
		int *xShapeInfo,
		T *dy,
		int *yShapeInfo,
		T *extraParams,
		T *result,
		int *resultShapeInfo,
		 int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets) {


	__shared__ functions::reduce3::Reduce3<T> * op;
	__shared__ functions::reduce3::Reduce3OpFactory<T> *reduce3OpFactory;

	__shared__ UnifiedSharedMemory *manager;

	if (threadIdx.x == 0) {
		extern __shared__ unsigned char shmem[];
		manager = new(shmem) UnifiedSharedMemory((int *) shmem);
		manager->init(sizeof(UnifiedSharedMemory), sizeof(functions::reduce3::Reduce3OpFactory<T>), sizeof(functions::reduce3::Reduce3<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));

		reduce3OpFactory = new(manager->getFactorySpace()) functions::reduce3::Reduce3OpFactory<T>();
		op = reduce3OpFactory->getOp(opNum, manager->getFunctionSpace());
	}
	__syncthreads();

	op->execScalarCuda(
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			allocationPointer, manager, tadOnlyShapeInfo);
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
		int *xShapeInfo,
		double *dy,
		int *yShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets) {
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
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets);

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
		int *xShapeInfo,
		float *dy,
		int *yShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets) {
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
			postProcessOrNot, allocationPointer, tadOnlyShapeInfo, tadOffsets);

}

extern "C"
__global__ void reduce3ScalarFloat(
		int opNum,
		float *dx,
		int *xShapeInfo,
		float *dy,
		int *yShapeInfo,
		float *extraParams,
		float *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets) {
	reduce3ScalarGeneric<float>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			allocationPointer, tadOnlyShapeInfo, tadOffsets);

}

extern "C"
__global__ void reduce3ScalarDouble(
		int opNum,
		double *dx,
		int *xShapeInfo,
		double *dy,
		int *yShapeInfo,
		double *extraParams,
		double *result,
		int *resultShapeInfo,
		int *dimension,
		int dimensionLength,
		int postProcessOrNot, int *allocationPointer, int *tadOnlyShapeInfo, int *tadOffsets) {
	reduce3ScalarGeneric<double>(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			extraParams,
			result,
			resultShapeInfo,
			allocationPointer, tadOnlyShapeInfo, tadOffsets);

}

#endif



#endif /* REDUCE3_H_ */
