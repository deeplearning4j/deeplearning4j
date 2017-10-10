
#ifndef REDUCE_H
#define REDUCE_H
#include <dll.h>
//#include <string>
#include <helpers/sharedmem.h>
#include <stdio.h>
#include <helpers/shape.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <templatemath.h>
#include <helper_cuda.h>
#include <nd4jmalloc.h>
#include <pairwise_util.h>
#include <ops/ops.h>
#include <ops/special_accumulation_ops.h>
#include <op_boilerplate.h>

#pragma once
#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#include "legacy_ops.h"

//an op for the kernel
namespace functions {
    namespace reduce {

/**
 * A reduce function
 * reduces a vector down to
 * a subset of itself
 * via aggregating member
 * elements.
 */
        template<typename T>
        class ReduceFunction {
        public:
#ifdef __CUDACC__
            template<typename OpType>
			static inline __device__ void transformCuda1D(T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecialCuda(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
                    return;
                }

				//shared memory space for storing intermediate results
				__shared__ T *sPartials;// = (T *)manager->getSharedReductionBuffer();
				__shared__ int tadLength;
				__shared__ int tadEWS;
				__shared__ int numTads;
				if (threadIdx.x == 0) {
                    extern __shared__ unsigned char shmem[];
                    sPartials = (T *) shmem;
					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);
					numTads = shape::length(xShapeInfo) / tadLength;
				}
				__syncthreads();

				for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
					Nd4jIndex tadOffsetForBlock = tadOffsets[r];
					T *rX = dx + tadOffsetForBlock;

					sPartials[threadIdx.x] = OpType::startingValue(rX);

                    if (tadEWS >= 1) {
					    for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						    sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(rX[i * tadEWS], extraParams), extraParams);
					    }
					} else {
                        __shared__ int tadRank;
				        __shared__ int *tadShape;
				        __shared__ int *tadStride;
				        int xCoord[MAX_RANK];
				        if (threadIdx.x == 0) {
                            tadRank = shape::rank(tadOnlyShapeInfo);
                            tadShape = shape::shapeOf(tadOnlyShapeInfo);
                            tadStride = shape::stride(tadOnlyShapeInfo);
				        }
    				    __syncthreads();

                        for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						    shape::ind2subC(tadRank, tadShape, i, xCoord);
						    Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

						    sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffset], extraParams), extraParams);
					    }
					}


					__syncthreads();

					// aggregate. do NOT reduce for elements > tadLength
					aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);


					__syncthreads();
					if (threadIdx.x == 0) {
						result[r] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
					}
				}
			}

template<typename OpType>
			static inline __device__ void execScalarCuda(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo) {
				int elementWiseStride = shape::elementWiseStride(xShapeInfo);

				int n = shape::length(xShapeInfo);

				int tid = blockDim.x * blockIdx.x + threadIdx.x;

				//shared memory space for storing intermediate results
				T *sPartials = (T *)manager->getSharedReductionBuffer();

				sPartials[threadIdx.x] = OpType::startingValue(dx);

				if (elementWiseStride >= 1) {
					for (int i = tid; i < n; i += (blockDim.x * gridDim.x)) {
						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[i * elementWiseStride], extraParams), extraParams);
					}
				}
				else {
				    __shared__ int rank;
				    __shared__ int *xShape;
				    __shared__ int *xStride;
				    if (threadIdx.x == 0) {
                        rank = shape::rank(xShapeInfo);
                        xShape = shape::shapeOf(xShapeInfo);
                        xStride = shape::stride(xShapeInfo);
				    }
				    __syncthreads();

					int ind2sub[MAX_RANK];

					for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
						shape::ind2subC(rank, xShape, i, ind2sub);

						Nd4jIndex offset = shape::getOffset(0, xShape, xStride, ind2sub, rank);
						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[offset], extraParams), extraParams);
					}
				}

				__syncthreads();
				aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, n), extraParams);


				__syncthreads();

				if (gridDim.x > 1) {
					unsigned int *tc = (unsigned int *)reductionBuffer;
					__shared__ bool amLast;
					tid = threadIdx.x;
					if (threadIdx.x == 0) {
						reductionBuffer[blockIdx.x] = sPartials[0];//this->postProcess(sPartials[0],n,extraParams);
					}
					__threadfence();
					__syncthreads();

					if (threadIdx.x == 0) {
						unsigned int ticket = atomicInc(&tc[16384], gridDim.x);
						amLast = (ticket == gridDim.x - 1);
					}

					__syncthreads();

					if (amLast) {
						tc[16384] = 0;

						sPartials[threadIdx.x] = OpType::startingValue(dx);

						for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
							sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], reductionBuffer[i], extraParams);
						}
						__syncthreads();



						aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(gridDim.x, blockDim.x), extraParams);

						__syncthreads();
						if (threadIdx.x == 0) {
							result[0] = OpType::postProcess(sPartials[0], n, extraParams);
						}
					}
				}
				else {
					if (threadIdx.x == 0) {
						unsigned int *tc = (unsigned *)reductionBuffer;
						tc[16384] = 0;
						result[0] = OpType::postProcess(sPartials[0], n, extraParams);
					}
				}
			}
			/**
			 * Kernel invocation for reduce
			 * @param n the length of the buffer
			 * @param dx the input
			 * @param xShapeInfo the shape information for the input
			 * @param extraParams extra parameters (starting value,..)
			 * @param result the result buffer
			 * @param resultShapeInfo the shapeinformation for the result buffer
			 * @param gpuInformation the gpu information (shared memory allocated,..)
			 * @param dimension the dimension to do reduce along long
			 * @param dimensionLength the length of the dimension buffer
			 * @param postProcessOrNot whether to reduce or not
			 */
template<typename OpType>
			static inline __device__ void transformCuda3D(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecialCuda(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
                    return;
                }

				//shared memory space for storing intermediate results
				__shared__ T *sPartials; // = (T *)manager->getSharedReductionBuffer();

				__shared__ int tadLength;
				__shared__ int tadRank;
				__shared__ int numTads;
				__shared__ int *tadShape;
				__shared__ int *tadStride;
				if (threadIdx.x == 0) {
				    extern __shared__ unsigned char shmem[];
				    sPartials = (T *) shmem;
					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadRank = shape::rank(tadOnlyShapeInfo);
					numTads = shape::length(xShapeInfo) / tadLength;

					tadShape = shape::shapeOf(tadOnlyShapeInfo);
					tadStride = shape::stride(tadOnlyShapeInfo);
				}
				__syncthreads();

				int xCoord[3];

				for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
					Nd4jIndex tadOffsetForBlock = tadOffsets[r];

					sPartials[threadIdx.x] = OpType::startingValue(dx + tadOffsetForBlock);

					for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						shape::ind2subC(tadRank, tadShape, i, xCoord);
						Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffset], extraParams), extraParams);
					}
					__syncthreads();

					// aggregate. do NOT reduce for elements > tadLength
					aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);

					__syncthreads();
					if (threadIdx.x == 0)
						result[r] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
				}
			}

template<typename OpType>
			static inline __device__ void transformCudaXD(
				T *dx,
				int *xShapeInfo,
				T *extraParams,
				T *result,
				int *resultShapeInfo,
				int *dimension,
				int dimensionLength,
				T *reductionBuffer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo,
				Nd4jIndex *tadOffsets) {

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecialCuda(dx, xShapeInfo, extraParams, result, resultShapeInfo, dimension, dimensionLength, reductionBuffer, manager, tadOnlyShapeInfo, tadOffsets);
                    return;
                }

				//shared memory space for storing intermediate results
				__shared__ T *sPartials;

				//                __shared__ shape::TAD *tad;
				__shared__ int tadLength;
				__shared__ int tadRank;
				__shared__ int numTads;
				__shared__ int *tadShape;
				__shared__ int *tadStride;
				if (threadIdx.x == 0) {
				    extern __shared__ unsigned char shmem[];
				    sPartials = (T *) shmem;
					tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
					tadRank = shape::rank(tadOnlyShapeInfo);
					numTads = shape::length(xShapeInfo) / tadLength;

					tadShape = shape::shapeOf(tadOnlyShapeInfo);
					tadStride = shape::stride(tadOnlyShapeInfo);
				}
				__syncthreads();

				int xCoord[MAX_RANK];

				for (int r = blockIdx.x; r < numTads; r += gridDim.x) {
					Nd4jIndex tadOffsetForBlock = tadOffsets[r];

					sPartials[threadIdx.x] = OpType::startingValue(dx + tadOffsetForBlock);

					for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
						shape::ind2subC(tadRank, tadShape, i, xCoord);
						Nd4jIndex xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

						sPartials[threadIdx.x] = OpType::update(sPartials[threadIdx.x], OpType::op(dx[xOffset], extraParams), extraParams);
					}
					__syncthreads();

					// aggregate. do NOT reduce for elements > tadLength
					aggregatePartials<OpType>(sPartials, threadIdx.x, nd4j::math::nd4j_min<int>(blockDim.x, tadLength), extraParams);


					__syncthreads();
					if (threadIdx.x == 0)
						result[r] = OpType::postProcess(sPartials[threadIdx.x], tadLength, extraParams);
				}
			}

			/**
			 *
			 * @param sPartialsRef
			 * @param tid
			 * @param extraParams
			 */
template<typename OpType>
			__device__ static inline void aggregatePartials(T *sPartials, int tid, int numItems, T *extraParams) {
				// start the shared memory loop on the next power of 2 less
				// than the block size.  If block size is not a power of 2,
				// accumulate the intermediate sums in the remainder range.
				int floorPow2 = numItems;

				if (floorPow2 & (floorPow2 - 1)) {
					while (floorPow2 & (floorPow2 - 1)) {
						floorPow2 &= floorPow2 - 1;
					}
					if (tid >= floorPow2) {
						sPartials[tid - floorPow2] = OpType::update(sPartials[tid - floorPow2], sPartials[tid], extraParams);
					}

					__syncthreads();
				}


				for (int activeThreads = floorPow2 >> 1; activeThreads; activeThreads >>= 1) {
					if (tid < activeThreads && tid + activeThreads < numItems) {
						sPartials[tid] = OpType::update(sPartials[tid], sPartials[tid + activeThreads], extraParams);
					}
                    __syncthreads();
				}

			}

#endif

            static T execScalar(const int opNum, T *x, int *xShapeInfo, T *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_OPS);
            }

            static void exec(const int opNum,
                             T *x,
                             int *xShapeInfo,
                             T *extraParams,
                             T *result,
                             int *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset) {
                DISPATCH_BY_OPNUM(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParams,
                                               result,
                                               resultShapeInfoBuffer,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset),
                                  REDUCE_OPS);
            }

            /**
             * Reduce down to 1 number
             * @param x the input
             * @param xShapeInfo the shape information
             * for the input
             * @param extraParams the extra params
             * @return
             */
            template<typename OpType>
#ifdef __CUDACC__
            __host__
#endif
            static T execScalar(T *x, int *xShapeInfo, T *extraParams) {
                const Nd4jIndex length = shape::length(xShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                if (xElementWiseStride >= 1) {
                    return execScalar<OpType>(x, xElementWiseStride, length, extraParams);
                }
                else {
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];

                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *xStride = shape::stride(xShapeInfo);
                    T start = OpType::startingValue(x);
                    int rank = shape::rank(xShapeInfo);

                    if (PrepareOneRawArrayIter<T>(rank,
                                                  xShape,
                                                  x,
                                                  xStride,
                                                  &rank,
                                                  shapeIter,
                                                  &x,
                                                  xStridesIter) >= 0) {

                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                const T *xIter = x;
                                start = OpType::update(start, OpType::op(xIter[0], extraParams), extraParams);
                            }
                        ND4J_RAW_ITER_ONE_NEXT(dim,
                                               rank,
                                               coord,
                                               shapeIter,
                                               x,
                                               xStridesIter);
                        start = OpType::postProcess(start, shape::length(xShapeInfo), extraParams);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }

                    return start;

                }

            }

            /**
             * Execute on the cpu
             * @param x the input data
             * @param xShapeInfo the shape information for x
             * @param extraParams the extra parameters
             * @param result the result buffer
             * @param resultShapeInfoBuffer the shape information
             * @param dimension the dimension to perform
             * the reduce along long
             * @param dimensionLength the length of the dimension buffer
             */


            template<typename OpType>
#ifdef __CUDACC__
            __host__
#endif
            static void exec(T *x,
                             int *xShapeInfo,
                             T *extraParams,
                             T *result,
                             int *resultShapeInfoBuffer,
                             int *dimension,
                             int dimensionLength,
                             int *tadShapeInfo,
                             Nd4jIndex *tadOffset) {

                int resultLength = shape::length(resultShapeInfoBuffer);

                //pre squeezed: this is for keeping the pointer to the original
                //shape information for tad offset
                //the squeezed information doesn't render the right strides for
                //tad offset
                // || tad.wholeThing
                if (resultLength == 1 || dimension == nullptr || dimensionLength == shape::rank(xShapeInfo)) {
                    result[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
                    return;
                }

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecial(x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength, tadShapeInfo, tadOffset);
                    return;
                }

                int *tadOnlyShapeInfo = tadShapeInfo;
                Nd4jIndex *tadOffsets = tadOffset;
                shape::TAD *tad = nullptr;

                if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
                    tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    if (tad->dimensionLength < 1) {
                        delete tad;
                        return;
                    }

                    tadOnlyShapeInfo = tad->tadOnlyShapeInfo;
                    tadOffsets = tad->tadOffsets;
                }


                const int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                int numTads = shape::length(xShapeInfo) / tadLength;
                int tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);

                int tadsPerThread = resultLength / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                if (tadEWS > 0 && (numTads == 1 || shape::isVector(tadOnlyShapeInfo) || shape::isScalar(tadOnlyShapeInfo))) {

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < resultLength; i++) {
                        T *iter = x + tadOffsets[i];
                        T start = OpType::startingValue(iter);
                        if (tadEWS == 1) {

// FIXME: proper reduction should be used here
                            for (int j = 0; j < tadLength; j++) {
                                start = OpType::update(start, OpType::op(iter[j], extraParams), extraParams);

                            }
                        }
                        else {
// FIXME: proper reduction to be used here
                            for (int j = 0; j < tadLength; j++) {
                                start = OpType::update(start, OpType::op(iter[j * tadEWS], extraParams), extraParams);
                            }
                        }
                        result[i] = OpType::postProcess(start, tadLength, extraParams);
                    }
                }
                else {
                    int *tadShape = shape::shapeOf(tadOnlyShapeInfo);
                    int *tadStride = shape::stride(tadOnlyShapeInfo);
                    int tadRank = shape::rank(tadOnlyShapeInfo);

#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < resultLength; i++) {
                        Nd4jIndex offset = tadOffsets[i];
                        int xCoord[MAX_RANK];

                        T start = OpType::startingValue(x + offset);

                        for (int j = 0; j < tadLength; j++) {
                            shape::ind2subC(tadRank, tadShape, j, xCoord);
                            Nd4jIndex xOffset = shape::getOffset(offset, tadShape, tadStride, xCoord, tadRank);

                            start = OpType::update(start, OpType::op(x[xOffset], extraParams), extraParams);
                        }

                        result[i] = OpType::postProcess(start, tadLength, extraParams);;
                    }
                }

                if (tad != nullptr)
                    delete tad;
            }





            /**
            * CPU implementation
            * @param x the input data
            * @param xShapeInfo the shape information for
            * the input data
            * @param extraParams the extra parameters for the problem
            * @param result the result buffer
            * @param resultShapeInfo the shape information
            */
            template<typename OpType>
#ifdef __CUDACC__
            __host__
#endif
            static void exec(T *x,
                             int *xShapeInfo,
                             T *extraParams,
                             T *result,
                             int *resultShapeInfo) {
                return execScalar<OpType>(x, xShapeInfo, extraParams);
            }



            /**
            * Reduce down to 1 number
            * @param x the input
            * @param xShapeInfo the shape information
            * for the input
            * @param extraParams the extra params
            * @return
            */
            template<typename OpType>
#ifdef __CUDACC__
            __host__
#endif
            static T execScalar(const T *x, int xElementWiseStride, Nd4jIndex length, T *extraParams) {
                T startingVal = OpType::startingValue(x);
                if (xElementWiseStride == 1) {
                    if (length < ELEMENT_THRESHOLD) {
                        T local = OpType::startingValue(x);

// FIXME: proper reduction to be used here
                        for (Nd4jIndex i = 0; i < length; i++) {
                            T curr = OpType::op(x[i], extraParams);
                            local = OpType::update(local, curr, extraParams);

                        }
                        local = OpType::postProcess(local, length, extraParams);

                        return local;
                    }

                    else {
                        T finalVal = startingVal;
                        BlockInformation info(length, ELEMENT_THRESHOLD);
                        T *blocks = new T[info.threads];

#pragma omp parallel num_threads(info.threads) if (info.threads > 1) proc_bind(AFFINITY) default(shared)
                        {
                            T local = OpType::startingValue(x);
                            for (int i = omp_get_thread_num(); i < info.chunks; i += info.threads) {
                                Nd4jIndex newOffset = (i * info.items);
                                const T *chunk = x + newOffset;
                                Nd4jIndex itemsToLoop = info.items;
                                if (i * info.items >= length) {
                                    break;
                                }

                                //handle modulo case
                                if (newOffset + info.items >= length) {
                                    itemsToLoop = length - newOffset;
                                }

// FIXME: proper reduction should be used here
                                for (Nd4jIndex j = 0; j < itemsToLoop && i * info.items + j < length; j++) {
                                    T curr = OpType::op(chunk[j], extraParams);
                                    local = OpType::update(local, curr, extraParams);
                                }

                            }

                            blocks[omp_get_thread_num()] = local;
                        }

// FIXME: proper reduction should be used here
                        for (int i = 0; i < info.threads; i++) {
                            finalVal = OpType::update(finalVal, blocks[i], extraParams);
                        }


                        finalVal = OpType::postProcess(finalVal, length, extraParams);
                        delete[] blocks;
                        return finalVal;

                    }

                }

                else {
                    if (length < ELEMENT_THRESHOLD) {
                        T local = OpType::startingValue(x);

// FIXME: proper reduction should be used here
                        for (Nd4jIndex i = 0; i < length; i++) {
                            T curr = OpType::op(x[i * xElementWiseStride], extraParams);
                            local = OpType::update(local, curr, extraParams);

                        }

                        local = OpType::postProcess(local, length, extraParams);

                        return local;
                    }

                    T finalVal = startingVal;
                    BlockInformation info(length, ELEMENT_THRESHOLD);
                    T *blocks = new T[info.threads];


#pragma omp parallel num_threads(info.threads) if (info.threads > 1) proc_bind(AFFINITY) default(shared)
                    {
                        T local = OpType::startingValue(x);
                        for (int i = omp_get_thread_num(); i < info.chunks; i += info.threads) {
                            Nd4jIndex newOffset = (i * info.items) * xElementWiseStride;
                            const T *chunk = x + newOffset;
                            Nd4jIndex itemsToLoop = info.items;
                            if (i * info.items >= length)
                                break;

// FIXME: proper reduction should be used here
                            for (Nd4jIndex j = 0; j < itemsToLoop && i * info.items + j < length; j++) {
                                T curr = OpType::op(chunk[j * xElementWiseStride], extraParams);
                                local = OpType::update(local, curr, extraParams);
                            }
                        }

                        blocks[omp_get_thread_num()] = local;
                    }

// FIXME: proper reduction should be used here
                    for (int i = 0; i < info.threads; i++) {
                        finalVal = OpType::update(finalVal, blocks[i], extraParams);
                    }

                    finalVal = OpType::postProcess(finalVal, length, extraParams);
                    delete[] blocks;
                    return finalVal;

                }

            }
        };

#ifdef __CUDACC__
        /**
    *
    * @param extraParams
    * @param sPartials
    * @param sMemSize
    */
        template<typename T>
        __device__ void initializeShared(T *extraParams, T **sPartials, int sMemSize) {
            int sPartialsLength = sMemSize / sizeof(T);
            T *sPartialsDeref = (T *) *sPartials;
            for (int i = 0; i < sPartialsLength; i++) {
                sPartialsDeref[i] = extraParams[0];
            }
        }

#endif

    }

}


#ifdef __CUDACC__

/**
 * Interface for the c and driver api
 * @param op the operation number
 * @param n the length of the problem
 * @param dx  the input information
 * @param xShapeInfo the shape information
 * @param extraParams the extra parameters
 * @param result the result data
 * @param resultShapeInfo the result shape information
 * @param gpuInformation the gpu information
 * @param dimension the dimension to do reduce along long
 * @param dimensionLength the length of the dimension buffer
 * @param postProcessOrNot whether to pre process or not
 */
template <typename T, typename OpClass>
__device__ void reduceSimpleGeneric(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer, int *tadOnlyShapeInfo, Nd4jIndex *tadOffsets) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceFunction<T>), sizeof(shape::TAD), shape::rank(xShapeInfo));
    }
    __syncthreads();


    functions::reduce::ReduceFunction<T>::template transformCudaXD<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            reductionBuffer,
            manager,
            tadOnlyShapeInfo,
            tadOffsets);
}

template <typename T, typename OpClass>
__device__ void reduceSimpleGeneric1D(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer,
        int *tadOnlyShapeInfo,
        Nd4jIndex *tadOffsets) {

    functions::reduce::ReduceFunction<T>::template transformCuda1D<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            reductionBuffer, nullptr, tadOnlyShapeInfo, tadOffsets);
}

template <typename T, typename OpClass>
__device__ void reduceSimpleGeneric3D(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer,
        int *tadOnlyShapeInfo,
        Nd4jIndex *tadOffsets) {

    functions::reduce::ReduceFunction<T>::template transformCuda3D<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            reductionBuffer,
            nullptr,
            tadOnlyShapeInfo,
            tadOffsets);
}

template <typename T, typename OpClass>
__device__ void reduceScalarGeneric(
        T *dx,
        int *xShapeInfo,
        T *extraParams,
        T *result,
        int *resultShapeInfo,
        int *dimension,
        int dimensionLength,
        T *reductionBuffer, int *tadOnlyShapeInfo) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
        manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::reduce::ReduceFunction<T>), sizeof(shape::TAD), 0);
    }
    __syncthreads();

    functions::reduce::ReduceFunction<T>::template execScalarCuda<OpClass>(
            dx,
            xShapeInfo,
            extraParams,
            result,
            resultShapeInfo,
            reductionBuffer,
            manager,
            tadOnlyShapeInfo);
};



/*

*/

// reduceScalar
DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, float, INPUT(float *x, int *xShapeInfo, float *extraParams, float *z, int *zShapeInfo, int *dimension, int dimensionLength, float *reductionBuffer, int *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, double, INPUT(double *x, int *xShapeInfo, double *extraParams, double *z, int *zShapeInfo, int *dimension, int dimensionLength, double *reductionBuffer, int *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceScalarSimple_, reduceScalarGeneric, float16, INPUT(float16 *x, int *xShapeInfo, float16 *extraParams, float16 *z, int *zShapeInfo, int *dimension, int dimensionLength, float16 *reductionBuffer, int *tadOnlyShapeInfo), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, reductionBuffer, tadOnlyShapeInfo), OPS_A(REDUCE_OPS))

// reduce1D
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric1D_, reduceSimpleGeneric1D, float, INPUT(float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric1D_, reduceSimpleGeneric1D, double, INPUT(double *x, int *xShape, double *extraParams, double *z, int *zShape, int *dimension, int dimensionLength, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric1D_, reduceSimpleGeneric1D, float16, INPUT(float16 *x, int *xShape, float16 *extraParams, float16 *z, int *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))

// reduce3D
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric3D_, reduceSimpleGeneric3D, float, INPUT(float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric3D_, reduceSimpleGeneric3D, double, INPUT(double *x, int *xShape, double *extraParams, double *z, int *zShape, int *dimension, int dimensionLength, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGeneric3D_, reduceSimpleGeneric3D, float16, INPUT(float16 *x, int *xShape, float16 *extraParams, float16 *z, int *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))

// reduceXD
DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, float, INPUT(float *x, int *xShape, float *extraParams, float *z, int *zShape, int *dimension, int dimensionLength, float *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, double, INPUT(double *x, int *xShape, double *extraParams, double *z, int *zShape, int *dimension, int dimensionLength, double *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))
DISPATCH_KERNEL_SIMPLE(reduceSimpleGenericXD_, reduceSimpleGeneric, float16, INPUT(float16 *x, int *xShape, float16 *extraParams, float16 *z, int *zShape, int *dimension, int dimensionLength, float16 *reductionPointer, int *tadShapeInfo, Nd4jIndex *tadOffsets), PARAMS(x, xShape, extraParams, z, zShape, dimension, dimensionLength, reductionPointer, tadShapeInfo, tadOffsets), OPS_A(REDUCE_OPS))


#endif

#endif

