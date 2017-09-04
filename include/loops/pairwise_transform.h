/*
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_
#ifdef _OPENMP
#include <omp.h>
#endif
#include <templatemath.h>
#include <helper_cuda.h>
#include <helpers/shape.h>
#include <pairwise_util.h>
#include <dll.h>
#include <stdio.h>
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

#define PAIRWISE_TRANSFORM_OPS \
        (0, simdOps::Add),\
        (1, simdOps::Copy),\
        (2, simdOps::Divide),\
        (3, simdOps::EqualTo),\
        (4, simdOps::GreaterThan),\
        (5, simdOps::LessThan),\
        (6, simdOps::Multiply),\
        (7, simdOps::Pow),\
        (8, simdOps::ReverseSubtract),\
        (9, simdOps::Subtract),\
        (10,simdOps::Epsilon),\
        (11,simdOps::GreaterThanOrEqual),\
        (12,simdOps::LessThanOrEqual),\
        (13,simdOps::Max),\
        (14,simdOps::Min),\
        (15,simdOps::NotEqualTo),\
        (16,simdOps::Copy),\
        (17,simdOps::Axpy),\
        (18,simdOps::ReverseDivide),\
        (45,simdOps::CompareAndSet),\
        (46,simdOps::CompareAndReplace),\
        (56,simdOps::And),\
        (57,simdOps::Or),\
        (58,simdOps::Xor),\
        (59,simdOps::Remainder),\
        (60,simdOps::FMod),\
        (69,simdOps::Atan2)






namespace functions {
    namespace pairwise_transforms {

/**
 * Transforms involving 2 arrays
 */
        template<typename T>
        class PairWiseTransform {
        public:

#ifdef __CUDACC__


		static inline __device__ void transformCuda(
				const int opNum,
				Nd4jIndex n,
				T *dx,
				T *y,
				int incx,
				int incy,
				T *extraParams,
				T *result,
				int incz,
				int *allocationPointer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo), PAIRWISE_TRANSFORM_OPS);
				
			}


			static inline __device__ void transformCuda(
				const int opNum,
				T *dx,
				int *xShapeBuffer,
				T *y,
				int *yShapeBuffer,
				T *result,
				int *resultShapeBuffer,
				T *extraParams,
				int *allocationPointer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo) {
                            DISPATCH_BY_OPNUM(transformCuda, PARAMS(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo), PAIRWISE_TRANSFORM_OPS);
			}


			static inline __device__ void transformCuda(
				const int opNum,
				T *dx,
				int *xShapeBuffer,
				T *y,
				int *yShapeBuffer,
				T *result,
				int *resultShapeBuffer,
				T *extraParams,
				int *indexes,
				int *yIndexes,
				int *resultIndexes,
				int *allocationPointer,
				UnifiedSharedMemory *manager,
				int *tadOnlyShapeInfo) {
                            DISPATCH_BY_OPNUM(transform, PARAMS(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo), PAIRWISE_TRANSFORM_OPS);
			}
            /**
	 *
	 */
	virtual __inline__ __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			Nd4jIndex n,
			int *indexes,
			int *allocationPointer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo) {
		transform(dx,
				xShapeBuffer,
				y,
				yShapeBuffer,
				result,
				resultShapeBuffer,
				extraParams,
				indexes,
				indexes,
				indexes, allocationPointer, manager, tadOnlyShapeInfo);
	}

	/**
	 *
	 */
template<typename OpType>
	static inline __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *indexes,
			int *yIndexes,
			int *resultIndexes,
			int *allocationPointer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		Nd4jIndex n = shape::length(xShapeBuffer);

		for (Nd4jIndex i = tid; i < n; i += gridDim.x * blockDim.x) {
			result[resultIndexes[i]] = OpType::op(dx[indexes[i]],y[yIndexes[i]], extraParams);
		}
	}


	/**
	 *
	 */
	virtual __inline__ __device__ void transform(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams,
			int *indexes,
			int *yIndexes,
			int *allocationPointer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo) {
		transform(dx,
				xShapeBuffer,
				y,
				yShapeBuffer,
				result,
				resultShapeBuffer,
				extraParams,
				indexes,
				yIndexes,
				indexes, allocationPointer, manager, tadOnlyShapeInfo);
	}

	/**
	 *
	 */
template<typename OpType>
	static inline __device__ void transformCuda(
			T *dx,
			int *xShapeBuffer,
			T *y,
			int *yShapeBuffer,
			T *result,
			int *resultShapeBuffer,
			T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		__shared__ int xRank;
		__shared__ int yRank;
		__shared__ int resultRank;

		__shared__ int xEWS;
		__shared__ int yEWS;
		__shared__ int zEWS;

		__shared__ char xOrder;
		__shared__ char yOrder;
		__shared__ char zOrder;

		__shared__ bool xRow;
		__shared__ bool yRow;
		__shared__ bool zRow;

		if (threadIdx.x == 0) {
		    xRank = shape::rank(xShapeBuffer);
		    yRank = shape::rank(yShapeBuffer);
		    resultRank = shape::rank(resultShapeBuffer);

		    xEWS = shape::elementWiseStride(xShapeBuffer);
		    yEWS = shape::elementWiseStride(yShapeBuffer);
		    zEWS = shape::elementWiseStride(resultShapeBuffer);

		    xOrder = shape::order(xShapeBuffer);
		    yOrder = shape::order(yShapeBuffer);
		    zOrder = shape::order(resultShapeBuffer);

		    xRow = shape::isRowVector(xShapeBuffer);
		    yRow = shape::isRowVector(yShapeBuffer);
		    zRow = shape::isRowVector(resultShapeBuffer);

		}
		__syncthreads();

		Nd4jIndex n = shape::length(xShapeBuffer);
		if((xEWS >= 1 && yEWS == xEWS && zEWS == xEWS &&  xOrder == yOrder && zOrder == xOrder) || (xEWS >= 1 && yEWS == xEWS && zEWS == xEWS && xRow && yRow && zRow)) {
			// TODO: this is wrong, and should be moved to host side
			transformCuda<OpType>(
					n,
					dx,
					y,
					xEWS,
					yEWS,
					extraParams,
					result,
					zEWS, allocationPointer, manager, tadOnlyShapeInfo);

		}

		else {

			int xCoord[MAX_RANK];
			int yCoord[MAX_RANK];

			if (dx == result) {
				for (Nd4jIndex i = tid; i < n; i += gridDim.x * blockDim.x) {
					shape::ind2subC(xRank,shape::shapeOf(xShapeBuffer), i, xCoord);
					shape::ind2subC(yRank,shape::shapeOf(yShapeBuffer), i, yCoord);

					Nd4jIndex xOffset = shape::getOffset(0, shape::shapeOf(xShapeBuffer), shape::stride(xShapeBuffer), xCoord, xRank);
					Nd4jIndex yOffset = shape::getOffset(0, shape::shapeOf(yShapeBuffer), shape::stride(yShapeBuffer), yCoord, yRank);
					result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
				}
			} else {
    			int resultCoord[MAX_RANK];

				for (Nd4jIndex i = tid; i < n; i += gridDim.x * blockDim.x) {
					shape::ind2subC(xRank,shape::shapeOf(xShapeBuffer), i, xCoord);
					shape::ind2subC(yRank,shape::shapeOf(yShapeBuffer), i, yCoord);
					shape::ind2subC(resultRank,shape::shapeOf(resultShapeBuffer), i, resultCoord);

					Nd4jIndex xOffset = shape::getOffset(0, shape::shapeOf(xShapeBuffer), shape::stride(xShapeBuffer), xCoord, xRank);
					Nd4jIndex yOffset = shape::getOffset(0, shape::shapeOf(yShapeBuffer), shape::stride(yShapeBuffer), yCoord, yRank);
					Nd4jIndex resultOffset = shape::getOffset(0, shape::shapeOf(resultShapeBuffer), shape::stride(resultShapeBuffer), resultCoord, resultRank);
					result[resultOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
				}
			}

		}
	}

	/**
	 *
	 * @param n
	 * @param xOffset
	 * @param yOffset
	 * @param resultOffset
	 * @param dx
	 * @param dy
	 * @param incx
	 * @param incy
	 * @param params
	 * @param result
	 * @param incz
	 * @param blockSize
	 */
template<typename OpType>
	static inline __device__ void transformCuda(
			Nd4jIndex n,
			T *dx,
			T *dy,
			int incx,
			int incy,
			T *params,
			T *result,
			int incz,int *allocationPointer,
			UnifiedSharedMemory *manager,
			int *tadOnlyShapeInfo) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (incx == incy && incy == incz && incx == 1) {
			for (Nd4jIndex i = tid; i < n; i += gridDim.x * blockDim.x) {
				result[i] = OpType::op(dx[i], dy[i], params);
			}
		} else {
			for (Nd4jIndex i = tid; i < n; i += gridDim.x * blockDim.x) {
				result[i * incz] = OpType::op(dx[i * incx], dy[i * incy], params);
			}
		}
	}

#endif
        public:
			static void exec(
				const int opNum,
				T *dx,
				int *xShapeBuffer,
				T *y,
				int *yShapeBuffer,
				T *result,
				int *resultShapeBuffer,
				T *extraParams,
				int *indexes,
				int *yIndexes,
				int *resultIndexes) {
                            DISPATCH_BY_OPNUM(exec, PARAMS(dx,
                                                           xShapeBuffer,
                                                           y,
                                                           yShapeBuffer,
                                                           result, resultShapeBuffer,
                                                           extraParams,
                                                           indexes,
                                                           yIndexes,
                                                           resultIndexes), PAIRWISE_TRANSFORM_OPS);
			}

			static void exec(
				const int opNum,
				T *dx,
				int *xShapeBuffer,
				T *y,
				int *yShapeBuffer,
				T *result,
				int *resultShapeBuffer,
				T *extraParams) {
				DISPATCH_BY_OPNUM(exec, PARAMS(dx,
                                               xShapeBuffer,
                                               y,
                                               yShapeBuffer,
                                               result,
                                               resultShapeBuffer,
                                               extraParams),
                                  PAIRWISE_TRANSFORM_OPS);
			}
			
			static void exec(
				const int opNum,
				T *dx,
				Nd4jIndex xStride,
				T *y,
				Nd4jIndex yStride,
				T *result,
				Nd4jIndex resultStride,
				T *extraParams,
				Nd4jIndex n) {
				DISPATCH_BY_OPNUM(exec, PARAMS(dx,
                                               xStride,
                                               y,
                                               yStride,
                                               result,
                                               resultStride,
                                               extraParams,
                                               n), PAIRWISE_TRANSFORM_OPS);
			}

			template<typename OpType>
			static void exec(
                    T *dx,
                    int *xShapeBuffer,
                    T *y,
                    int *yShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams,
                    int *indexes,
                    int *yIndexes,
                    int *resultIndexes) {
                Nd4jIndex n = shape::length(xShapeBuffer);

#pragma omp parallel for simd schedule(guided) proc_bind(AFFINITY) default(shared)
                for (Nd4jIndex i = 0; i < n; i++) {
                    result[resultIndexes[i]] = OpType::op(dx[indexes[i]], y[yIndexes[i]], extraParams);

                }
            }

			template<typename OpType>
			static void exec(
                    T *dx,
                    int *xShapeBuffer,
                    T *y,
                    int *yShapeBuffer,
                    T *result,
                    int *resultShapeBuffer,
                    T *extraParams) {
                Nd4jIndex n = shape::length(xShapeBuffer);
                int xElementWiseStride = shape::elementWiseStride(xShapeBuffer);
                int yElementWiseStride = shape::elementWiseStride(yShapeBuffer);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);

                bool sameShape = shape::shapeEquals(shape::rank(xShapeBuffer), shape::shapeOf(xShapeBuffer),
                                                    shape::rank(yShapeBuffer), shape::shapeOf(yShapeBuffer));



                if (xElementWiseStride >= 1 &&
                    yElementWiseStride >= 1 &&
                    resultElementWiseStride >= 1 &&
                    shape::order(xShapeBuffer) == shape::order(yShapeBuffer) &&
                    shape::order(resultShapeBuffer) == shape::order(xShapeBuffer) &&
                    sameShape &&  xElementWiseStride == yElementWiseStride) {

                    exec<OpType>(dx,
                         xElementWiseStride,
                         y,
                         yElementWiseStride,
                         result,
                         resultElementWiseStride,
                         extraParams,
                         n);
                }
                    //not same shape
                else if (!sameShape && shape::order(xShapeBuffer) == shape::order(yShapeBuffer) &&
                         shape::order(resultShapeBuffer) == shape::order(xShapeBuffer) && xElementWiseStride >= 1 &&
                         yElementWiseStride >= 1 &&
                         resultElementWiseStride >= 1 && xElementWiseStride == yElementWiseStride) {

                    exec<OpType>(dx,
                         xElementWiseStride,
                         y,
                         yElementWiseStride,
                         result,
                         resultElementWiseStride,
                         extraParams,
                         shape::length(yShapeBuffer));
                }

                else if (sameShape) {
                    int rank = shape::rank(xShapeBuffer);
                    int *xShape = shape::shapeOf(xShapeBuffer);

                    int *xStride = shape::stride(xShapeBuffer);
                    int *yStride = shape::stride(yShapeBuffer);
                    int *resultStride = shape::stride(resultShapeBuffer);

                    // tad-oriented rotation technically

                    int tadsPerThread = xShape[0] / TAD_THRESHOLD;
                    int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                    num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
for (Nd4jIndex i = 0; i < xShape[0]; i++) {
                    T *dxLocal = dx + xStride[0] * i;
                    T *yLocal = y + yStride[0] * i;
                    T *resultLocal = result + resultStride[0] * i;

                    int rankLocal = rank - 1;
                    int *xShapeLocal = xShape + 1;

                    int *xStrideLocal = xStride + 1;
                    int *yStrideLocal = yStride + 1;
                    int *resultStrideLocal = resultStride + 1;

                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int yStridesIter[MAX_RANK];
                    int resultStridesIter[MAX_RANK];
                    if (PrepareThreeRawArrayIter<T>(rankLocal,
                                                    xShapeLocal,
                                                    dxLocal,
                                                    xStrideLocal,
                                                    yLocal,
                                                    yStrideLocal,
                                                    resultLocal,
                                                    resultStrideLocal,
                                                    rankLocal,
                                                    shapeIter,
                                                    &dxLocal,
                                                    xStridesIter,
                                                    &yLocal,
                                                    yStridesIter,
                                                    &resultLocal,
                                                    resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rankLocal, coord, shapeIter); {
                                // Process the innermost dimension
                                T *xIter = dxLocal;
                                T *yIter = yLocal;
                                T *resultIter = resultLocal;
                                resultIter[0] = OpType::op(xIter[0], yIter[0], extraParams);
                            }
                        ND4J_RAW_ITER_THREE_NEXT(dim,
                                                 rankLocal,
                                                 coord,
                                                 shapeIter,
                                                 dxLocal,
                                                 xStridesIter,
                                                 yLocal,
                                                 yStridesIter,
                                                 resultLocal,
                                                 resultStridesIter);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }
                    }

                }

                else {
                    Nd4jIndex len = shape::length(xShapeBuffer);
                    int xRank = shape::rank(xShapeBuffer);
                    int yRank = shape::rank(yShapeBuffer);
                    int resultRank = shape::rank(resultShapeBuffer);

                    int *xShape = shape::shapeOf(xShapeBuffer);
                    int *xStride = shape::stride(xShapeBuffer);

                    int *yShape = shape::shapeOf(yShapeBuffer);
                    int *yStride = shape::stride(yShapeBuffer);

                    int *resultShape = shape::shapeOf(resultShapeBuffer);
                    int *resultStride = shape::stride(resultShapeBuffer);

                    int elementsPerThread = n / ELEMENT_THRESHOLD;
                    int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                    num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                    int xCoord[MAX_RANK];
                    int yCoord[MAX_RANK];

                    if(dx == result) {
#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(xCoord, yCoord)
                        for (Nd4jIndex i = 0; i < len; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(yRank,yShape, i, yCoord);

                            Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                            result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);

                        }
                    }
                    else {
                        int resultCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(xCoord, yCoord, resultCoord)
                        for (Nd4jIndex i = 0; i < len; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(yRank,yShape, i, yCoord);
                            shape::ind2subC(resultRank,resultShape, i, resultCoord);

                            Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                            Nd4jIndex resultOffset = shape::getOffset(0, resultShape, resultStride, resultCoord, resultRank);
                            result[resultOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);

                        }
                    }
                }
            }

            template<typename OpType>
            static void exec(T *dx,
                             Nd4jIndex xStride,
                             T *y,
                             Nd4jIndex yStride,
                             T *result,
                             Nd4jIndex resultStride,
                             T *extraParams,
                             const Nd4jIndex n) {
                int elementsPerThread = n / ELEMENT_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

                int span = (n / _threads) + 8;

                if (xStride == 1 && yStride == 1 && resultStride == 1) {
#pragma omp parallel num_threads(_threads) if (_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jIndex tid = omp_get_thread_num();
                        Nd4jIndex start = span * tid;
                        Nd4jIndex end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i] = OpType::op(dx[i], y[i], extraParams);
                        }
                    }
                }
                else {
#pragma omp parallel num_threads(_threads) if (_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jIndex tid = omp_get_thread_num();
                        Nd4jIndex start = span * tid;
                        Nd4jIndex end = span * (tid + 1);
                        if (end > n) end = n;

#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i * resultStride] = OpType::op(dx[i * xStride], y[i * yStride], extraParams);
                        }
                    }
                }
            }
        };
    }
}

#ifdef __CUDACC__

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template <typename T>
__device__ void pairWiseTransformGeneric(
		int opNum,
		T *dx,
		T *dy,
		T *params,
		T *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank, int *allocationPointer, int *tadOnlyShapeInfo) {

	__shared__ UnifiedSharedMemory *manager;

     if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::pairwise_transforms::PairWiseTransform<T>), sizeof(shape::TAD), xRank);
    }
    __syncthreads();

	functions::pairwise_transforms::PairWiseTransform<T>::transformCuda(
		opNum,
	    dx,
	    xShapeInfo,
	    dy,
	    yShapeInfo,
	    result,
	    resultShapeInfo,
	    params,
	    allocationPointer,
	    manager, tadOnlyShapeInfo);
}


/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformDouble(
		int opNum,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<double>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformFloat(
		int opNum,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo);

}

extern "C" __global__ void pairWiseTransformHalf(
		int opNum,
		float16 *dx,
		float16 *dy,
		float16 *params,
		float16 *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float16>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank, allocationPointer, tadOnlyShapeInfo);

}

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template <typename T>
__device__ void pairWiseTransformGeneric(
		int opNum,
		T *dx,
		T *dy,
		T *params,
		T *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {

	__shared__ UnifiedSharedMemory *manager;

     if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::pairwise_transforms::PairWiseTransform<T>), sizeof(shape::TAD), xRank);
    }
    __syncthreads();

	functions::pairwise_transforms::PairWiseTransform<T>::transformCuda(
			opNum,
			dx,
			xShapeInfo,
			dy,
			yShapeInfo,
			result,
			resultShapeInfo,
			params,
			xIndexes,
			yIndexes,
			resultIndexes,
			allocationPointer,
			manager,
			tadOnlyShapeInfo);

}


/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformDoubleIndex(
		int opNum,
		double *dx,
		double *dy,
		double *params,
		double *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<double>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, tadOnlyShapeInfo);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformFloatIndex(
		int opNum,
		float *dx,
		float *dy,
		float *params,
		float *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, tadOnlyShapeInfo);
}

extern "C" __global__ void pairWiseTransformHalfIndex(
		int opNum,
		float16 *dx,
		float16 *dy,
		float16 *params,
		float16 *result,
		int *xShapeInfo, int xRank,
		int *yShapeInfo, int yRank,
		int *resultShapeInfo, int zRank,
		int *xIndexes,
		int *yIndexes,
		int *resultIndexes, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformGeneric<float16>(
			opNum,
			dx,
			dy,
			params,
			result,
			xShapeInfo, xRank,
			yShapeInfo, yRank,
			resultShapeInfo, zRank,
			xIndexes,
			yIndexes,
			resultIndexes, allocationPointer, tadOnlyShapeInfo);
}

/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
template<typename T>
__device__ void pairWiseTransformStridedGeneric(
		const int opNum,
		Nd4jIndex n,
		T *dx,
		T *dy,
		int incx,
		int incy,
		T *params,
		T *result,
		int incz, int *allocationPointer, int *tadOnlyShapeInfo) {

	__shared__ UnifiedSharedMemory *manager;

     if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::pairwise_transforms::PairWiseTransform<T>), sizeof(shape::TAD), 0);
	}
	__syncthreads();

	functions::pairwise_transforms::PairWiseTransform<T>::transformCuda(
		opNum,
		n,
		dx,
		dy,
		incx,
		incy,
		params,
		result,
		incz,
		allocationPointer,
		manager,
		tadOnlyShapeInfo);

}



/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformStridedDouble(
		int opNum,
		Nd4jIndex n,
		double *dx,
		double *dy,
		int incx,
		int incy,
		double *params,
		double *result,
		int incz, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformStridedGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer, tadOnlyShapeInfo);
}
/**
 * The api for the driver interface
 * @param opNum the op number
 * @param n the length of the problem
 * @param xOffset the offset for x
 * @param yOffset the offset for y
 * @param resultOffset the offset for result
 * @param dx the input
 * @param dy the pair wise array
 * @param incx the stride for x
 * @param incy the stride for y
 * @param params the parameters for the problem
 * @param result the result buffer
 * @param incz the result stride
 * @param blockSize the block size
 */
extern "C" __global__ void pairWiseTransformStridedFloat(
		int opNum,
		Nd4jIndex n,
		float *dx,
		float *dy,
		int incx,
		int incy,
		float *params,
		float *result,
		int incz, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformStridedGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer, tadOnlyShapeInfo);
}


extern "C" __global__ void pairWiseTransformStridedHalf(
		int opNum,
		Nd4jIndex n,
		float16 *dx,
		float16 *dy,
		int incx,
		int incy,
		float16 *params,
		float16 *result,
		int incz, int *allocationPointer, int *tadOnlyShapeInfo) {
	pairWiseTransformStridedGeneric<float16>(
			opNum,
			n,
			dx,
			dy,
			incx,
			incy,
			params,
			result,
			incz, allocationPointer, tadOnlyShapeInfo);
}


#endif


#endif /* PAIRWISE_TRANSFORM_H_ */
