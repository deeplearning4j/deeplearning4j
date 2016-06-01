/*
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_
#ifdef __JNI__
#include <jni.h>
#endif
#include <omp.h>
#include <templatemath.h>
#include <helper_cuda.h>
#include <shape.h>
#include <pairwise_util.h>
#include <dll.h>
#include <stdio.h>
#include <ops.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
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
				const int op,
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
				if (op == 0)
					transformCuda<simdOps::Add<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 1)
					transformCuda<simdOps::Copy<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 2)
					transformCuda<simdOps::Divide<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 3)
					transformCuda<simdOps::EqualTo<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 4)
					transformCuda<simdOps::GreaterThan<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 5)
					transformCuda<simdOps::LessThan<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 6)
					transformCuda<simdOps::Multiply<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 7)
					transformCuda<simdOps::ReverseDivide<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 8)
					transformCuda<simdOps::ReverseSubtract<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 9)
					transformCuda<simdOps::Subtract<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 10)
					transformCuda<simdOps::Epsilon<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 11)
					transformCuda<simdOps::GreaterThanOrEqual<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 12)
					transformCuda<simdOps::LessThanOrEqual<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 13)
					transformCuda<simdOps::Max<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 14)
					transformCuda<simdOps::Min<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 15)
					transformCuda<simdOps::NotEqualTo<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 16)
					transformCuda<simdOps::Copy<T>>(n, dx, y, incx, incy, extraParams, result, incz, allocationPointer, manager, tadOnlyShapeInfo);
				else
					printf("[ERROR] Unknow opNum %d for pairwise transform\n", op);
			}


			static inline __device__ void transformCuda(
				const int op,
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
				if (op == 0)
					transformCuda<simdOps::Add<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 1)
					transformCuda<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 2)
					transformCuda<simdOps::Divide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 3)
					transformCuda<simdOps::EqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 4)
					transformCuda<simdOps::GreaterThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 5)
					transformCuda<simdOps::LessThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 6)
					transformCuda<simdOps::Multiply<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 7)
					transformCuda<simdOps::ReverseDivide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 8)
					transformCuda<simdOps::ReverseSubtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 9)
					transformCuda<simdOps::Subtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 10)
					transformCuda<simdOps::Epsilon<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 11)
					transformCuda<simdOps::GreaterThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 12)
					transformCuda<simdOps::LessThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 13)
					transformCuda<simdOps::Max<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 14)
					transformCuda<simdOps::Min<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 15)
					transformCuda<simdOps::NotEqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 16)
					transformCuda<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, allocationPointer, manager, tadOnlyShapeInfo);
				else
					printf("[ERROR] Unknow opNum %d for pairwise transform\n", op);
			}


			static inline __device__ void transformCuda(
				const int op,
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
				if (op == 0)
					transform<simdOps::Add<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 1)
					transform<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 2)
					transform<simdOps::Divide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 3)
					transform<simdOps::EqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 4)
					transform<simdOps::GreaterThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 5)
					transform<simdOps::LessThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 6)
					transform<simdOps::Multiply<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 7)
					transform<simdOps::ReverseDivide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 8)
					transform<simdOps::ReverseSubtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 9)
					transform<simdOps::Subtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 10)
					transform<simdOps::Epsilon<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 11)
					transform<simdOps::GreaterThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 12)
					transform<simdOps::LessThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 13)
					transform<simdOps::Max<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 14)
					transform<simdOps::Min<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 15)
					transform<simdOps::NotEqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else if (op == 16)
					transform<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes, allocationPointer, manager, tadOnlyShapeInfo);
				else
					printf("[ERROR] Unknow opNum %d for pairwise transform\n", op);
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

		for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
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

		int xRank = shape::rank(xShapeBuffer);
		int yRank = shape::rank(yShapeBuffer);
		int resultRank = shape::rank(resultShapeBuffer);

		Nd4jIndex n = shape::length(xShapeBuffer);
		if(shape::elementWiseStride(xShapeBuffer) >= 1 && shape::elementWiseStride(yShapeBuffer) >= 1 && shape::elementWiseStride(resultShapeBuffer) >= 1 && shape::order(xShapeBuffer) == shape::order(yShapeBuffer) && shape::order(resultShapeBuffer) == shape::order(xShapeBuffer)) {

			// TODO: this is wrong, and should be moved to host side
			transformCuda<OpType>(
					n,
					dx,
					y,
					shape::elementWiseStride(xShapeBuffer),
					shape::elementWiseStride(yShapeBuffer),
					extraParams,
					result,
					shape::elementWiseStride(resultShapeBuffer), allocationPointer, manager, tadOnlyShapeInfo);

		}

		else {

			int xCoord[MAX_RANK];
			int yCoord[MAX_RANK];

			if (dx == result) {
				for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
					shape::ind2subC(xRank,shape::shapeOf(xShapeBuffer), i, xCoord);
					shape::ind2subC(yRank,shape::shapeOf(yShapeBuffer), i, yCoord);

					Nd4jIndex xOffset = shape::getOffset(0, shape::shapeOf(xShapeBuffer), shape::stride(xShapeBuffer), xCoord, xRank);
					Nd4jIndex yOffset = shape::getOffset(0, shape::shapeOf(yShapeBuffer), shape::stride(yShapeBuffer), yCoord, yRank);
					result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);
				}
			} else {
    			int resultCoord[MAX_RANK];

				for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
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
			for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
				result[i] = OpType::op(dx[i], dy[i], params);
			}
		} else {
			for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
				result[i * incz] = OpType::op(dx[i * incx], dy[i * incy], params);
			}
		}
	}

#endif
        public:
			static void exec(
				const int op,
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
				if (op == 0)
					exec<simdOps::Add<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 1)
					exec<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 2)
					exec<simdOps::Divide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 3)
					exec<simdOps::EqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 4)
					exec<simdOps::GreaterThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 5)
					exec<simdOps::LessThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 6)
					exec<simdOps::Multiply<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 7)
					exec<simdOps::ReverseDivide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 8)
					exec<simdOps::ReverseSubtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 9)
					exec<simdOps::Subtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 10)
					exec<simdOps::Epsilon<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 11)
					exec<simdOps::GreaterThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 12)
					exec<simdOps::LessThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 13)
					exec<simdOps::Max<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 14)
					exec<simdOps::Min<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 15)
					exec<simdOps::NotEqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else if (op == 16)
					exec<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams, indexes, yIndexes, resultIndexes);
				else
					printf("[ERROR] Unknow opNum %d for pairwise transform\n", op);
			}

			static void exec(
				const int op,
				T *dx,
				int *xShapeBuffer,
				T *y,
				int *yShapeBuffer,
				T *result,
				int *resultShapeBuffer,
				T *extraParams) {
				if (op == 0)
					exec<simdOps::Add<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 1)
					exec<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 2)
					exec<simdOps::Divide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 3)
					exec<simdOps::EqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 4)
					exec<simdOps::GreaterThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 5)
					exec<simdOps::LessThan<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 6)
					exec<simdOps::Multiply<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 7)
					exec<simdOps::ReverseDivide<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 8)
					exec<simdOps::ReverseSubtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 9)
					exec<simdOps::Subtract<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 10)
					exec<simdOps::Epsilon<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 11)
					exec<simdOps::GreaterThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 12)
					exec<simdOps::LessThanOrEqual<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 13)
					exec<simdOps::Max<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 14)
					exec<simdOps::Min<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 15)
					exec<simdOps::NotEqualTo<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else if (op == 16)
					exec<simdOps::Copy<T>>(dx, xShapeBuffer, y, yShapeBuffer, result, resultShapeBuffer, extraParams);
				else
					printf("[ERROR] Unknow opNum %d for pairwise transform\n", op);
			}
			
			static void exec(
				const int op,
				T *dx,
				Nd4jIndex xStride,
				T *y,
				Nd4jIndex yStride,
				T *result,
				Nd4jIndex resultStride,
				T *extraParams,
				Nd4jIndex n) {
				if (op == 0)
					exec<simdOps::Add<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 1)
					exec<simdOps::Copy<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 2)
					exec<simdOps::Divide<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 3)
					exec<simdOps::EqualTo<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 4)
					exec<simdOps::GreaterThan<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 5)
					exec<simdOps::LessThan<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 6)
					exec<simdOps::Multiply<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 7)
					exec<simdOps::ReverseDivide<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 8)
					exec<simdOps::ReverseSubtract<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 9)
					exec<simdOps::Subtract<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 10)
					exec<simdOps::Epsilon<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 11)
					exec<simdOps::GreaterThanOrEqual<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 12)
					exec<simdOps::LessThanOrEqual<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 13)
					exec<simdOps::Max<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 14)
					exec<simdOps::Min<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 15)
					exec<simdOps::NotEqualTo<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else if (op == 16)
					exec<simdOps::Copy<T>>(dx, xStride, y, yStride, result, resultStride, extraParams, n);
				else
					printf("[ERROR] Unknow opNum %d for pairwise transform\n", op);
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

#pragma omp parallel for simd schedule(guided)
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

                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int yStridesIter[MAX_RANK];
                    int resultStridesIter[MAX_RANK];
                    if (PrepareThreeRawArrayIter<T>(rank,
                                                    xShape,
                                                    dx,
                                                    xStride,
                                                    y,
                                                    yStride,
                                                    result,
                                                    resultStride,
                                                    rank,
                                                    shapeIter,
                                                    &dx,
                                                    xStridesIter,
                                                    &y,
                                                    yStridesIter,
                                                    &result,
                                                    resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                T *xIter = dx;
                                T *yIter = y;
                                T *resultIter = result;
                                resultIter[0] = OpType::op(xIter[0], yIter[0], extraParams);
                            }
                        ND4J_RAW_ITER_THREE_NEXT(dim,
                                                 rank,
                                                 coord,
                                                 shapeIter,
                                                 dx,
                                                 xStridesIter,
                                                 y,
                                                 yStridesIter,
                                                 result,
                                                 resultStridesIter);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }

                }

                else {
                    Nd4jIndex len = shape::length(xShapeBuffer);
                    int xRank = shape::rank(xShapeBuffer);
                    int yRank = shape::rank(yShapeBuffer);
                    int resultRank = shape::rank(resultShapeBuffer);
                    int *xCoord = new int[xRank];
                    int *yCoord = new int[yRank];
                    int *resultCoord = new int[resultRank];

                    int *xShape = shape::shapeOf(xShapeBuffer);
                    int *xStride = shape::stride(xShapeBuffer);

                    int *yShape = shape::shapeOf(yShapeBuffer);
                    int *yStride = shape::stride(yShapeBuffer);

                    int *resultShape = shape::shapeOf(resultShapeBuffer);
                    if(dx == result) {
                        for (Nd4jIndex i = 0; i < len; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(yRank,yShape, i, yCoord);
                            shape::ind2subC(resultRank,resultShape, i, resultCoord);

                            Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                            result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);

                        }
                    }
                    else {
                        for (Nd4jIndex i = 0; i < len; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(yRank,yShape, i, yCoord);
                            shape::ind2subC(resultRank,resultShape, i, resultCoord);

                            Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            Nd4jIndex yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                            Nd4jIndex resultOffset = shape::getOffset(0, resultShape, resultShape, resultCoord, resultRank);
                            result[resultOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);

                        }
                    }


                    delete[] xCoord;
                    delete[] yCoord;
                    delete []resultCoord;
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
                              Nd4jIndex n) {
                if (xStride == 1 && yStride == 1 && resultStride == 1) {

#pragma omp parallel for simd schedule(guided) if (n > 2048)
                        for (Nd4jIndex i = 0; i < n; i++) {
                            result[i] = OpType::op(dx[i], y[i], extraParams);
                        }
                }
                else {
#pragma omp parallel for simd schedule(guided) if (n > 2048)
                        for (Nd4jIndex i = 0; i < n; i++) {
                            result[i * resultStride] = OpType::op(dx[i * xStride],
                                                          y[i * yStride], extraParams);
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
__global__ void pairWiseTransformDoubleIndex(
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
__global__ void pairWiseTransformFloatIndex(
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
__global__ void pairWiseTransformStridedDouble(
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
__global__ void pairWiseTransformStridedFloat(
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



#endif


#endif /* PAIRWISE_TRANSFORM_H_ */
