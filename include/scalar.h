/*
 * scalar.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef SCALAR_H_
#define SCALAR_H_
#include <dll.h>

#ifdef __JNI__
#include <jni.h>
#endif
#include <op.h>
#include <templatemath.h>
#include <ops.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#endif
namespace functions {
    namespace scalar {
/**
 * Apply a scalar
 *  operation to an array
 */
        template<typename T>
        class ScalarTransform: public virtual functions::ops::Op<T> {

        public:

#ifdef __CUDACC__


            /**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual __inline__ __device__ void transform(
			Nd4jIndex n,
			T scalar,
			T *dy,
			T *params,
			T *result,
			int *indexes, int *allocationBuffer, UnifiedSharedMemory *manager) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		Nd4jIndex i = blockIdx.x * blockDim.x + tid;

		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i+= totalThreads) {
			result[indexes[i]] = op(dy[indexes[i]],scalar, params);
		}
	}


	/**
	 * Cuda implementation of transform
	 * @param dx
	 * @param xShapeInfo
	 * @param result
	 * @param resultShapeInfo
	 * @param extraParams
	 * @param n
	 */
	virtual __inline__ __device__ void transformCuda(
			T scalar,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result, int *resultShapeInfo, int *allocationBuffer, UnifiedSharedMemory *manager) {
		int *xShape = shape::shapeOf(shapeInfo);
		int *xStride = shape::stride(shapeInfo);
		char xOrder = shape::order(shapeInfo);
		int xRank = shape::rank(shapeInfo);
		int xOffset = shape::offset(shapeInfo);
		int xElementWiseStride = shape::elementWiseStride(shapeInfo);
        int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
        int *zShape = shape::shapeOf(resultShapeInfo);
        int *zStride = shape::stride(resultShapeInfo);
        int zRank = shape::rank(resultShapeInfo);

		int totalThreads = gridDim.x * blockDim.x;
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		__shared__ int length;
		if(threadIdx.x == 0)
			length = shape::length(shapeInfo);
		__syncthreads();


		if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && xOrder == shape::order(resultShapeInfo)) {
			transformCuda(
					length,
					scalar,
					dy,
					xElementWiseStride,
					params,
					result,resultElementWiseStride, allocationBuffer, manager);
		}
		else {
			/* equal, positive, non-unit increments. */
			/*
			long allocSize = sizeof(int) * xRank;
			int *xIdx = shape::cuMalloc(manager->getT1ShapeBuffer(), allocSize);
            */
            int xIdx[MAX_RANK];

#pragma unroll
			for (int i = tid; i < length; i+= totalThreads) {
				shape::ind2sub(xRank, xShape, i,xIdx);
				int xOffset2 = shape::getOffset(0, xShape, xStride, xIdx, xRank);
				int resultOffset = shape::getOffset(0, zShape, zStride, xIdx, zRank);
				result[resultOffset] = op(dy[xOffset2],scalar, params);
			}
		}



	}


	/**
	 *
	 * @param n
	 * @param idx
	 * @param dx
	 * @param dy
	 * @param incy
	 * @param params
	 * @param result
	 * @param blockSize
	 */
	virtual
	__inline__ __device__ void transformCuda(
			Nd4jIndex n,
			T dx,
			T *dy,
			int incy,
			T *params,
			T *result,int resultStride, int *allocationBuffer, UnifiedSharedMemory *manager) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		Nd4jIndex i = tid;
		if(incy == 1) {
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i] = op(dy[i],dx, params);
			}
		}
		else {
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i * resultStride] = op(dy[i * incy],dx, params);
			}
		}


	}
#endif

		void transform(const int op,
			T *x,
			int *xShapeInfo,
			T *result,
			int *resultShapeInfo,
			T scalar,
			T *extraParams,
			int *indexes,
			int *resultIndexes) {
			if (op == 0)
				transform<simdOps::Add>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 1)
				transform<simdOps::Subtract>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 2)
				transform<simdOps::Multiply>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 3)
				transform<simdOps::Divide>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 4)
				transform<simdOps::ReverseDivide>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 5)
				transform<simdOps::ReverseSubtract>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 6)
				transform<simdOps::Max>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 7)
				transform<simdOps::LessThan>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 8)
				transform<simdOps::GreaterThan>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 9)
				transform<simdOps::EqualTo>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 10)
				transform<simdOps::LessThanOrEqual>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 11)
				transform<simdOps::NotEqualTo>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 12)
				transform<simdOps::Min>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 13)
				transform<simdOps::Copy>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 14)
				transform<simdOps::Mod>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 15)
				transform<simdOps::ReverseMod>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else if (op == 16)
				transform<simdOps::GreaterThanOrEqual>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes);
			else
				printf("[ERROR] Unknown opNum=%d for scalar", op);

		}

		void transform(const int op, T *x, int xStride, T *result, int resultStride,
			T scalar, T *extraParams, const Nd4jIndex n) {
			if (op == 0)
				transform<simdOps::Add>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 1)
				transform<simdOps::Subtract>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 2)
				transform<simdOps::Multiply>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 3)
				transform<simdOps::Divide>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 4)
				transform<simdOps::ReverseDivide>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 5)
				transform<simdOps::ReverseSubtract>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 6)
				transform<simdOps::Max>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 7)
				transform<simdOps::LessThan>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 8)
				transform<simdOps::GreaterThan>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 9)
				transform<simdOps::EqualTo>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 10)
				transform<simdOps::LessThanOrEqual>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 11)
				transform<simdOps::NotEqualTo>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 12)
				transform<simdOps::Min>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 13)
				transform<simdOps::Copy>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 14)
				transform<simdOps::Mod>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 15)
				transform<simdOps::ReverseMod>(x, xStride, result, resultStride, scalar, extraParams, n);
			else if (op == 16)
				transform<simdOps::GreaterThanOrEqual>(x, xStride, result, resultStride, scalar, extraParams, n);
			else
				printf("[ERROR] Unknown opNum=%d for scalar", op);
		}

		void transform(const int op,
			T *x,
			int *xShapeInfo,
			T *result,
			int *resultShapeInfo,
			T scalar, T *extraParams) {
			if (op == 0)
				transform<simdOps::Add>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 1)
				transform<simdOps::Subtract>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 2)
				transform<simdOps::Multiply>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 3)
				transform<simdOps::Divide>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 4)
				transform<simdOps::ReverseDivide>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 5)
				transform<simdOps::ReverseSubtract>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 6)
				transform<simdOps::Max>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 7)
				transform<simdOps::LessThan>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 8)
				transform<simdOps::GreaterThan>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 9)
				transform<simdOps::EqualTo>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 10)
				transform<simdOps::LessThanOrEqual>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 11)
				transform<simdOps::NotEqualTo>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 12)
				transform<simdOps::Min>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 13)
				transform<simdOps::Copy>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 14)
				transform<simdOps::Mod>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 15)
				transform<simdOps::ReverseMod>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else if (op == 16)
				transform<simdOps::GreaterThanOrEqual>(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams);
			else
				printf("[ERROR] Unknown opNum=%d for scalar", op);
		}

            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */
		template<template <typename> typename OpType>
          void transform(T *x,
                           int *xShapeInfo,
                           T *result,
                           int *resultShapeInfo,
                           T scalar,
                           T *extraParams,
                           int *indexes,
                           int *resultIndexes) {
                const Nd4jIndex n = shape::length(xShapeInfo);
#pragma omp parallel for simd schedule(guided) if (n > 2048)
                for (Nd4jIndex i = 0; i < n; i++) {
                    result[resultIndexes[i]] = OpType<T>::op(x[indexes[i]], scalar,extraParams);
                }
            }




            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */
          /*  void transform(T *x, int *xShapeInfo, T *result, int *resultShapeInfo,
                           T scalar, T *extraParams,int *indexes) {
                transform(x,
                          xShapeInfo,
                          result,
                          resultShapeInfo,
                          scalar,
                          extraParams,
                          indexes,
                          indexes);
            }*/


            /**
         * CPU implementation of scalar operation
         * @param x the input
         * @param xStride the stride for the input
         * @param result the result buffer
         * @param resultStride the stride for the result
         * @param scalar the scalar to apply
         * @param extraParams the extra parameters where
         * neccssary
         * @param n the number of elements to loop over
         */
		  template<template <typename> typename OpType>
            void transform(T *x,
                           int *xShapeInfo,
                           T *result,
                           int *resultShapeInfo,
                           T scalar, T *extraParams) {
                char xOrdering = shape::order(xShapeInfo);
                char resultOrdering = shape::order(resultShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
                if(xOrdering != resultOrdering || xElementWiseStride < 1 || resultElementWiseStride < 0) {
                    int shapeIter[MAX_RANK];
                    int coord[MAX_RANK];
                    int dim;
                    int xStridesIter[MAX_RANK];
                    int resultStridesIter[MAX_RANK];
                    int *xShape = shape::shapeOf(xShapeInfo);
                    int *xStride = shape::stride(xShapeInfo);
                    int *resultStride = shape::stride(resultShapeInfo);
                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<T>(rank,
                                                 xShape,
                                                 x,
                                                 xStride,
                                                 result,
                                                 resultStride,
                                                 &rank,
                                                 shapeIter,
                                                 &x,
                                                 xStridesIter,
                                                 &result,
                                                 resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                T *xIter = x;
                                T *resultIter = result;
                                resultIter[0] = OpType<T>::op(xIter[0],scalar,extraParams);
                            } ND4J_RAW_ITER_TWO_NEXT(dim,
                                                     rank,
                                                     coord,
                                                     shapeIter,
                                                     x,
                                                     xStridesIter,
                                                     result,
                                                     resultStridesIter);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }

                }
                else {
                    const Nd4jIndex n = shape::length(xShapeInfo);


                    if(xElementWiseStride >= 1 && resultElementWiseStride >= 1) {
                        transform<OpType>(x,xElementWiseStride,result,resultElementWiseStride,scalar,extraParams,n);
                    }
                    else {
                        int *xShape = shape::shapeOf(xShapeInfo);
                        int *resultShape = shape::shapeOf(resultShapeInfo);

                        int *xStride = shape::stride(xShapeInfo);
                        int *resultStride = shape::stride(resultShapeInfo);
                        int xRank = shape::rank(xShapeInfo);
                        int resultRank = shape::rank(resultShapeInfo);

                        int xOffset = shape::offset(xShapeInfo);
                        int resultOffset = shape::offset(resultShapeInfo);

#pragma omp parallel for simd schedule(guided) if (n > 2048)
                        for (Nd4jIndex i = 0; i < n; i++) {
                            int *xIdx = shape::ind2sub(xRank, xShape, i);
                            int *resultIdx = shape::ind2sub(resultRank, resultShape, i);
                            int xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
                            int resultOffset2 = shape::getOffset(resultOffset, resultShape, resultStride, resultIdx, resultRank);

                            result[resultOffset2] = OpType<T>::op(x[xOffset2], scalar,extraParams);

                            delete[] xIdx;
                            delete[] resultIdx;

                        }

                    }
                }


            }


            /**
             * CPU implementation of scalar operation
             * @param x the input
             * @param xStride the stride for the input
             * @param result the result buffer
             * @param resultStride the stride for the result
             * @param scalar the scalar to apply
             * @param extraParams the extra parameters where
             * neccssary
             * @param n the number of elements to loop over
             */
			template<template <typename> typename OpType>
            void transform(T *x, int xStride, T *result, int resultStride,
                           T scalar, T *extraParams, const Nd4jIndex n) {
                if (xStride == 1 && resultStride == 1) {
#pragma omp parallel for simd schedule(guided) if (n > 2048)
                    for (Nd4jIndex i = 0; i < n; i++) {
                        result[i] = OpType<T>::op(x[i], scalar, extraParams);
                    }
                }

                else {
#pragma omp parallel for schedule(guided) if (n > 2048)
                    for (Nd4jIndex i = 0; i < n; i++) {
                        result[i * resultStride] = OpType<T>::op(x[i * xStride], scalar,
                                                      extraParams);

                    }
                }

            }
            virtual inline
#ifdef __CUDACC__
            __host__ __device__
#endif
            void aggregateExtraParams(T **extraParamsTotal,T **extraParamsLocal) {
                //no extra params aggregation needs to happen
            }
#ifdef __CUDACC__
            __host__ __device__
#endif
            virtual inline ~ScalarTransform() {
            }
#ifdef __CUDACC__
            __host__ __device__
#endif
            ScalarTransform() {
            }
        };
    }
}
#ifdef __CUDACC__

template <typename T>
__device__ void scalarGeneric(
		int opNum,
		Nd4jIndex n,
		T dx,
		T *dy,
		int incy, T *params,
		T *result,int resultStride, int *allocationBuffer) {

	__shared__ functions::scalar::ScalarTransform<T> *op;
	__shared__  functions::scalar::ScalarOpFactory<T> *scalarDoubleOpFactory;

	__shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), sizeof(functions::scalar::ScalarOpFactory<T>), sizeof(functions::scalar::ScalarTransform<T>), sizeof(shape::TAD), 0);

		scalarDoubleOpFactory = new(manager->getFactorySpace()) functions::scalar::ScalarOpFactory<T>();
		op = scalarDoubleOpFactory->getOp(opNum, manager->getFunctionSpace());
	}
	__syncthreads();


	op->transformCuda(n,dx,dy,incy,params,result,resultStride,allocationBuffer, manager);
}

__global__ void scalarDouble(
		int opNum,
		Nd4jIndex n,
		double dx,
		double *dy,
		int incy, double *params,
		double *result,int resultStride, int *allocationBuffer) {
	scalarGeneric<double>(
			opNum,
			n,
			dx,
			dy,
			incy,
			params,
			result,resultStride, allocationBuffer);
}

 __global__ void scalarFloat(int opNum,
		Nd4jIndex n,float dx, float *dy, int incy, float *params, float *result,int resultStride, int *allocationBuffer) {
	scalarGeneric<float>(
			opNum,
			n,
			dx,
			dy,
			incy,
			params,
			result,resultStride, allocationBuffer);
}


template <typename T>
__device__ void scalarGenericIndexes(
        int opNum,
        Nd4jIndex n,
        T dx,
        T *dy,
        T *params,
        T *result,int *indexes, int *allocationBuffer) {

    __shared__ functions::scalar::ScalarTransform<T> *op;
    __shared__  functions::scalar::ScalarOpFactory<T> *scalarDoubleOpFactory;

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), sizeof(functions::scalar::ScalarOpFactory<T>), sizeof(functions::scalar::ScalarTransform<T>), sizeof(shape::TAD), 0);

        scalarDoubleOpFactory = new(manager->getFactorySpace()) functions::scalar::ScalarOpFactory<T>();
        op = scalarDoubleOpFactory->getOp(opNum, manager->getFunctionSpace());
    }
    __syncthreads();

    op->transform(n,dx,dy,params,result,indexes, allocationBuffer, manager);
}

 __global__ void scalarDoubleIndexes(
        int opNum,
        Nd4jIndex n,
        double dx,
        double *dy,
        double *params,
        double *result,int *indexes, int *allocationBuffer) {
    scalarGenericIndexes<double>(opNum,
                                 n,
                                 dx,
                                 dy,
                                 params,
                                 result,
                                 indexes, allocationBuffer);
}

 __global__ void scalarFloatIndexes(
        int opNum,
        Nd4jIndex n,
        float dx,
        float *dy,
        float *params,
        float *result,
        int *indexes, int *allocationBuffer) {
    scalarGenericIndexes<float>(opNum,
                                 n,
                                 dx,
                                 dy,
                                 params,
                                 result,
                                 indexes, allocationBuffer);
}









template <typename T>
__device__ void scalarGeneric(
		int opNum,
		T dx,
		T *dy,
		int *xShapeInfo,int xRank,
		T *params,
		T *result,int *resultShapeInfo, int zRank, int *allocationBuffer) {

	__shared__ functions::scalar::ScalarTransform<T> *op;
	__shared__  functions::scalar::ScalarOpFactory<T> *scalarDoubleOpFactory;

	__shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), sizeof(functions::scalar::ScalarOpFactory<T>), sizeof(functions::scalar::ScalarTransform<T>), sizeof(shape::TAD), 0);

		scalarDoubleOpFactory = new(manager->getFactorySpace()) functions::scalar::ScalarOpFactory<T>();
		op = scalarDoubleOpFactory->getOp(opNum, manager->getFunctionSpace());
	}
	__syncthreads();


	op->transformCuda(
	    dx,
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationBuffer,
	    manager);
}

extern "C" __global__ void scalarDouble(
		int opNum,
		double dx,
		double *dy,
		int *shapeInfo, int xRank, double *params,
		double *result,int *resultShapeInfo, int zRank, int *allocationBuffer) {
	scalarGeneric<double>(
			opNum,
			dx,
			dy,
			shapeInfo, xRank,
			params,
			result,resultShapeInfo, zRank, allocationBuffer);
}

extern "C" __global__ void scalarFloat(
		int opNum,
		float dx,
		float *dy,
		int *shapeInfo, int xRank,
		float *params,
		float *result,int *resultShapeInfo, int zRank, int *allocationBuffer) {
	scalarGeneric<float>(
			opNum,
			dx,
			dy,
			shapeInfo, xRank,
			params,
			result,resultShapeInfo, zRank, allocationBuffer);
}

#endif
#endif /* SCALAR_H_ */
