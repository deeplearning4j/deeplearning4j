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
#include <templatemath.h>
#include <ops/ops.h>
#include <op_boilerplate.h>
#include "helpers/logger.h"

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <types/float16.h>
#endif

#define SCALAR_OPS \
        (0, simdOps::Add),\
        (1, simdOps::Subtract),\
        (2, simdOps::Multiply),\
        (3, simdOps::Divide),\
        (4, simdOps::ReverseDivide),\
        (5, simdOps::ReverseSubtract),\
        (6, simdOps::Max),\
        (7, simdOps::LessThan),\
        (8, simdOps::GreaterThan),\
        (9, simdOps::EqualTo),\
        (10,simdOps::LessThanOrEqual),\
        (11,simdOps::NotEqualTo),\
        (12,simdOps::Min),\
        (13,simdOps::Copy),\
        (14,simdOps::Mod),\
        (15,simdOps::ReverseMod),\
        (16,simdOps::GreaterThanOrEqual),\
        (17,simdOps::Remainder),\
        (18,simdOps::FMod)

namespace functions {
    namespace scalar {
/**
 * Apply a scalar
 *  operation to an array
 */
        template<typename T>
        class ScalarTransform {

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
template<typename OpType>
	static inline __device__ void transform(
			Nd4jIndex n,
			T scalar,
			T *dy,
			T *params,
			T *result,
			int *indexes,
			int *allocationBuffer,
			UnifiedSharedMemory *manager) {
		int totalThreads = gridDim.x * blockDim.x;
		int tid = threadIdx.x;
		Nd4jIndex i = blockIdx.x * blockDim.x + tid;

		/* equal, positive, non-unit increments. */
#pragma unroll
		for (; i < n; i+= totalThreads) {
			result[indexes[i]] = OpType::op(dy[indexes[i]],scalar, params);
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
template<typename OpType>
	static inline __device__ void transformCuda(
			T scalar,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationBuffer,
			UnifiedSharedMemory *manager) {

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
			transformCuda<OpType>(
					length,
					scalar,
					dy,
					xElementWiseStride,
					params,
					result,resultElementWiseStride, allocationBuffer, manager);
		}
		else {
            int xIdx[MAX_RANK];

#pragma unroll
			for (Nd4jIndex i = tid; i < length; i+= totalThreads) {
				shape::ind2sub(xRank, xShape, i,xIdx);
				int xOffset2 = shape::getOffset(0, xShape, xStride, xIdx, xRank);
				int resultOffset = shape::getOffset(0, zShape, zStride, xIdx, zRank);
			    result[resultOffset] = OpType::op(dy[xOffset2],scalar, params);
			}
		}
	}
/**
  * ScalarOp along dimension
**/
template<typename OpType>
    static inline __device__ void transformCuda(T *x,
                                  int *xShapeInfo,
                                  T *extraParams,
                                  T *z,
                                  int *zShapeInfo,
                                  T *scalars,
                                  int *dimension,
                                  int dimensionLength,
                                  int *tadShapeInfo,
                                  Nd4jIndex *tadOffsets,
                                  int *tadShapeInfoZ,
                                  Nd4jIndex *tadOffsetsZ) {


                if (tadShapeInfoZ == nullptr) {
                    tadShapeInfoZ = tadShapeInfo;
                    tadOffsetsZ = tadOffsets;
                }

                // tad preparation
                int tadEWS = shape::elementWiseStride(tadShapeInfo);
                int zEWS = shape::elementWiseStride(tadShapeInfo);
                int tadRank = shape::rank(tadShapeInfo);
                int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                int numTads =shape::length(xShapeInfo) / tadLength;

                // main loop, rolling over tads
                for (int r = blockIdx.x; r < numTads; r+=gridDim.x) {
                    Nd4jIndex offset = tadOffsets[r];
                    Nd4jIndex offsetZ = tadOffsetsZ[r];
                    T scalar = scalars[r];

                    if (tadEWS >= 1 && zEWS >= 1) {
                        T *oZ = z + offsetZ;
                        T *oX = x + offset;

                       for (int f = threadIdx.x; f < tadLength; f+= blockDim.x) {
                            oZ[f] = OpType::op(oX[f], scalar, extraParams);
                        }
                    } else {
                        // ind2sub loop
                        printf("Super-bad loop visited. Shouldn't ever happen\n");
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
template<typename OpType>
	static inline __device__ void transformCuda(
			Nd4jIndex n,
			T dx,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationBuffer,
			UnifiedSharedMemory *manager) {

		int totalThreads = gridDim.x * blockDim.x;
		int tid = blockIdx.x * blockDim.x + threadIdx.x;

		Nd4jIndex i = tid;
		if(incy == 1 && resultStride == 1) {
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i] = OpType::op(dy[i],dx, params);
			}
		}
		else {
#pragma unroll
			for (; i < n; i += totalThreads) {
				result[i * resultStride] = OpType::op(dy[i * incy],dx, params);
			}
		}
	}

		static inline __device__ void transformCuda(
			const int opNum,
			T scalar,
			T *dy,
			int *shapeInfo,
			T *params,
			T *result,
			int *resultShapeInfo,
			int *allocationBuffer,
			UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(scalar, dy, shapeInfo, params, result, resultShapeInfo, allocationBuffer, manager), SCALAR_OPS);
                    }


		static inline __device__ void transform(
			const int opNum,
			Nd4jIndex n,
			T scalar,
			T *dy,
			T *params,
			T *result,
			int *indexes,
			int *allocationBuffer,
			UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transform, PARAMS(n, scalar, dy, params, result, indexes, allocationBuffer, manager), SCALAR_OPS);
		}


		static inline __device__ void transformCuda(
			const int opNum,
			Nd4jIndex n,
			T dx,
			T *dy,
			int incy,
			T *params,
			T *result,
			int resultStride,
			int *allocationBuffer,
			UnifiedSharedMemory *manager) {
                    DISPATCH_BY_OPNUM(transformCuda, PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer, manager), SCALAR_OPS);
		}
#endif

            static void transform(int opNum,
                                  T *x,
                                  int *xShapeInfo,
                                  T *extraParams,
                                  T *z,
                                  int *zShapeInfo,
                                  T *scalars,
                                  int *dimension,
                                  int dimensionLength,
                                  int *tadShapeInfo,
                                  Nd4jIndex *tadOffsets,
                                  int *tadShapeInfoZ,
                                  Nd4jIndex *tadOffsetsZ) {

                DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_OPS);
            }

            static void transform(const int opNum,
                                  T *x,
                                  int *xShapeInfo,
                                  T *result,
                                  int *resultShapeInfo,
                                  T scalar,
                                  T *extraParams,
                                  int *indexes,
                                  int *resultIndexes) {
                DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams, indexes, resultIndexes), SCALAR_OPS);
            }

            static void transform(const int opNum, T *x, int xStride, T *result, int resultStride,
                                  T scalar, T *extraParams, const Nd4jIndex n) {
                DISPATCH_BY_OPNUM(transform,
                                  PARAMS(x,
                                         xStride,
                                         result,
                                         resultStride,
                                         scalar,
                                         extraParams, n),
                                  SCALAR_OPS);
            }

            static void transform(const int opNum,
                                  T *x,
                                  int *xShapeInfo,
                                  T *result,
                                  int *resultShapeInfo,
                                  T scalar, T *extraParams) {
                DISPATCH_BY_OPNUM(transform, PARAMS(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams), SCALAR_OPS);
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


            template<typename OpType>
            static void transform(T *x,
                                  int *xShapeInfo,
                                  T *result,
                                  int *resultShapeInfo,
                                  T scalar,
                                  T *extraParams,
                                  int *indexes,
                                  int *resultIndexes) {
                const Nd4jIndex n = shape::length(xShapeInfo);
#pragma omp parallel for simd schedule(guided) if (n > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                for (Nd4jIndex i = 0; i < n; i++) {
                    result[resultIndexes[i]] = OpType::op(x[indexes[i]], scalar,extraParams);
                }
            }

            /*
             * ScalarOp along dimension
             */
            template<typename OpType>
            static void transform(T *x,
                                  int *xShapeInfo,
                                  T *extraParams,
                                  T *z,
                                  int *zShapeInfo,
                                  T *scalars,
                                  int *dimension,
                                  int dimensionLength,
                                  int *tadShapeInfo,
                                  Nd4jIndex *tadOffsets,
                                  int *tadShapeInfoZ,
                                  Nd4jIndex *tadOffsetsZ) {


                if (tadShapeInfoZ == nullptr) {
                    tadShapeInfoZ = tadShapeInfo;
                    tadOffsetsZ = tadOffsets;
                }

                // tad preparation
                int tadEWS = shape::elementWiseStride(tadShapeInfo);
                int zEWS = shape::elementWiseStride(tadShapeInfo);
                //int tadRank = shape::rank(tadShapeInfo);
                int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                int numTads =shape::length(xShapeInfo) / tadLength;

                int tadsPerThread = numTads / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                // main loop, rolling along tads
#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                for (int r = 0; r < numTads; r++) {

                    Nd4jIndex offset = tadOffsets[r];
                    Nd4jIndex offsetZ = tadOffsetsZ[r];
                    T scalar = scalars[r];

                    if (tadEWS >= 1 && zEWS >= 1) {
                        T *oZ = z + offsetZ;
                        T *oX = x + offset;

                        if (tadEWS == 1 && zEWS == 1) {

#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oZ[f] = OpType::op(oX[f], scalar, extraParams);
                            }
                        } else {

// TODO: nested loop should be used here probably, instead of simd
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oZ[f * zEWS] = OpType::op(oX[f * tadEWS], scalar, extraParams);
                            }
                        }

                    } else {
                        // ind2sub loop
                        printf("Super-bad loop visited. Shouldn't ever happen\n");
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

            template<typename OpType>
            static  void transform(T *x,
                                   int *xShapeInfo,
                                   T *result,
                                   int *resultShapeInfo,
                                   T scalar, T *extraParams) {
                char xOrdering = shape::order(xShapeInfo);
                char resultOrdering = shape::order(resultShapeInfo);
                int xElementWiseStride = shape::elementWiseStride(xShapeInfo);

                nd4j_logger("Launching scalar: xOrder: %i; zOrder: %i; xEWS: %i\n", xOrdering, resultOrdering, xElementWiseStride);

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
                                resultIter[0] = OpType::op(xIter[0],scalar,extraParams);
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

#pragma omp parallel for simd schedule(guided) if (n > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                        for (Nd4jIndex i = 0; i < n; i++) {
                            int *xIdx = shape::ind2sub(xRank, xShape, i);
                            int *resultIdx = shape::ind2sub(resultRank, resultShape, i);
                            Nd4jIndex xOffset2 = shape::getOffset(xOffset, xShape, xStride, xIdx, xRank);
                            Nd4jIndex resultOffset2 = shape::getOffset(resultOffset, resultShape, resultStride, resultIdx, resultRank);

                            result[resultOffset2] = OpType::op(x[xOffset2], scalar,extraParams);

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

            template<typename OpType>
            static void transform(T *x, int xStride, T *result, int resultStride,
                                  T scalar, T *extraParams, const Nd4jIndex n) {

                Nd4jIndex elementsPerThread = n / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                Nd4jIndex span = (n / num_threads) + 8;

                if (xStride == 1 && resultStride == 1) {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jIndex tid = omp_get_thread_num();
                        Nd4jIndex start = span * tid;
                        Nd4jIndex end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i] = OpType::op(x[i], scalar, extraParams);
                        }
                    }
                }

                else {
#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jIndex tid = omp_get_thread_num();
                        Nd4jIndex start = span * tid;
                        Nd4jIndex end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jIndex i = start; i < end; i++) {
                            result[i * resultStride] = OpType::op(x[i * xStride], scalar, extraParams);
                        }
                    }
                }

            }
        };
    }
}
#ifdef __CUDACC__

template <typename T, typename OpType>
__device__ void scalarAlongDimensionGeneric(T *x,
                                  int *xShapeInfo,
                                  T *extraParams,
                                  T *z,
                                  int *zShapeInfo,
                                  T *scalars,
                                  int *dimension,
                                  int dimensionLength,
                                  int *tadShapeInfo,
                                  Nd4jIndex *tadOffsets,
                                  int *tadShapeInfoZ,
                                  Nd4jIndex *tadOffsetsZ) {

    functions::scalar::ScalarTransform<T>::template transformCuda<OpType>(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ);
}

template <typename T, typename OpClass>
__device__ void scalarSimpleGeneric(
		Nd4jIndex n,
		T dx,
		T *dy,
		int incy, T *params,
		T *result,int resultStride, int *allocationBuffer) {

	functions::scalar::ScalarTransform<T>::template transformCuda<OpClass>(
		n,
		dx,
		dy,
		incy,
		params,
		result,
		resultStride,
		allocationBuffer,
		NULL);
}


template <typename T>
__device__ void scalarGenericIndexes(
        int opNum,
        Nd4jIndex n,
        T dx,
        T *dy,
        T *params,
        T *result,int *indexes, int *allocationBuffer) {

    __shared__ UnifiedSharedMemory *manager;

    if (threadIdx.x == 0) {
        extern __shared__ unsigned char shmem[];
        manager = new(shmem) UnifiedSharedMemory((int *) shmem);
	    manager->init(sizeof(UnifiedSharedMemory), 0, sizeof(functions::scalar::ScalarTransform<T>), sizeof(shape::TAD), 0);
    }
    __syncthreads();

    functions::scalar::ScalarTransform<T>::transform(
    	opNum,
    	n,
    	dx,
    	dy,
    	params,
    	result,
    	indexes,
    	allocationBuffer,
    	manager);
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









template <typename T, typename OpClass>
__device__ void scalarSimpleGeneric(
		T dx,
		T *dy,
		int *xShapeInfo,
		T *params,
		T *result,
		int *resultShapeInfo,
		int *allocationBuffer) {

	functions::scalar::ScalarTransform<T>::template transformCuda<OpClass>(
	    dx,
	    dy,
	    xShapeInfo,
	    params,
	    result,
	    resultShapeInfo,
	    allocationBuffer,
	    NULL);
}



// ScalarOp Along Dimension kernels
DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, float, INPUT(float *x, int *xShapeInfo, float *extraParams, float *z, int *zShapeInfo, float *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, double, INPUT(double *x, int *xShapeInfo, double *extraParams, double *z, int *zShapeInfo, double *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarAlongDimension_, scalarAlongDimensionGeneric, float16, INPUT(float16 *x, int *xShapeInfo, float16 *extraParams, float16 *z, int *zShapeInfo, float16 *scalars, int *dimension, int dimensionLength, int *tadShapeInfo, Nd4jIndex *tadOffsets, int *tadShapeInfoZ, Nd4jIndex *tadOffsetsZ), PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), OPS_A(SCALAR_OPS))

// scalar shape
DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, float, INPUT(float dx, float *dy, int *xShapeInfo, float *params, float *result, int *resultShapeInfo, int *allocationBuffer), PARAMS(dx, dy, xShapeInfo, params, result, resultShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, double, INPUT(double dx, double *dy, int *xShapeInfo, double *params, double *result, int *resultShapeInfo, int *allocationBuffer), PARAMS(dx, dy, xShapeInfo, params, result, resultShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleShaped_, scalarSimpleGeneric, float16, INPUT(float16 dx, float16 *dy, int *xShapeInfo, float16 *params, float16 *result, int *resultShapeInfo, int *allocationBuffer), PARAMS(dx, dy, xShapeInfo, params, result, resultShapeInfo, allocationBuffer), OPS_A(SCALAR_OPS))

// scalar strided
DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, float, INPUT(Nd4jIndex n, float dx, float *dy, int incy, float *params, float *result,int resultStride, int *allocationBuffer), PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, double, INPUT(Nd4jIndex n, double dx, double *dy, int incy, double *params, double *result,int resultStride, int *allocationBuffer), PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer), OPS_A(SCALAR_OPS))
DISPATCH_KERNEL_SIMPLE(scalarSimpleStrided_, scalarSimpleGeneric, float16, INPUT(Nd4jIndex n, float16 dx, float16 *dy, int incy, float16 *params, float16 *result,int resultStride, int *allocationBuffer), PARAMS(n, dx, dy, incy, params, result, resultStride, allocationBuffer), OPS_A(SCALAR_OPS))

#endif
#endif /* SCALAR_H_ */
