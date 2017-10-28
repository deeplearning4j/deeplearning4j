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


#include "legacy_ops.h"


namespace functions {
    namespace pairwise_transforms {

/**
 * Transforms involving 2 arrays
 */
        template<typename T>
        class PairWiseTransform {
        public:

#ifdef __CUDACC__

            static __host__ void execudaCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *dx, int xStride, T *y, int yStride, T *result, int resultStride, T *extraParams, Nd4jIndex n);

            static __host__ void execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *dx, int *xShapeInfo, T *y, int *yShapeInfo, T *result, int *resultShapeInfo, T *extraParams);

            static __device__ void transformCuda(const int opNum, Nd4jIndex n, T *dx, T *y, int incx, int incy, T *extraParams, T *result, int incz, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *indexes, int *yIndexes, int *resultIndexes, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo);


            template<typename OpType>
	        static __device__ void transformCuda(T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo);

            template<typename OpType>
	        static __device__ void transformCuda(Nd4jIndex n, T *dx, T *dy, int incx, int incy, T *params, T *result, int incz, int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo);

            template<typename OpType>
	        static __device__ void transform(T *dx, int *xShapeBuffer, T *y, int *yShapeBuffer, T *result, int *resultShapeBuffer, T *extraParams, int *indexes, int *yIndexes, int *resultIndexes,  int *allocationPointer, UnifiedSharedMemory *manager, int *tadOnlyShapeInfo);


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

                if (shape::isScalar(yShapeBuffer)) {
                    if (xElementWiseStride == 1 && resultElementWiseStride == 1) {
                        for (int e = 0; e < n; e++) {
                            result[e] = OpType::op(dx[e], y[0], extraParams);
                        }
                    } else {
                        int xCoord[MAX_RANK];
                        int resultCoord[MAX_RANK];

                        int xRank = shape::rank(xShapeBuffer);
                        int resultRank = shape::rank(resultShapeBuffer);

                        int *xShape = shape::shapeOf(xShapeBuffer);
                        int *xStride = shape::stride(xShapeBuffer);

                        int *resultShape = shape::shapeOf(resultShapeBuffer);
                        int *resultStride = shape::stride(resultShapeBuffer);

                        int elementsPerThread = n / ELEMENT_THRESHOLD;
                        int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                        num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(xCoord, resultCoord)
                        for (Nd4jIndex i = 0; i < n; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(resultRank,resultShape, i, resultCoord);

                            Nd4jIndex xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            Nd4jIndex resultOffset = shape::getOffset(0, resultShape, resultStride, resultCoord, resultRank);
                            result[resultOffset] = OpType::op(dx[xOffset], y[0], extraParams);
                        }
                    }

                    return;
                }

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
                    Nd4jIndex len = n;
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



#endif


#endif /* PAIRWISE_TRANSFORM_H_ */
