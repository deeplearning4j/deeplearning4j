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
        template<typename X, typename Y>
        class PairWiseTransform {
        public:

#ifdef __CUDACC__

            static __host__ void execudaCudaStrided(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *dx, Nd4jLong xStride, T *y, Nd4jLong yStride, T *result, Nd4jLong resultStride, T *extraParams, Nd4jLong n);

            static __host__ void execudaCudaShaped(dim3& launchDims, Nd4jPointer *extraPointers, int opNum, T *dx, Nd4jLong *xShapeInfo, T *y, Nd4jLong *yShapeInfo, T *result, Nd4jLong *resultShapeInfo, T *extraParams);

            static __device__ void transformCuda(const int opNum, Nd4jLong n, T *dx, T *y, Nd4jLong incx, Nd4jLong incy, T *extraParams, T *result, Nd4jLong incz, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            static __device__ void transformCuda(const int opNum, T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, Nd4jLong *indexes, Nd4jLong *yIndexes, Nd4jLong *resultIndexes, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);


            template<typename OpType>
	        static __device__ void transformCuda(T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            template<typename OpType>
	        static __device__ void transformCuda(Nd4jLong n, T *dx, T *dy, Nd4jLong incx, Nd4jLong incy, T *params, T *result, Nd4jLong incz, int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);

            template<typename OpType>
	        static __device__ void transform(T *dx, Nd4jLong *xShapeBuffer, T *y, Nd4jLong *yShapeBuffer, T *result, Nd4jLong *resultShapeBuffer, T *extraParams, Nd4jLong *indexes, Nd4jLong *yIndexes, Nd4jLong *resultIndexes,  int *allocationPointer, UnifiedSharedMemory *manager, Nd4jLong *tadOnlyShapeInfo);


#endif
        public:
			static void exec(
				const int opNum,
				X *dx,
				Nd4jLong *xShapeBuffer,
				Y *y,
				Nd4jLong *yShapeBuffer,
				X *result,
				Nd4jLong *resultShapeBuffer,
				X *extraParams,
				Nd4jLong *indexes,
				Nd4jLong *yIndexes,
				Nd4jLong *resultIndexes) {
                            DISPATCH_BY_OPNUM_TT(exec, PARAMS(dx,
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
				X *dx,
				Nd4jLong *xShapeBuffer,
				Y *y,
				Nd4jLong *yShapeBuffer,
				X *result,
				Nd4jLong *resultShapeBuffer,
				X *extraParams) {
				DISPATCH_BY_OPNUM_TT(exec, PARAMS(dx,
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
				X *dx,
				Nd4jLong xStride,
				Y *y,
				Nd4jLong yStride,
				X *result,
				Nd4jLong resultStride,
				X *extraParams,
				Nd4jLong n) {
				DISPATCH_BY_OPNUM_TT(exec, PARAMS(dx,
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
                    X *dx,
                    Nd4jLong* xShapeBuffer,
                    Y *y,
                    Nd4jLong* yShapeBuffer,
                    X *result,
                    Nd4jLong* resultShapeBuffer,
                    X *extraParams,
                    Nd4jLong *indexes,
                    Nd4jLong *yIndexes,
                    Nd4jLong *resultIndexes) {
                Nd4jLong n = shape::length(xShapeBuffer);

#pragma omp parallel for simd schedule(guided) proc_bind(AFFINITY) default(shared)
                for (Nd4jLong i = 0; i < n; i++) {
                    result[resultIndexes[i]] = OpType::op(dx[indexes[i]], y[yIndexes[i]], extraParams);

                }
            }

			template<typename OpType>
			static void exec(
                    X *dx,
                    Nd4jLong* xShapeBuffer,
                    Y *y,
                    Nd4jLong* yShapeBuffer,
                    X *result,
                    Nd4jLong* resultShapeBuffer,
                    X *extraParams) {
                auto n = shape::length(xShapeBuffer);
                auto xElementWiseStride = shape::elementWiseStride(xShapeBuffer);
                auto yElementWiseStride = shape::elementWiseStride(yShapeBuffer);
                auto resultElementWiseStride = shape::elementWiseStride(resultShapeBuffer);

                if (shape::isScalar(yShapeBuffer)) {
                    if (xElementWiseStride == 1 && resultElementWiseStride == 1) {
                        for (int e = 0; e < n; e++) {
                            result[e] = OpType::op(dx[e], y[0], extraParams);
                        }
                    } else {
                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong resultCoord[MAX_RANK];

                        int xRank = shape::rank(xShapeBuffer);
                        int resultRank = shape::rank(resultShapeBuffer);

                        Nd4jLong *xShape = shape::shapeOf(xShapeBuffer);
                        Nd4jLong *xStride = shape::stride(xShapeBuffer);

                        Nd4jLong *resultShape = shape::shapeOf(resultShapeBuffer);
                        Nd4jLong *resultStride = shape::stride(resultShapeBuffer);

                        int elementsPerThread = n / ELEMENT_THRESHOLD;
                        int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                        num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(xCoord, resultCoord)
                        for (Nd4jLong i = 0; i < n; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(resultRank,resultShape, i, resultCoord);

                            Nd4jLong xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            Nd4jLong resultOffset = shape::getOffset(0, resultShape, resultStride, resultCoord, resultRank);
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
                    Nd4jLong *xShape = shape::shapeOf(xShapeBuffer);

                    Nd4jLong *xStride = shape::stride(xShapeBuffer);
                    Nd4jLong *yStride = shape::stride(yShapeBuffer);
                    Nd4jLong *resultStride = shape::stride(resultShapeBuffer);

                    // tad-oriented rotation technically

                    int tadsPerThread = xShape[0] / TAD_THRESHOLD;
                    int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                    num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
for (Nd4jLong i = 0; i < xShape[0]; i++) {
                    auto dxLocal = dx + xStride[0] * i;
                    auto yLocal = y + yStride[0] * i;
                    auto resultLocal = result + resultStride[0] * i;

                    int rankLocal = rank - 1;
                    auto xShapeLocal = xShape + 1;

                    auto xStrideLocal = xStride + 1;
                    auto yStrideLocal = yStride + 1;
                    auto resultStrideLocal = resultStride + 1;

                    Nd4jLong shapeIter[MAX_RANK];
                    Nd4jLong coord[MAX_RANK];
                    int dim;
                    Nd4jLong xStridesIter[MAX_RANK];
                    Nd4jLong yStridesIter[MAX_RANK];
                    Nd4jLong resultStridesIter[MAX_RANK];
                    if (PrepareThreeRawArrayIter<X, Y>(rankLocal,
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
                                auto xIter = dxLocal;
                                auto yIter = yLocal;
                                auto resultIter = resultLocal;
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
                    Nd4jLong len = n;
                    int xRank = shape::rank(xShapeBuffer);
                    int yRank = shape::rank(yShapeBuffer);
                    int resultRank = shape::rank(resultShapeBuffer);

                    auto xShape = shape::shapeOf(xShapeBuffer);
                    auto xStride = shape::stride(xShapeBuffer);

                    auto yShape = shape::shapeOf(yShapeBuffer);
                    auto yStride = shape::stride(yShapeBuffer);

                    auto resultShape = shape::shapeOf(resultShapeBuffer);
                    auto resultStride = shape::stride(resultShapeBuffer);

                    int elementsPerThread = n / ELEMENT_THRESHOLD;
                    int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                    num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                    Nd4jLong xCoord[MAX_RANK];
                    Nd4jLong yCoord[MAX_RANK];

                    if(dx == result) {
#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(xCoord, yCoord)
                        for (Nd4jLong i = 0; i < len; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(yRank,yShape, i, yCoord);

                            auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                            result[xOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);

                        }
                    }
                    else {
                        Nd4jLong resultCoord[MAX_RANK];

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(xCoord, yCoord, resultCoord)
                        for (Nd4jLong i = 0; i < len; i++) {
                            shape::ind2subC(xRank,xShape, i, xCoord);
                            shape::ind2subC(yRank,yShape, i, yCoord);
                            shape::ind2subC(resultRank,resultShape, i, resultCoord);

                            auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                            auto yOffset = shape::getOffset(0, yShape, yStride, yCoord, yRank);
                            auto resultOffset = shape::getOffset(0, resultShape, resultStride, resultCoord, resultRank);
                            result[resultOffset] = OpType::op(dx[xOffset], y[yOffset], extraParams);

                        }
                    }
                }
            }

            template<typename OpType>
            static void exec(X *dx,
                             Nd4jLong xStride,
                             Y *y,
                             Nd4jLong yStride,
                             X *result,
                             Nd4jLong resultStride,
                             X *extraParams,
                             const Nd4jLong n) {
                int elementsPerThread = n / ELEMENT_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

                int span = (n / _threads) + 8;

                if (xStride == 1 && yStride == 1 && resultStride == 1) {
                    if (_threads > 1) {
#pragma omp parallel num_threads(_threads) if (_threads>1) proc_bind(AFFINITY) default(shared)
                        {
                            Nd4jLong tid = omp_get_thread_num();
                            Nd4jLong start = span * tid;
                            Nd4jLong end = span * (tid + 1);
                            if (end > n) end = n;
#pragma omp simd
                            for (Nd4jLong i = start; i < end; i++) {
                                result[i] = OpType::op(dx[i], y[i], extraParams);
                            }
                        }
                    } else {
#pragma omp simd
                        for (Nd4jLong i = 0; i < n; i++) {
                            result[i] = OpType::op(dx[i], y[i], extraParams);
                        }
                    }
                }
                else {
                    if (_threads > 1) {
#pragma omp parallel num_threads(_threads) if (_threads>1) proc_bind(AFFINITY) default(shared)
                        {
                            Nd4jLong tid = omp_get_thread_num();
                            Nd4jLong start = span * tid;
                            Nd4jLong end = span * (tid + 1);
                            if (end > n) end = n;

#pragma omp simd
                            for (Nd4jLong i = start; i < end; i++) {
                                result[i * resultStride] = OpType::op(dx[i * xStride], y[i * yStride], extraParams);
                            }
                        }
                    } else {
#pragma omp simd
                        for (Nd4jLong i = 0; i < n; i++) {
                            result[i * resultStride] = OpType::op(dx[i * xStride], y[i * yStride], extraParams);
                        }
                    }
                }
            }
        };
    }
}

#endif /* PAIRWISE_TRANSFORM_H_ */
