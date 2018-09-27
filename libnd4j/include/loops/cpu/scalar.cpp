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
// Created by raver119 on 08.10.2017.
//

#include "../scalar.h"
#include <op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
    namespace scalar {


        template<typename X, typename Y, typename Z>
        template<typename OpType>
        void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, void *vscalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<X *>(vz);
            auto scalars = reinterpret_cast<Y *>(vscalars);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

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

                auto offset = tadOffsets[r];
                auto offsetZ = tadOffsetsZ[r];
                auto scalar = scalars[r];

                if (tadEWS >= 1 && zEWS >= 1) {
                    auto oZ = z + offsetZ;
                    auto oX = x + offset;

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

        template<typename X, typename Y, typename Z>
        void ScalarTransform<X,Y,Z>::transform(int opNum,
                              void *x,
                              Nd4jLong *xShapeInfo,
                              void *extraParams,
                              void *z,
                              Nd4jLong *zShapeInfo,
                              void *scalars,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *tadShapeInfo,
                              Nd4jLong *tadOffsets,
                              Nd4jLong *tadShapeInfoZ,
                              Nd4jLong *tadOffsetsZ) {
            DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_OPS);
        }


        template<typename X, typename Y, typename Z>
        void ScalarTransform<X, Y, Z>::transform(const int opNum,
                void *x,
                Nd4jLong xStride,
                void *result,
                Nd4jLong resultStride,
                void *scalar,
                void *extraParams,
                const Nd4jLong n) {
            DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xStride, result, resultStride, scalar, extraParams, n), SCALAR_OPS);
        }

        template<typename X, typename Y, typename Z>
        void ScalarTransform<X, Y, Z>::transform(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *result,
                Nd4jLong *resultShapeInfo,
                void *scalar,
                void *extraParams) {
            DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, result, resultShapeInfo, scalar, extraParams), SCALAR_OPS);
        }

        template<typename X, typename Y, typename Z>
        template<typename OpType>
        void ScalarTransform<X, Y, Z>::transform(void *vx,
                               Nd4jLong *xShapeInfo,
                               void *vz,
                               Nd4jLong *resultShapeInfo,
                               void *vscalar,
                               void *vextraParams) {
            auto x = reinterpret_cast<X *>(vx);
            auto result = reinterpret_cast<X *>(vz);
            auto scalar = reinterpret_cast<Y *>(vscalar)[0];
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

            char xOrdering = shape::order(xShapeInfo);
            char resultOrdering = shape::order(resultShapeInfo);
            int xElementWiseStride = shape::elementWiseStride(xShapeInfo);

            // nd4j_logger("Launching scalar: xOrder: %i; zOrder: %i; xEWS: %i\n", xOrdering, resultOrdering, xElementWiseStride);

            if (xElementWiseStride == 1 && shape::elementWiseStride(resultShapeInfo) == 1 && xOrdering == resultOrdering) {
                transform<OpType>(x, 1, result, 1, vscalar, extraParams, shape::length(xShapeInfo));
                return;
            }

            int resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);
            if(xOrdering != resultOrdering || xElementWiseStride < 1 || resultElementWiseStride < 0) {
                Nd4jLong shapeIter[MAX_RANK];
                Nd4jLong coord[MAX_RANK];
                int dim;
                Nd4jLong xStridesIter[MAX_RANK];
                Nd4jLong resultStridesIter[MAX_RANK];
                auto xShape = shape::shapeOf(xShapeInfo);
                auto xStride = shape::stride(xShapeInfo);
                auto resultStride = shape::stride(resultShapeInfo);
                int rank = shape::rank(xShapeInfo);
                if(PrepareTwoRawArrayIter<X, X>(rank,
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
                            auto xIter = x;
                            auto resultIter = result;
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
                const Nd4jLong n = shape::length(xShapeInfo);

                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1) {
                    transform<OpType>(x,xElementWiseStride,result,resultElementWiseStride,vscalar,extraParams,n);
                }
                else {
                    Nd4jLong xIdx[MAX_RANK];
                    Nd4jLong resultIdx[MAX_RANK];

                    auto xShape = shape::shapeOf(xShapeInfo);
                    auto resultShape = shape::shapeOf(resultShapeInfo);

                    auto xStride = shape::stride(xShapeInfo);
                    auto resultStride = shape::stride(resultShapeInfo);
                    int xRank = shape::rank(xShapeInfo);
                    int resultRank = shape::rank(resultShapeInfo);
                    if (vx == vz) {
#pragma omp parallel for schedule(guided) if (n > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                        for (Nd4jLong i = 0; i < n; i++) {
                            shape::ind2sub(xRank, xShape, i, n, xIdx);

                            auto xOffset2 = shape::getOffset(0, xShape, xStride, xIdx, xRank);

                            result[xOffset2] = OpType::op(x[xOffset2], scalar, extraParams);
                        }
                    } else {
#pragma omp parallel for schedule(guided) if (n > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                        for (Nd4jLong i = 0; i < n; i++) {
                            shape::ind2sub(xRank, xShape, i, n, xIdx);
                            shape::ind2sub(resultRank, resultShape, i, n, resultIdx);

                            auto xOffset2 = shape::getOffset(0, xShape, xStride, xIdx, xRank);
                            auto resultOffset2 = shape::getOffset(0, resultShape, resultStride, resultIdx, resultRank);

                            result[resultOffset2] = OpType::op(x[xOffset2], scalar, extraParams);
                        }
                    }
                }

                }
            }


            template<typename X, typename Y, typename Z>
            template<typename OpType>
            void ScalarTransform<X, Y, Z>::transform(void *vx,
                    Nd4jLong xStride,
                    void *vz,
                    Nd4jLong resultStride,
                    void *vscalar,
                    void *vextraParams,
                    const Nd4jLong n) {
                auto x = reinterpret_cast<X *>(vx);
                auto result = reinterpret_cast<X *>(vz);
                auto scalar = reinterpret_cast<Y *>(vscalar)[0];
                auto extraParams = reinterpret_cast<Z *>(vextraParams);
/*
                Nd4jLong elementsPerThread = n / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());
*/
                int num_threads = 1;
                Nd4jLong span = 100;// (n / num_threads) + 8;

                if (xStride == 1 && resultStride == 1) {
                    if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                        {
                            Nd4jLong tid = omp_get_thread_num();
                            Nd4jLong start = span * tid;
                            Nd4jLong end = span * (tid + 1);
                            if (end > n) end = n;
#pragma omp simd
                            for (Nd4jLong i = start; i < end; i++) {
                                result[i] = OpType::op(x[i], scalar, extraParams);
                            }
                        }
                    } else {
#pragma omp simd
                        for (Nd4jLong i = 0; i < n; i++) {
                            result[i] = OpType::op(x[i], scalar, extraParams);
                        }
                    }
                }

                else {
                    if (num_threads > 1) {
#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                        {
                            Nd4jLong tid = omp_get_thread_num();
                            Nd4jLong start = span * tid;
                            Nd4jLong end = span * (tid + 1);
                            if (end > n) end = n;
#pragma omp simd
                            for (Nd4jLong i = start; i < end; i++) {
                                result[i * resultStride] = OpType::op(x[i * xStride], scalar, extraParams);
                            }
                        }
                    } else {
#pragma omp simd
                        for (Nd4jLong i = 0; i < n; i++) {
                            result[i * resultStride] = OpType::op(x[i * xStride], scalar, extraParams);
                        }
                    }
                }
            }


        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_0)
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_1)
        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT ScalarTransform, , PAIRWISE_TYPES_2)
    }
}