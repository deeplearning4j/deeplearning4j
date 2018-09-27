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
// Created by remote on 2018-09-20.
//

#include <loops/pairwise_transform.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
    namespace pairwise_transforms {

        template <typename X, typename Y, typename Z>
        void PairWiseTransform<X, Y, Z>::exec(
                const int opNum,
                void *dx,
                Nd4jLong *xShapeBuffer,
                void *y,
                Nd4jLong *yShapeBuffer,
                void *result,
                Nd4jLong *resultShapeBuffer,
                void *extraParams) {
            DISPATCH_BY_OPNUM_TT(exec, PARAMS(dx,
                                              xShapeBuffer,
                                              y,
                                              yShapeBuffer,
                                              result,
                                              resultShapeBuffer,
                                              extraParams),
                                 PAIRWISE_TRANSFORM_OPS);
        };


        template <typename X, typename Y, typename Z>
        template <typename OpType>
        void PairWiseTransform<X, Y, Z>::exec(
                void *vx,
                Nd4jLong* xShapeBuffer,
                void *vy,
                Nd4jLong* yShapeBuffer,
                void *vresult,
                Nd4jLong* resultShapeBuffer,
                void *vextraParams) {
            auto dx = reinterpret_cast<X *>(vx);
            auto y = reinterpret_cast<Y *>(vy);
            auto result = reinterpret_cast<Z *>(vresult);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

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

                    auto xShape = shape::shapeOf(xShapeBuffer);
                    auto xStride = shape::stride(xShapeBuffer);

                    auto resultShape = shape::shapeOf(resultShapeBuffer);
                    auto resultStride = shape::stride(resultShapeBuffer);

                    int elementsPerThread = n / ELEMENT_THRESHOLD;
                    int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                    num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared) private(xCoord, resultCoord)
                    for (Nd4jLong i = 0; i < n; i++) {
                        shape::ind2subC(xRank,xShape, i, xCoord);
                        shape::ind2subC(resultRank,resultShape, i, resultCoord);

                        auto xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                        auto resultOffset = shape::getOffset(0, resultShape, resultStride, resultCoord, resultRank);
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
                auto xShape = shape::shapeOf(xShapeBuffer);
                auto xStride = shape::stride(xShapeBuffer);
                auto yStride = shape::stride(yShapeBuffer);
                auto resultStride = shape::stride(resultShapeBuffer);

                // tad-oriented rotation technically

                int tadsPerThread = xShape[0] / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                Nd4jLong shapeIter[MAX_RANK];
                Nd4jLong coord[MAX_RANK];
                Nd4jLong xStridesIter[MAX_RANK];
                Nd4jLong yStridesIter[MAX_RANK];
                Nd4jLong resultStridesIter[MAX_RANK];

#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared) private(shapeIter,coord,xStridesIter,yStridesIter,resultStridesIter)
                for (Nd4jLong i = 0; i < xShape[0]; i++) {
                    auto dxLocal = dx + xStride[0] * i;
                    auto yLocal = y + yStride[0] * i;
                    auto resultLocal = result + resultStride[0] * i;

                    int rankLocal = rank - 1;
                    auto xShapeLocal = xShape + 1;

                    auto xStrideLocal = xStride + 1;
                    auto yStrideLocal = yStride + 1;
                    auto resultStrideLocal = resultStride + 1;

                    int dim;
                    if (PrepareThreeRawArrayIter<X, Y, Z>(rankLocal,
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

        //BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , LIBND4J_TYPES, LIBND4J_TYPES);
    }
}
