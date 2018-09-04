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
// Created by raver119 on 18.12.17.
//

#include <types/types.h>
#include <op_boilerplate.h>
#include <loops/summarystatsreduce.h>
#include <helpers/shape.h>
#include <helpers/TAD.h>

namespace functions {
    namespace summarystats {


        template <typename X>
        X SummaryStatsReduce<X>::execScalar(const int opNum,
                const bool biasCorrected,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams) {
            RETURNING_DISPATCH_BY_OPNUM_T(execScalar, PARAMS(biasCorrected, x, xShapeInfo, extraParams), SUMMARY_STATS_OPS);
        }

        template <typename X>
        void SummaryStatsReduce<X>::exec(const int opNum,
                const bool biasCorrected,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams,
                void *result,
                Nd4jLong *resultShapeInfoBuffer,
                int *dimension,
                int dimensionLength) {
            DISPATCH_BY_OPNUM_T(exec, PARAMS(biasCorrected, x, xShapeInfo, extraParams, result, resultShapeInfoBuffer, dimension, dimensionLength), SUMMARY_STATS_OPS);
        }

        template <typename X>
        template <typename OpType >
        X SummaryStatsReduce<X>::execScalar(const bool biasCorrected,
                void *vx,
                Nd4jLong *xShapeInfo,
                void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            SummaryStatsData<X> startingIndex;
            startingIndex.initialize();
            Nd4jLong length = shape::length(xShapeInfo);
            int xElementWiseStride = shape::elementWiseStride(xShapeInfo);
            if (xElementWiseStride == 1) {
                for (Nd4jLong i = 0; i < length; i++) {
                    SummaryStatsData<X> curr;
                    curr.initWithValue(x[i]);
                    startingIndex = update(startingIndex, curr,
                                           extraParams);
                }

                return (X) OpType::getValue(biasCorrected, startingIndex);
            }
            else {
                Nd4jLong xCoords[MAX_RANK];

                auto xShape = shape::shapeOf(xShapeInfo);
                auto xStride = shape::stride(xShapeInfo);
                int xRank = shape::rank(xShapeInfo);


                for (Nd4jLong i = 0; i < length; i++) {
                    shape::ind2subC(xRank, xShape, i, length, xCoords);
                    auto xOffset = shape::getOffset(0, xShape, xStride, xCoords, xRank);

                    SummaryStatsData<X> curr;
                    curr.initWithValue(x[xOffset]);
                    startingIndex = update(startingIndex, curr, extraParams);
                }

                return (X)OpType::getValue(biasCorrected, startingIndex);
            }
        }

        template <typename X>
        template <typename OpType >
        void SummaryStatsReduce<X>::exec(const bool biasCorrected,
                void *vx,
                Nd4jLong *xShapeInfo,
                void *vextraParams,
                void *vresult,
                Nd4jLong *resultShapeInfoBuffer,
                int *dimension,
                int dimensionLength) {
            auto x = reinterpret_cast<X *>(vx);
            auto result = reinterpret_cast<X *>(vresult);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            if (shape::isScalar(resultShapeInfoBuffer)) {
                result[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                return;
            }


            shape::TAD tad(xShapeInfo, dimension, dimensionLength);
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();

            //no-op
            if (tad.dimensionLength < 1)
                return;

            int resultLength = shape::length(resultShapeInfoBuffer);
            //pre squeezed: this is for keeping the pointer to the original
            //shape information for tad offset
            //the squeezed information doesn't render the right strides for
            //tad offset
            if (resultLength == 1 || dimensionLength == shape::rank(xShapeInfo) || tad.wholeThing) {
                result[0] = execScalar<OpType>(biasCorrected, x, xShapeInfo, extraParams);
                return;
            }

            if (!(shape::elementWiseStride(tad.tadOnlyShapeInfo) > 0 && (tad.numTads == 1 || shape::isVector(tad.tadOnlyShapeInfo) ||
                                                                         shape::isScalar(tad.tadOnlyShapeInfo) || tad.wholeThing)) && !(dimensionLength > 1)) {

                /**
                 * The element wise stride belong longs to a reduction index.
                 * When used out of order, we can get rid of the data
                 * dependencies and rely on using the max dimension
                 * specified for stride instead.
                 * Say we take the sum(0,1) along long arr
                 * we can use arr.stride(1) as a representation
                 * along long which to iterate.
                 */

                auto tadShapeShapeInfo = tad.tadOnlyShapeInfo;

                auto xShape = shape::shapeOf(tadShapeShapeInfo);
                auto xStride = shape::stride(tadShapeShapeInfo);
                int rank = shape::rank(tadShapeShapeInfo);
#pragma omp parallel for schedule(guided) default(shared)
                for (int i = 0; i < resultLength; i++) {
                    auto offset = tad.tadOffsets[i];
                    Nd4jLong shapeIter[MAX_RANK];
                    Nd4jLong coord[MAX_RANK];
                    int dim;
                    int rankIter = rank;
                    Nd4jLong xStridesIter[MAX_RANK];
                    auto xPointer = x + offset;
                    SummaryStatsData<X> comp;
                    comp.initWithValue(0.0);
                    if (PrepareOneRawArrayIter<X>(rankIter,
                                                  xShape,
                                                  xPointer,
                                                  xStride,
                                                  &rankIter,
                                                  shapeIter,
                                                  &xPointer,
                                                  xStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter); {
                                /* Process the innermost dimension */
                                SummaryStatsData<X> comp2;
                                comp2.initWithValue(xPointer[0]);
                                comp = update(comp, comp2, extraParams);
                            } ND4J_RAW_ITER_ONE_NEXT(dim,
                                                     rank,
                                                     coord,
                                                     shapeIter,
                                                     xPointer,
                                                     xStridesIter);
                    }
                    else {
                        printf("Unable to prepare array\n");
                    }

                    result[i] = OpType::getValue(biasCorrected, comp);
                }
            }
            else {
                if (dimensionLength == 1) {
                    auto tadElementWiseStride = shape::elementWiseStride(tad.tadOnlyShapeInfo);
                    auto tadLength = shape::length(tad.tadOnlyShapeInfo);

#pragma omp parallel for schedule(guided) default(shared)
                    for (int i = 0; i < resultLength; i++) {
                        Nd4jLong baseOffset = tad.tadOffsets[i];
                        SummaryStatsData<X> comp;
                        comp.initWithValue(x[baseOffset]);
// FIXME: reduction to be used here
                        for (int j = 1; j < tadLength; j++) {
                            SummaryStatsData<X> comp2;
                            comp2.initWithValue(x[baseOffset + (tadElementWiseStride * j)]);
                            comp = update(comp, comp2, extraParams);
                        }

                        result[i] = OpType::getValue(biasCorrected, comp);
                    }
                } else {
                    auto tadShapeShapeInfo = tad.tadOnlyShapeInfo;

                    auto tadShape = shape::shapeOf(tadShapeShapeInfo);
                    auto tadStride = shape::stride(tadShapeShapeInfo);
                    auto tadRank = shape::rank(tadShapeShapeInfo);
                    auto tadLength = shape::length(tad.tadOnlyShapeInfo);

#pragma omp parallel for schedule(guided) default(shared)
                    for (int r = 0; r < resultLength; r++) {
                        Nd4jLong xCoord[MAX_RANK];
                        auto tadOffsetForBlock = tad.tadOffsets[r];

                        SummaryStatsData<X> comp;
                        comp.initWithValue(x[tadOffsetForBlock]);

// FIXME: reduction should be fixed
                        for (int i = 1; i < tadLength; i ++) {
                            shape::ind2subC(tadRank, tadShape, i, tadLength, xCoord);
                            auto xOffset = shape::getOffset(tadOffsetForBlock, tadShape, tadStride, xCoord, tadRank);

                            SummaryStatsData <X> indexVal2;
                            indexVal2.initWithValue(x[xOffset]);

                            comp = update(comp, OpType::op(indexVal2, extraParams), extraParams);
                        }
                        result[r] = OpType::getValue(biasCorrected, comp);
                    }
                }
            }
        }


        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT SummaryStatsReduce, , LIBND4J_TYPES);
    }
}