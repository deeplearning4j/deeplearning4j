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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#include <loops/broadcasting.h>
#include <loops/legacy_ops.h>

namespace functions {
    namespace broadcast {

        template <typename X, typename Y>
        void Broadcast<X, Y>::exec(const int opNum,
                             X *x,
                             Nd4jLong *xShapeInfo,
                             Y *y,
                             Nd4jLong *yShapeInfo,
                             X *result,
                             Nd4jLong *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset,
                             Nd4jLong *tadShapeInfoZ,
                             Nd4jLong *tadOffsetZ) {
            DISPATCH_BY_OPNUM_TT(exec, PARAMS(x,
                                               xShapeInfo,
                                               y,
                                               yShapeInfo,
                                               result,
                                               resultShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset,
                                               tadShapeInfoZ,
                                               tadOffsetZ), BROADCAST_OPS);
        }

        template <typename X, typename  Y>
        template<typename OpType>
        void Broadcast<X, Y>::exec(X *x,
                             Nd4jLong *xShapeInfo,
                             Y *y,
                             Nd4jLong *yShapeInfo,
                             X *result,
                             Nd4jLong *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset,
                             Nd4jLong *tadShapeInfoZ,
                             Nd4jLong *tadOffsetZ) {


                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                auto tadShapeShapeInfo = tadShapeInfo;
                auto tadOffsets = tadOffset;
                shape::TAD *tad = nullptr;

                if (tadShapeInfo == nullptr || tadOffsets == nullptr) {
                    tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    tadShapeShapeInfo = tad->tadOnlyShapeInfo;
                    tadOffsets = tad->tadOffsets;
                }

                //int *resultStride = shape::stride(tadShapeShapeInfo);
                auto tadEWS = shape::elementWiseStride(tadShapeShapeInfo);
                auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                auto yStride = shape::elementWiseStride(yShapeInfo);
                auto tads = shape::length(xShapeInfo) / tadLength;

                if (tadShapeInfoZ == nullptr) {
                    tadShapeInfoZ = tadShapeShapeInfo;
                    tadOffsetZ = tadOffsets;
                }

                auto zEWS = shape::elementWiseStride(tadShapeInfoZ);

                int tadsPerThread = tads / TAD_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(_threads) if (_threads > 1) proc_bind(AFFINITY) default(shared)
                for (int i = 0; i < tads; i++) {
                    auto offset = tadOffsets[i];
                    auto offsetZ = tadOffsetZ[i];

                    if (tadEWS > 0 && yStride > 0 && zEWS > 0 && dimensionLength == 1) {
                        auto oRes = result + offsetZ;
                        auto oX = x + offset;

                        if (tadEWS == 1 && yStride == 1 && zEWS == 1) {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oRes[f] = OpType::op(oX[f], y[f]);
                            }
                        } else {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) {
                                oRes[f * zEWS] = OpType::op(oX[f * tadEWS], y[f * yStride]);
                            }
                        }
                    }
                    else {
                        auto zShape = shape::shapeOf(tadShapeInfoZ);
                        auto zStrides = shape::stride(tadShapeInfoZ);
                        int zRank = shape::rank(tadShapeInfoZ);

                        auto xShape = shape::shapeOf(tadShapeShapeInfo);
                        auto xStrides = shape::stride(tadShapeShapeInfo);
                        int xRank = shape::rank(tadShapeShapeInfo);

                        auto yShape = shape::shapeOf(yShapeInfo);
                        auto yStrides = shape::stride(yShapeInfo);
                        int yRank = shape::rank(yShapeInfo);

                        Nd4jLong xCoord[MAX_RANK];
                        Nd4jLong yCoord[MAX_RANK];
                        Nd4jLong zCoord[MAX_RANK];


                        // TODO: cover this codebranch with tests
                        // all this stuff already happens within thread
                        for (int f = 0; f < tadLength; f++) {
                            if (shape::order(tadShapeShapeInfo) == 'c') {
                                shape::ind2subC(xRank, xShape, f, tadLength, xCoord);
                                shape::ind2subC(yRank, yShape, f, tadLength, yCoord);
                            } else {
                                shape::ind2sub(xRank, xShape, f, tadLength, xCoord);
                                shape::ind2sub(yRank, yShape, f, tadLength, yCoord);
                            }

                            if (shape::order(tadShapeInfoZ) == 'c')
                                shape::ind2subC(zRank, zShape, f, tadLength, zCoord);
                            else
                                shape::ind2sub(zRank, zShape, f, tadLength, zCoord);

                            auto xOffset = shape::getOffset(offset, xShape, xStrides, xCoord, xRank);
                            auto zOffset = shape::getOffset(offsetZ, zShape, zStrides, zCoord, zRank);
                            auto yOffset = shape::getOffset(0, yShape, yStrides, yCoord, yRank);

                            result[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                        }
                    }
                }

                if (tad != nullptr)
                    delete tad;
        }

        //BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT Broadcast, , LIBND4J_TYPES);

        //BUILD_CALL_1(template void Broadcast<float, float>::exec, float, (float*, Nd4jLong*, float*, Nd4jLong*, float*, Nd4jLong*, int*, int, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*), BROADCAST_OPS)
        //BUILD_CALL_1(template void Broadcast<float16, float16>::exec, float16, (float16*, Nd4jLong*, float16*, Nd4jLong*, float16*, Nd4jLong*, int*, int, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*), BROADCAST_OPS)
        //BUILD_CALL_1(template void Broadcast<double, double>::exec, double, (double*, Nd4jLong*, double*, Nd4jLong*, double*, Nd4jLong*, int*, int, Nd4jLong*, Nd4jLong*, Nd4jLong*, Nd4jLong*), BROADCAST_OPS)
    }
}