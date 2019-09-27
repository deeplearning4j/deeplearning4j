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

#include <ops/ops.h>
#include <loops/pairwise_transform.h>
#include <types/types.h>
#include <LoopKind.h>
#include <templatemath.h>
#include <helpers/shape.h>
#include <op_boilerplate.h>
#include <OmpLaunchHelper.h>
#include <execution/Threads.h>

using namespace simdOps;

namespace functions {
    namespace pairwise_transforms {

        template <typename X, typename Y, typename Z>
        void PairWiseTransform<X, Y, Z>::exec(
                const int opNum,
                void *x,
                Nd4jLong xEws,
                void *y,
                Nd4jLong yEws,
                void *z,
                Nd4jLong zEws,
                void *extraParams,
                Nd4jLong n) {
            DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x,
                                               xEws,
                                               y,
                                               yEws,
                                               z,
                                               zEws,
                                               extraParams,
                                               n), PAIRWISE_TRANSFORM_OPS);
        };



        template <typename X, typename Y, typename Z>
        template <typename OpType>
        void PairWiseTransform<X, Y, Z>::exec(void *vx, Nd4jLong xEws,
                                              void *vy, Nd4jLong yEws,
                                              void *vz, Nd4jLong zEws,
                                              void *vextraParams,
                                              const Nd4jLong n) {

            auto x = reinterpret_cast<X *>(vx);
            auto y = reinterpret_cast<Y *>(vy);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

            if (xEws == 1 && yEws == 1 && zEws == 1) {

                auto f = PRAGMA_THREADS_FOR {
                    PRAGMA_OMP_SIMD
                    for (uint64_t i = start; i < stop; i += increment)
                        z[i] = OpType::op(x[i], y[i], extraParams);
                };

                samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
            }
            else {

                auto f = PRAGMA_THREADS_FOR {
                    PRAGMA_OMP_SIMD
                    for (uint64_t i = start; i < stop; i += increment)
                        z[i*zEws] = OpType::op(x[i*xEws], y[i*yEws], extraParams);
                };

                samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
            }
        }

        template <typename X, typename Y, typename Z>
        void PairWiseTransform<X, Y, Z>::exec(
                const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *y,
                Nd4jLong *yShapeInfo,
                void *z,
                Nd4jLong *zShapeInfo,
                void *extraParams) {
            DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x,
                                              xShapeInfo,
                                              y,
                                              yShapeInfo,
                                              z,
                                              zShapeInfo,
                                              extraParams),
                                 PAIRWISE_TRANSFORM_OPS);
        };


        template <typename X, typename Y, typename Z>
        template <typename OpType>
        void PairWiseTransform<X, Y, Z>::exec(
                void *vx,
                Nd4jLong* xShapeInfo,
                void *vy,
                Nd4jLong* yShapeInfo,
                void *vz,
                Nd4jLong* zShapeInfo,
                void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto y = reinterpret_cast<Y *>(vy);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

            auto n = shape::length(xShapeInfo);
            auto xEws = shape::elementWiseStride(xShapeInfo);
            auto yEws = shape::elementWiseStride(yShapeInfo);
            auto zEws = shape::elementWiseStride(zShapeInfo);


            if (shape::isScalar(yShapeInfo)) {

                uint xShapeInfoCast[MAX_RANK];
                const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for(uint64_t i = start; i < stop; i += increment)  {
                            auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                            z[offset] = OpType::op(x[offset], y[0], extraParams);
                        }
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
                }
                else {
                    uint zShapeInfoCast[MAX_RANK];
                    const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for(uint64_t i = start; i < stop; i += increment)  {
                            auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                            auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
                            z[zOffset] = OpType::op(x[xOffset], y[0], extraParams);
                        }
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
                }
                return;
            }



            const nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);
            const bool sameShapesXY = shape::shapeEquals(xShapeInfo, yShapeInfo);

            if ((kindOfLoop == nd4j::LoopKind::EWS1 || kindOfLoop == nd4j::LoopKind::EWSNONZERO) && sameShapesXY) {
                exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, n);
            }
            else if ((kindOfLoop == nd4j::LoopKind::EWS1 || kindOfLoop == nd4j::LoopKind::EWSNONZERO) && !sameShapesXY) { //not same shape
                exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, shape::length(yShapeInfo));
            }
            else {

                if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo) && shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {

                    uint xShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for (uint64_t i = start; i < stop; i += increment)  {
                            auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                            z[offset] = OpType::op(x[offset], y[offset], extraParams);
                        }
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
                }
                else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {

                    uint xShapeInfoCast[MAX_RANK];
                    uint zShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for (uint64_t i = start; i < stop; i += increment)  {
                            auto offset  = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                            auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
                            z[zOffset] = OpType::op(x[offset], y[offset], extraParams);
                        }
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
                }
                else if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {

                    uint xShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for (uint64_t i = start; i < stop; i += increment)  {
                            auto offset  = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                            auto yOffset = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, canCastY);
                            z[offset] = OpType::op(x[offset], y[yOffset], extraParams);
                        }
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
                }
                else if(shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo)) {

                    uint xShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for (uint64_t i = start; i < stop; i += increment)  {
                            auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                            auto offset  = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, canCastY);
                            z[offset] = OpType::op(x[xOffset], y[offset], extraParams);
                        }
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(),0, n, 1);
                }
                else {

                    uint xShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    uint zShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
                    bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for (unsigned int i = start; i < stop; i += increment)  {
                            auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                            auto yOffset = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, canCastY);
                            auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
                            z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
                        }
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(), 0, n, 1);
                }
            }
        }
    }
}
