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
// @author raver119@gmail.com
//

#include "../scalar_int.h"
#include <op_boilerplate.h>
#include <types/types.h>
#include <LoopKind.h>
#include <execution/Threads.h>

#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
    namespace scalar {


        template<typename X>
        template<typename OpType>
        void ScalarIntTransform<X>::transform(void *vx, Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vz,  Nd4jLong *zShapeInfo,
                                                void *vscalars,
                                                int *dimension, int dimensionLength,
                                                Nd4jLong *xTadShapeInfo, Nd4jLong *xTadOffsets,
                                                Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {

            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<X *>(vz);
            auto scalars = reinterpret_cast<X *>(vscalars);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            if (zTadShapeInfo == nullptr) {
                zTadShapeInfo = xTadShapeInfo;
                zTadOffsets   = xTadOffsets;
            }

            // tad preparation
            const int xTadEws    = shape::elementWiseStride(xTadShapeInfo);
            const int zTadEws    = shape::elementWiseStride(zTadShapeInfo);
            const int tadLength  = shape::tadLength(xShapeInfo, dimension, dimensionLength);
            const int numTads    = shape::length(xShapeInfo) / tadLength;

            nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXZ(xTadShapeInfo, zTadShapeInfo);

            if (kindOfLoop != nd4j::LoopKind::EWS1 && kindOfLoop != nd4j::LoopKind::EWSNONZERO) {
                printf("ScalarIntTransform<X>::transform: super-bad loop visited. Shouldn't ever happen\n");
                return;
            }

            int num_threads = nd4j::math::nd4j_min<int>(numTads, nd4j::Environment::getInstance()->maxThreads());

            if (kindOfLoop == nd4j::LoopKind::EWS1) {
                auto func = PRAGMA_THREADS_FOR {
                    for (unsigned int r = start; r < stop; r += increment) {
                        auto oZ = z + zTadOffsets[r];
                        auto oX = x + xTadOffsets[r];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++)
                            oZ[f] = OpType::op(oX[f], scalars[r], extraParams);
                    }
                };

                samediff::Threads::parallel_for(func, num_threads, 0, numTads, 1);
            }
            else {
                auto func = PRAGMA_THREADS_FOR {
                    for (unsigned int r = start; r < stop; r += increment) {
                        auto oZ = z + zTadOffsets[r];
                        auto oX = x + xTadOffsets[r];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++)
                            oZ[f * zTadEws] = OpType::op(oX[f * xTadEws], scalars[r], extraParams);
                    }
                };

                samediff::Threads::parallel_for(func, num_threads, 0, numTads, 1);
            }
        }

        template<typename X>
        void ScalarIntTransform<X>::transform(int opNum,
                              void *x,
                              Nd4jLong *xShapeInfo,
                              void *extraParams,
                              void *z,
                              Nd4jLong *zShapeInfo,
                              void *scalars,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *xTadShapeInfo,
                              Nd4jLong *xTadOffsets,
                              Nd4jLong *zTadShapeInfo,
                              Nd4jLong *zTadOffsets) {
            DISPATCH_BY_OPNUM_T(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets), SCALAR_INT_OPS);
        }


        template<typename X>
        void ScalarIntTransform<X>::transform(const int opNum,
                void *x,
                Nd4jLong xEws,
                void *z,
                Nd4jLong zEws,
                void *scalar,
                void *extraParams,
                const Nd4jLong n) {
            DISPATCH_BY_OPNUM_T(transform, PARAMS(x, xEws, z, zEws, scalar, extraParams, n), SCALAR_INT_OPS);
        }

        template<typename X>
        void ScalarIntTransform<X>::transform(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *z,
                Nd4jLong *zShapeInfo,
                void *scalar,
                void *extraParams) {
            DISPATCH_BY_OPNUM_T(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams), SCALAR_INT_OPS);
        }

        template<typename X>
        template<typename OpType>
        void ScalarIntTransform<X>::transform(void *vx,
                               Nd4jLong *xShapeInfo,
                               void *vz,
                               Nd4jLong *zShapeInfo,
                               void *vscalar,
                               void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<X *>(vz);
            auto scalar = reinterpret_cast<X *>(vscalar)[0];
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            auto xEws = shape::elementWiseStride(xShapeInfo);
            auto zEws = shape::elementWiseStride(zShapeInfo);
            auto len = shape::length(xShapeInfo);

            nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

            if (kindOfLoop == nd4j::LoopKind::EWS1 || kindOfLoop == nd4j::LoopKind::EWSNONZERO) {
                transform<OpType>(x, xEws, z, zEws, vscalar, extraParams, len);
                return;
            }

            uint xShapeInfoCast[MAX_RANK];
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo<uint>(xShapeInfo, xShapeInfoCast);

            if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {

                auto f = PRAGMA_THREADS_FOR {
                    PRAGMA_OMP_SIMD
                    for (uint64_t i = start; i < stop; i += increment) {
                        auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                        z[offset] = OpType::op(x[offset], scalar, extraParams);
                    }
                };

                samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(), 0, len, 1);
            }
            else {

                uint zShapeInfoCast[MAX_RANK];
                const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo, zShapeInfoCast);

                auto f = PRAGMA_THREADS_FOR {
                    PRAGMA_OMP_SIMD
                    for (uint64_t i = start; i < stop; i += increment) {
                        auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                        auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
                        z[zOffset] = OpType::op(x[xOffset], scalar, extraParams);
                    }
                };

                samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(), 0, len, 1);
            }
        }


            template<typename X>
            template<typename OpType>
            void ScalarIntTransform<X>::transform(void *vx,
                    Nd4jLong xEws,
                    void *vz,
                    Nd4jLong zEws,
                    void *vscalar,
                    void *vextraParams,
                    const Nd4jLong len) {

                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<X *>(vz);
                auto scalar = reinterpret_cast<X *>(vscalar)[0];
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                if (xEws == 1 && zEws == 1) {

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for (uint64_t i = start; i < stop; i += increment)
                            z[i] = OpType::op(x[i], scalar, extraParams);
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(), 0, len, 1);
                }
                else {

                    auto f = PRAGMA_THREADS_FOR {
                        PRAGMA_OMP_SIMD
                        for (uint64_t i = start; i < stop; i += increment)
                            z[i * zEws] = OpType::op(x[i * xEws], scalar, extraParams);
                    };

                    samediff::Threads::parallel_for(f, nd4j::Environment::getInstance()->maxThreads(), 0, len, 1);
                }
            }

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT ScalarIntTransform, , INTEGER_TYPES);

}
}
