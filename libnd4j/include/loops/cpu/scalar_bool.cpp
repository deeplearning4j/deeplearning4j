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

#include "../scalar_bool.h"
#include <op_boilerplate.h>
#include <types/types.h>
#include <LoopKind.h>

#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
    namespace scalar {


        template<typename X, typename Z>
        template<typename OpType>
        void ScalarBoolTransform<X, Z>::transform(void *vx, Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vz,  Nd4jLong *zShapeInfo,
                                                void *vscalars,
                                                int *dimension, int dimensionLength,
                                                Nd4jLong *xTadShapeInfo, Nd4jLong *xTadOffsets,
                                                Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {

            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<Z *>(vz);
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
                printf("ScalarBoolTransform<X, Z>::transform: super-bad loop visited. Shouldn't ever happen\n");
                return;
            }

            int num_threads = nd4j::math::nd4j_min<int>(numTads, omp_get_max_threads());

            if (kindOfLoop == nd4j::LoopKind::EWS1) {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(num_threads)
                for (unsigned int r = 0; r < numTads; r++) {
                    auto oZ = z + zTadOffsets[r];
                    auto oX = x + xTadOffsets[r];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++)
                        oZ[f] = OpType::op(oX[f], scalars[r], extraParams);
                }
            }
            else { // kindOfLoop != nd4j::LoopKind::EWSNONZERO
                PRAGMA_OMP_PARALLEL_FOR_THREADS(num_threads)
                for (unsigned int r = 0; r < numTads; r++) {
                    auto oZ = z + zTadOffsets[r];
                    auto oX = x + xTadOffsets[r];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++)
                        oZ[f * zTadEws] = OpType::op(oX[f * xTadEws], scalars[r], extraParams);
                }
            }
        }

        template<typename X, typename Y>
        void ScalarBoolTransform<X,Y>::transform(int opNum,
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
            DISPATCH_BY_OPNUM_TT(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets), SCALAR_BOOL_OPS);
        }


        template<typename X, typename Y>
        void ScalarBoolTransform<X, Y>::transform(const int opNum,
                void *x,
                Nd4jLong xEws,
                void *z,
                Nd4jLong zEws,
                void *scalar,
                void *extraParams,
                const Nd4jLong n) {
            DISPATCH_BY_OPNUM_TT(transform, PARAMS(x, xEws, z, zEws, scalar, extraParams, n), SCALAR_BOOL_OPS);
        }

        template<typename X, typename Y>
        void ScalarBoolTransform<X, Y>::transform(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *z,
                Nd4jLong *zShapeInfo,
                void *scalar,
                void *extraParams) {
            DISPATCH_BY_OPNUM_TT(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams), SCALAR_BOOL_OPS);
        }

        template<typename X, typename Z>
        template<typename OpType>
        void ScalarBoolTransform<X, Z>::transform(void *vx,
                               Nd4jLong *xShapeInfo,
                               void *vz,
                               Nd4jLong *zShapeInfo,
                               void *vscalar,
                               void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<Z *>(vz);
            auto scalar = reinterpret_cast<X *>(vscalar)[0];
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            auto xEws = shape::elementWiseStride(xShapeInfo);
            auto zEws = shape::elementWiseStride(zShapeInfo);
            auto len = shape::length(xShapeInfo);

            // nd4j_logger("Launching scalar: xOrder: %i; zOrder: %i; xEWS: %i\n", xOrder, zOrder, xEws);

            nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

            if (kindOfLoop == nd4j::LoopKind::EWS1 || kindOfLoop == nd4j::LoopKind::EWSNONZERO) {
                transform<OpType>(x, xEws, z, zEws, vscalar, extraParams, len);
                return;
            }

            uint xShapeInfoCast[MAX_RANK];
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo<uint>(xShapeInfo, xShapeInfoCast);

            nd4j::OmpLaunchHelper info(len);

            if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {

                PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                {
                    auto threadNum = omp_get_thread_num();
                    auto threadOffset = info.getThreadOffset(threadNum);
                    auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                    PRAGMA_OMP_SIMD
                    for (unsigned int i = 0; i < ulen; i++) {
                        auto offset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, canCastX);
                        z[offset] = OpType::op(x[offset], scalar, extraParams);
                    }
                }
            }
            else {

                uint zShapeInfoCast[MAX_RANK];
                const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo, zShapeInfoCast);

                PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                {
                    auto threadNum = omp_get_thread_num();
                    auto threadOffset = info.getThreadOffset(threadNum);
                    auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                    PRAGMA_OMP_SIMD
                    for (unsigned int i = 0; i < ulen; i++) {
                        auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, canCastX);
                        auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, canCastZ);
                        z[zOffset] = OpType::op(x[xOffset], scalar, extraParams);
                    }
                }
            }
        }


            template<typename X, typename Z>
            template<typename OpType>
            void ScalarBoolTransform<X, Z>::transform(void *vx,
                    Nd4jLong xEws,
                    void *vz,
                    Nd4jLong zEws,
                    void *vscalar,
                    void *vextraParams,
                    const Nd4jLong len) {

                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<Z *>(vz);
                auto scalar = reinterpret_cast<X *>(vscalar)[0];
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                nd4j::OmpLaunchHelper info(len);

                if (xEws == 1 && zEws == 1) {

                    PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                    {
                        auto threadNum = omp_get_thread_num();
                        auto threadOffset = info.getThreadOffset(threadNum);
                        auto xi = x + threadOffset;
                        auto zi = z + threadOffset;
                        auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                        PRAGMA_OMP_SIMD
                        for (unsigned int i = 0; i < ulen; i++)
                            zi[i] = OpType::op(xi[i], scalar, extraParams);
                    }
                }
                else {

                    PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                    {
                        auto threadNum = omp_get_thread_num();
                        auto threadOffset = info.getThreadOffset(threadNum);
                        auto xi = x + xEws * threadOffset;
                        auto zi = z + zEws * threadOffset;
                        auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                        PRAGMA_OMP_SIMD
                        for (unsigned int i = 0; i < ulen; i++)
                            zi[i * zEws] = OpType::op(xi[i * xEws], scalar, extraParams);
                    }
                }
            }

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT ScalarBoolTransform, , LIBND4J_TYPES, BOOL_TYPES);

}
}
