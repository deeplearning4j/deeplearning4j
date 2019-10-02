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
#include <LoopKind.h>
#include <execution/Threads.h>
#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
namespace scalar    {


////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vz, Nd4jLong *zShapeInfo,
                                                void *vscalars,
                                                int *dimension, int dimensionLength,
                                                Nd4jLong *xTadShapeInfo, Nd4jLong *xTadOffsets,
                                                Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {

    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto scalars = reinterpret_cast<Y *>(vscalars);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    if (zTadShapeInfo == nullptr) {
        zTadShapeInfo = xTadShapeInfo;
        zTadOffsets   = xTadOffsets;
    }

    const int xTadEws    = shape::elementWiseStride(xTadShapeInfo);
    const int zTadEws    = shape::elementWiseStride(zTadShapeInfo);
    const int tadLength  = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    const int numTads    = shape::length(xShapeInfo) / tadLength;

    nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXZ(xTadShapeInfo, zTadShapeInfo);

    if (kindOfLoop != nd4j::LoopKind::EWS1 && kindOfLoop != nd4j::LoopKind::EWSNONZERO) {
        printf("ScalarTransform<X, Z>::transform: super-bad loop visited. Shouldn't ever happen\n");
        return;
    }

    int num_threads = nd4j::math::nd4j_min<int>(numTads, omp_get_max_threads());

    if (kindOfLoop == nd4j::LoopKind::EWS1) {
        for (uint64_t r = 0; r < numTads; r++) {
            auto oZ = z + zTadOffsets[r];
            auto oX = x + xTadOffsets[r];

            PRAGMA_OMP_SIMD
            for (unsigned int f = 0; f < tadLength; f++)
                oZ[f] = OpType::op(oX[f], scalars[r], extraParams);
        };
    }
    else {
        for (uint64_t r = 0; r < numTads; r++) {
            auto oZ = z + zTadOffsets[r];
            auto oX = x + xTadOffsets[r];

            PRAGMA_OMP_SIMD
            for (unsigned int f = 0; f < tadLength; f++)
                oZ[f * zTadEws] = OpType::op(oX[f * xTadEws], scalars[r], extraParams);
        };
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X,Y,Z>::transform(int opNum,
                              void *x, Nd4jLong *xShapeInfo,
                              void *extraParams,
                              void *z, Nd4jLong *zShapeInfo,
                              void *scalars,
                              int *dimension, int dimensionLength,
                              Nd4jLong *xTadShapeInfo, Nd4jLong *xTadOffsets,
                              Nd4jLong *zTadShapeInfo, Nd4jLong *zTadOffsets) {

    DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, xTadShapeInfo, xTadOffsets, zTadShapeInfo, zTadOffsets), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(const int opNum,
                                        void *x, Nd4jLong xStride,
                                        void *z, Nd4jLong zStride,
                                        void *scalar,
                                        void *extraParams,
                                        const Nd4jLong n, bool allowParallelism) {

    DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xStride, z, zStride, scalar, extraParams, n, allowParallelism), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void ScalarTransform<X, Y, Z>::transform(const int opNum,
                                        void *x, Nd4jLong *xShapeInfo,
                                        void *z, Nd4jLong *zShapeInfo,
                                        void *scalar,
                                        void *extraParams, bool allowParallelism) {

    DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams, allowParallelism), SCALAR_OPS);
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong *xShapeInfo,
                                        void *vz, Nd4jLong *zShapeInfo,
                                        void *vscalar,
                                        void *vextraParams, bool allowParallelism) {

    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto scalar = reinterpret_cast<Y *>(vscalar)[0];
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    const auto len = shape::length(xShapeInfo);
    const auto xEws = shape::elementWiseStride(xShapeInfo);
    const auto zEws = shape::elementWiseStride(zShapeInfo);

    nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

    if (kindOfLoop == nd4j::LoopKind::EWS1 || kindOfLoop == nd4j::LoopKind::EWSNONZERO) {
        transform<OpType>(x, xEws, z, zEws, vscalar, extraParams, len, allowParallelism);
    }
    else {

        uint xShapeInfoCast[MAX_RANK];
        const bool canCastX = nd4j::DataTypeUtils::castShapeInfo<uint>(xShapeInfo, xShapeInfoCast);

        nd4j::OmpLaunchHelper info(len, allowParallelism ? -1 : 1);

        if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
            PRAGMA_OMP_SIMD
            for (uint64_t i = 0; i < len; i++) {
                auto offset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                z[offset] = OpType::op(x[offset], scalar, extraParams);
            };
        }
        else {
            uint zShapeInfoCast[MAX_RANK];
            const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo, zShapeInfoCast);

            PRAGMA_OMP_SIMD
            for (uint64_t i = 0; i < len; i++) {
                auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
                z[zOffset] = OpType::op(x[xOffset], scalar, extraParams);
            };
        }
    }
}

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong xEws,
                                        void *vz, Nd4jLong zEws,
                                        void *vscalar,
                                        void *vextraParams,
                                        const Nd4jLong len, bool allowParallelism) {

    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto scalar = reinterpret_cast<Y *>(vscalar)[0];
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    nd4j::OmpLaunchHelper info(len, allowParallelism ? -1 : 1);

    if (xEws == 1 && zEws == 1) {
        PRAGMA_OMP_SIMD
        for (uint64_t i = 0; i < len; i++)
            z[i] = OpType::op(x[i], scalar, extraParams);
    }
    else {
        PRAGMA_OMP_SIMD
        for (uint64_t i = 0; i < len; i++)
            z[i * zEws] = OpType::op(x[i * xEws], scalar, extraParams);
    }
}



}
}
