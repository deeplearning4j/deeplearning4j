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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 14.03.2019
//

#include <helpers/Loops.h>

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E>
template <typename OpType>
void nd4j::TransformLoops<X,Z,E>::loopXZ(const X* x, const Nd4jLong* xShapeInfo,
                   Z* z, const Nd4jLong* zShapeInfo,
                   E* extraParams) {

    const LoopKind kindOfLoop = deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

    const Nd4jLong* xShape  = shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo));
    const Nd4jLong* xStride = shape::stride(const_cast<Nd4jLong*>(xShapeInfo));
    const Nd4jLong* zStride = shape::stride(const_cast<Nd4jLong*>(zShapeInfo));

    const Nd4jLong len = shape::length(xShapeInfo);

    OmpLaunchHelper thredsInfo(len);

    switch (kindOfLoop) {

        //*********************************************//
        case EWS1: {
            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                const auto xi = x + threadOffset;
                const auto zi = z + threadOffset;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i] = OpType::op(xi[i], extraParams);
            }
        }
            break;

            //*********************************************//
        case EWSNONZERO: {
            const uint xEws = shape::elementWiseStride(xShapeInfo);
            const uint zEws = shape::elementWiseStride(zShapeInfo);

            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                const auto xi = x + threadOffset * xEws;
                auto zi = z + threadOffset * zEws;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i*zEws] = OpType::op(xi[i*xEws], extraParams);
            }
        }
            break;

            //*********************************************//
        case Z_EWSNONZERO: {
            const uint zEws = shape::elementWiseStride(zShapeInfo);
            uint castXShapeInfo[MAX_RANK];
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo<uint>(xShapeInfo, castXShapeInfo);

            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                auto zi = z + threadOffset * zEws;

                if (zEws > 1) {

                    PRAGMA_OMP_SIMD
                    for (uint i = 0; i < lenPerThread; i++) {
                        const auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, castXShapeInfo, len, canCastX);
                        zi[i * zEws] = OpType::op(x[xOffset], extraParams);
                    }
                } else {
                    PRAGMA_OMP_SIMD
                    for (uint i = 0; i < lenPerThread; i++) {
                        const auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, castXShapeInfo, len, canCastX);
                        zi[i] = OpType::op(x[xOffset], extraParams);
                    }
                }
            }
        }
            break;

            //*********************************************//
        case RANK1: {
            PRAGMA_OMP_PARALLEL_FOR
            for (uint i0 = 0; i0 < len; ++i0)
                z[i0 * zStride[0]] = OpType::op(x[i0 * xStride[0]], extraParams);
        }
            break;

            //*********************************************//
        case RANK2: {
            auto uXShape0 = static_cast<uint>(xShape[0]);
            auto uXShape1 = static_cast<uint>(xShape[1]);

            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (uint i0 = 0; i0 < uXShape0; ++i0) {

                auto z0 = i0 * zStride[0];
                auto x0 = i0 * xStride[0];
                for (uint i1 = 0; i1 < uXShape1; ++i1)
                    z[z0 + i1 * zStride[1]] = OpType::op(x[x0 + i1 * xStride[1]], extraParams);
            }
        }
            break;

            //*********************************************//
        case RANK3: {
            auto uXShape0 = static_cast<uint>(xShape[0]);
            auto uXShape1 = static_cast<uint>(xShape[1]);
            auto uXShape2 = static_cast<uint>(xShape[2]);

            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
            for (uint i0 = 0; i0 < uXShape0; ++i0)
                for (uint i1 = 0; i1 < uXShape1; ++i1) {

                    auto z0 = i0 * zStride[0] + i1 * zStride[1];
                    auto x0 = i0 * xStride[0] + i1 * xStride[1];

                    for (uint i2 = 0; i2 < uXShape2; ++i2)
                        z[z0 + i2 * zStride[2]] = OpType::op(x[x0 + i2 * xStride[2]], extraParams);
                }
        }
            break;

            //*********************************************//
        case RANK4: {
            auto uXShape0 = static_cast<uint>(xShape[0]);
            auto uXShape1 = static_cast<uint>(xShape[1]);
            auto uXShape2 = static_cast<uint>(xShape[2]);
            auto uXShape3 = static_cast<uint>(xShape[3]);

            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
            for (uint i0 = 0; i0 < uXShape0; ++i0)
                for (uint i1 = 0; i1 < uXShape1; ++i1)
                    for (uint i2 = 0; i2 < uXShape2; ++i2) {

                        auto x0 = i0 * xStride[0] + i1 * xStride[1] + i2 * xStride[2];
                        auto z0 = i0 * zStride[0] + i1 * zStride[1] + i2 * zStride[2];

                        for (uint i3 = 0; i3 < uXShape3; ++i3)
                            z[z0 + i3 * zStride[3]] = OpType::op(x[x0 + i3 * xStride[3]], extraParams);
                    }
        }
            break;

            //*********************************************//
        case RANK5: {
            auto uXShape0 = static_cast<uint>(xShape[0]);
            auto uXShape1 = static_cast<uint>(xShape[1]);
            auto uXShape2 = static_cast<uint>(xShape[2]);
            auto uXShape3 = static_cast<uint>(xShape[3]);
            auto uXShape4 = static_cast<uint>(xShape[4]);

            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
            for (uint i0 = 0; i0 < uXShape0; ++i0)
                for (uint i1 = 0; i1 < uXShape1; ++i1)
                    for (uint i2 = 0; i2 < uXShape2; ++i2) {

                        auto z0 = i0 * zStride[0] + i1 * zStride[1] + i2 * zStride[2];
                        auto x0 = i0 * xStride[0] + i1 * xStride[1] + i2 * xStride[2];

                        for (uint i3 = 0; i3 < uXShape3; ++i3) {

                            auto z1 = z0 + i3 * zStride[3];
                            auto x1 = x0 + i3 * xStride[3];

                            for (uint i4 = 0; i4 < uXShape4; ++i4)
                                z[z1 + i4 * zStride[4]] = OpType::op(x[x1 + i4 * xStride[4]], extraParams);

                        }
                    }
        }
            break;

            //*********************************************//
        default: {
            uint xShapeInfoCast[MAX_RANK];
            uint zShapeInfoCast[MAX_RANK];

            bool canCastX = DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
            bool canCastZ = DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                auto threadNum = omp_get_thread_num();
                auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++) {
                    auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);
                    auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, len, canCastZ);
                    z[zOffset] = OpType::op(x[xOffset], extraParams);
                }
            }
        }
    }
}