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

#include <system/op_boilerplate.h>
#include <loops/broadcasting_int.h>
#include <loops/legacy_ops.h>
#include <types/types.h>
#include <helpers/LoopKind.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>

using namespace simdOps;

namespace functions {
    namespace broadcast {

        template <typename X>
        void BroadcastInt<X>::exec(const int opNum,
                             void *x,
                             Nd4jLong *xShapeInfo,
                             void *y,
                             Nd4jLong *yShapeInfo,
                             void *z,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *xTadShapeInfo,
                             Nd4jLong *xTadOffset,
                             Nd4jLong *zTadShapeInfo,
                             Nd4jLong *zTadOffset,
                             uint64_t start,
                             uint64_t stop) {
            DISPATCH_BY_OPNUM_T(exec, PARAMS(x,
                                               xShapeInfo,
                                               y,
                                               yShapeInfo,
                                               z,
                                               zShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               xTadShapeInfo,
                                               xTadOffset,
                                               zTadShapeInfo,
                                               zTadOffset, start, stop), BROADCAST_INT_OPS);
        }

        template <typename X>
        void BroadcastInt<X>::exec(const int opNum,
                             const void *x, const Nd4jLong *xShapeInfo,
                             const void *y, const Nd4jLong *yShapeInfo,
                                   void *z, const Nd4jLong *zShapeInfo) {

            DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo), BROADCAST_INT_OPS);
        }

        template <typename X>
        void BroadcastInt<X>::execInverse(const int opNum,
                             void *x,
                             Nd4jLong *xShapeInfo,
                             void *y,
                             Nd4jLong *yShapeInfo,
                             void *z,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *xTadShapeInfo,
                             Nd4jLong *xTadOffset,
                             Nd4jLong *zTadShapeInfo,
                             Nd4jLong *zTadOffset,
                             uint64_t start,
                             uint64_t stop) {
            DISPATCH_BY_OPNUM_T(execInverse, PARAMS(x,
                                               xShapeInfo,
                                               y,
                                               yShapeInfo,
                                               z,
                                               zShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               xTadShapeInfo,
                                               xTadOffset,
                                               zTadShapeInfo,
                                               zTadOffset, start, stop), BROADCAST_INT_OPS);
        }

        template <typename X>
        template<typename OpType>
        void BroadcastInt<X>::exec(void *vx,
                             Nd4jLong *xShapeInfo,
                             void *vy,
                             Nd4jLong *yShapeInfo,
                             void *vz,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *xTadShapeInfo,
                             Nd4jLong *xTadOffset,
                             Nd4jLong *zTadShapeInfo,
                             Nd4jLong *zTadOffset,
                             uint64_t start,
                             uint64_t stop) {

                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<X *>(vy);
                auto z = reinterpret_cast<X *>(vz);

                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                auto xTadShapeShapeInfo = xTadShapeInfo;
                auto tadOffsets = xTadOffset;

                if (xTadShapeInfo == nullptr || tadOffsets == nullptr) {
                    auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);

                    xTadShapeShapeInfo = tadPack.primaryShapeInfo();
                    tadOffsets = tadPack.primaryOffsets();
                }

                //int *resultStride = shape::stride(xTadShapeShapeInfo);
                unsigned int tadLength = shape::length(xTadShapeShapeInfo);//shape::length(xTadShapeShapeInfo);
                unsigned int tads = shape::length(xShapeInfo) / tadLength;

                if (zTadShapeInfo == nullptr) {
                    zTadShapeInfo = xTadShapeShapeInfo;
                    zTadOffset = tadOffsets;
                }

                auto lenZ = shape::length(zTadShapeInfo);
                auto lenY = shape::length(yShapeInfo);

                int tadsPerThread = tads / TAD_THRESHOLD;
                int threads = sd::math::nd4j_max<int>(1, tadsPerThread);
                threads = sd::math::nd4j_min<int>(threads, sd::Environment::getInstance()->maxThreads());

                auto xEws = shape::elementWiseStride(xTadShapeShapeInfo);
                auto yEws = shape::elementWiseStride(yShapeInfo);
                auto zEws = shape::elementWiseStride(zTadShapeInfo);

                const sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXYZ(xTadShapeShapeInfo, yShapeInfo, zTadShapeInfo);

                if (kindOfLoop == sd::LoopKind::EWS1) {
                    for (auto i = start; i < stop; i ++) {
                        auto oX = x + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++)
                            oZ[f] = OpType::op(oX[f], y[f]);
                    };
                }
                else if(kindOfLoop == sd::LoopKind::EWSNONZERO) {
                    for (auto i = start; i < stop; i ++) {
                        auto oX = x + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++)
                            oZ[f * zEws] = OpType::op(oX[f * xEws], y[f * yEws]);
                    };
                }
                else if(shape::haveSameShapeAndStrides(xTadShapeShapeInfo, yShapeInfo) && shape::haveSameShapeAndStrides(xTadShapeShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            oZ[offset] = OpType::op(oX[offset], y[offset]);
                        }
                    };
                }
                else if(shape::haveSameShapeAndStrides(xTadShapeShapeInfo, yShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastZ = sd::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                            oZ[zOffset] = OpType::op(oX[offset], y[offset]);
                        }
                    };
                }
                else if(shape::haveSameShapeAndStrides(xTadShapeShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto yOffset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, canCastY);
                            oZ[offset] = OpType::op(oX[offset], y[yOffset]);
                        }
                    };
                }
                else if(shape::haveSameShapeAndStrides(yShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto xOffset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto offset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, canCastY);
                            oZ[offset] = OpType::op(oX[xOffset], y[offset]);
                        }
                    };
                }
                else {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
                    bool canCastZ = sd::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto xOffset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto yOffset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, canCastY);
                            auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                            oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
                        }
                    };
                }
        }


        template <typename X>
        template<typename OpType>
        void BroadcastInt<X>::execInverse(void *vx,
                             Nd4jLong *xShapeInfo,
                             void *vy,
                             Nd4jLong *yShapeInfo,
                             void *vz,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *yTadShapeInfo,
                             Nd4jLong *yTadOffset,
                             Nd4jLong *zTadShapeInfo,
                             Nd4jLong *zTadOffset,
                             uint64_t start,
                             uint64_t stop) {

                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<X *>(vy);
                auto z = reinterpret_cast<X *>(vz);

                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                auto yTadShapeShapeInfo = yTadShapeInfo;
                auto tadOffsets = yTadOffset;

                if (yTadShapeInfo == nullptr || tadOffsets == nullptr) {
                    auto tadPack = sd::ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dimension, dimensionLength);

                    yTadShapeShapeInfo = tadPack.primaryShapeInfo();
                    tadOffsets = tadPack.primaryOffsets();
                }

                //int *resultStride = shape::stride(yTadShapeShapeInfo);
                unsigned int tadLength = shape::length(yTadShapeShapeInfo);
                unsigned int tads = shape::length(yShapeInfo) / tadLength;

                if (zTadShapeInfo == nullptr) {
                    zTadShapeInfo = yTadShapeShapeInfo;
                    zTadOffset = tadOffsets;
                }

                auto lenZ = shape::length(zTadShapeInfo);
                auto lenX = shape::length(xShapeInfo);

                int tadsPerThread = tads / TAD_THRESHOLD;
                int threads = sd::math::nd4j_max<int>(1, tadsPerThread);
                threads = sd::math::nd4j_min<int>(threads, sd::Environment::getInstance()->maxThreads());

                auto yEws = shape::elementWiseStride(yTadShapeShapeInfo);
                auto xEws = shape::elementWiseStride(xShapeInfo);
                auto zEws = shape::elementWiseStride(zTadShapeInfo);

                const sd::LoopKind::Kind kindOfLoop = sd::LoopKind::deduceKindOfLoopXYZ(yTadShapeShapeInfo, xShapeInfo, zTadShapeInfo);

                if (kindOfLoop == sd::LoopKind::EWS1) {
                    for (auto i = start; i < stop; i ++) {
                        auto oY = y + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++)
                            oZ[f] = OpType::op(x[f], oY[f]);
                    };
                }
                else if(kindOfLoop == sd::LoopKind::EWSNONZERO) {
                    for (auto i = start; i < stop; i ++) {
                        auto oY = y + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (uint f = 0; f < tadLength; f++)
                            oZ[f * zEws] = OpType::op(x[f * xEws], oY[f * yEws]);
                    };
                }
                else if(shape::haveSameShapeAndStrides(yTadShapeShapeInfo, xShapeInfo) && shape::haveSameShapeAndStrides(yTadShapeShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oY = y + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (uint f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                            oZ[offset] = OpType::op(x[offset], oY[offset]);
                        }
                    };
                }
                else if(shape::haveSameShapeAndStrides(yTadShapeShapeInfo, xShapeInfo)) {

                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastZ = sd::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oY = y + tadOffsets[i];

                        for (uint f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                            auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                            oZ[zOffset] = OpType::op(x[offset], oY[offset]);
                        }
                    };
                }
                else if(shape::haveSameShapeAndStrides(yTadShapeShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint xShapeInfoCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oY = y + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (uint f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                            auto xOffset = shape::indexOffset(f, xShapeInfo, xShapeInfoCast, canCastX);
                            oZ[offset] = OpType::op(x[xOffset], oY[offset]);
                        }
                    };
                }
                else if(shape::haveSameShapeAndStrides(xShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint xShapeInfoCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oY = y + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (uint f = 0; f < tadLength; f++) {
                            auto yOffset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                            auto offset = shape::indexOffset(f, xShapeInfo, xShapeInfoCast, canCastX);
                            oZ[offset] = OpType::op(x[offset], oY[yOffset]);
                        }
                    };
                }
                else {
                    uint xShapeInfoCast[MAX_RANK];
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    bool canCastX = sd::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    bool canCastY = sd::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastZ = sd::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);

                    for (auto i = start; i < stop; i ++) {
                        auto oZ = z + zTadOffset[i];
                        auto oY = y + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (uint f = 0; f < tadLength; f++) {
                            auto xOffset = shape::indexOffset(f, xShapeInfo, xShapeInfoCast, canCastX);
                            auto yOffset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                            auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                            oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset]);
                        }
                    };
                }
        }


////////////////////////////////////////////////////////////////////////
template <typename X>
template<typename OpType>
void BroadcastInt<X>::exec(const void *vx, const Nd4jLong *xShapeInfo,
                                 const void *vy, const Nd4jLong *yShapeInfo,
                                       void *vz, const Nd4jLong *zShapeInfo) {

    const X* x = reinterpret_cast<const X*>(vx);
    const X* y = reinterpret_cast<const X*>(vy);
          X* z = reinterpret_cast<X*>(vz);

    const int rank   = shape::rank(zShapeInfo);    // xRank = yRank = zRank
    const char zOrder = shape::order(zShapeInfo);

    uint xAxis0 = shape::sizeAt(xShapeInfo, zOrder == 'c' ? 0 : rank-1);
    uint xAxis1 = shape::sizeAt(xShapeInfo, zOrder == 'c' ? 1 : rank-2);
    uint xAxis2 = rank > 2 ? shape::sizeAt(xShapeInfo, zOrder == 'c' ? 2 : rank - 3) : 0;
    uint xAxis3 = rank > 3 ? shape::sizeAt(xShapeInfo, zOrder == 'c' ? 3 : rank - 4) : 0;
    uint xAxis4 = rank > 4 ? shape::sizeAt(xShapeInfo, zOrder == 'c' ? 4 : rank - 5) : 0;
    Nd4jLong xStrd0 = shape::strideAt(xShapeInfo, zOrder == 'c' ? 0 : rank-1);
    Nd4jLong xStrd1 = shape::strideAt(xShapeInfo, zOrder == 'c' ? 1 : rank-2);
    Nd4jLong xStrd2 = rank > 2 ? shape::strideAt(xShapeInfo, zOrder == 'c' ? 2 : rank - 3) : 0;
    Nd4jLong xStrd3 = rank > 3 ? shape::strideAt(xShapeInfo, zOrder == 'c' ? 3 : rank - 4) : 0;
    Nd4jLong xStrd4 = rank > 4 ? shape::strideAt(xShapeInfo, zOrder == 'c' ? 4 : rank - 5) : 0;

    uint yAxis0 = shape::sizeAt(yShapeInfo, zOrder == 'c' ? 0 : rank-1);
    uint yAxis1 = shape::sizeAt(yShapeInfo, zOrder == 'c' ? 1 : rank-2);
    uint yAxis2 = rank > 2 ? shape::sizeAt(yShapeInfo, zOrder == 'c' ? 2 : rank - 3) : 0;
    uint yAxis3 = rank > 3 ? shape::sizeAt(yShapeInfo, zOrder == 'c' ? 3 : rank - 4) : 0;
    uint yAxis4 = rank > 4 ? shape::sizeAt(yShapeInfo, zOrder == 'c' ? 4 : rank - 5) : 0;
    Nd4jLong yStrd0 = shape::strideAt(yShapeInfo, zOrder == 'c' ? 0 : rank-1);
    Nd4jLong yStrd1 = shape::strideAt(yShapeInfo, zOrder == 'c' ? 1 : rank-2);
    Nd4jLong yStrd2 = rank > 2 ? shape::strideAt(yShapeInfo, zOrder == 'c' ? 2 : rank - 3) : 0;
    Nd4jLong yStrd3 = rank > 3 ? shape::strideAt(yShapeInfo, zOrder == 'c' ? 3 : rank - 4) : 0;
    Nd4jLong yStrd4 = rank > 4 ? shape::strideAt(yShapeInfo, zOrder == 'c' ? 4 : rank - 5) : 0;

    uint zAxis0 = shape::sizeAt(zShapeInfo, zOrder == 'c' ? 0 : rank-1);
    uint zAxis1 = shape::sizeAt(zShapeInfo, zOrder == 'c' ? 1 : rank-2);
    uint zAxis2 = rank > 2 ? shape::sizeAt(zShapeInfo, zOrder == 'c' ? 2 : rank - 3) : 0;
    uint zAxis3 = rank > 3 ? shape::sizeAt(zShapeInfo, zOrder == 'c' ? 3 : rank - 4) : 0;
    uint zAxis4 = rank > 4 ? shape::sizeAt(zShapeInfo, zOrder == 'c' ? 4 : rank - 5) : 0;
    Nd4jLong zStrd0 = shape::strideAt(zShapeInfo, zOrder == 'c' ? 0 : rank-1);
    Nd4jLong zStrd1 = shape::strideAt(zShapeInfo, zOrder == 'c' ? 1 : rank-2);
    Nd4jLong zStrd2 = rank > 2 ? shape::strideAt(zShapeInfo, zOrder == 'c' ? 2 : rank - 3) : 0;
    Nd4jLong zStrd3 = rank > 3 ? shape::strideAt(zShapeInfo, zOrder == 'c' ? 3 : rank - 4) : 0;
    Nd4jLong zStrd4 = rank > 4 ? shape::strideAt(zShapeInfo, zOrder == 'c' ? 4 : rank - 5) : 0;

    switch (rank) {

        case 1: {

            auto func = PRAGMA_THREADS_FOR{

                if(zStrd0 == 1 && xStrd0 <= 1 && yStrd0 <= 1)
                    for (auto i0 = start; i0 < stop; ++i0)
                        z[i0] = OpType::op(x[xStrd0 ? i0 : 0], y[yStrd0 ? i0 : 0]);
                else
                    for (auto i0 = start; i0 < stop; ++i0)
                        z[i0 * zStrd0] = OpType::op(x[i0 * xStrd0], y[i0 * yStrd0]);
            };
            samediff::Threads::parallel_tad(func, 0, zAxis0);
        }
        break;

        case 2: {

            auto func = PRAGMA_THREADS_FOR{

                for (auto i0 = start; i0 < stop; ++i0) {

                    auto x0 = x + i0 * xStrd0;
                    auto y0 = y + i0 * yStrd0;
                    auto z0 = z + i0 * zStrd0;

                    if(zStrd1 == 1 && xStrd1 <= 1 && yStrd1 <= 1)
                        for (uint i1 = 0; i1 < zAxis1; ++i1)
                            z0[i1] = OpType::op(x0[xStrd1 ? i1 : 0], y0[yStrd1 ? i1 : 0]);
                    else
                        for (uint i1 = 0; i1 < zAxis1; ++i1)
                            z0[i1 * zStrd1] = OpType::op(x0[i1 * xStrd1], y0[i1 * yStrd1]);
                }
            };
            samediff::Threads::parallel_tad(func, 0, zAxis0);
        }
        break;

        case 3: {


            auto func = PRAGMA_THREADS_FOR_2D {

                for (auto i0 = start_x; i0 < stop_x; ++i0) {
                    for (auto i1 = start_y; i1 < stop_y; ++i1) {

                        auto x1 = x + i0 * xStrd0 + i1 * xStrd1;
                        auto y1 = y + i0 * yStrd0 + i1 * yStrd1;
                        auto z1 = z + i0 * zStrd0 + i1 * zStrd1;

                        if(zStrd2 == 1 && xStrd2 <= 1 && yStrd2 <= 1)
                            for (uint i2 = 0; i2 < zAxis2; ++i2)
                                z1[i2] = OpType::op(x1[xStrd2 ? i2 : 0], y1[yStrd2 ? i2 : 0]);
                        else
                            for (uint i2 = 0; i2 < zAxis2; ++i2)
                                z1[i2 * zStrd2] = OpType::op(x1[i2 * xStrd2], y1[i2 * yStrd2]);
                    }
                }
            };
            samediff::Threads::parallel_for(func, 0,zAxis0,1,  0,zAxis1,1);
        }
        break;

        case 4: {

            auto func = PRAGMA_THREADS_FOR_3D {

                for (auto i0 = start_x; i0 < stop_x; ++i0) {
                    for (auto i1 = start_y; i1 < stop_y; ++i1) {
                        for (auto i2 = start_z; i2 < stop_z; ++i2) {

                            auto x2 = x + i0 * xStrd0 + i1 * xStrd1 + i2 * xStrd2;
                            auto y2 = y + i0 * yStrd0 + i1 * yStrd1 + i2 * yStrd2;
                            auto z2 = z + i0 * zStrd0 + i1 * zStrd1 + i2 * zStrd2;

                            if(zStrd3 == 1 && xStrd3 <= 1 && yStrd3 <= 1)
                                for (uint i3 = 0; i3 < zAxis3; ++i3)
                                    z2[i3] = OpType::op(x2[xStrd3 ? i3 : 0], y2[yStrd3 ? i3 : 0]);
                            else
                                for (uint i3 = 0; i3 < zAxis3; ++i3)
                                    z2[i3 * zStrd3] = OpType::op(x2[i3 * xStrd3], y2[i3 * yStrd3]);
                        }
                    }
                }
            };
            samediff::Threads::parallel_for(func,  0,zAxis0,1,  0,zAxis1,1,  0,zAxis2,1);
        }
        break;

        case 5: {

            auto func = PRAGMA_THREADS_FOR_3D {

                for (auto i0 = start_x; i0 < stop_x; ++i0) {
                    for (auto i1 = start_y; i1 < stop_y; ++i1) {
                        for (auto i2 = start_z; i2 < stop_z; ++i2) {
                            for (uint i3 = 0; i3 < zAxis3; ++i3) {

                                auto x3 = x + i0 * xStrd0 + i1 * xStrd1 + i2 * xStrd2 + i3 * xStrd3;
                                auto y3 = y + i0 * yStrd0 + i1 * yStrd1 + i2 * yStrd2 + i3 * yStrd3;
                                auto z3 = z + i0 * zStrd0 + i1 * zStrd1 + i2 * zStrd2 + i3 * zStrd3;

                                if(zStrd4 == 1 && xStrd4 <= 1 && yStrd4 <= 1)
                                    for (uint i4 = 0; i4 < zAxis4; ++i4)
                                        z3[i4] = OpType::op(x3[xStrd4 ? i4 : 0], y3[yStrd4 ? i4 : 0]);
                                else
                                    for (uint i4 = 0; i4 < zAxis4; ++i4)
                                        z3[i4 * zStrd4] = OpType::op(x3[i4 * xStrd4], y3[i4 * yStrd4]);
                            }
                        }
                    }
                }
            };
            samediff::Threads::parallel_for(func,  0,zAxis0,1,  0,zAxis1,1,  0,zAxis2,1);
        }
        break;

        default: {

            auto func = PRAGMA_THREADS_FOR{

                Nd4jLong xCoords[MAX_RANK], yCoords[MAX_RANK], zCoords[MAX_RANK];

                for (auto i = start; i < stop; ++i) {

                    shape::index2coords(i, zShapeInfo, zCoords);

                    for (uint j = 0; j < rank; ++j) {
                        xCoords[j] = shape::sizeAt(xShapeInfo, j) == 1 ? 0 : zCoords[j];
                        yCoords[j] = shape::sizeAt(yShapeInfo, j) == 1 ? 0 : zCoords[j];
                    }

                    const auto xOffset = shape::getOffset(xShapeInfo, xCoords);
                    const auto yOffset = shape::getOffset(yShapeInfo, yCoords);
                    const auto zOffset = shape::getOffset(zShapeInfo, zCoords);

                    z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                }
            };

            samediff::Threads::parallel_for(func, 0, shape::length(zShapeInfo));
        }
    }
}

//BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT BroadcastInt, , INTEGER_TYPES);
}
}