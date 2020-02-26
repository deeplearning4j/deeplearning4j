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
#include <types/types.h>
#include <LoopKind.h>
#include <helpers/ConstantTadHelper.h>
#include <execution/Threads.h>
#include <helpers/ShapeUtils.h>

using namespace simdOps;

namespace functions {
    namespace broadcast {

        template <typename X, typename Y, typename Z>
        void Broadcast<X, Y, Z>::execInverse(const int opNum,
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
            DISPATCH_BY_OPNUM_TTT(execInverse, PARAMS(x,
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
                                               zTadOffset, start, stop), BROADCAST_OPS);
        }

        template <typename X, typename Y, typename Z>
        void Broadcast<X, Y, Z>::exec(const int opNum,
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
                             nd4j::LoopKind::Kind loopKind,
                             uint64_t start,
                             uint64_t stop) {
            DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x,
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
                                               zTadOffset, loopKind, start, stop), BROADCAST_OPS);
        }

        template <typename X, typename  Y, typename Z>
        template<typename OpType>
        void Broadcast<X, Y, Z>::exec(void *vx,
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
                             nd4j::LoopKind::Kind loopKind,
                             uint64_t start,
                             uint64_t stop) {

                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<Y *>(vy);
                auto z = reinterpret_cast<Z *>(vz);

                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                auto xTadShapeShapeInfo = xTadShapeInfo;
                auto tadOffsets = xTadOffset;

                if (xTadShapeInfo == nullptr || tadOffsets == nullptr) {
                    auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);

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

                auto xEws = shape::elementWiseStride(xTadShapeShapeInfo);
                auto yEws = shape::elementWiseStride(yShapeInfo);
                auto zEws = shape::elementWiseStride(zTadShapeInfo);


                const nd4j::LoopKind::Kind kindOfLoop =
                    (loopKind == nd4j::LoopKind::BROADCAST_SCALAR_X ||
                        loopKind == nd4j::LoopKind::BROADCAST_SCALAR_Y ||
                        loopKind == nd4j::LoopKind::BROADCAST_3D ||
                        loopKind == nd4j::LoopKind::BROADCAST_4D ||
                        loopKind == nd4j::LoopKind::BROADCAST_5D)
                    ? loopKind : nd4j::LoopKind::deduceKindOfLoopXYZ(xTadShapeShapeInfo, yShapeInfo, zTadShapeInfo);

                if (kindOfLoop == nd4j::LoopKind::EWS1) {
                    for (auto i = start; i < stop; i++) {
                        auto oX = x + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++)
                            oZ[f] = OpType::op(oX[f], y[f]);
                    }
                }
                else if(kindOfLoop == nd4j::LoopKind::EWSNONZERO){
                    for (auto i = start; i < stop; i++) {
                        auto oX = x + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++)
                            oZ[f * zEws] = OpType::op(oX[f * xEws], y[f * yEws]);
                    }
                } else if(kindOfLoop == nd4j::LoopKind::BROADCAST_SCALAR_X){
                    // this loop effectively turns broadcast into series of scalar ops
                    auto loopLength = yShapeInfo[shape::rank(yShapeInfo)];

                    for (auto i = start; i < stop; i++) {
                        auto oY = y + (i * loopLength);
                        auto oZ = z + (i * loopLength);

                        const auto oX = x[i];

                        PRAGMA_OMP_SIMD
                        for (Nd4jLong f = 0; f < loopLength; f++)
                            oZ[f] = OpType::op(oX, oY[f]);
                    }
                } else if(kindOfLoop == nd4j::LoopKind::BROADCAST_SCALAR_Y){
                    // this loop effectively turns broadcast into series of scalar ops
                    auto loopLength = xShapeInfo[shape::rank(xShapeInfo)];

                    for (auto i = start; i < stop; i++) {
                        auto oX = x + (i * loopLength);
                        auto oZ = z + (i * loopLength);

                        const auto oY = y[i];

                        PRAGMA_OMP_SIMD
                        for (Nd4jLong f = 0; f < loopLength; f++)
                            oZ[f] = OpType::op(oX[f], oY);
                    }
                }
                else if (kindOfLoop == nd4j::LoopKind::BROADCAST_3D) {

                    int xRank = shape::rank(xShapeInfo);
                    int yRank = shape::rank(yShapeInfo);

                    auto  xStrides = shape::stride(xShapeInfo);
                    auto  zStrides = shape::stride(zShapeInfo);

                    Nd4jLong  yStrides[3] = { 0,0,0 };
                    nd4j::ShapeUtils::copyCertainStridesFromShapeInfo(yShapeInfo, xRank, dimensionLength, dimension, yStrides);

                    uint64_t nSize1 = shape::sizeAt(zShapeInfo, 1);
                    uint64_t nSize2 = shape::sizeAt(zShapeInfo, 2);

                    for (auto index0 = start; index0 < stop; index0++) {

                        PRAGMA_OMP_SIMD
                            for (uint64_t index1 = 0; index1 < nSize1; index1++) {
                                for (uint64_t index2 = 0; index2 < nSize2; index2++) {
                                    auto rX = x + (xStrides[0] * index0 + xStrides[1] * index1 + xStrides[2] * index2);
                                    auto rY = y + (yStrides[0] * index0 + yStrides[1] * index1 + yStrides[2] * index2);
                                    auto rZ = z + (zStrides[0] * index0 + zStrides[1] * index1 + zStrides[2] * index2);
                                    *rZ = OpType::op(*rX, *rY);
                                }
                            }

                    }

                }
                else if (kindOfLoop == nd4j::LoopKind::BROADCAST_4D) {

                    int xRank = shape::rank(xShapeInfo);
                    int yRank = shape::rank(yShapeInfo);

                    auto  xStrides = shape::stride(xShapeInfo);
                    auto  zStrides = shape::stride(zShapeInfo);

                    Nd4jLong  yStrides[4] = { 0,0,0,0 };
                    nd4j::ShapeUtils::copyCertainStridesFromShapeInfo(yShapeInfo, xRank, dimensionLength, dimension, yStrides);

                    uint64_t nSize1 = shape::sizeAt(zShapeInfo, 1);
                    uint64_t nSize2 = shape::sizeAt(zShapeInfo, 2);
                    uint64_t nSize3 = shape::sizeAt(zShapeInfo, 3);

                    for (auto i = start; i < stop; i++) {

                        uint64_t index0 = i / nSize1;
                        uint64_t index1 = i % nSize1;

                        PRAGMA_OMP_SIMD
                            for (uint64_t index2 = 0; index2 < nSize2; index2++) {
                                for (uint64_t index3 = 0; index3 < nSize3; index3++) {
                                    auto rX = x + (xStrides[0] * index0 + xStrides[1] * index1 + xStrides[2] * index2 + xStrides[3] * index3);
                                    auto rY = y + (yStrides[0] * index0 + yStrides[1] * index1 + yStrides[2] * index2 + yStrides[3] * index3);
                                    auto rZ = z + (zStrides[0] * index0 + zStrides[1] * index1 + zStrides[2] * index2 + zStrides[3] * index3);
                                    *rZ = OpType::op(*rX, *rY);
                                }
                            }
                    }

                }
                else if (kindOfLoop == nd4j::LoopKind::BROADCAST_5D) {

                    int xRank = shape::rank(xShapeInfo);
                    int yRank = shape::rank(yShapeInfo);

                    auto  xStrides = shape::stride(xShapeInfo);
                    auto  zStrides = shape::stride(zShapeInfo);

                    Nd4jLong  yStrides[5] = { 0,0,0,0,0 };
                    nd4j::ShapeUtils::copyCertainStridesFromShapeInfo(yShapeInfo, xRank, dimensionLength, dimension, yStrides);

                    uint32_t nSize1 = shape::sizeAt(zShapeInfo, 1);
                    uint32_t nSize2 = shape::sizeAt(zShapeInfo, 2);
                    uint32_t nSize3 = shape::sizeAt(zShapeInfo, 3);
                    uint32_t nSize4 = shape::sizeAt(zShapeInfo, 4);

                    for (auto i = start; i < stop; i++) {

                        uint32_t index0 = i / nSize1;
                        uint32_t index1 = i % nSize1;

                        PRAGMA_OMP_SIMD
                            for (uint32_t index2 = 0; index2 < nSize2; index2++) {
                                for (uint32_t index3 = 0; index3 < nSize3; index3++) {
                                    for (uint32_t index4 = 0; index4 < nSize4; index4++) {
                                        auto rX = x + (xStrides[0] * index0 + xStrides[1] * index1 + xStrides[2] * index2 + xStrides[3] * index3 + xStrides[4] * index4);
                                        auto rY = y + (yStrides[0] * index0 + yStrides[1] * index1 + yStrides[2] * index2 + yStrides[3] * index3 + yStrides[4] * index4);
                                        auto rZ = z + (zStrides[0] * index0 + zStrides[1] * index1 + zStrides[2] * index2 + zStrides[3] * index3 + zStrides[4] * index4);

                                        *rZ = OpType::op(*rX, *rY);
                                    }
                                }
                            }
                    }

                }
                else if(shape::haveSameShapeAndStrides(xTadShapeShapeInfo, yShapeInfo) && shape::haveSameShapeAndStrides(xTadShapeShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);

                    for (auto i = start; i < stop; i++) {
                        auto oX = x + tadOffsets[i];
                        auto oZ = z + zTadOffset[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            oZ[offset] = OpType::op(oX[offset], y[offset]);
                        }
                    }
                }
                else if(shape::haveSameShapeAndStrides(xTadShapeShapeInfo, yShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);


                    for (auto i = start; i < stop; i++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                            oZ[zOffset] = OpType::op(oX[offset], y[offset]);
                        }
                    }
                }
                else if(shape::haveSameShapeAndStrides(xTadShapeShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    for (auto i = start; i < stop; i++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto yOffset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, canCastY);
                            oZ[offset] = OpType::op(oX[offset], y[yOffset]);
                        }
                    }
                }
                else if(shape::haveSameShapeAndStrides(yShapeInfo, zTadShapeInfo)) {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    for (auto i = start; i < stop; i++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto xOffset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto offset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, canCastY);
                            oZ[offset] = OpType::op(oX[xOffset], y[offset]);
                        }
                    }
                }
                else {
                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xTadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
                    bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);

                    for (auto i = start; i < stop; i++) {
                        auto oZ = z + zTadOffset[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto xOffset = shape::indexOffset(f, xTadShapeShapeInfo, tadShapeShapeInfoCast, canCastX);
                            auto yOffset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, canCastY);
                            auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                            oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
                        }
                    }
                }
        }



        template <typename X, typename  Y, typename Z>
        template<typename OpType>
        void Broadcast<X, Y, Z>::execInverse(void *vx,
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
            auto y = reinterpret_cast<Y *>(vy);
            auto z = reinterpret_cast<Z *>(vz);

            //decompose in to several sub tads after
            //moving all dimensions (in sorted order)
            //to the back.
            //permuted version of the x shape info for setting up the tad problem
            auto yTadShapeShapeInfo = yTadShapeInfo;
            auto tadOffsets = yTadOffset;

            if (yTadShapeInfo == nullptr || tadOffsets == nullptr) {
                auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dimension, dimensionLength);

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
            int threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
            threads = nd4j::math::nd4j_min<int>(threads, nd4j::Environment::getInstance()->maxThreads());

            auto yEws = shape::elementWiseStride(yTadShapeShapeInfo);
            auto xEws = shape::elementWiseStride(xShapeInfo);
            auto zEws = shape::elementWiseStride(zTadShapeInfo);

            const nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXYZ(yTadShapeShapeInfo, xShapeInfo, zTadShapeInfo);

            if(kindOfLoop == nd4j::LoopKind::EWS1) {
                for (auto i = start; i < stop; i++) {
                    auto oY = y + tadOffsets[i];
                    auto oZ = z + zTadOffset[i];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++)
                        oZ[f] = OpType::op(x[f], oY[f]);
                }
            }
            else if(kindOfLoop == nd4j::LoopKind::EWSNONZERO) {
                for (auto i = start; i < stop; i++) {
                    auto oY = y + tadOffsets[i];
                    auto oZ = z + zTadOffset[i];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++)
                        oZ[f * zEws] = OpType::op(x[f * xEws], oY[f * yEws]);
                };
            }
            else if(shape::haveSameShapeAndStrides(yTadShapeShapeInfo, xShapeInfo) && shape::haveSameShapeAndStrides(yTadShapeShapeInfo, zTadShapeInfo)) {
                uint tadShapeShapeInfoCast[MAX_RANK];
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);

                for (auto i = start; i < stop; i++) {
                    auto oY = x + tadOffsets[i];
                    auto oZ = z + zTadOffset[i];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++) {
                        auto offset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                        oZ[offset] = OpType::op(x[offset], oY[offset]);
                    }
                };
            }
            else if(shape::haveSameShapeAndStrides(yTadShapeShapeInfo, xShapeInfo)) {
                uint tadShapeShapeInfoCast[MAX_RANK];
                uint tadShapeInfoZCast[MAX_RANK];
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);
                bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);

                for (auto i = start; i < stop; i++) {
                    auto oZ = z + zTadOffset[i];
                    auto oY = y + tadOffsets[i];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++) {
                        auto offset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                        auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                        oZ[zOffset] = OpType::op(x[offset], oY[offset]);
                    }
                };
            }
            else if(shape::haveSameShapeAndStrides(yTadShapeShapeInfo, zTadShapeInfo)) {
                uint tadShapeShapeInfoCast[MAX_RANK];
                uint xShapeInfoCast[MAX_RANK];
                bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);

                for (auto i = start; i < stop; i++) {
                    auto oZ = z + zTadOffset[i];
                    auto oY = y + tadOffsets[i];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++) {
                        auto offset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                        auto xOffset = shape::indexOffset(f, yShapeInfo, xShapeInfoCast, canCastX);
                        oZ[offset] = OpType::op(x[xOffset], oY[offset]);
                    }
                };
            }
            else if(shape::haveSameShapeAndStrides(xShapeInfo, zTadShapeInfo)) {
                uint tadShapeShapeInfoCast[MAX_RANK];
                uint xShapeInfoCast[MAX_RANK];
                bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);

                for (auto i = start; i < stop; i++) {
                    auto oZ = z + zTadOffset[i];
                    auto oY = y + tadOffsets[i];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++) {
                        auto yOffset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                        auto offset = shape::indexOffset(f, xShapeInfo, xShapeInfoCast, canCastX);
                        oZ[offset] = OpType::op(x[offset], oY[yOffset]);
                    }
                };
            }
            else {
                uint tadShapeShapeInfoCast[MAX_RANK];
                uint tadShapeInfoZCast[MAX_RANK];
                uint xShapeInfoCast[MAX_RANK];
                bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yTadShapeShapeInfo, tadShapeShapeInfoCast);
                bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zTadShapeInfo, tadShapeInfoZCast);

                for (auto i = start; i < stop; i++) {
                    auto oZ = z + zTadOffset[i];
                    auto oY = y + tadOffsets[i];

                    PRAGMA_OMP_SIMD
                    for (unsigned int f = 0; f < tadLength; f++) {
                        auto xOffset = shape::indexOffset(f, xShapeInfo, xShapeInfoCast, canCastX);
                        auto yOffset = shape::indexOffset(f, yTadShapeShapeInfo, tadShapeShapeInfoCast, canCastY);
                        auto zOffset = shape::indexOffset(f, zTadShapeInfo, tadShapeInfoZCast, canCastZ);
                        oZ[zOffset] = OpType::op(x[xOffset], oY[yOffset]);
                    }
                };
            }
        }
    }
}