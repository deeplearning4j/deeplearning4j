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

using namespace simdOps;

namespace functions {
    namespace broadcast {

        template <typename X, typename Y, typename Z>
        void Broadcast<X, Y, Z>::exec(const int opNum,
                             void *x,
                             Nd4jLong *xShapeInfo,
                             void *y,
                             Nd4jLong *yShapeInfo,
                             void *z,
                             Nd4jLong *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset,
                             Nd4jLong *tadShapeInfoZ,
                             Nd4jLong *tadOffsetZ) {
            DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x,
                                               xShapeInfo,
                                               y,
                                               yShapeInfo,
                                               z,
                                               resultShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset,
                                               tadShapeInfoZ,
                                               tadOffsetZ), BROADCAST_OPS);
        }

        template <typename X, typename  Y, typename Z>
        template<typename OpType>
        void Broadcast<X, Y, Z>::exec(void *vx,
                             Nd4jLong *xShapeInfo,
                             void *vy,
                             Nd4jLong *yShapeInfo,
                             void *vz,
                             Nd4jLong *resultShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset,
                             Nd4jLong *tadShapeInfoZ,
                             Nd4jLong *tadOffsetZ) {

                auto x = reinterpret_cast<X *>(vx);
                auto y = reinterpret_cast<Y *>(vy);
                auto z = reinterpret_cast<Z *>(vz);

                //decompose in to several sub tads after
                //moving all dimensions (in sorted order)
                //to the back.
                //permuted version of the x shape info for setting up the tad problem
                auto tadShapeShapeInfo = tadShapeInfo;
                auto tadOffsets = tadOffset;
                shape::TAD *tad = nullptr;

                if (tadShapeInfo == nullptr || tadOffsets == nullptr) {
                    tad = new shape::TAD();
                    tad->init(xShapeInfo, dimension, dimensionLength);
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    tadShapeShapeInfo = tad->tadOnlyShapeInfo;
                    tadOffsets = tad->tadOffsets;
                }

                //int *resultStride = shape::stride(tadShapeShapeInfo);                
                unsigned int tadLength = shape::length(tadShapeShapeInfo);
                unsigned int tads = shape::length(xShapeInfo) / tadLength;

                if (tadShapeInfoZ == nullptr) {
                    tadShapeInfoZ = tadShapeShapeInfo;
                    tadOffsetZ = tadOffsets;
                }

                auto lenZ = shape::length(tadShapeInfoZ);
                auto lenY = shape::length(yShapeInfo);

                int tadsPerThread = tads / TAD_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

                if (shape::order(tadShapeShapeInfo) == shape::order(yShapeInfo) && shape::order(tadShapeInfoZ) == shape::order(yShapeInfo)) {
                    auto xEws = shape::elementWiseStride(tadShapeShapeInfo);
                    auto yEws = shape::elementWiseStride(yShapeInfo);
                    auto zEws = shape::elementWiseStride(tadShapeInfoZ);

                    if (xEws == 1 && yEws == 1 && zEws == 1) {
                        PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                        for (int i = 0; i < tads; i++) {
                            auto oX = x + tadOffsets[i];
                            auto oZ = z + tadOffsetZ[i];

                            PRAGMA_OMP_SIMD
                            for (unsigned int f = 0; f < tadLength; f++)
                                oZ[f] = OpType::op(oX[f], y[f]);
                        }
                    } else {
                        PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                        for (int i = 0; i < tads; i++) {
                            auto oX = x + tadOffsets[i];
                            auto oZ = z + tadOffsetZ[i];

                            PRAGMA_OMP_SIMD
                            for (unsigned int f = 0; f < tadLength; f++)
                                oZ[f * zEws] = OpType::op(oX[f * xEws], y[f * yEws]);
                        }
                    }
                } else if(shape::haveSameOffsets(tadShapeShapeInfo, yShapeInfo) && shape::haveSameOffsets(tadShapeShapeInfo, tadShapeInfoZ)) {

                    uint tadShapeShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadShapeShapeInfo, tadShapeShapeInfoCast);

                    PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                    for (int i = 0; i < tads; i++) {
                        
                        auto oX = x + tadOffsets[i];
                        auto oZ = z + tadOffsetZ[i];

                        PRAGMA_OMP_SIMD
                        for (unsigned int f = 0; f < tadLength; f++) {
                            auto offset = shape::indexOffset(f, tadShapeShapeInfo, tadShapeShapeInfoCast, tadLength, canCastX);
                            oZ[offset] = OpType::op(oX[offset], y[offset]);
                        }
                    }
                }
                else if(shape::haveSameOffsets(tadShapeShapeInfo, yShapeInfo)) {

                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(tadShapeInfoZ, tadShapeInfoZCast);

                    PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                    for (int i = 0; i < tads; i++) {
                    
                        auto oZ = z + tadOffsetZ[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (int f = 0; f < tadLength; f++) {
                            auto offset  = shape::indexOffset(f, tadShapeShapeInfo, tadShapeShapeInfoCast, tadLength, canCastX);
                            auto zOffset = shape::indexOffset(f, tadShapeInfoZ, tadShapeInfoZCast, lenZ, canCastZ);
                            oZ[zOffset] = OpType::op(oX[offset], y[offset]);
                        }
                    }
                }
                else if(shape::haveSameOffsets(tadShapeShapeInfo, tadShapeInfoZ)) {

                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                    for (int i = 0; i < tads; i++) {
                    
                        auto oZ = z + tadOffsetZ[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (int f = 0; f < tadLength; f++) {
                            auto offset  = shape::indexOffset(f, tadShapeShapeInfo, tadShapeShapeInfoCast, tadLength, canCastX);
                            auto yOffset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, lenY, canCastY);
                            oZ[offset] = OpType::op(oX[offset], y[yOffset]);
                        }
                    }
                }
                else if(shape::haveSameOffsets(yShapeInfo, tadShapeInfoZ)) {

                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

                    PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                    for (int i = 0; i < tads; i++) {
                    
                        auto oZ = z + tadOffsetZ[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (int f = 0; f < tadLength; f++) {
                            auto xOffset  = shape::indexOffset(f, tadShapeShapeInfo, tadShapeShapeInfoCast, tadLength, canCastX);
                            auto offset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, lenY, canCastY);
                            oZ[offset] = OpType::op(oX[xOffset], y[offset]);
                        }
                    }
                }
                else {

                    uint tadShapeShapeInfoCast[MAX_RANK];
                    uint tadShapeInfoZCast[MAX_RANK];
                    uint yShapeInfoCast[MAX_RANK];
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadShapeShapeInfo, tadShapeShapeInfoCast);
                    bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
                    bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(tadShapeInfoZ, tadShapeInfoZCast);

                    PRAGMA_OMP_PARALLEL_FOR_THREADS(_threads)
                    for (int i = 0; i < tads; i++) {
                    
                        auto oZ = z + tadOffsetZ[i];
                        auto oX = x + tadOffsets[i];

                        PRAGMA_OMP_SIMD
                        for (int f = 0; f < tadLength; f++) {
                            auto xOffset  = shape::indexOffset(f, tadShapeShapeInfo, tadShapeShapeInfoCast, tadLength, canCastX);
                            auto yOffset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, lenY, canCastY);
                            auto zOffset  = shape::indexOffset(f, tadShapeInfoZ, tadShapeInfoZCast, lenZ, canCastZ);
                            oZ[zOffset] = OpType::op(oX[xOffset], y[yOffset]);
                        }
                    }
                }

















                if (tad != nullptr)
                    delete tad;
        }
    }
}