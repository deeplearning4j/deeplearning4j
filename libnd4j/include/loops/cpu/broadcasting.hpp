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
                auto lenZ = shape::length(tadShapeInfoZ);
                auto lenY = shape::length(yShapeInfo);

                int tadsPerThread = tads / TAD_THRESHOLD;
                int _threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

#pragma omp parallel for schedule(guided) num_threads(_threads) if (_threads > 1) proc_bind(AFFINITY) default(shared)
                for (int i = 0; i < tads; i++) {
                    auto offset = tadOffsets[i];
                    auto offsetZ = tadOffsetZ[i];

                    if (tadEWS > 0 && yStride > 0 && zEWS > 0 && dimensionLength == 1) {
                        auto oZ = z + offsetZ;
                        auto oX = x + offset;

                        if (tadEWS == 1 && yStride == 1 && zEWS == 1) {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++)
                                oZ[f] = OpType::op(oX[f], y[f]);                            
                        } 
                        else {
#pragma omp simd
                            for (int f = 0; f < tadLength; f++) 
                                oZ[f * zEWS] = OpType::op(oX[f * tadEWS], y[f * yStride]);                            
                        }
                    }
                    else {

                        // TODO: cover this codebranch with tests
                        // all this stuff already happens within thread                        
                        for (int f = 0; f < tadLength; f++) {                            

                            auto xOffset = offset + shape::getIndexOffset(f, tadShapeShapeInfo, tadLength);
                            auto zOffset = offsetZ + shape::getIndexOffset(f, tadShapeInfoZ, lenZ);
                            auto yOffset = shape::getIndexOffset(f, yShapeInfo, lenY);

                            z[zOffset] = OpType::op(x[xOffset], y[yOffset]);
                        }
                    }
                }

                if (tad != nullptr)
                    delete tad;
        }
    }
}