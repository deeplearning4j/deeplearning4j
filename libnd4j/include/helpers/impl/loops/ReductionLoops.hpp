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
template<typename X, typename Z, typename E>
template <typename OpType>
void nd4j::ReductionLoops<X, Z, E>::loopTadXZ(const X* x, const Nd4jLong* xShapeInfo,
                      Z* z, const Nd4jLong* zShapeInfo,
                      const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                      const int* dimsToExclude,
                      const int dimsLen,
                      E* extraParams) {

    const LoopKind kindOfLoop = deduceKindOfLoopTadXZ(xShapeInfo, zShapeInfo, tadShapeInfo);

    const Nd4jLong zLen   = shape::length(zShapeInfo);
    const Nd4jLong tadLen = shape::length(tadShapeInfo);

    const uint tadEws = shape::elementWiseStride(tadShapeInfo);
    const uint zEws   = shape::elementWiseStride(zShapeInfo);

    const Nd4jLong* tadShape  = shape::shapeOf(const_cast<Nd4jLong*>(tadShapeInfo));
    const Nd4jLong* tadStride = shape::stride(const_cast<Nd4jLong*>(tadShapeInfo));

    int numThreads = OmpLaunchHelper::tadThreads(tadLen, zLen);

    switch (kindOfLoop) {
        //*********************************************//
        // case SMALLARR2DX: {
        //         shape::printShapeInfoLinear(xShapeInfo);
        //     shape::printShapeInfoLinear(zShapeInfo);
        //     const auto xLen = zLen * tadLen;
        //     for (uint i = 0; i < xLen; ++i) {
        //         const auto zOffset = shape::subArrayOffset(i, xShapeInfo, zShapeInfo, dimsToExclude, dimsLen);
        //         const uint tadInd = (i / tadEws) % tadLen;
        //         auto startVal = tadInd ? z[zOffset] : static_cast<Z>(OpType::startingValue(x));
        //         z[zOffset] = OpType::update(startVal, OpType::op(x[i], extraParams), extraParams);
        //         if(tadInd == tadLen - 1)
        //             z[zOffset] = OpType::postProcess(z[zOffset], tadLen, extraParams);
        //         printf("%u - %lld\n", i, zOffset);
        //     }
        // }
        case SMALLARR2DX: {
            const auto uTadLen        = static_cast<uint>(tadLen);
            const auto uZLenMinusOne  = static_cast<uint>(zLen - 1);
            const auto xLen           = static_cast<uint>(zLen * uTadLen);
            const auto sv             = static_cast<Z>(OpType::startingValue(x));

            for (uint i = 0; i <= uZLenMinusOne; i++)
                z[i] = OpType::startingValue(x);

            uint zOffset = 0;
            for (uint i = 0; i < xLen; ++i) {
                z[zOffset] = OpType::update(z[zOffset], OpType::op(x[i], extraParams), extraParams);
                zOffset = zOffset == uZLenMinusOne ? 0 : zOffset + 1;
            }

            for (uint i = 0; i <= uZLenMinusOne; i++)
                z[i] = OpType::postProcess(z[i], tadLen, extraParams);
        }
            break;

            //*********************************************//
        case EWS1: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint j = 0; j < tadLen; j++)
                    start = OpType::update(start, OpType::op(tad[j], extraParams), extraParams);

                z[i] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case EWSNONZERO: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint j = 0; j < tadLen; j++)
                    start = OpType::update(start, OpType::op(tad[j * tadEws], extraParams), extraParams);

                z[i * zEws] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case RANK1: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint i0 = 0; i0 < tadLen; ++i0)
                    start = OpType::update(start, OpType::op(tad[i0 * tadStride[0]], extraParams), extraParams);

                z[i] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case RANK2: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                        start = OpType::update(start, OpType::op(tad[i0*tadStride[0] + i1*tadStride[1]], extraParams), extraParams);

                z[i] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case RANK3: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                        for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                            start = OpType::update(start, OpType::op(tad[i0*tadStride[0] + i1*tadStride[1] + i2*tadStride[2]], extraParams), extraParams);

                z[i] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case RANK4: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                        for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                            for (uint i3 = 0; i3 < tadShape[3]; ++i3)
                                start = OpType::update(start, OpType::op(tad[i0*tadStride[0] + i1*tadStride[1] + i2*tadStride[2] + i3*tadStride[3]], extraParams), extraParams);

                z[i] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case RANK5: {

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; ++i) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                    for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                        for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                            for (uint i3 = 0; i3 < tadShape[3]; ++i3)
                                for (uint i4 = 0; i4 < tadShape[4]; ++i4)
                                    start = OpType::update(start, OpType::op(tad[i0*tadStride[0] + i1*tadStride[1] + i2*tadStride[2] + i3*tadStride[3] + i4*tadStride[4] ], extraParams), extraParams);

                z[i] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case X_EWSNONZERO: {
            uint castZShapeInfo[MAX_RANK];
            const bool canCastZ   = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo,   castZShapeInfo);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint j = 0; j < tadLen; j++)
                    start = OpType::update(start, OpType::op(tad[j * tadEws], extraParams), extraParams);

                auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                z[zOffset] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        case Z_EWSNONZERO: {
            uint castTadShapeInfo[MAX_RANK];
            const bool canCastTad = nd4j::DataTypeUtils::castShapeInfo<uint>(tadShapeInfo, castTadShapeInfo);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint j = 0; j < tadLen; j++) {
                    auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                    start = OpType::update(start, OpType::op(tad[tadOffset], extraParams), extraParams);
                }

                z[i * zEws] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
            break;

            //*********************************************//
        default: {
            uint castTadShapeInfo[MAX_RANK];
            uint castZShapeInfo[MAX_RANK];
            const bool canCastTad = nd4j::DataTypeUtils::castShapeInfo<uint>(tadShapeInfo, castTadShapeInfo);
            const bool canCastZ   = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo,   castZShapeInfo);

            PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
            for (uint i = 0; i < zLen; i++) {
                auto tad = x + tadOffsets[i];
                auto start = OpType::startingValue(tad);

                for (uint j = 0; j < tadLen; j++) {
                    auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                    start = OpType::update(start, OpType::op(tad[tadOffset], extraParams), extraParams);
                }

                auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                z[zOffset] = OpType::postProcess(start, tadLen, extraParams);
            }
        }
    }
}