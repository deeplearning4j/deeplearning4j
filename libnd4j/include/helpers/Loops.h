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

#ifndef LIBND4J_LOOPS_H
#define LIBND4J_LOOPS_H

#include <functional>
#include <pointercast.h>
#include <shape.h>
#include <LoopKind.h>
#include <OmpLaunchHelper.h>
#include <DataTypeUtils.h>
#include <ops.h>
#include <indexreduce.h>
#include <helpers/ConstantTadHelper.h>
#include <openmp_pragmas.h>
#include <execution/Threads.h>

namespace nd4j {

    template <typename X, typename Z, typename E>
    class ND4J_EXPORT ReductionLoops {
    protected:
    public:

        template <typename OpType>
        static FORCEINLINE void loopReduce(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, E* extraParams);
    };

    template <typename X, typename Z>
    class ReductionFloatLoops : public ReductionLoops<X,Z,Z> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, Z* extraParams);

        template <typename OpType>
        static void innerloopReduce(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, Z* extraParams);
    };

    template <typename X, typename Z>
    class ND4J_EXPORT ReductionBoolLoops : public ReductionLoops<X,Z,X> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);

        template <typename OpType>
        static void innerloopReduce(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };

    template <typename X, typename Z>
    class ND4J_EXPORT ReductionLongLoops : public ReductionLoops<X,Z,X> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);

        template <typename OpType>
        static void innerloopReduce(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };

    template <typename X>
    class ND4J_EXPORT ReductionSameLoops : public ReductionLoops<X,X,X> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, X* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);

        template <typename OpType>
        static void innerloopReduce(X* x, Nd4jLong* xShapeInfo, X* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };


    template <typename X, typename Z>
    class ND4J_EXPORT IndexReductionLoops {
    private:
    public:
        static void wrapIndexReduce(const int opNum, void* x, Nd4jLong* xShapeInfo, void* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, void* extraParams);

        template <typename OpType>
        static void loopIndexReduce(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };


    template <typename X, typename Z, typename E>
    class ND4J_EXPORT TransformLoops {

    public:

        template<typename OpType>
        static FORCEINLINE void loopTransform(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, E* extraParams);
    };

    template <typename X, typename Z>
    class ND4J_EXPORT Reduction3Loops {
    public:

        template <typename OpType>
        static FORCEINLINE void loopReduce3(X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, int* dims, int dimsLen, Z* extraParams);

        template <typename OpType>
        static FORCEINLINE void loopReduce3All(X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* xTadShapeInfo, Nd4jLong* xTadOffsets, Nd4jLong* yTadShapeInfo, Nd4jLong* yTadOffsets, Z* extraParams);

        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, int* dims, int dimsLen, Z* extraParams);

        static void wrapperAll(const int opNum, X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* xTadShapeInfo, Nd4jLong* xTadOffsets, Nd4jLong* yTadShapeInfo, Nd4jLong* yTadOffsets, Z* extraParams);

        template <typename OpType>
        static void innerloopReduce3(X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, int* dims, int dimsLen, Z* extraParams);

        template <typename OpType>
        static void innerloopReduce3All(X* x, Nd4jLong* xShapeInfo, X* y, Nd4jLong* yShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* xTadShapeInfo, Nd4jLong* xTadOffsets, Nd4jLong* yTadShapeInfo, Nd4jLong* yTadOffsets, Z* extraParams);
    };




/*
//////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void Loops::loopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                    const Y* y, const Nd4jLong* yShapeInfo,
                          Z* z, const Nd4jLong* zShapeInfo,
                          Z* extraParams,
                          std::function<Z(X,Y,Z*)> op) {

    const LoopKind::Kind kindOfLoop = LoopKind::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);

    const Nd4jLong* xShape  = shape::shapeOf(xShapeInfo);
    const Nd4jLong* xStride = shape::stride(xShapeInfo);
    const Nd4jLong* yStride = shape::stride(yShapeInfo);
    const Nd4jLong* zStride = shape::stride(zShapeInfo);

    const Nd4jLong len = shape::length(xShapeInfo);

    OmpLaunchHelper threadsInfo(len);

    switch (kindOfLoop) {

        case LoopKind::EWS1: {
            PRAGMA_OMP_PARALLEL_THREADS(threadsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = threadsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(threadsInfo.getItersPerThread(threadNum));

                const auto xi = x + threadOffset;
                const auto yi = y + threadOffset;
                          auto zi = z + threadOffset;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i] = op(xi[i], yi[i], extraParams);
            }
        }
            break;

        case LoopKind::EWSNONZERO: {
            const uint xEws = shape::elementWiseStride(xShapeInfo);
            const uint yEws = shape::elementWiseStride(yShapeInfo);
            const uint zEws = shape::elementWiseStride(zShapeInfo);

            PRAGMA_OMP_PARALLEL_THREADS(threadsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = threadsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(threadsInfo.getItersPerThread(threadNum));
                const auto xi = x + threadOffset * xEws;
                const auto yi = y + threadOffset * yEws;
                      auto zi = z + threadOffset * zEws;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i*zEws] = op(xi[i*xEws], yi[i*yEws], extraParams);
            }
        }
            break;

        case LoopKind::RANK1: {
            PRAGMA_OMP_PARALLEL_FOR
            for (uint i0 = 0; i0 < len; ++i0)
                z[i0 * zStride[0]] = op(x[i0 * xStride[0]], y[i0 * yStride[0]], extraParams);
        }
            break;

        case LoopKind::RANK2: {
            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    z[i0 * zStride[0] + i1 * zStride[1]] = op(x[i0 * xStride[0] + i1 * xStride[1]], y[i0 * yStride[0] + i1 * yStride[1]], extraParams);
        }
            break;

        case LoopKind::RANK3: {
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]] = op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]], y[i0*yStride[0]+i1*yStride[1]+i2*yStride[2]], extraParams);
        }
            break;

        case LoopKind::RANK4: {
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        for (uint i3 = 0; i3 < xShape[3]; ++i3)
                            z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]+i3*zStride[3]] = op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]+i3*xStride[3]], y[i0*yStride[0]+i1*yStride[1]+i2*yStride[2]+i3*yStride[3]], extraParams);
        }
            break;

        case LoopKind::RANK5: {
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(4)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        for (uint i3 = 0; i3 < xShape[3]; ++i3)
                            for (uint i4 = 0; i4 < xShape[4]; ++i4)
                                z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]+i3*zStride[3]+i4*zStride[4]] = op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]+i3*xStride[3]+i4*xStride[4]], y[i0*yStride[0]+i1*yStride[1]+i2*yStride[2]+i3*yStride[3]+i4*yStride[4]], extraParams);
        }
            break;

        default: {
            uint xShapeInfoCast[MAX_RANK];
            uint yShapeInfoCast[MAX_RANK];
            uint zShapeInfoCast[MAX_RANK];

            bool canCastX = DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
            bool canCastY = DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
            bool canCastZ = DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

            PRAGMA_OMP_PARALLEL_THREADS(threadsInfo._numThreads)
            {
                auto threadNum = omp_get_thread_num();
                auto threadOffset = threadsInfo.getThreadOffset(threadNum);
                auto lenPerThread = static_cast<uint>(threadsInfo.getItersPerThread(threadNum));
                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++) {
                    auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, canCastX);
                    auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, canCastY);
                    auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, canCastZ);
                    z[zOffset] = op(x[xOffset], y[yOffset], extraParams);
                }
            }
        }
    }
}
*/



//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename Z, typename E>
    template <typename OpType>
    void nd4j::ReductionLoops<X, Z, E>::loopReduce(X* x, Nd4jLong* xShapeInfo,
                                                  Z* z, Nd4jLong* zShapeInfo,
                                                  Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets,
                                                  E* extraParams) {

        const LoopKind::Kind kindOfLoop = LoopKind::deduceKindOfLoopTadXZ(xShapeInfo, zShapeInfo, tadShapeInfo);

        const Nd4jLong zLen   = shape::length(zShapeInfo);
        const Nd4jLong tadLen = shape::length(tadShapeInfo);

        const uint tadEws = shape::elementWiseStride(tadShapeInfo);
        const uint zEws   = shape::elementWiseStride(zShapeInfo);

        const Nd4jLong* tadShape  = shape::shapeOf(tadShapeInfo);
        const Nd4jLong* tadStride = shape::stride(tadShapeInfo);

        int numThreads = OmpLaunchHelper::tadThreads(tadLen, zLen);

        switch (kindOfLoop) {

            //*********************************************//
            // case LoopKind::SMALLARR2DX: {
            //     shape::printShapeInfoLinear(xShapeInfo);
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
            case LoopKind::SMALLARR2DX: {
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
            case LoopKind::EWS1: {
                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint j = 0; j < tadLen; j++)
                        s = OpType::update(s, OpType::op(tad[j], extraParams), extraParams);

                    z[i] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::EWSNONZERO: {
                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint j = 0; j < tadLen; j++)
                        s = OpType::update(s, OpType::op(tad[j * tadEws], extraParams), extraParams);

                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK1: {
                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint i0 = 0; i0 < tadLen; ++i0)
                        s = OpType::update(s, OpType::op(tad[i0 * tadStride[0]], extraParams), extraParams);

                    z[i] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK2: {
                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            s = OpType::update(s, OpType::op(tad[i0 * tadStride[0] + i1 * tadStride[1]], extraParams), extraParams);

                    z[i] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK3: {
                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                                s = OpType::update(s, OpType::op(tad[i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2]], extraParams), extraParams);

                    z[i] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK4: {
                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3)
                                    s = OpType::update(s, OpType::op(tad[i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3]], extraParams), extraParams);

                    z[i] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK5: {
                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3)
                                    for (uint i4 = 0; i4 < tadShape[4]; ++i4)
                                        s = OpType::update(s, OpType::op(tad[i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3] + i4 * tadStride[4]], extraParams), extraParams);

                    z[i] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::X_EWSNONZERO: {
                uint castZShapeInfo[MAX_RANK];
                const bool canCastZ   = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo,   castZShapeInfo);

                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint j = 0; j < tadLen; j++)
                        s = OpType::update(s, OpType::op(tad[j * tadEws], extraParams), extraParams);

                    auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, canCastZ);
                    z[zOffset] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::Z_EWSNONZERO: {
                uint castTadShapeInfo[MAX_RANK];
                const bool canCastTad = nd4j::DataTypeUtils::castShapeInfo<uint>(tadShapeInfo, castTadShapeInfo);

                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, canCastTad);
                        s = OpType::update(s, OpType::op(tad[tadOffset], extraParams), extraParams);
                    }

                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            default: {
                auto innertadOffsets = new Nd4jLong[tadLen];
                shape::calcOffsets(tadShapeInfo, innertadOffsets);

                uint castZShapeInfo[MAX_RANK];
                const bool canCastZ   = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo,   castZShapeInfo);

                for (auto i = 0; i < zLen; i++) {
                    auto tad = x + tadOffsets[i];
                    auto s = OpType::startingValue(tad);

                    for (uint j = 0; j < tadLen; j++)
                        s = OpType::update(s, OpType::op(tad[innertadOffsets[j]], extraParams), extraParams);

                    auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, canCastZ);
                    z[zOffset] = OpType::postProcess(s, tadLen, extraParams);
                };

                delete[] innertadOffsets;
            }
        }
    }



    //////////////////////////////////////////////////////////////////////////////
    template <typename X, typename Z, typename E>
    template <typename OpType>
    void nd4j::TransformLoops<X,Z,E>::loopTransform(X* x, Nd4jLong* xShapeInfo,
                                             Z* z, Nd4jLong* zShapeInfo,
                                             E* extraParams) {

        const LoopKind::Kind kindOfLoop = LoopKind::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

        const Nd4jLong* xShape  = shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo));
        const Nd4jLong* xStride = shape::stride(const_cast<Nd4jLong*>(xShapeInfo));
        const Nd4jLong* zStride = shape::stride(const_cast<Nd4jLong*>(zShapeInfo));

        const Nd4jLong len = shape::length(xShapeInfo);

        OmpLaunchHelper threadsInfo(len);

        switch (kindOfLoop) {

            //*********************************************//
            case LoopKind::EWS1: {
                    for (auto i = 0; i < len; i++)
                        z[i] = OpType::op(x[i], extraParams);
                }
                break;

            //*********************************************//
            case LoopKind::EWSNONZERO: {
                    const uint xEws = shape::elementWiseStride(xShapeInfo);
                    const uint zEws = shape::elementWiseStride(zShapeInfo);

                    for (auto i = 0; i < len; i++)
                        z[i*zEws] = OpType::op(x[i*xEws], extraParams);
                }
                break;

                //*********************************************//
            case LoopKind::Z_EWSNONZERO: {
                    const uint zEws = shape::elementWiseStride(zShapeInfo);
                    uint castXShapeInfo[MAX_RANK];
                    const bool canCastX = nd4j::DataTypeUtils::castShapeInfo<uint>(xShapeInfo, castXShapeInfo);

                    if (zEws > 1) {
                        for (auto i = 0; i < len; i++) {
                            const auto xOffset = shape::indexOffset(i, xShapeInfo, castXShapeInfo, canCastX);
                            z[i * zEws] = OpType::op(x[xOffset], extraParams);
                        }
                    } else {
                        for (auto i = 0; i < len; i++) {
                            const auto xOffset = shape::indexOffset(i, xShapeInfo, castXShapeInfo, canCastX);
                            z[i] = OpType::op(x[xOffset], extraParams);
                        }
                    }
                }
                break;

                //*********************************************//
            case LoopKind::RANK1: {
                    for (auto i0 = 0; i0 < len; i0++)
                        z[i0 * zStride[0]] = OpType::op(x[i0 * xStride[0]], extraParams);
                }
                break;

                //*********************************************//
            case LoopKind::RANK2: {
                    auto uXShape0 = static_cast<uint>(xShape[0]);
                    auto uXShape1 = static_cast<uint>(xShape[1]);

                    for (auto i0 = 0; i0 < uXShape0; i0++) {
                        auto z0 = i0 * zStride[0];
                        auto x0 = i0 * xStride[0];

                        for (uint i1 = 0; i1 < uXShape1; ++i1)
                            z[z0 + i1 * zStride[1]] = OpType::op(x[x0 + i1 * xStride[1]], extraParams);
                    }
                }
                break;

                //*********************************************//
            case LoopKind::RANK3: {
                    auto uXShape0 = static_cast<uint>(xShape[0]);
                    auto uXShape1 = static_cast<uint>(xShape[1]);
                    auto uXShape2 = static_cast<uint>(xShape[2]);

                    for (uint i0 = 0; i0 < uXShape0; i0++)
                        for (uint i1 = 0; i1 < uXShape1; i1++) {
                            auto z0 = i0 * zStride[0] + i1 * zStride[1];
                            auto x0 = i0 * xStride[0] + i1 * xStride[1];

                            for (uint i2 = 0; i2 < uXShape2; ++i2)
                                z[z0 + i2 * zStride[2]] = OpType::op(x[x0 + i2 * xStride[2]], extraParams);
                        }
                }
                break;

                //*********************************************//
            case LoopKind::RANK4: {
                    auto uXShape0 = static_cast<uint>(xShape[0]);
                    auto uXShape1 = static_cast<uint>(xShape[1]);
                    auto uXShape2 = static_cast<uint>(xShape[2]);
                    auto uXShape3 = static_cast<uint>(xShape[3]);

                    for (uint i0 = 0; i0 < uXShape0; i0++)
                        for (uint i1 = 0; i1 < uXShape1; i1++)
                            for (uint i2 = 0; i2 < uXShape2; i2++) {
                                auto x0 = i0 * xStride[0] + i1 * xStride[1] + i2 * xStride[2];
                                auto z0 = i0 * zStride[0] + i1 * zStride[1] + i2 * zStride[2];

                                for (uint i3 = 0; i3 < uXShape3; ++i3)
                                    z[z0 + i3 * zStride[3]] = OpType::op(x[x0 + i3 * xStride[3]], extraParams);
                            }
                }
                break;

                //*********************************************//
            case LoopKind::RANK5: {
                    auto uXShape0 = static_cast<uint>(xShape[0]);
                    auto uXShape1 = static_cast<uint>(xShape[1]);
                    auto uXShape2 = static_cast<uint>(xShape[2]);
                    auto uXShape3 = static_cast<uint>(xShape[3]);
                    auto uXShape4 = static_cast<uint>(xShape[4]);


                    for (uint i0 = 0; i0 < uXShape0; i0++)
                        for (uint i1 = 0; i1 < uXShape1; i1++)
                            for (uint i2 = 0; i2 < uXShape2; i2++) {

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


                    for (auto i = 0; i < len; i++) {
                        auto xOffset = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, canCastX);
                        auto zOffset = shape::indexOffset(i, zShapeInfo, zShapeInfoCast, canCastZ);
                        z[zOffset] = OpType::op(x[xOffset], extraParams);
                    }
                }

        }
    }


//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename Z>
    template <typename OpType>
    void nd4j::Reduction3Loops<X, Z>::loopReduce3(X* x, Nd4jLong* xShapeInfo,
                                                  X* y, Nd4jLong* yShapeInfo,
                                                  Z* z, Nd4jLong* zShapeInfo,
                                                  int* dims, int dimsLen,
                                                  Z* extraParameters) {

        // both tads have same shape, however strides and ews may differ

        Z param0(OpType::startingValue(x)), param1(OpType::startingValue(x)), param2(extraParameters ? extraParameters[0] : OpType::startingValue(x));

        const Nd4jLong xLen = shape::length(xShapeInfo);
        const Nd4jLong yLen = shape::length(yShapeInfo);

        Nd4jLong *xTadShapeInfo = nullptr, *yTadShapeInfo = nullptr, *xTadOffsets = nullptr, *yTadOffsets = nullptr;
        TadPack tadPackX, tadPackY;
        std::vector<Nd4jLong> zeroOffsets;

        if(xLen == yLen) {
            tadPackX      = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dims, dimsLen);
            tadPackY      = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dims, dimsLen);
            xTadShapeInfo = tadPackX.primaryShapeInfo();
            yTadShapeInfo = tadPackY.primaryShapeInfo();
            xTadOffsets   = tadPackX.primaryOffsets();
            yTadOffsets   = tadPackY.primaryOffsets();
        }
        else if(yLen > xLen) {
            tadPackY      = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dims, dimsLen);
            xTadShapeInfo = xShapeInfo;
            yTadShapeInfo = tadPackY.primaryShapeInfo();
            yTadOffsets   = tadPackY.primaryOffsets();
        }
        else {
            tadPackX      = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dims, dimsLen);
            yTadShapeInfo = yShapeInfo;
            xTadShapeInfo = tadPackX.primaryShapeInfo();
            xTadOffsets   = tadPackX.primaryOffsets();
        }


        const LoopKind::Kind kindOfLoop = LoopKind::deduceKindOfLoopTadXYZ(xTadShapeInfo, yTadShapeInfo, zShapeInfo);

        const auto xTadEws = shape::elementWiseStride(xTadShapeInfo);
        const auto yTadEws = shape::elementWiseStride(yTadShapeInfo);
        const auto zEws    = shape::elementWiseStride(zShapeInfo);

        const auto zLen   = shape::length(zShapeInfo);
        const auto tadLen = shape::length(xTadShapeInfo);

        const auto tadShape    = shape::shapeOf(xTadShapeInfo);
        const auto xTadStride  = shape::stride(xTadShapeInfo);
        const auto yTadStride  = shape::stride(xTadShapeInfo);

        int numThreads = OmpLaunchHelper::tadThreads(tadLen, zLen);

        switch (kindOfLoop) {

            //*********************************************//
            case LoopKind::EWS1: {
                Z extraParams[3];
                for (auto i = 0; i < zLen; i++) {
                    extraParams[0] = param0;
                    extraParams[1] = param1;
                    extraParams[2] = param2;

                    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                    auto s = OpType::startingValue(xTad);

                    for (uint j = 0; j < tadLen; ++j)
                        s = OpType::update(s, OpType::op(xTad[j], yTad[j], extraParams), extraParams);

                    z[i] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::EWSNONZERO: {
                Z extraParams[3];
                for (auto i = 0; i < zLen; i++) {
                    extraParams[0] = param0;
                    extraParams[1] = param1;
                    extraParams[2] = param2;

                    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                    auto s = OpType::startingValue(xTad);

                    for (uint j = 0; j < tadLen; ++j)
                        s = OpType::update(s, OpType::op(xTad[j * xTadEws], yTad[j * yTadEws], extraParams), extraParams);

                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK1: {
                Z extraParams[3];
                for (auto i = 0; i < zLen; i++) {
                    extraParams[0] = param0;
                    extraParams[1] = param1;
                    extraParams[2] = param2;

                    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                    auto s = OpType::startingValue(xTad);

                    for (uint i0 = 0; i0 < tadLen; ++i0) {
                        const auto xTadOffset = i0 * xTadStride[0];
                        const auto yTadOffset = i0 * yTadStride[0];
                        s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                    }

                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK2: {
                Z extraParams[3];
                for (auto i = 0; i < zLen; i++) {
                    extraParams[0] = param0;
                    extraParams[1] = param1;
                    extraParams[2] = param2;

                    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                    auto s = OpType::startingValue(xTad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1];
                            const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1];
                            s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                        }
                    }
                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK3: {
                Z extraParams[3];
                for (auto i = 0; i < zLen; i++) {
                    extraParams[0] = param0;
                    extraParams[1] = param1;
                    extraParams[2] = param2;

                    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                    auto s = OpType::startingValue(xTad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1] + i2 * xTadStride[2];
                                const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1] + i2 * yTadStride[2];
                                s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                            }
                        }
                    }
                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK4: {
                Z extraParams[3];
                for (auto i = 0; i < zLen; i++) {
                    extraParams[0] = param0;
                    extraParams[1] = param1;
                    extraParams[2] = param2;

                    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                    auto s = OpType::startingValue(xTad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                    const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1] + i2 * xTadStride[2] + i3 * xTadStride[3];
                                    const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1] + i2 * yTadStride[2] + i3 * yTadStride[3];
                                    s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                                }
                            }
                        }
                    }
                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK5: {
                Z extraParams[3];
                for (auto i = 0; i < zLen; i++) {
                    extraParams[0] = param0;
                    extraParams[1] = param1;
                    extraParams[2] = param2;

                    const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                    const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                    auto s = OpType::startingValue(xTad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                    for (uint i4 = 0; i4 < tadShape[4]; ++i4) {
                                        const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1] + i2 * xTadStride[2] + i3 * xTadStride[3] + i4 * xTadStride[4];
                                        const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1] + i2 * yTadStride[2] + i3 * yTadStride[3] + i4 * yTadStride[4];
                                        s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                                    }
                                }
                            }
                        }
                    }
                    z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                };
            }
            break;

            //*********************************************//
            default: {
                uint castXTadShapeInfo[MAX_RANK];
                const bool canCastXTad = nd4j::DataTypeUtils::castShapeInfo<uint>(xTadShapeInfo, castXTadShapeInfo);

                if(shape::haveSameShapeAndStrides(xTadShapeInfo, yTadShapeInfo)) {
                    Z extraParams[3];
                    for (auto i = 0; i < zLen; i++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                        const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                        auto s = OpType::startingValue(xTad);

                        for (uint j = 0; j < tadLen; ++j) {
                            const auto tadOffset = shape::indexOffset(j, xTadShapeInfo, castXTadShapeInfo, canCastXTad);
                            s = OpType::update(s, OpType::op(xTad[tadOffset], yTad[tadOffset], extraParams), extraParams);
                        }

                        z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                    };
                }
                else {
                    uint castYTadShapeInfo[MAX_RANK];
                    const bool canCastYTad = nd4j::DataTypeUtils::castShapeInfo<uint>(yTadShapeInfo, castYTadShapeInfo);

                    Z extraParams[3];
                    for (auto i = 0; i < zLen; i++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = xTadOffsets ? x + xTadOffsets[i] : x;
                        const auto yTad = yTadOffsets ? y + yTadOffsets[i] : y;
                        auto s = OpType::startingValue(xTad);

                        for (uint j = 0; j < tadLen; ++j) {
                            const auto xTadOffset = shape::indexOffset(j, xTadShapeInfo, castXTadShapeInfo, canCastXTad);
                            const auto yTadOffset = shape::indexOffset(j, yTadShapeInfo, castYTadShapeInfo, canCastYTad);
                            s = OpType::update(s, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                        }
                        z[i * zEws] = OpType::postProcess(s, tadLen, extraParams);
                    };
                }
            }
        }
    }

//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename Z>
    template <typename OpType>
    void nd4j::Reduction3Loops<X, Z>::loopReduce3All(X* x, Nd4jLong* xShapeInfo,
                                                     X* y, Nd4jLong* yShapeInfo,
                                                     Z* z, Nd4jLong* zShapeInfo,
                                                     Nd4jLong* xTadShapeInfo, Nd4jLong* xTadOffsets,
                                                     Nd4jLong* yTadShapeInfo, Nd4jLong* yTadOffsets,
                                                     Z* extraParameters) {

        // both tads have same shape, however strides and ews may differ

        Z param0(OpType::startingValue(x)), param1(OpType::startingValue(x)), param2(extraParameters ? extraParameters[0] : OpType::startingValue(x));

        const LoopKind::Kind kindOfLoop = LoopKind::deduceKindOfLoopTadXYZ(xTadShapeInfo, yTadShapeInfo, zShapeInfo);

        const auto xTadEws = shape::elementWiseStride(xTadShapeInfo);
        const auto yTadEws = shape::elementWiseStride(yTadShapeInfo);
        const auto zEws    = shape::elementWiseStride(zShapeInfo);

        const auto zLen   = shape::length(zShapeInfo);
        const auto tadLen = shape::length(xTadShapeInfo);

        const auto numXTads = shape::length(xShapeInfo) / tadLen;
        const auto numYTads = shape::length(yShapeInfo) / tadLen;

        const auto tadShape    = shape::shapeOf(xTadShapeInfo);
        const auto xTadStride  = shape::stride(xTadShapeInfo);
        const auto yTadStride  = shape::stride(yTadShapeInfo);

        const auto startVal = OpType::startingValue(x);

        int numThreads = OmpLaunchHelper::tadThreads(tadLen, numXTads*numYTads);

        switch (kindOfLoop) {
            //*********************************************//
            case LoopKind::EWS1: {
                Z extraParams[3];
                for (auto ix = 0; ix < numXTads; ix++) {
                    for (auto iy = 0; iy < numYTads; iy++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = x + xTadOffsets[ix];
                        const auto yTad = y + yTadOffsets[iy];
                        const auto zInd = ix * numYTads + iy;
                        auto start = startVal;

                        for (uint j = 0; j < tadLen; ++j)
                            start = OpType::update(start, OpType::op(xTad[j], yTad[j], extraParams), extraParams);

                        z[zInd] = OpType::postProcess(start, tadLen, extraParams);
                    }
                };
            }
            break;

            //*********************************************//
            case LoopKind::EWSNONZERO: {
                Z extraParams[3];
                for (auto ix = 0; ix < numXTads; ix++) {
                    for (auto iy = 0; iy < numYTads; iy++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = x + xTadOffsets[ix];
                        const auto yTad = y + yTadOffsets[iy];
                        const auto zInd = ix * numYTads + iy;
                        auto start = startVal;

                        for (uint j = 0; j < tadLen; ++j)
                            start = OpType::update(start, OpType::op(xTad[j * xTadEws], yTad[j * yTadEws], extraParams), extraParams);

                        z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                    }
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK1: {
                Z extraParams[3];
                for (auto ix = 0; ix < numXTads; ix++) {
                    for (auto iy = 0; iy < numYTads; iy++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = x + xTadOffsets[ix];
                        const auto yTad = y + yTadOffsets[iy];
                        const auto zInd = ix * numYTads + iy;
                        auto start = startVal;

                        for (uint i0 = 0; i0 < tadLen; ++i0) {
                            const auto xTadOffset = i0 * xTadStride[0];
                            const auto yTadOffset = i0 * yTadStride[0];
                            start = OpType::update(start, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                        }
                        z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                    }
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK2: {
                Z extraParams[3];
                for (auto ix = 0; ix < numXTads; ix++) {
                    for (auto iy = 0; iy < numYTads; iy++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = x + xTadOffsets[ix];
                        const auto yTad = y + yTadOffsets[iy];
                        const auto zInd = ix * numYTads + iy;
                        auto start = startVal;

                        for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                            for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                                const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1];
                                const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1];
                                start = OpType::update(start, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                            }
                        }
                        z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                    }
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK3: {
                Z extraParams[3];
                for (auto ix = 0; ix < numXTads; ix++) {
                    for (auto iy = 0; iy < numYTads; iy++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = x + xTadOffsets[ix];
                        const auto yTad = y + yTadOffsets[iy];
                        const auto zInd = ix * numYTads + iy;
                        auto start = startVal;

                        for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                            for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                                for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                    const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1] + i2 * xTadStride[2];
                                    const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1] + i2 * yTadStride[2];
                                    start = OpType::update(start, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                                }
                            }
                        }
                        z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                    }
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK4: {
                Z extraParams[3];
                for (auto ix = 0; ix < numXTads; ix++) {
                    for (auto iy = 0; iy < numYTads; iy++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = x + xTadOffsets[ix];
                        const auto yTad = y + yTadOffsets[iy];
                        const auto zInd = ix * numYTads + iy;
                        auto start = startVal;

                        for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                            for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                                for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                    for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                        const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1] + i2 * xTadStride[2] + i3 * xTadStride[3];
                                        const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1] + i2 * yTadStride[2] + i3 * yTadStride[3];
                                        start = OpType::update(start, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                                    }
                                }
                            }
                        }
                        z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                    }
                };
            }
            break;

            //*********************************************//
            case LoopKind::RANK5: {
                Z extraParams[3];
                for (auto ix = 0; ix < numXTads; ix++) {
                    for (auto iy = 0; iy < numYTads; iy++) {
                        extraParams[0] = param0;
                        extraParams[1] = param1;
                        extraParams[2] = param2;

                        const auto xTad = x + xTadOffsets[ix];
                        const auto yTad = y + yTadOffsets[iy];
                        const auto zInd = ix * numYTads + iy;
                        auto start = startVal;

                        for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                            for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                                for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                    for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                        for (uint i4 = 0; i4 < tadShape[4]; ++i4) {
                                            const auto xTadOffset = i0 * xTadStride[0] + i1 * xTadStride[1] + i2 * xTadStride[2] + i3 * xTadStride[3] + i4 * xTadStride[4];
                                            const auto yTadOffset = i0 * yTadStride[0] + i1 * yTadStride[1] + i2 * yTadStride[2] + i3 * yTadStride[3] + i4 * yTadStride[4];
                                            start = OpType::update(start, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                                        }
                                    }
                                }
                            }
                        }
                        z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                    }
                };
            }
            break;

            //*********************************************//
            default: {
                uint castXTadShapeInfo[MAX_RANK];
                const bool canCastXTad = nd4j::DataTypeUtils::castShapeInfo<uint>(xTadShapeInfo, castXTadShapeInfo);

                if(shape::haveSameShapeAndStrides(xTadShapeInfo, yTadShapeInfo)) {
                    Z extraParams[3];
                    for (auto ix = 0; ix < numXTads; ix++) {
                        for (auto iy = 0; iy < numYTads; iy++) {
                            extraParams[0] = param0;
                            extraParams[1] = param1;
                            extraParams[2] = param2;

                            const auto xTad = x + xTadOffsets[ix];
                            const auto yTad = y + yTadOffsets[iy];
                            const auto zInd = ix * numYTads + iy;
                            auto start = startVal;

                            for (uint j = 0; j < tadLen; ++j) {
                                const auto tadOffset = shape::indexOffset(j, xTadShapeInfo, castXTadShapeInfo, canCastXTad);
                                start = OpType::update(start, OpType::op(xTad[tadOffset], yTad[tadOffset], extraParams), extraParams);
                            }
                            z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                        }
                    };
                }
                else {
                    uint castYTadShapeInfo[MAX_RANK];
                    const bool canCastYTad = nd4j::DataTypeUtils::castShapeInfo<uint>(yTadShapeInfo, castYTadShapeInfo);

                    Z extraParams[3];
                    for (auto ix = 0; ix < numXTads; ix++) {
                        for (auto iy = 0; iy < numYTads; iy++) {
                            extraParams[0] = param0;
                            extraParams[1] = param1;
                            extraParams[2] = param2;

                            const auto xTad = x + xTadOffsets[ix];
                            const auto yTad = y + yTadOffsets[iy];
                            const auto zInd = ix * numYTads + iy;
                            auto start = startVal;

                            for (uint j = 0; j < tadLen; ++j) {
                                const auto xTadOffset = shape::indexOffset(j, xTadShapeInfo, castXTadShapeInfo, canCastXTad);
                                const auto yTadOffset = shape::indexOffset(j, yTadShapeInfo, castYTadShapeInfo, canCastYTad);
                                start = OpType::update(start, OpType::op(xTad[xTadOffset], yTad[yTadOffset], extraParams), extraParams);
                            }

                            z[zInd * zEws] = OpType::postProcess(start, tadLen, extraParams);
                        }
                    };
                }
            }
        }
    }



}


#endif //LIBND4J_LOOPS_H
