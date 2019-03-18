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
#include <OmpLaunchHelper.h>
#include <DataTypeUtils.h>
#include <ops.h>
#include <indexreduce.h>
#include <openmp_pragmas.h>

namespace nd4j {

    class Loops {
    private:
        enum LoopKind {EWS1, EWSNONZERO, RANK1, RANK2, RANK3, RANK4, RANK5, X_EWSNONZERO, Z_EWSNONZERO, COMMON};

        //////////////////////////////////////////////////////////////////////////////
        FORCEINLINE static LoopKind deduceKindOfLoopXYZ(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo);

        //////////////////////////////////////////////////////////////////////////////
        FORCEINLINE static LoopKind deduceKindOfLoopXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo);

        //////////////////////////////////////////////////////////////////////////////
        FORCEINLINE static LoopKind deduceKindOfLoopTadXZ(const Nd4jLong* tadShapeInfo, const Nd4jLong* zShapeInfo);

    public:
        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename Y, typename Z> 
        FORCEINLINE static void loopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                            const Y* y, const Nd4jLong* yShapeInfo,
                                  Z* z, const Nd4jLong* zShapeInfo,
                                  Z* extraParams,
                            std::function<Z(X,Y,Z*)> op);

        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename Z, typename E>
        FORCEINLINE static void loopTadXZ(const X* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                                                Z* z, const Nd4jLong* zShapeInfo,
                                                E* extraParams,
                                                std::function<X(const X*)>      startVal, 
                                                std::function<Z(Z,Z,E*)>        update,
                                                std::function<Z(X,E*)>          op,
                                                std::function<Z(Z,Nd4jLong,E*)> postPr);

        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename E>
        FORCEINLINE static void loopIndexTadXZ(const X* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                                              Nd4jLong* z, const Nd4jLong* zShapeInfo,
                                              E* extraParams,
                                              std::function<functions::indexreduce::IndexValue<X>(X*)> startVal, 
                                              std::function<functions::indexreduce::IndexValue<X>(functions::indexreduce::IndexValue<X>&, functions::indexreduce::IndexValue<X>, E*)> update);
    };



    //////////////////////////////////////////////////////////////////////////////
    FORCEINLINE Loops::LoopKind Loops::deduceKindOfLoopXYZ(const Nd4jLong* tadShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo) {

        const int xRank = shape::rank(tadShapeInfo);

        const Nd4jLong xEws = shape::elementWiseStride(tadShapeInfo);
        const Nd4jLong yEws = shape::elementWiseStride(yShapeInfo);
        const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

        int temp;
        const bool xVector = shape::isCommonVector(tadShapeInfo, temp);
        const bool yVector = shape::isCommonVector(yShapeInfo, temp);
        const bool zVector = shape::isCommonVector(zShapeInfo, temp);

        const char xOrder = shape::order(tadShapeInfo);
        const char yOrder = shape::order(yShapeInfo);
        const char zOrder = shape::order(zShapeInfo);

        const bool shapesSame = shape::shapeEquals(tadShapeInfo, yShapeInfo) && shape::shapeEquals(tadShapeInfo, zShapeInfo);

        if (xEws == 1 && yEws == 1 && zEws == 1 && ((xOrder == yOrder && xOrder == zOrder) || ((xVector || xOrder == 'c') && (yVector || yOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWS1;

        if(xEws > 0 && yEws > 0 && zEws > 0     && ((xOrder == yOrder && xOrder == zOrder) || ((xVector || xOrder == 'c') && (yVector || yOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWSNONZERO;

        if(xRank == 1 && shapesSame)
            return RANK1;

        if(xRank == 2 && shapesSame)
            return RANK2;

        if(xRank == 3 && shapesSame)
            return RANK3;

        if(xRank == 4 && shapesSame)
            return RANK4;

        if(xRank == 5 && shapesSame)
            return RANK5;

        return COMMON;
    }

//////////////////////////////////////////////////////////////////////////////
    FORCEINLINE Loops::LoopKind Loops::deduceKindOfLoopXZ(const Nd4jLong* tadShapeInfo, const Nd4jLong* zShapeInfo) {


        const int xRank = shape::rank(tadShapeInfo);

        const Nd4jLong xEws = shape::elementWiseStride(tadShapeInfo);
        const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

        const char xOrder = shape::order(tadShapeInfo);
        const char zOrder = shape::order(zShapeInfo);

        int temp;
        const bool xVector = shape::isCommonVector(tadShapeInfo, temp);
        const bool zVector = shape::isCommonVector(zShapeInfo, temp);

        const bool shapesSame = shape::shapeEquals(tadShapeInfo, zShapeInfo);

        if(xEws == 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWS1;

        if(xEws > 0 && zEws > 0   && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWSNONZERO;

        if(xRank == 1 && shapesSame)
            return RANK1;

        if(xRank == 2 && shapesSame)
            return RANK2;

        if(xRank == 3 && shapesSame)
            return RANK3;

        if(xRank == 4 && shapesSame)
            return RANK4;

        if(xRank == 5 && shapesSame)
            return RANK5;

        if(xEws > 0 && (xOrder == 'c' || xVector) && zEws == 0)
            return X_EWSNONZERO;

        if(zEws > 0 && (zOrder == 'c' || zVector) && xEws == 0)
            return Z_EWSNONZERO;

        return COMMON;
    }

//////////////////////////////////////////////////////////////////////////////
    FORCEINLINE Loops::LoopKind Loops::deduceKindOfLoopTadXZ(const Nd4jLong* tadShapeInfo, const Nd4jLong* zShapeInfo) {

        const int xRank = shape::rank(tadShapeInfo);

        const Nd4jLong tadEws = shape::elementWiseStride(tadShapeInfo);
        const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

        const char xOrder = shape::order(tadShapeInfo);
        const char zOrder = shape::order(zShapeInfo);

        int temp;
        const bool xVector = shape::isCommonVector(tadShapeInfo, temp);
        const bool zVector = shape::isCommonVector(zShapeInfo, temp);

        if(tadEws == 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWS1;

        if(tadEws > 0 && zEws > 0   && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
            return EWSNONZERO;

        if(xRank == 1 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK1;

        if(xRank == 2 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK2;

        if(xRank == 3 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK3;

        if(xRank == 4 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK4;

        if(xRank == 5 && zEws == 1 && (zVector || zOrder == 'c'))
            return RANK5;

        if(tadEws > 0 && (xVector || xOrder == 'c') && zEws == 0)
            return X_EWSNONZERO;

        if(zEws > 0 && (zOrder == 'c' || zVector) && tadEws == 0)
            return Z_EWSNONZERO;

        return COMMON;
    }


//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename Y, typename Z>
    FORCEINLINE void Loops::loopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                        const Y* y, const Nd4jLong* yShapeInfo,
                        Z* z, const Nd4jLong* zShapeInfo,
                        Z* extraParams,
                        std::function<Z(X,Y,Z*)> op) {

        const LoopKind kindOfLoop = Loops::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);

        const Nd4jLong len = shape::length(xShapeInfo);

        OmpLaunchHelper thredsInfo(len);

        switch (kindOfLoop) {

            //*********************************************//
            case EWS1: {
                PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
                {
                    const auto threadNum = omp_get_thread_num();
                    const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                    const auto ulen = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                    PRAGMA_OMP_SIMD
                    for (uint i = 0; i < ulen; i++)
                        z[i] = op(x[i], y[i], extraParams);
                }
            }
                break;

                //*********************************************//
            case EWSNONZERO: {
                const uint xEws = shape::elementWiseStride(xShapeInfo);
                const uint yEws = shape::elementWiseStride(yShapeInfo);
                const uint zEws = shape::elementWiseStride(zShapeInfo);

                PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
                {
                    const auto threadNum = omp_get_thread_num();
                    const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                    const auto ulen = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                    PRAGMA_OMP_SIMD
                    for (uint i = 0; i < ulen; i++)
                        z[i*zEws] = op(x[i*xEws], y[i*yEws], extraParams);
                }
            }
                break;

                //*********************************************//
            case RANK1: {
                const auto xStride0 = shape::stride(xShapeInfo)[0];
                const auto yStride0 = shape::stride(yShapeInfo)[0];
                const auto zStride0 = shape::stride(zShapeInfo)[0];

                PRAGMA_OMP_PARALLEL_FOR
                for (uint i0 = 0; i0 < len; ++i0)
                    z[i0 * zStride0] = op(x[i0 * xStride0], y[i0 * yStride0], extraParams);
            }
                break;

                //*********************************************//
            case RANK2: {
                const auto xStride0 = shape::stride(xShapeInfo)[0];
                const auto xStride1 = shape::stride(xShapeInfo)[1];
                const auto yStride0 = shape::stride(yShapeInfo)[0];
                const auto yStride1 = shape::stride(yShapeInfo)[1];
                const auto zStride0 = shape::stride(zShapeInfo)[0];
                const auto zStride1 = shape::stride(zShapeInfo)[1];

                PRAGMA_OMP_PARALLEL_FOR_SIMD
                for (uint i0 = 0; i0 < xShapeInfo[1]; ++i0)
                    for (uint i1 = 0; i1 < xShapeInfo[2]; ++i1)
                        z[i0 * zStride0 + i1 * zStride1] = op(x[i0 * xStride0 + i1 * xStride1], y[i0 * yStride0 + i1 * yStride1], extraParams);
            }
                break;

                //*********************************************//
            case RANK3: {
                const auto xStride0 = shape::stride(xShapeInfo)[0];
                const auto xStride1 = shape::stride(xShapeInfo)[1];
                const auto xStride2 = shape::stride(xShapeInfo)[2];
                const auto yStride0 = shape::stride(yShapeInfo)[0];
                const auto yStride1 = shape::stride(yShapeInfo)[1];
                const auto yStride2 = shape::stride(yShapeInfo)[2];
                const auto zStride0 = shape::stride(zShapeInfo)[0];
                const auto zStride1 = shape::stride(zShapeInfo)[1];
                const auto zStride2 = shape::stride(zShapeInfo)[2];

                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
                for (uint i0 = 0; i0 < xShapeInfo[1]; ++i0)
                    for (uint i1 = 0; i1 < xShapeInfo[2]; ++i1)
                        for (uint i2 = 0; i2 < xShapeInfo[3]; ++i2)
                            z[i0*zStride0+i1*zStride1+i2*zStride2] = op(x[i0*xStride0+i1*xStride1+i2*xStride2], x[i0*yStride0+i1*yStride1+i2*yStride2], extraParams);
            }
                break;

                //*********************************************//
            case RANK4: {
                const auto xStride0 = shape::stride(xShapeInfo)[0];
                const auto xStride1 = shape::stride(xShapeInfo)[1];
                const auto xStride2 = shape::stride(xShapeInfo)[2];
                const auto xStride3 = shape::stride(xShapeInfo)[3];
                const auto yStride0 = shape::stride(yShapeInfo)[0];
                const auto yStride1 = shape::stride(yShapeInfo)[1];
                const auto yStride2 = shape::stride(yShapeInfo)[2];
                const auto yStride3 = shape::stride(yShapeInfo)[3];
                const auto zStride0 = shape::stride(zShapeInfo)[0];
                const auto zStride1 = shape::stride(zShapeInfo)[1];
                const auto zStride2 = shape::stride(zShapeInfo)[2];
                const auto zStride3 = shape::stride(zShapeInfo)[3];

                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
                for (uint i0 = 0; i0 < xShapeInfo[1]; ++i0)
                    for (uint i1 = 0; i1 < xShapeInfo[2]; ++i1)
                        for (uint i2 = 0; i2 < xShapeInfo[3]; ++i2)
                            for (uint i3 = 0; i3 < xShapeInfo[4]; ++i3)
                                z[i0*zStride0+i1*zStride1+i2*zStride2+i3*zStride3] = op(x[i0*xStride0+i1*xStride1+i2*xStride2+i3*xStride3], y[i0*yStride0+i1*yStride1+i2*yStride2+i3*yStride3], extraParams);
            }
                break;

                //*********************************************//
            case RANK5: {
                const auto xStride0 = shape::stride(xShapeInfo)[0];
                const auto xStride1 = shape::stride(xShapeInfo)[1];
                const auto xStride2 = shape::stride(xShapeInfo)[2];
                const auto xStride3 = shape::stride(xShapeInfo)[3];
                const auto xStride4 = shape::stride(xShapeInfo)[4];
                const auto yStride0 = shape::stride(yShapeInfo)[0];
                const auto yStride1 = shape::stride(yShapeInfo)[1];
                const auto yStride2 = shape::stride(yShapeInfo)[2];
                const auto yStride3 = shape::stride(yShapeInfo)[3];
                const auto yStride4 = shape::stride(yShapeInfo)[4];
                const auto zStride0 = shape::stride(zShapeInfo)[0];
                const auto zStride1 = shape::stride(zShapeInfo)[1];
                const auto zStride2 = shape::stride(zShapeInfo)[2];
                const auto zStride3 = shape::stride(zShapeInfo)[3];
                const auto zStride4 = shape::stride(zShapeInfo)[4];

                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(4)
                for (uint i0 = 0; i0 < xShapeInfo[1]; ++i0)
                    for (uint i1 = 0; i1 < xShapeInfo[2]; ++i1)
                        for (uint i2 = 0; i2 < xShapeInfo[3]; ++i2)
                            for (uint i3 = 0; i3 < xShapeInfo[4]; ++i3)
                                for (uint i4 = 0; i4 < xShapeInfo[5]; ++i4)
                                    z[i0*zStride0+i1*zStride1+i2*zStride2+i3*zStride3 + i4*zStride4] = op(x[i0*xStride0+i1*xStride1+i2*xStride2+i3*xStride3+i4*xStride4], y[i0*yStride0+i1*yStride1+i2*yStride2+i3*yStride3+i4*yStride4], extraParams);
            }
                break;

                //*********************************************//
            default: {
                uint xShapeInfoCast[MAX_RANK];
                uint yShapeInfoCast[MAX_RANK];
                uint zShapeInfoCast[MAX_RANK];

                bool canCastX = DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                bool canCastY = DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
                bool canCastZ = DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

                PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
                {
                    auto threadNum = omp_get_thread_num();
                    auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                    auto ulen = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                    PRAGMA_OMP_SIMD
                    for (uint i = 0; i < ulen; i++) {
                        auto tadOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);
                        auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, len, canCastY);
                        auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, len, canCastZ);
                        z[zOffset] = op(x[tadOffset], y[yOffset], extraParams);
                    }
                }
            }
        }
    }


//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename Z, typename E>
    FORCEINLINE  void Loops::loopTadXZ(const X* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                                      Z* z, const Nd4jLong* zShapeInfo,
                                      E* extraParams,
                                      std::function<X(const X*)>      startVal,
                                      std::function<Z(Z,Z,E*)>        update,
                                      std::function<Z(X,E*)>          op,
                                      std::function<Z(Z,Nd4jLong,E*)> postPr) {

        const LoopKind kindOfLoop = Loops::deduceKindOfLoopTadXZ(tadShapeInfo, zShapeInfo);

        const Nd4jLong zLen   = shape::length(zShapeInfo);
        const Nd4jLong tadLen = shape::length(tadShapeInfo);

        const uint tadEws = shape::elementWiseStride(tadShapeInfo);
        const uint zEws   = shape::elementWiseStride(zShapeInfo);

        const Nd4jLong* tadShape  = shape::shapeOf(const_cast<Nd4jLong*>(tadShapeInfo));
        const Nd4jLong* tadStride = shape::stride(const_cast<Nd4jLong*>(tadShapeInfo));

        int numThreads = OmpLaunchHelper::tadThreads(tadLen, zLen);


        switch (kindOfLoop) {

            //*********************************************//
            case EWS1: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = x + tadOffsets[i];
                    auto start = startVal(tad);

                    for (uint j = 0; j < tadLen; j++)
                        start = update(start, op(tad[j], extraParams), extraParams);

                    z[i] = postPr(start, tadLen, extraParams);
                }
            }
                break;

                //*********************************************//
            case EWSNONZERO: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = x + tadOffsets[i];
                    auto start = startVal(tad);

                    for (uint j = 0; j < tadLen; j++)
                        start = update(start, op(tad[j * tadEws], extraParams), extraParams);

                    z[i * zEws] = postPr(start, tadLen, extraParams);
                }
            }
                break;

                //*********************************************//
            case RANK1: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = x + tadOffsets[i];
                    auto start = startVal(tad);

                    for (uint i0 = 0; i0 < tadLen; ++i0)
                        start = update(start, op(tad[i0 * tadStride[0]], extraParams), extraParams);

                    z[i] = postPr(start, tadLen, extraParams);
                }
            }
                break;

                //*********************************************//
            case RANK2: {

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = x + tadOffsets[i];
                    auto start = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            start = update(start, op(tad[i0*tadStride[0] + i1*tadStride[1]], extraParams), extraParams);

                    z[i] = postPr(start, tadLen, extraParams);
                }
            }
                break;

                //*********************************************//
            case RANK3: {

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = x + tadOffsets[i];
                    auto start = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                                start = update(start, op(tad[i0*tadStride[0] + i1*tadStride[1] + i2*tadStride[2]], extraParams), extraParams);

                    z[i] = postPr(start, tadLen, extraParams);
                }
            }
                break;

                //*********************************************//
            case RANK4: {

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = x + tadOffsets[i];
                    auto start = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3)
                                    start = update(start, op(tad[i0*tadStride[0] + i1*tadStride[1] + i2*tadStride[2] + i3*tadStride[3]], extraParams), extraParams);

                    z[i] = postPr(start, tadLen, extraParams);
                }
            }
                break;

                //*********************************************//
            case RANK5: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = x + tadOffsets[i];
                    auto start = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0)
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1)
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2)
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3)
                                    for (uint i4 = 0; i4 < tadShape[4]; ++i4)
                                        start = update(start, op(tad[i0*tadStride[0] + i1*tadStride[1] + i2*tadStride[2] + i3*tadStride[3] + i4*tadStride[4] ], extraParams), extraParams);

                    z[i] = postPr(start, tadLen, extraParams);
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
                    auto start = startVal(tad);

                    for (uint j = 0; j < tadLen; j++)
                        start = update(start, op(tad[j * tadEws], extraParams), extraParams);

                    auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                    z[zOffset] = postPr(start, tadLen, extraParams);
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
                    auto start = startVal(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                        start = update(start, op(tad[tadOffset], extraParams), extraParams);
                    }

                    z[i * zEws] = postPr(start, tadLen, extraParams);
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
                    auto start = startVal(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                        start = update(start, op(tad[tadOffset], extraParams), extraParams);
                    }

                    auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                    z[zOffset] = postPr(start, tadLen, extraParams);
                }
            }
        }
    }

//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename E>
    FORCEINLINE  void Loops::loopIndexTadXZ(const X* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                                            Nd4jLong* z, const Nd4jLong* zShapeInfo,
                                            E* extraParams,
                                            std::function<functions::indexreduce::IndexValue<X>(X*)> startVal, 
                                            std::function<functions::indexreduce::IndexValue<X>(functions::indexreduce::IndexValue<X>&, functions::indexreduce::IndexValue<X>, E*)> update) {

        const LoopKind kindOfLoop = Loops::deduceKindOfLoopTadXZ(tadShapeInfo, zShapeInfo);

        const Nd4jLong zLen   = shape::length(zShapeInfo);
        const Nd4jLong tadLen = shape::length(tadShapeInfo);

        const uint tadEws = shape::elementWiseStride(tadShapeInfo);
        const uint zEws   = shape::elementWiseStride(zShapeInfo);

        const Nd4jLong* tadShape  = shape::shapeOf(const_cast<Nd4jLong*>(tadShapeInfo));
        const Nd4jLong* tadStride = shape::stride(const_cast<Nd4jLong*>(tadShapeInfo));

        int tadsPerThread = zLen / TAD_THRESHOLD;
        int numThreads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
        numThreads = nd4j::math::nd4j_min<int>(numThreads, omp_get_max_threads());

        switch (kindOfLoop) {

            //*********************************************//
            case EWS1: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint j = 0; j < tadLen; j++)
                        indexValue = update(indexValue, functions::indexreduce::IndexValue<X>(tad[j], j), extraParams);

                    z[i] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case EWSNONZERO: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint j = 0; j < tadLen; j++)
                        indexValue = update(indexValue, functions::indexreduce::IndexValue<X>(tad[j * tadEws], j), extraParams);

                    z[i * zEws] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case RANK1: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint i0 = 0; i0 < tadLen; ++i0) {
                        functions::indexreduce::IndexValue<X> comp(tad[i0 * tadStride[0]], i0);
                        indexValue = update(indexValue, comp, extraParams);
                    }

                    z[i] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case RANK2: {
                
                Nd4jLong newStride[2];
                shape::updateStrides(2, tadShape, newStride, 'c');

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1];
                            const auto tadIndex  = i0 * newStride[0] + i1;

                            functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                            indexValue = update(indexValue, comp, extraParams);
                        }
                    }
                    z[i] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case RANK3: {

                Nd4jLong newStride[3];
                shape::updateStrides(3, tadShape, newStride, 'c');

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2];
                                const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2;
                                functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                indexValue = update(indexValue, comp, extraParams);
                            }
                        }
                    }
                    z[i] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case RANK4: {

                Nd4jLong newStride[4];
                shape::updateStrides(4, tadShape, newStride, 'c');

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                    const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3];
                                    const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3;
                                    functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                    indexValue = update(indexValue, comp, extraParams);
                                }
                            }
                        }
                    }
                    z[i] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case RANK5: {
                
                Nd4jLong newStride[5];
                shape::updateStrides(5, tadShape, newStride, 'c');

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; ++i) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                    for (uint i4 = 0; i4 < tadShape[4]; ++i4) {
                                        const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3] + i4 * tadStride[4];
                                        const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3 * newStride[3] + i4;
                                        functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                        indexValue = update(indexValue, comp, extraParams);
                                    }
                                }
                            }
                        }
                    }
                    z[i] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case X_EWSNONZERO: {

                uint castZShapeInfo[MAX_RANK];
                const bool canCastZ   = nd4j::DataTypeUtils::castShapeInfo<uint>(zShapeInfo,   castZShapeInfo);

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint j = 0; j < tadLen; j++)
                        indexValue = update(indexValue, functions::indexreduce::IndexValue<X>(tad[j * tadEws], j), extraParams);

                    auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                    z[zOffset] = indexValue.index;
                }
            }
                break;

            //*********************************************//
            case Z_EWSNONZERO: {

                uint castTadShapeInfo[MAX_RANK];
                const bool canCastTad = nd4j::DataTypeUtils::castShapeInfo<uint>(tadShapeInfo, castTadShapeInfo);

                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                        indexValue = update(indexValue, functions::indexreduce::IndexValue<X>(tad[tadOffset], j), extraParams);
                    }
                    z[i * zEws] = indexValue.index;
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

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = startVal(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                        indexValue = update(indexValue, functions::indexreduce::IndexValue<X>(tad[tadOffset], j), extraParams);
                    }

                    auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                    z[zOffset] = indexValue.index;
                }
            }
        }
    }











}


#endif //LIBND4J_LOOPS_H
