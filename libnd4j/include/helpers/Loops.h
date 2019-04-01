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
    enum LoopKind {SMALLARR2DX, EWS1, EWSNONZERO, RANK1, RANK2, RANK3, RANK4, RANK5, X_EWSNONZERO, Z_EWSNONZERO, COMMON};


    template <typename X, typename Z, typename E>
    class ND4J_EXPORT ReductionLoops {
    protected:
    public:
        //////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE LoopKind deduceKindOfLoopTadXZ(Nd4jLong* xShapeInfo, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo);
        //////////////////////////////////////////////////////////////////////////////
        template <typename OpType>
        static FORCEINLINE void loopTadXZ(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, E* extraParams);
    };

    template <typename X, typename Z>
    class ReductionFloatLoops : public ReductionLoops<X,Z,Z> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, Z* extraParams);

        template <typename OpType>
        static void innerloopTadXZ(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, Z* extraParams);
    };

    template <typename X, typename Z>
    class ND4J_EXPORT ReductionBoolLoops : public ReductionLoops<X,Z,X> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);

        template <typename OpType>
        static void innerloopTadXZ(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };

    template <typename X, typename Z>
    class ND4J_EXPORT ReductionLongLoops : public ReductionLoops<X,Z,X> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);

        template <typename OpType>
        static void innerloopTadXZ(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };

    template <typename X>
    class ND4J_EXPORT ReductionSameLoops : public ReductionLoops<X,X,X> {
    public:
        static void wrapper(const int opNum, X* x, Nd4jLong* xShapeInfo, X* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);

        template <typename OpType>
        static void innerloopTadXZ(X* x, Nd4jLong* xShapeInfo, X* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };


    template <typename X>
    class ND4J_EXPORT IndexReductionLoops {
    private:
    public:
        static void wrapXZ(const int opNum, void* x, Nd4jLong* xShapeInfo, Nd4jLong* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, void* extraParams);

        //////////////////////////////////////////////////////////////////////////////
        template <typename OpType>
        static void loopIndexTadXZ(X* x, Nd4jLong* xShapeInfo, Nd4jLong* z, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets, X* extraParams);
    };


    template <typename X, typename Z, typename E>
    class ND4J_EXPORT TransformLoops {
    private:
        //////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE LoopKind deduceKindOfLoopXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo);
    public:

        //////////////////////////////////////////////////////////////////////////////
        template<typename OpType, bool doParallel>
        static FORCEINLINE void loopXZ(X* x, Nd4jLong* xShapeInfo, Z* z, Nd4jLong* zShapeInfo, E* extraParams);
    };

    /*
class ND4J_EXPORT Loops {

    private:

        //////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE LoopKind deduceKindOfLoopXYZ(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo);



    public:
        //////////////////////////////////////////////////////////////////////////////
        template<typename X, typename Y, typename Z> 
        static void loopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                            const Y* y, const Nd4jLong* yShapeInfo,
                                  Z* z, const Nd4jLong* zShapeInfo,
                                  Z* extraParams,
                            std::function<Z(X,Y,Z*)> op);


};
*/

/*
//////////////////////////////////////////////////////////////////////////////
Loops::LoopKind Loops::deduceKindOfLoopXYZ(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo) {
    
    const int xRank = shape::rank(xShapeInfo);
    
    const Nd4jLong xEws = shape::elementWiseStride(xShapeInfo);
    const Nd4jLong yEws = shape::elementWiseStride(yShapeInfo);
    const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);
    
    int temp;
    const bool xVector = shape::isCommonVector(xShapeInfo, temp);
    const bool yVector = shape::isCommonVector(yShapeInfo, temp);
    const bool zVector = shape::isCommonVector(zShapeInfo, temp);
    
    const char xOrder = shape::order(xShapeInfo);
    const char yOrder = shape::order(yShapeInfo);
    const char zOrder = shape::order(zShapeInfo);
    
    const bool shapesSame = shape::shapeEquals(xShapeInfo, yShapeInfo) && shape::shapeEquals(xShapeInfo, zShapeInfo);
    
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
*/

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E>
LoopKind TransformLoops<X, Z, E>::deduceKindOfLoopXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo) {

    const int xRank = shape::rank(xShapeInfo);
    const Nd4jLong xEws = shape::elementWiseStride(xShapeInfo);
    const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

    int temp;
    const bool xVector = shape::isCommonVector(xShapeInfo, temp);
    const bool zVector = shape::isCommonVector(zShapeInfo, temp);

    const char xOrder = shape::order(xShapeInfo);
    const char zOrder = shape::order(zShapeInfo);

    const bool shapesSame = shape::shapeEquals(xShapeInfo, zShapeInfo);

    if (xEws == 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
        return EWS1;
    if(xEws > 0 && zEws > 0 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
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
    if(xEws > 0 && (xVector || xOrder == 'c'))
        return X_EWSNONZERO;
    if(zEws > 0 && (zVector || zOrder == 'c'))
        return Z_EWSNONZERO;
    return COMMON;
}

//////////////////////////////////////////////////////////////////////////////
template <typename X, typename Z, typename E>
LoopKind ReductionLoops<X, Z, E>::deduceKindOfLoopTadXZ(Nd4jLong* xShapeInfo, Nd4jLong* zShapeInfo, Nd4jLong* tadShapeInfo) {

    const int xRank = shape::rank(xShapeInfo);
    const int tRank = shape::rank(tadShapeInfo);

    const Nd4jLong xEws = shape::elementWiseStride(xShapeInfo);
    const Nd4jLong tEws = shape::elementWiseStride(tadShapeInfo);
    const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

    const char xOrder = shape::order(xShapeInfo);
    const char tOrder = shape::order(tadShapeInfo);
    const char zOrder = shape::order(zShapeInfo);

    int temp;
    const bool tVector = shape::isCommonVector(tadShapeInfo, temp);
    const bool zVector = shape::isCommonVector(zShapeInfo, temp);

    if(shape::length(tadShapeInfo) * shape::length(zShapeInfo) <= Environment::getInstance()->elementwiseThreshold() && shape::rank(xShapeInfo) == 2 && xEws == 1 && xOrder == 'c' && xRank == 2 &&
        tEws > 1 && zEws == 1 && ((tOrder == zOrder) || ((tVector || tOrder == 'c') && (zVector || zOrder == 'c'))))
        return SMALLARR2DX;
    if(tEws == 1 && zEws == 1 && ((tOrder == zOrder) || ((tVector || tOrder == 'c') && (zVector || zOrder == 'c'))))
        return EWS1;
    if(tEws > 0 && zEws > 0   && ((tOrder == zOrder) || ((tVector || tOrder == 'c') && (zVector || zOrder == 'c'))))
        return EWSNONZERO;
    if(tRank == 1 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK1;
    if(tRank == 2 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK2;
    if(tRank == 3 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK3;
    if(tRank == 4 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK4;
    if(tRank == 5 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK5;
    if(tEws > 0 && (tVector || tOrder == 'c') && zEws == 0)
        return X_EWSNONZERO;
    if(zEws > 0 && (zOrder == 'c' || zVector) && tEws == 0)
        return Z_EWSNONZERO;
    return COMMON;
}
/*
//////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
void Loops::loopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                    const Y* y, const Nd4jLong* yShapeInfo,
                          Z* z, const Nd4jLong* zShapeInfo,
                          Z* extraParams,
                          std::function<Z(X,Y,Z*)> op) {

    const LoopKind kindOfLoop = Loops::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);

    const Nd4jLong* xShape  = shape::shapeOf(xShapeInfo);
    const Nd4jLong* xStride = shape::stride(xShapeInfo);
    const Nd4jLong* yStride = shape::stride(yShapeInfo);
    const Nd4jLong* zStride = shape::stride(zShapeInfo);

    const Nd4jLong len = shape::length(xShapeInfo);

    OmpLaunchHelper thredsInfo(len);

    switch (kindOfLoop) {

        case EWS1: {
            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                const auto xi = x + threadOffset;
                const auto yi = y + threadOffset;
                          auto zi = z + threadOffset;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i] = op(xi[i], yi[i], extraParams);
            }
        }
            break;

        case EWSNONZERO: {
            const uint xEws = shape::elementWiseStride(xShapeInfo);
            const uint yEws = shape::elementWiseStride(yShapeInfo);
            const uint zEws = shape::elementWiseStride(zShapeInfo);

            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));
                const auto xi = x + threadOffset * xEws;
                const auto yi = y + threadOffset * yEws;
                      auto zi = z + threadOffset * zEws;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i*zEws] = op(xi[i*xEws], yi[i*yEws], extraParams);
            }
        }
            break;

        case RANK1: {
            PRAGMA_OMP_PARALLEL_FOR
            for (uint i0 = 0; i0 < len; ++i0)
                z[i0 * zStride[0]] = op(x[i0 * xStride[0]], y[i0 * yStride[0]], extraParams);
        }
            break;
        
        case RANK2: {
            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    z[i0 * zStride[0] + i1 * zStride[1]] = op(x[i0 * xStride[0] + i1 * xStride[1]], y[i0 * yStride[0] + i1 * yStride[1]], extraParams);
        }
            break;
        
        case RANK3: {
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]] = op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]], y[i0*yStride[0]+i1*yStride[1]+i2*yStride[2]], extraParams);
        }
            break;

        case RANK4: {
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        for (uint i3 = 0; i3 < xShape[3]; ++i3)
                            z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]+i3*zStride[3]] = op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]+i3*xStride[3]], y[i0*yStride[0]+i1*yStride[1]+i2*yStride[2]+i3*yStride[3]], extraParams);
        }
            break;
        
        case RANK5: {
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

            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                auto threadNum = omp_get_thread_num();
                auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));
                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++) {
                    auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);
                    auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, len, canCastY);
                    auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, len, canCastZ);
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
    void nd4j::ReductionLoops<X, Z, E>::loopTadXZ(X* x, Nd4jLong* xShapeInfo,
                                                  Z* z, Nd4jLong* zShapeInfo,
                                                  Nd4jLong* tadShapeInfo, Nd4jLong* tadOffsets,
                                                  E* extraParams) {

        const LoopKind kindOfLoop = deduceKindOfLoopTadXZ(xShapeInfo, zShapeInfo, tadShapeInfo);

        const Nd4jLong zLen   = shape::length(zShapeInfo);
        const Nd4jLong tadLen = shape::length(tadShapeInfo);

        const uint tadEws = shape::elementWiseStride(tadShapeInfo);
        const uint zEws   = shape::elementWiseStride(zShapeInfo);

        const Nd4jLong* tadShape  = shape::shapeOf(tadShapeInfo);
        const Nd4jLong* tadStride = shape::stride(tadShapeInfo);

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



    //////////////////////////////////////////////////////////////////////////////
    template <typename X, typename Z, typename E>
    template <typename OpType, bool doParallel>
    void nd4j::TransformLoops<X,Z,E>::loopXZ(X* x, Nd4jLong* xShapeInfo,
                                             Z* z, Nd4jLong* zShapeInfo,
                                             E* extraParams) {

        const LoopKind kindOfLoop = deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

        const Nd4jLong* xShape  = shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo));
        const Nd4jLong* xStride = shape::stride(const_cast<Nd4jLong*>(xShapeInfo));
        const Nd4jLong* zStride = shape::stride(const_cast<Nd4jLong*>(zShapeInfo));

        const Nd4jLong len = shape::length(xShapeInfo);
        const uint ulen = static_cast<uint>(len);

        OmpLaunchHelper thredsInfo(len, doParallel ? -1 : 1);

        switch (kindOfLoop) {

            //*********************************************//
            case EWS1: {
                if (ulen > Environment::getInstance()->elementwiseThreshold()) {
                    PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(thredsInfo._numThreads)
                    for (uint i = 0; i < ulen; i++)
                        z[i] = OpType::op(x[i], extraParams);
                } else {
                    for (uint i = 0; i < ulen; i++)
                        z[i] = OpType::op(x[i], extraParams);
                }
            }
                break;

                //*********************************************//
            case EWSNONZERO: {
                const uint xEws = shape::elementWiseStride(xShapeInfo);
                const uint zEws = shape::elementWiseStride(zShapeInfo);

                if (ulen > Environment::getInstance()->elementwiseThreshold()) {
                    PRAGMA_OMP_PARALLEL_FOR_SIMD
                    for (uint i = 0; i < ulen; i++)
                        z[i * zEws] = OpType::op(x[i * xEws], extraParams);
                } else {
                    for (uint i = 0; i < ulen; i++)
                        z[i * zEws] = OpType::op(x[i * xEws], extraParams);
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
                PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(thredsInfo._numThreads)
                for (uint i0 = 0; i0 < len; ++i0)
                    z[i0 * zStride[0]] = OpType::op(x[i0 * xStride[0]], extraParams);
            }
                break;

                //*********************************************//
            case RANK2: {
                auto uXShape0 = static_cast<uint>(xShape[0]);
                auto uXShape1 = static_cast<uint>(xShape[1]);

                //PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(thredsInfo._numThreads)
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

                PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS_COLLAPSE(thredsInfo._numThreads, 2)
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

                PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS_COLLAPSE(thredsInfo._numThreads, 2)
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

                PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS_COLLAPSE(thredsInfo._numThreads, 3)
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







}


#endif //LIBND4J_LOOPS_H
