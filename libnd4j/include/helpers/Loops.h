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
        static FORCEINLINE LoopKind deduceKindOfLoopTadXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo);
        static void wrapFloatXZ(const int opNum, const void* x, const Nd4jLong* xShapeInfo, void* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, const int* dimsToExclude, const int dimsLen, void* extraParams);

        static void wrapSameXZ(const int opNum, const void* x, const Nd4jLong* xShapeInfo, void* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, const int* dimsToExclude, const int dimsLen, void* extraParams);

        static void wrapLongXZ(const int opNum, const void* x, const Nd4jLong* xShapeInfo, void* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, const int* dimsToExclude, const int dimsLen, void* extraParams);

        static void wrapBoolXZ(const int opNum, const void* x, const Nd4jLong* xShapeInfo, void* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, const int* dimsToExclude, const int dimsLen, void* extraParams);

        //////////////////////////////////////////////////////////////////////////////
        template <typename OpType>
        static void loopTadXZ(const X* x, const Nd4jLong* xShapeInfo, Z* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, const int* dimsToExclude, const int dimsLen, E* extraParams);
    };

    template <typename X>
    class ND4J_EXPORT IndexReductionLoops {
    private:
    public:
        static void wrapXZ(const int opNum, const void* x, const Nd4jLong* xShapeInfo, Nd4jLong* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, void* extraParams);

        //////////////////////////////////////////////////////////////////////////////
        template <typename OpType>
        static void loopIndexTadXZ(const X* x, const Nd4jLong* xShapeInfo, Nd4jLong* z, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, X* extraParams);
    };


    template <typename X, typename Z, typename E>
    class ND4J_EXPORT TransformLoops {
    private:
        //////////////////////////////////////////////////////////////////////////////
        static FORCEINLINE LoopKind deduceKindOfLoopXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo);
    public:

        //////////////////////////////////////////////////////////////////////////////
        template<typename OpType>
        static void loopXZ(const X* x, const Nd4jLong* xShapeInfo, Z* z, const Nd4jLong* zShapeInfo, E* extraParams);
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
LoopKind ReductionLoops<X, Z, E>::deduceKindOfLoopTadXZ(const Nd4jLong* xShapeInfo, const Nd4jLong* zShapeInfo, const Nd4jLong* tadShapeInfo) {

    const int tadRank = shape::rank(tadShapeInfo);

    const Nd4jLong tadEws = shape::elementWiseStride(tadShapeInfo);
    const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

    const char xOrder = shape::order(tadShapeInfo);
    const char zOrder = shape::order(zShapeInfo);

    int temp;
    const bool xVector = shape::isCommonVector(tadShapeInfo, temp);
    const bool zVector = shape::isCommonVector(zShapeInfo, temp);

    if(shape::length(tadShapeInfo) * shape::length(zShapeInfo) <= Environment::getInstance()->elementwiseThreshold() && shape::rank(xShapeInfo) == 2 &&
        tadEws > 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
        return SMALLARR2DX;
    if(tadEws == 1 && zEws == 1 && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
        return EWS1;
    if(tadEws > 0 && zEws > 0   && ((xOrder == zOrder) || ((xVector || xOrder == 'c') && (zVector || zOrder == 'c'))))
        return EWSNONZERO;
    if(tadRank == 1 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK1;
    if(tadRank == 2 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK2;
    if(tadRank == 3 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK3;
    if(tadRank == 4 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK4;
    if(tadRank == 5 && zEws == 1 && (zVector || zOrder == 'c'))
        return RANK5;
    if(tadEws > 0 && (xVector || xOrder == 'c') && zEws == 0)
        return X_EWSNONZERO;
    if(zEws > 0 && (zOrder == 'c' || zVector) && tadEws == 0)
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















}


#endif //LIBND4J_LOOPS_H
