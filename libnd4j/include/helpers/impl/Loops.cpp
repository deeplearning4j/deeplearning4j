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

#ifndef LIBND4J_LOOPS_CPP
#define LIBND4J_LOOPS_CPP

#include <Loops.h>
#include <shape.h>
#include <OmpLaunchHelper.h>
#include <DataTypeUtils.h>


using namespace nd4j;

//////////////////////////////////////////////////////////////////////////////
std::string Loops::deduceKindOfLoopXYZ(const Nd4jLong* xShapeInfo, const Nd4jLong* yShapeInfo, const Nd4jLong* zShapeInfo) {

    const int xRank = shape::rank(xShapeInfo);

    const Nd4jLong xEws = shape::elementWiseStride(xShapeInfo);
    const Nd4jLong yEws = shape::elementWiseStride(yShapeInfo);
    const Nd4jLong zEws = shape::elementWiseStride(zShapeInfo);

    const char xOrder = shape::order(xShapeInfo);
    const char yOrder = shape::order(yShapeInfo);
    const char zOrder = shape::order(zShapeInfo);    

    const bool allShapesSame = shape::shapeEquals(xShapeInfo, yShapeInfo) && shape::shapeEquals(xShapeInfo, zShapeInfo);

    if (xEws == 1 && yEws == 1 && zEws == 1 && xOrder == yOrder && xOrder == zOrder)
        return "allEws1_allOrdersSame";

    if(xEws > 1 && yEws > 1 && zEws > 1) // means all of them are vector-like
        return "allEwsGreater1";

    if(xRank == 1 && allShapesSame)
        return "rank1_allShapesSame";

    if(xRank == 2 && allShapesSame)
        return "rank2_allShapesSame";

    if(xRank == 3 && allShapesSame)
        return "rank3_allShapesSame";

    if(xRank == 4 && allShapesSame)
        return "rank4_allShapesSame";

    if(xRank == 5 && allShapesSame)
        return "rank5_allShapesSame";

    return "";
}

//////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z> 
void Loops::loopXYZ(const std::string& str, 
                            const X* x, const Nd4jLong* xShapeInfo,
                            const Y* y, const Nd4jLong* yShapeInfo,
                            const Z* z, const Nd4jLong* zShapeInfo,
                            const Z* extraParams,
                            std::function<Z(X,Y,Z*)> op) {

    const Nd4jLong len = shape::length(xShapeInfo);

    OmpLaunchHelper thredsInfo(len);

    //*********************************************//
    if(str == "allEws1_allOrdersSame") {

        PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
        {
            const auto threadNum = omp_get_thread_num();
            const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
            const auto ulen = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));
            
            PRAGMA_OMP_SIMD
            for (unsigned int i = 0; i < ulen; i++)
                z[i] = op(x[i], y[i], extraParams);
        }
    }
    //*********************************************//
    else if (str == "allEwsGreater1") {

        const uint xEws = shape::elementWiseStride(xShapeInfo);
        const uint yEws = shape::elementWiseStride(yShapeInfo);
        const uint zEws = shape::elementWiseStride(yShapeInfo);

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
    //*********************************************//
    else if (str == "rank1_allShapesSame") {

        const auto xStride0 = shape::stride(xShapeInfo)[0];
        const auto yStride0 = shape::stride(yShapeInfo)[0];
        const auto zStride0 = shape::stride(zShapeInfo)[0];

        PRAGMA_OMP_PARALLEL_FOR
        for (uint i0 = 0; i0 < len; ++i0) 
            z[i0 * zStride0] = op(x[i0 * xStride0], y[i0 * yStride0], extraParams);
    }
    //*********************************************//
    else if (str == "rank2_allShapesSame") {

        const auto xStride0 = shape::stride(xShapeInfo)[0];
        const auto xStride1 = shape::stride(xShapeInfo)[1];
        const auto yStride0 = shape::stride(yShapeInfo)[0];
        const auto yStride1 = shape::stride(yShapeInfo)[1];
        const auto zStride0 = shape::stride(zShapeInfo)[0];
        const auto zStride1 = shape::stride(zShapeInfo)[1];

        PRAGMA_OMP_PARALLEL_FOR_SIMD
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1) 
                z[i0 * zStride0 + i1 * zStride1] = op(x[i0 * xStride0 + i1 * xStride1], y[i0 * yStride0 + i1 * yStride1], extraParams);
    }
    //*********************************************//
    else if (str == "rank3_allShapesSame") {

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
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1)
                for (int i2 = 0; i2 < xShapeInfo[3]; ++i2)
                    z[i0*zStride0+i1*zStride1+i2*zStride2] = op(x[i0*xStride0+i1*xStride1+i2*xStride2], x[i0*yStride0+i1*yStride1+i2*yStride2], extraParams);
    }
    //*********************************************//
    else if (str == "rank4_allShapesSame") {

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
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1)
                for (int i2 = 0; i2 < xShapeInfo[3]; ++i2)
                    for (int i3 = 0; i3 < xShapeInfo[4]; ++i3)
                        z[i0*zStride0+i1*zStride1+i2*zStride2+i3*zStride3] = op(x[i0*xStride0+i1*xStride1+i2*xStride2+i3*xStride3], y[i0*yStride0+i1*yStride1+i2*yStride2+i3*yStride3], extraParams);
    }
    //*********************************************//
    else if (str == "rank4_allShapesSame") {
        
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
        for (int i0 = 0; i0 < xShapeInfo[1]; ++i0) 
            for (int i1 = 0; i1 < xShapeInfo[2]; ++i1)
                for (int i2 = 0; i2 < xShapeInfo[3]; ++i2)
                    for (int i3 = 0; i3 < xShapeInfo[4]; ++i3)
                        for (int i4 = 0; i4 < xShapeInfo[5]; ++i4)
                            z[i0*zStride0+i1*zStride1+i2*zStride2+i3*zStride3 + i4*zStride4] = op(x[i0*xStride0+i1*xStride1+i2*xStride2+i3*xStride3+i4*xStride4], y[i0*yStride0+i1*yStride1+i2*yStride2+i3*yStride3+i4*yStride4], extraParams);
    }
    //*********************************************//
    else {

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
                auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, len, canCastX);
                auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, len, canCastY);
                auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, len, canCastZ);
                z[zOffset] = op(x[xOffset], y[yOffset], extraParams);
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z> 
void Loops::runLoopXYZ(const X* x, const Nd4jLong* xShapeInfo,
                       const Y* y, const Nd4jLong* yShapeInfo,
                       const Z* z, const Nd4jLong* zShapeInfo,
                       const Z* extraParams,
                       std::function<Z(X,Y,Z*)> op) {

    const std::string kindOfLoop = Loops::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);

    Loops::loopXYZ<X,Y,Z>(kindOfLoop, x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, op);
}




#endif // LIBND4J_LOOPS_CPP
