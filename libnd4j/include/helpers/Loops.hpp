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


namespace nd4j {

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

            //*********************************************//
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

                //*********************************************//
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

                //*********************************************//
            case RANK1: {                

                PRAGMA_OMP_PARALLEL_FOR
                for (uint i0 = 0; i0 < len; ++i0)
                    z[i0 * zStride[0]] = op(x[i0 * xStride[0]], y[i0 * yStride[0]], extraParams);
            }
                break;

                //*********************************************//
            case RANK2: {                

                PRAGMA_OMP_PARALLEL_FOR_SIMD
                for (uint i0 = 0; i0 < xShape[0]; ++i0)
                    for (uint i1 = 0; i1 < xShape[1]; ++i1)
                        z[i0 * zStride[0] + i1 * zStride[1]] = op(x[i0 * xStride[0] + i1 * xStride[1]], y[i0 * yStride[0] + i1 * yStride[1]], extraParams);
            }
                break;

                //*********************************************//
            case RANK3: {

                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
                for (uint i0 = 0; i0 < xShape[0]; ++i0)
                    for (uint i1 = 0; i1 < xShape[1]; ++i1)
                        for (uint i2 = 0; i2 < xShape[2]; ++i2)
                            z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]] = op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]], y[i0*yStride[0]+i1*yStride[1]+i2*yStride[2]], extraParams);
            }
                break;

                //*********************************************//
            case RANK4: {

                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
                for (uint i0 = 0; i0 < xShape[0]; ++i0)
                    for (uint i1 = 0; i1 < xShape[1]; ++i1)
                        for (uint i2 = 0; i2 < xShape[2]; ++i2)
                            for (uint i3 = 0; i3 < xShape[3]; ++i3)
                                z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]+i3*zStride[3]] = op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]+i3*xStride[3]], y[i0*yStride[0]+i1*yStride[1]+i2*yStride[2]+i3*yStride[3]], extraParams);
            }
                break;

                //*********************************************//
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


//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename Z, typename E, typename OpType>
     void Loops::loopTadXZ(const X* x, const Nd4jLong* xShapeInfo,
                                 Z* z, const Nd4jLong* zShapeInfo,
                                 const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,
                                 const int* dimsToExclude,
                                 const int dimsLen,
                                 E* extraParams) {

        const LoopKind kindOfLoop = Loops::deduceKindOfLoopTadXZ(xShapeInfo, zShapeInfo, tadShapeInfo);

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

//////////////////////////////////////////////////////////////////////////////
    template<typename X, typename OpType>
    void Loops::loopIndexTadXZ(const X* x, const Nd4jLong* xShapeInfo,
                                Nd4jLong* z, const Nd4jLong* zShapeInfo,
                                const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets,                                            
                                X* extraParams) {

        LoopKind kindOfLoop = Loops::deduceKindOfLoopTadXZ(xShapeInfo, zShapeInfo, tadShapeInfo);
        if(kindOfLoop == SMALLARR2DX)
            kindOfLoop = EWSNONZERO;

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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        functions::indexreduce::IndexValue<X> comp(tad[j], j);
                        indexValue = OpType::update(indexValue, comp, extraParams);
                    }

                    z[i] = indexValue.index;
                }
            }
                break;

                //*********************************************//
            case EWSNONZERO: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        functions::indexreduce::IndexValue<X> comp(tad[j * tadEws], j);
                        indexValue = OpType::update(indexValue, comp, extraParams);
                    }

                    z[i * zEws] = indexValue.index;
                }
            }
                break;

                //*********************************************//
            case RANK1: {
                PRAGMA_OMP_PARALLEL_FOR_THREADS(numThreads)
                for (uint i = 0; i < zLen; i++) {

                    auto tad = const_cast<X*>(x) + tadOffsets[i];
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint i0 = 0; i0 < tadLen; ++i0) {
                        functions::indexreduce::IndexValue<X> comp(tad[i0 * tadStride[0]], i0);
                        indexValue = OpType::update(indexValue, comp, extraParams);
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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1];
                            const auto tadIndex  = i0 * newStride[0] + i1;

                            functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                            indexValue = OpType::update(indexValue, comp, extraParams);
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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2];
                                const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2;
                                functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                indexValue = OpType::update(indexValue, comp, extraParams);
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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                    const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3];
                                    const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3;
                                    functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                    indexValue = OpType::update(indexValue, comp, extraParams);
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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint i0 = 0; i0 < tadShape[0]; ++i0) {
                        for (uint i1 = 0; i1 < tadShape[1]; ++i1) {
                            for (uint i2 = 0; i2 < tadShape[2]; ++i2) {
                                for (uint i3 = 0; i3 < tadShape[3]; ++i3) {
                                    for (uint i4 = 0; i4 < tadShape[4]; ++i4) {
                                        const auto tadOffset = i0 * tadStride[0] + i1 * tadStride[1] + i2 * tadStride[2] + i3 * tadStride[3] + i4 * tadStride[4];
                                        const auto tadIndex  = i0 * newStride[0] + i1 * newStride[1] + i2 * newStride[2] + i3 * newStride[3] + i4;
                                        functions::indexreduce::IndexValue<X> comp(tad[tadOffset], tadIndex);
                                        indexValue = OpType::update(indexValue, comp, extraParams);
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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        functions::indexreduce::IndexValue<X> comp(tad[j * tadEws], j);
                        indexValue = OpType::update(indexValue, comp, extraParams);
                    }

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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                        functions::indexreduce::IndexValue<X> comp(tad[tadOffset], j);
                        indexValue = OpType::update(indexValue, comp, extraParams);
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
                    auto indexValue = OpType::startingIndexValue(tad);

                    for (uint j = 0; j < tadLen; j++) {
                        auto tadOffset = shape::indexOffset(j, tadShapeInfo, castTadShapeInfo, tadLen, canCastTad);
                        functions::indexreduce::IndexValue<X> comp(tad[tadOffset], j);
                        indexValue = OpType::update(indexValue, comp, extraParams);
                    }

                    auto zOffset = shape::indexOffset(i, zShapeInfo, castZShapeInfo, zLen, canCastZ);
                    z[zOffset] = indexValue.index;
                }
            }
        }
    }


//////////////////////////////////////////////////////////////////////////////
template<typename X, typename Z, typename E, typename OpType>
void Loops::loopXZ(const X* x, const Nd4jLong* xShapeInfo,
                         Z* z, const Nd4jLong* zShapeInfo,
                    E* extraParams) {

    const LoopKind kindOfLoop = Loops::deduceKindOfLoopXZ(xShapeInfo, zShapeInfo);

    const Nd4jLong* xShape  = shape::shapeOf(const_cast<Nd4jLong*>(xShapeInfo));
    const Nd4jLong* xStride = shape::stride(const_cast<Nd4jLong*>(xShapeInfo));        
    const Nd4jLong* zStride = shape::stride(const_cast<Nd4jLong*>(zShapeInfo));

    const Nd4jLong len = shape::length(xShapeInfo);

    OmpLaunchHelper thredsInfo(len);

    switch (kindOfLoop) {

        //*********************************************//
        case EWS1: {            
            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));
                
                const auto xi = x + threadOffset;
                const auto zi = z + threadOffset;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i] = OpType::op(xi[i], extraParams);                        
            }
        }
            break;

        //*********************************************//
        case EWSNONZERO: {            
            const uint xEws = shape::elementWiseStride(xShapeInfo);                
            const uint zEws = shape::elementWiseStride(zShapeInfo);

            PRAGMA_OMP_PARALLEL_THREADS(thredsInfo._numThreads)
            {
                const auto threadNum = omp_get_thread_num();
                const auto threadOffset = thredsInfo.getThreadOffset(threadNum);
                const auto lenPerThread = static_cast<uint>(thredsInfo.getItersPerThread(threadNum));

                const auto xi = x + threadOffset * xEws;
                      auto zi = z + threadOffset * zEws;

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++)
                    zi[i*zEws] = OpType::op(xi[i*xEws], extraParams);
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

                PRAGMA_OMP_SIMD
                for (uint i = 0; i < lenPerThread; i++) {
                    const auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, castXShapeInfo, len, canCastX);
                    zi[i*zEws] = OpType::op(x[xOffset], extraParams);
                }
            }
        }
            break;       

        //*********************************************//
        case RANK1: {                            
            PRAGMA_OMP_PARALLEL_FOR
            for (uint i0 = 0; i0 < len; ++i0)
                z[i0 * zStride[0]] = OpType::op(x[i0 * xStride[0]], extraParams);
        }
            break;

        //*********************************************//
        case RANK2: {            
            PRAGMA_OMP_PARALLEL_FOR_SIMD
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    z[i0 * zStride[0] + i1 * zStride[1]] = OpType::op(x[i0 * xStride[0] + i1 * xStride[1]], extraParams);
        }
            break;

        //*********************************************//
        case RANK3: {            
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]] = OpType::op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]], extraParams);
        }
            break;

        //*********************************************//
        case RANK4: {            
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(3)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        for (uint i3 = 0; i3 < xShape[3]; ++i3)
                            z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]+i3*zStride[3]] = OpType::op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]+i3*xStride[3]], extraParams);
        }
            break;

        //*********************************************//
        case RANK5: {            
            PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(4)
            for (uint i0 = 0; i0 < xShape[0]; ++i0)
                for (uint i1 = 0; i1 < xShape[1]; ++i1)
                    for (uint i2 = 0; i2 < xShape[2]; ++i2)
                        for (uint i3 = 0; i3 < xShape[3]; ++i3)
                            for (uint i4 = 0; i4 < xShape[4]; ++i4)
                                z[i0*zStride[0]+i1*zStride[1]+i2*zStride[2]+i3*zStride[3]+i4*zStride[4]] = OpType::op(x[i0*xStride[0]+i1*xStride[1]+i2*xStride[2]+i3*xStride[3]+i4*xStride[4]], extraParams);
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
//template void Loops::loopTadXZ<double, double>(const double* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, double* z, const Nd4jLong* zShapeInfo, double* extraParams, std::function<double(const double*)> startVal, std::function<double(double,double,double*)> update, std::function<double(double,double*)> op, std::function<double(double,Nd4jLong,double*)> postPr);
//template void Loops::loopTadXZ<float, float>(const float* x, const Nd4jLong* tadShapeInfo, const Nd4jLong* tadOffsets, float* z, const Nd4jLong* zShapeInfo, float* extraParams, std::function<float(const float*)> startVal, std::function<float(float,float,float*)> update, std::function<float(float,float*)> op, std::function<float(float,Nd4jLong,float*)> postPr);

#endif // LIBND4J_LOOPS_CPP

