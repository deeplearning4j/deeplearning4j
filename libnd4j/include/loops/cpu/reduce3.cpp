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

// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.11.2018


#include <types/types.h>
#include <op_boilerplate.h>
#include <loops/reduce3.h>
#include <loops/legacy_ops.h>
#include <helpers/ConstantTadHelper.h>

using namespace simdOps;

namespace functions {
namespace reduce3   {

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::execScalar(void *vx, Nd4jLong *xShapeInfo,
                                    void *vextraParams,
                                    void *vy, Nd4jLong *yShapeInfo,
                                    void *vz, Nd4jLong *zShapeInfo) {
    
    auto x = reinterpret_cast<X *>(vx);
    auto y = reinterpret_cast<X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    auto length = shape::length(xShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);
    auto xOrder = shape::order(xShapeInfo);
    auto yOrder = shape::order(yShapeInfo);

    Z extraParamsVals[3] = {(Z) 0.0f, (Z) 0.0f, (Z) 0.0f};
    // it's possible case for EqualsWithEps op
    if (extraParams != nullptr) 
        extraParamsVals[2] = extraParams[0];                
    
    uint xShapeInfoCast[MAX_RANK];
    const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

    Z startingVal = OpType::startingValue(x);
    const int maxThreads = nd4j::math::nd4j_min<int>(256, omp_get_max_threads());
    nd4j::OmpLaunchHelper t(length, maxThreads);
    Z intermediate[256];

    for (int e = 0; e < maxThreads; e++)
        intermediate[e] = startingVal;

    if (xEws == 1 && yEws == 1 && xOrder == yOrder) {
        PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(t._numThreads)
        for(unsigned int i = 0; i < length; i++)
            intermediate[omp_get_thread_num()] = OpType::update(intermediate[omp_get_thread_num()], OpType::op(x[i], y[i], extraParamsVals), extraParamsVals);

    } else if(shape::haveSameOffsets(xShapeInfo, yShapeInfo)) {

        PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(t._numThreads)
        for(unsigned int i = 0; i < length; i++) {            
            auto offset  = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, length, canCastX);
            intermediate[omp_get_thread_num()] = OpType::update(intermediate[omp_get_thread_num()], OpType::op(x[offset], y[offset], extraParamsVals), extraParamsVals);
        }
    } else {
        uint yShapeInfoCast[MAX_RANK];
        const bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

        PRAGMA_OMP_PARALLEL_FOR_SIMD_THREADS(t._numThreads)
        for(unsigned int i = 0; i < length; i++) {            
            auto xOffset  = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, length, canCastX);
            auto yOffset  = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, length, canCastY);
            intermediate[omp_get_thread_num()] = OpType::update(intermediate[omp_get_thread_num()], OpType::op(x[xOffset], y[yOffset], extraParamsVals), extraParamsVals);
        }
    }

    // merge step
    for (int e = 0; e < maxThreads; e++)
        startingVal = OpType::update(startingVal, intermediate[e], extraParams);

    z[0] = OpType::postProcess(startingVal, length, extraParamsVals);;
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::execScalar(const int opNum,
                                        void *vx, Nd4jLong *xShapeInfo,
                                        void *extraParamsVals,
                                        void *vy, Nd4jLong *yShapeInfo,
                                        void *vz, Nd4jLong *zShapeInfo) {
                
    DISPATCH_BY_OPNUM_TT(execScalar, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::exec(void *vx, Nd4jLong *xShapeInfo,
                    void *vextraParams,
                    void *vy, Nd4jLong *yShapeInfo,
                    void *vz, Nd4jLong *zShapeInfo,
                    int *dimension, int dimensionLength) {

    auto x = reinterpret_cast<X *>(vx);
    auto y = reinterpret_cast<X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);
    
    Z extraParamsVals[3] = {(Z) 0.0f, (Z) 0.0f, (Z) 0.0f};

    if(shape::isScalar(zShapeInfo)) {
        execScalar<OpType>(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo);
        return;
    }
    
    char xOrder = shape::order(xShapeInfo);
    char yOrder = shape::order(yShapeInfo);
    auto zLen = shape::length(zShapeInfo);
    auto tadLength = shape::tadLength(xShapeInfo,dimension,dimensionLength);

    nd4j::OmpLaunchHelper info(zLen);
    
    if(xOrder != yOrder) {
         
         if(shape::haveSameOffsets(xShapeInfo, yShapeInfo) && shape::haveSameOffsets(xShapeInfo, zShapeInfo)) {

            uint xShapeInfoCast[MAX_RANK];
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

             PRAGMA_OMP_PARALLEL_ARGS(num_threads(info._numThreads) private(extraParamsVals))
            {                
                auto threadNum = omp_get_thread_num();         
                auto threadOffset = info.getThreadOffset(threadNum);
                auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                for (Nd4jLong i = 0; i < ulen; i++) {
                    auto offset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, zLen, canCastX);
                    z[offset] = OpType::update(z[offset], OpType::op(x[offset], y[offset], extraParamsVals), extraParamsVals);
                }
            }
        }
        else if(shape::haveSameOffsets(xShapeInfo, yShapeInfo)) {

            uint xShapeInfoCast[MAX_RANK];
            uint zShapeInfoCast[MAX_RANK];        
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
            const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

            PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
            {                
                auto threadNum = omp_get_thread_num();         
                auto threadOffset = info.getThreadOffset(threadNum);
                auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                for (Nd4jLong i = 0; i < ulen; i++) {
                    auto offset  = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, zLen, canCastX);
                    auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, zLen, canCastZ);
                    z[zOffset] = OpType::update(z[zOffset], OpType::op(x[offset], y[offset], extraParamsVals), extraParamsVals);
                }
            }       
        }
        else if(shape::haveSameOffsets(xShapeInfo, zShapeInfo)) {

            uint xShapeInfoCast[MAX_RANK];
            uint yShapeInfoCast[MAX_RANK];
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
            const bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

             PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
            {                
                auto threadNum = omp_get_thread_num();         
                auto threadOffset = info.getThreadOffset(threadNum);
                auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                PRAGMA_OMP_SIMD
                for (Nd4jLong i = 0; i < ulen; i++) {
                    auto offset  = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, zLen, canCastX);
                    auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, zLen, canCastY);
                    z[offset] = OpType::update(z[offset], OpType::op(x[offset], y[yOffset], extraParamsVals), extraParamsVals);
                }
            }
        }
        else if(shape::haveSameOffsets(yShapeInfo, zShapeInfo)) {

            uint xShapeInfoCast[MAX_RANK];
            uint yShapeInfoCast[MAX_RANK];
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
            const bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

             PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
            {                
                auto threadNum = omp_get_thread_num();         
                auto threadOffset = info.getThreadOffset(threadNum);
                auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                PRAGMA_OMP_SIMD
                for (Nd4jLong i = 0; i < ulen; i++) {
                    auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, zLen, canCastX);
                    auto offset  = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, zLen, canCastY);
                    z[offset] = OpType::update(z[offset], OpType::op(x[xOffset], y[offset], extraParamsVals), extraParamsVals);
                }
            }
        }
        else {

            uint xShapeInfoCast[MAX_RANK];
            uint yShapeInfoCast[MAX_RANK];
            uint zShapeInfoCast[MAX_RANK];
            const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
            const bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);
            const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

             PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
            {                
                auto threadNum = omp_get_thread_num();         
                auto threadOffset = info.getThreadOffset(threadNum);
                auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                PRAGMA_OMP_SIMD
                for (Nd4jLong i = 0; i < ulen; i++) {
                    auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, zLen, canCastX);
                    auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, zLen, canCastY);
                    auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, zLen, canCastZ);
                    z[zOffset] = OpType::update(z[zOffset], OpType::op(x[xOffset], y[yOffset], extraParamsVals), extraParamsVals);
                }
            }   
        }

        auto zEws = shape::elementWiseStride(zShapeInfo);
        PRAGMA_OMP_PARALLEL_FOR
        for(Nd4jLong i = 0; i < zLen; i+=zEws) 
            z[i] = OpType::postProcess(z[i], tadLength, extraParamsVals);
    }
    else {
        
        auto startingVal = OpType::startingValue(x);

        auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);
        auto tadPackY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dimension, dimensionLength);

        /**
        * The element wise stride belong longs to a reduction index.
        * When used out of order, we can get rid of the data
        * dependencies and rely on using the max dimension
        * specified for stride instead.
        * Say we take the sum(0,1) along long arr
        * we can use arr.stride(1) as a representation
        * along long which to iterate.
        */
        int largerElementWiseStride;
        int smallerElementWiseStride;
        auto xEws = shape::elementWiseStride(tadPackX.primaryShapeInfo());
        auto yEws = shape::elementWiseStride(tadPackY.primaryShapeInfo());
        int tadLength;
        Nd4jLong xModLength;
        Nd4jLong yModLength;
        Nd4jLong *iterationTadInfo;
        bool xTadBigger;
        
        if(shape::length(xShapeInfo) > shape::length(yShapeInfo)) {
            tadLength = shape::length(tadPackX.primaryShapeInfo());
            iterationTadInfo = tadPackX.primaryShapeInfo();
            largerElementWiseStride = shape::elementWiseStride(xShapeInfo);
            smallerElementWiseStride = shape::elementWiseStride(yShapeInfo);
            xModLength = 1;
            yModLength = tadLength;
            xTadBigger = true;
        }
        else {
            tadLength = shape::length(tadPackY.primaryShapeInfo());
            iterationTadInfo = tadPackY.primaryShapeInfo();
            largerElementWiseStride = shape::elementWiseStride(yShapeInfo);
            smallerElementWiseStride = shape::elementWiseStride(xShapeInfo);
            xModLength = tadLength;
            yModLength = 1;
            xTadBigger = false;
        }
        
        if (largerElementWiseStride >= 1 && smallerElementWiseStride >= 1 && xEws >= 1 && yEws >= 1) {

            if(shape::length(xShapeInfo) == shape::length(yShapeInfo)) {

                PRAGMA_OMP_PARALLEL_FOR
                for (Nd4jLong i = 0; i < zLen; i++) {
                    
                    Z *localExtraParams = nullptr;
                    
                    if (OpType::extraParamsLen > 0)
                        localExtraParams = new Z[OpType::extraParamsLen];
                    
                    for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
                        localExtraParams[extraParamsIdx] = startingVal;
                                
                    auto offset = tadPackX.primaryOffsets()[i];
                    auto yOffset = tadPackY.primaryOffsets()[i];
                    auto sv = OpType::op(x[offset], y[yOffset], localExtraParams);

                    for (int j = 1; j < tadLength; j++) {
                        auto xIdx = (offset + xEws * j);
                        auto yIdx = (yOffset + yEws * j);
                        sv = OpType::update(sv, OpType::op(x[xIdx],y[yIdx],localExtraParams), localExtraParams);
                    }

                    z[i] = OpType::postProcess(sv, tadLength, localExtraParams);

                    if (localExtraParams != nullptr)
                        delete[] localExtraParams;
                }
            }
            else {
                
                int tadsPerThread = zLen / TAD_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                auto xShapeInf = xTadBigger ? tadPackX.primaryShapeInfo() : xShapeInfo;
                auto yShapeInf = !xTadBigger ? tadPackY.primaryShapeInfo() : yShapeInfo;

                uint yShapeInfoCast[MAX_RANK];
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInf, yShapeInfoCast);

                uint xShapeInfoCast[MAX_RANK];
                bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInf, xShapeInfoCast);

                PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(num_threads(num_threads) if (num_threads>1) private(extraParamsVals))
                for (int i = 0; i < zLen; i++) {

                    extraParamsVals[0] = (Z) 0.f;
                    extraParamsVals[1] = (Z) 0.f;
                    extraParamsVals[2] = (Z) 0.f;

                    Nd4jLong xOffset = xTadBigger ? tadPackX.primaryOffsets()[i] : 0;
                    Nd4jLong yOffset = !xTadBigger ? tadPackY.primaryOffsets()[i] : 0;

                    auto start = OpType::startingValue(x);

                    auto tX = x + xOffset;
                    auto tY = y + yOffset;

                    if (xEws == 1 && yEws == 1) {

                        for (unsigned int j = 0; j < tadLength; j++)
                            start = OpType::update(start, OpType::op(tX[j], tY[j], extraParamsVals), extraParamsVals);

                    } else if(shape::haveSameOffsets(xShapeInf, yShapeInf)) {

                        for (unsigned int j = 0; j < tadLength; j++) {                            
                            auto offset = shape::indexOffset(j, xShapeInf, xShapeInfoCast, tadLength, canCastX);
                            start = OpType::update(start, OpType::op(tX[offset], tY[offset],extraParamsVals), extraParamsVals);
                        }

                    }
                    else {

                        for (unsigned int j = 0; j < tadLength; j++) {                            
                            auto xOffset2 = shape::indexOffset(j, xShapeInf, xShapeInfoCast, tadLength, canCastX);
                            auto yOffset2 = shape::indexOffset(j, yShapeInf, yShapeInfoCast, tadLength, canCastY);
                            start = OpType::update(start, OpType::op(tX[xOffset2], tY[yOffset2],extraParamsVals), extraParamsVals);
                        }

                    } 

                    z[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                }   
            }
        } 
        else {

            auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(xShapeInfo, dimension, dimensionLength);
            
            int tadsPerThread = zLen / TAD_THRESHOLD;
            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

            uint xShapeInfoCast[MAX_RANK];            
            bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadPackX.primaryShapeInfo(), xShapeInfoCast);

            if(shape::haveSameOffsets(xShapeInfo, yShapeInfo)) {

                PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(num_threads(num_threads) if (num_threads>1) private(extraParamsVals))
                for (unsigned int i = 0; i < zLen; i++) {

                    extraParamsVals[0] = (Z) 0.f;
                    extraParamsVals[1] = (Z) 0.f;
                    extraParamsVals[2] = (Z) 0.f;
                
                    auto offset = tadPackX.primaryOffsets()[i];
                    auto start = OpType::startingValue(x + offset);

                    auto tX = x + offset;
                    auto tY = y + offset;

                    for (unsigned int j = 0; j < tadLength; j++) {
                        auto offset = shape::indexOffset(j, tadPackX.primaryShapeInfo(), xShapeInfoCast, tadLength, canCastX);
                        start = OpType::update(start, OpType::op(tX[offset], tY[offset], extraParamsVals), extraParamsVals);
                    }

                    z[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                }                
            }
            else {

                auto tadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(yShapeInfo, dimension, dimensionLength);
                
                uint yShapeInfoCast[MAX_RANK];
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(tadPackY.primaryShapeInfo(), yShapeInfoCast);

                PRAGMA_OMP_PARALLEL_FOR_SIMD_ARGS(num_threads(num_threads) if (num_threads>1) private(extraParamsVals))
                for (unsigned int i = 0; i < zLen; i++) {

                    extraParamsVals[0] = (Z) 0.f;
                    extraParamsVals[1] = (Z) 0.f;
                    extraParamsVals[2] = (Z) 0.f;
                
                    auto xOffset = tadPackX.primaryOffsets()[i];
                    auto yOffset = tadPackY.primaryOffsets()[i];
                    auto start = OpType::startingValue(x + xOffset);

                    auto tX = x + xOffset;
                    auto tY = y + yOffset;

                    for (unsigned int j = 0; j < tadLength; j++) {
                        auto xOffset2 = shape::indexOffset(j, tadPackX.primaryShapeInfo(), xShapeInfoCast, tadLength, canCastX);
                        auto yOffset2 = shape::indexOffset(j, tadPackY.primaryShapeInfo(), yShapeInfoCast, tadLength, canCastY);
                        start = OpType::update(start, OpType::op(tX[xOffset2], tY[yOffset2], extraParamsVals), extraParamsVals);
                    }

                    z[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>::exec(void *vx, Nd4jLong *xShapeInfo,
                        void *vextraParams,
                        void *vy, Nd4jLong *yShapeInfo,
                        void *vz, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    auto x = reinterpret_cast<X *>(vx);
    auto y = reinterpret_cast<X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<Z *>(vextraParams);
    
    auto startingVal = OpType::startingValue(x);

    auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    auto tads = shape::length(xShapeInfo) / tadLength;

    uint tadShapeInfoCast[MAX_RANK];
    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(tadShapeInfo, tadShapeInfoCast);

    if(shape::haveSameOffsets(tadShapeInfo, yShapeInfo)) {

        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong r = 0; r < tads; r++) {
            
            Nd4jLong offset = tadOffsets[r];
            Z *localExtraParams = nullptr;
            auto sv = OpType::startingValue(x);

            if (OpType::extraParamsLen > 0)
                localExtraParams = new Z[OpType::extraParamsLen];

            for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
                localExtraParams[extraParamsIdx] = startingVal;

            for (Nd4jLong f = 0; f < tadLength; f++) {
                auto yOffset = shape::indexOffset(f, tadShapeInfo, tadShapeInfoCast, tadLength, canCastX);
                auto xOffset = offset + yOffset;                
                sv = OpType::update(sv, OpType::op(x[xOffset], y[yOffset], localExtraParams), localExtraParams);
            }

            z[r] = OpType::postProcess(sv, tadLength, localExtraParams);

            if (localExtraParams != nullptr)
                delete[] localExtraParams;
        }        
    }
    else {

        uint yShapeInfoCast[MAX_RANK];
        bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong r = 0; r < tads; r++) {
            
            Nd4jLong offset = tadOffsets[r];
            Z *localExtraParams = nullptr;
            auto sv = OpType::startingValue(x);

            if (OpType::extraParamsLen > 0)
                localExtraParams = new Z[OpType::extraParamsLen];

            for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
                localExtraParams[extraParamsIdx] = startingVal;

            for (Nd4jLong f = 0; f < tadLength; f++) {
                auto xOffset = offset + shape::indexOffset(f, tadShapeInfo, tadShapeInfoCast, tadLength, canCastX);
                auto yOffset = shape::indexOffset(f, yShapeInfo, yShapeInfoCast, tadLength, canCastY);
                sv = OpType::update(sv, OpType::op(x[xOffset], y[yOffset], localExtraParams), localExtraParams);
            }

            z[r] = OpType::postProcess(sv, tadLength, localExtraParams);

            if (localExtraParams != nullptr)
                delete[] localExtraParams;
        }
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Z>
template<typename OpType>
void Reduce3<X,Z>:: execAll(void *vx, Nd4jLong *xShapeInfo,
                            void *vextraParams,
                            void *vy, Nd4jLong *yShapeInfo,
                            void *vz, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength, 
                            Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets, 
                            Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

    auto x = reinterpret_cast<X *>(vx);
    auto y = reinterpret_cast<X *>(vy);
    auto z = reinterpret_cast<Z *>(vz);
    auto extraParams = reinterpret_cast<X *>(vextraParams);

    auto xTadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    auto yTadLength = shape::tadLength(yShapeInfo, dimension, dimensionLength);

    auto xTads = shape::length(xShapeInfo) / xTadLength;
    auto yTads = shape::length(yShapeInfo) / yTadLength;
    auto startingVal = OpType::startingValue(x);

    uint xTadShapeInfoCast[MAX_RANK];
    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xTadShapeInfo, xTadShapeInfoCast);
    
    if (shape::haveSameOffsets(xTadShapeInfo, yTadShapeInfo) ) {
        
        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong r = 0; r < xTads; r++) {
        
            Nd4jLong xOffset = xOffsets[r];
            auto lX = x + xOffset;

            for (Nd4jLong g = 0; g < yTads; g++) {
            
                auto yOffset = yOffsets[g];
                auto lY = y + yOffset;
                auto ri = (r * yTads) + g;
                auto sv = OpType::startingValue(x);

                Z *localExtraParams = nullptr;
                if (OpType::extraParamsLen > 0)
                    localExtraParams = new Z[OpType::extraParamsLen];

                for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
                    localExtraParams[extraParamsIdx] = startingVal;

                for (int f = 0; f < xTadLength; f++) {                            
                    auto offset = shape::indexOffset(f, xTadShapeInfo, xTadShapeInfoCast, xTadLength, canCastX);                    
                    sv = OpType::update(sv, OpType::op(lX[offset], lY[offset], localExtraParams), localExtraParams);
                }

                z[ri] = OpType::postProcess(sv, xTadLength, localExtraParams);

                if (localExtraParams != nullptr)
                    delete[] localExtraParams;
            }
        }
    }
    else {

        uint yTadShapeInfoCast[MAX_RANK];
        bool canCastY = canCastX ? nd4j::DataTypeUtils::castShapeInfo(yTadShapeInfo, yTadShapeInfoCast) : false;
        
        PRAGMA_OMP_PARALLEL_FOR
        for (Nd4jLong r = 0; r < xTads; r++) {
        
            Nd4jLong xOffset = xOffsets[r];
            auto lX = x + xOffset;

            for (Nd4jLong g = 0; g < yTads; g++) {
            
                auto yOffset = yOffsets[g];
                auto lY = y + yOffset;
                auto ri = (r * yTads) + g;
                auto sv = OpType::startingValue(x);

                Z *localExtraParams = nullptr;
                if (OpType::extraParamsLen > 0)
                    localExtraParams = new Z[OpType::extraParamsLen];

                for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
                    localExtraParams[extraParamsIdx] = startingVal;

                for (int f = 0; f < xTadLength; f++) {
                    auto xO = shape::indexOffset(f, yTadShapeInfo, xTadShapeInfoCast, xTadLength, canCastX);
                    auto yO = shape::indexOffset(f, yTadShapeInfo, yTadShapeInfoCast, xTadLength, canCastY);
                    sv = OpType::update(sv, OpType::op(lX[xO], lY[yO], localExtraParams), localExtraParams);
                }

                z[ri] = OpType::postProcess(sv, xTadLength, localExtraParams);

                if (localExtraParams != nullptr)
                    delete[] localExtraParams;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::exec( const int opNum,
                        void *vx, Nd4jLong *xShapeInfo,
                        void *extraParamsVals,
                        void *vy, Nd4jLong *yShapeInfo,
                        void *vz, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength) {

    DISPATCH_BY_OPNUM_TT(exec, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension, dimensionLength), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::exec( const int opNum,
                        void *vx, Nd4jLong *xShapeInfo,
                        void *extraParamsVals,
                        void *vy, Nd4jLong *yShapeInfo,
                        void *vz, Nd4jLong *zShapeInfo,
                        int *dimension, int dimensionLength,
                        Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

    DISPATCH_BY_OPNUM_TT(exec, PARAMS(vx,xShapeInfo,extraParamsVals,vy, yShapeInfo,vz,zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffsets), REDUCE3_OPS);
}


//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y>
void Reduce3<X,Y>::execAll(const int opNum,
                            void *vx, Nd4jLong *xShapeInfo,
                            void *extraParamsVals,
                            void *vy, Nd4jLong *yShapeInfo,
                            void *vz, Nd4jLong *zShapeInfo,
                            int *dimension, int dimensionLength,
                            Nd4jLong *xTadShapeInfo, Nd4jLong *xOffsets,
                            Nd4jLong *yTadShapeInfo, Nd4jLong *yOffsets) {

    DISPATCH_BY_OPNUM_TT(execAll, PARAMS(vx, xShapeInfo, extraParamsVals, vy, yShapeInfo, vz, zShapeInfo, dimension, dimensionLength, xTadShapeInfo, xOffsets, yTadShapeInfo, yOffsets), REDUCE3_OPS);
}




BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT Reduce3, , LIBND4J_TYPES, FLOAT_TYPES);

}
}