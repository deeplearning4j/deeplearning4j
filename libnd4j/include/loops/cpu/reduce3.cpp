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

    auto startingVal = OpType::startingValue(x);
    auto length = shape::length(xShapeInfo);
    auto xEws = shape::elementWiseStride(xShapeInfo);
    auto yEws = shape::elementWiseStride(yShapeInfo);

    Z extraParamsVals[3] = {(X) 0.0f, (X) 0.0f, (X) 0.0f};
    // it's possible case for EqualsWithEps op
    if (extraParams != nullptr) 
        extraParamsVals[2] = extraParams[0];                
    
    uint xShapeInfoCast[MAX_RANK];
    const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

    if(shape::haveSameOffsets(xShapeInfo, yShapeInfo)) {

        for(unsigned int i = 0; i < length; i++) {            
            auto offset  = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, length, canCastX);
            startingVal = OpType::update(startingVal, OpType::op(x[offset], y[offset], extraParamsVals), extraParamsVals);
        }
    }
    else {
        uint yShapeInfoCast[MAX_RANK];
        const bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInfo, yShapeInfoCast);

        for(unsigned int i = 0; i < length; i++) {            
            auto xOffset  = shape::indexOffset(i, xShapeInfo, xShapeInfoCast, length, canCastX);
            auto yOffset  = shape::indexOffset(i, yShapeInfo, yShapeInfoCast, length, canCastY);
            startingVal = OpType::update(startingVal, OpType::op(x[xOffset], y[yOffset], extraParamsVals), extraParamsVals);
        }
    }

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

             PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
            {                
                auto threadNum = omp_get_thread_num();         
                auto threadOffset = info.getThreadOffset(threadNum);

                for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
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

                for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
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

                for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
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

                for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
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

                for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
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
        
        shape::TAD xTad;
        xTad.init(xShapeInfo, dimension, dimensionLength);
        xTad.createTadOnlyShapeInfo();
        xTad.createOffsets();

        shape::TAD yTad;
        yTad.init(yShapeInfo, dimension, dimensionLength);
        yTad.createTadOnlyShapeInfo();
        yTad.createOffsets();

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
        auto xEws = shape::elementWiseStride(xTad.tadOnlyShapeInfo);
        auto yEws = shape::elementWiseStride(yTad.tadOnlyShapeInfo);
        int tadLength;
        Nd4jLong xModLength;
        Nd4jLong yModLength;
        Nd4jLong *iterationTadInfo;
        bool xTadBigger;
        
        if(shape::length(xShapeInfo) > shape::length(yShapeInfo)) {
            tadLength = shape::length(xTad.tadOnlyShapeInfo);
            iterationTadInfo = xTad.tadOnlyShapeInfo;
            largerElementWiseStride = shape::elementWiseStride(xShapeInfo);
            smallerElementWiseStride = shape::elementWiseStride(yShapeInfo);
            xModLength = 1;
            yModLength = tadLength;
            xTadBigger = true;
        }
        else {
            tadLength = shape::length(yTad.tadOnlyShapeInfo);
            iterationTadInfo = yTad.tadOnlyShapeInfo;
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
                                
                    auto offset = xTad.tadOffsets[i];
                    auto yOffset = yTad.tadOffsets[i];
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

                PRAGMA_OMP_PARALLEL_FOR_THREADS(num_threads)
                for (int i = 0; i < zLen; i++) {
                
                    Nd4jLong xOffset = xTadBigger ? xTad.tadOffsets[i] : 0;
                    Nd4jLong yOffset = !xTadBigger ? yTad.tadOffsets[i] : 0;
                    auto xShapeInf = xTadBigger ? xTad.tadOnlyShapeInfo : xShapeInfo;
                    auto yShapeInf = !xTadBigger ? yTad.tadOnlyShapeInfo : yShapeInfo;
                    auto start = OpType::startingValue(x);

                    uint xShapeInfoCast[MAX_RANK];                    
                    bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInf, xShapeInfoCast);

                    auto tX = x + xOffset;
                    auto tY = y + yOffset;

                    if(shape::haveSameOffsets(xShapeInf, yShapeInf)) {

                        for (unsigned int j = 0; j < tadLength; j++) {                            
                            auto offset = shape::indexOffset(j, xShapeInf, xShapeInfoCast, tadLength, canCastX);
                            start = OpType::update(start, OpType::op(tX[offset], tY[offset],extraParams), extraParamsVals);
                        }
                    }
                    else {
                        uint yShapeInfoCast[MAX_RANK];                    
                        bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yShapeInf, yShapeInfoCast);

                        for (unsigned int j = 0; j < tadLength; j++) {                            
                            auto xOffset2 = shape::indexOffset(j, xShapeInf, xShapeInfoCast, tadLength, canCastX);
                            auto yOffset2 = shape::indexOffset(j, yShapeInf, yShapeInfoCast, tadLength, canCastY);
                            start = OpType::update(start, OpType::op(tX[xOffset2], tY[yOffset2],extraParams), extraParamsVals);
                        }
                    } 

                    z[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                }   
            }
        } 
        else {
        
            shape::TAD xTad;
            xTad.init(xShapeInfo, dimension, dimensionLength);
            xTad.createTadOnlyShapeInfo();
            xTad.createOffsets();
            
            int tadsPerThread = zLen / TAD_THRESHOLD;
            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

            uint xShapeInfoCast[MAX_RANK];            
            bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xTad.tadOnlyShapeInfo, xShapeInfoCast);            

            if(shape::haveSameOffsets(xShapeInfo, yShapeInfo)) {

                PRAGMA_OMP_PARALLEL_FOR_THREADS(num_threads)
                for (unsigned int i = 0; i < zLen; i++) {
                
                    auto offset = xTad.tadOffsets[i];                    
                    auto start = OpType::startingValue(x + offset);

                    auto tX = x + offset;
                    auto tY = y + offset;

                    for (unsigned int j = 0; j < tadLength; j++) {
                        auto offset = shape::indexOffset(j, xTad.tadOnlyShapeInfo, xShapeInfoCast, tadLength, canCastX);                        
                        start = OpType::update(start, OpType::op(tX[offset], tY[offset], extraParamsVals), extraParamsVals);
                    }

                    z[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
                }                
            }
            else {
                
                shape::TAD yTad;
                yTad.init(yShapeInfo, dimension, dimensionLength);
                yTad.createTadOnlyShapeInfo();
                yTad.createOffsets();
                
                uint yShapeInfoCast[MAX_RANK];
                bool canCastY = nd4j::DataTypeUtils::castShapeInfo(yTad.tadOnlyShapeInfo, yShapeInfoCast);

#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
                for (unsigned int i = 0; i < zLen; i++) {
                
                    auto xOffset = xTad.tadOffsets[i];
                    auto yOffset = yTad.tadOffsets[i];
                    auto start = OpType::startingValue(x + xOffset);

                    auto tX = x + xOffset;
                    auto tY = y + yOffset;

                    for (unsigned int j = 0; j < tadLength; j++) {
                        auto xOffset2 = shape::indexOffset(j, xTad.tadOnlyShapeInfo, xShapeInfoCast, tadLength, canCastX);
                        auto yOffset2 = shape::indexOffset(j, yTad.tadOnlyShapeInfo, yShapeInfoCast, tadLength, canCastY);
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

        #pragma  omp parallel for proc_bind(AFFINITY) default(shared)
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

#pragma  omp parallel for proc_bind(AFFINITY) default(shared)
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
        
        #pragma  omp parallel for proc_bind(AFFINITY) default(shared)
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
        
        #pragma  omp parallel for proc_bind(AFFINITY) default(shared)
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