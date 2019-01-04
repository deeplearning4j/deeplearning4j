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
                
    auto xOrder = shape::order(xShapeInfo);
    auto yOrder = shape::order(yShapeInfo);
    if(xOrder == yOrder && (xEws  >=1 && yEws >= 1) && shape::strideDescendingCAscendingF(xShapeInfo) && shape::strideDescendingCAscendingF(yShapeInfo)) {

        if (xEws == 1 && yEws == 1) {

            // TODO:: proper reduction required here
            for(int i = 0; i < length; i++) 
                startingVal = OpType::update(startingVal, OpType::op(x[i],y[i], extraParamsVals),extraParamsVals);                        

            z[0] = OpType::postProcess(startingVal, length, extraParamsVals);

        }
        else {
            // TODO:: proper reduction required here
            for(Nd4jLong i = 0; i < length; i++) 
                startingVal = OpType::update(startingVal, OpType::op(x[i * xEws],y[i * yEws], extraParamsVals), extraParamsVals);
                        
            z[0] =  OpType::postProcess(startingVal, length, extraParamsVals);
        }

    }
    else {
        for(unsigned int i = 0 ;i < length; i++) {
            auto offset  = shape::getIndexOffset(i, xShapeInfo, length);
            auto yOffset = shape::getIndexOffset(i, yShapeInfo, length);
            startingVal = OpType::update(startingVal, OpType::op(x[offset], y[yOffset], extraParamsVals), extraParamsVals);
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

    shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
    xTad.createTadOnlyShapeInfo();
    xTad.createOffsets();

    shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
    yTad.createTadOnlyShapeInfo();
    yTad.createOffsets();

    if(!shape::equalsSoft(xTad.tadOnlyShapeInfo, yTad.tadOnlyShapeInfo) || (xTad.numTads != yTad.numTads && yTad.numTads != 1))
        throw std::runtime_error("Reduce3<X,Z>::exec function: arrays tads are inconsistent !");

    const auto tadLen    = shape::length(xTad.tadOnlyShapeInfo);
    const auto xTadOrder = shape::order(xTad.tadOnlyShapeInfo);
    const auto yTadOrder = shape::order(yTad.tadOnlyShapeInfo);
    const auto xTadEws   = shape::elementWiseStride(xTad.tadOnlyShapeInfo);
    const auto yTadEws   = shape::elementWiseStride(yTad.tadOnlyShapeInfo);
    const auto zLen      = shape::length(zShapeInfo);    

    nd4j::OmpLaunchHelper info(zLen);

    if(xTadEws > 1 && yTadEws > 1 && xTadOrder == yTadOrder) {

        // #pragma omp  parallel for schedule(guided) num_threads(info._numThreads) if (info._numThreads > 1) proc_bind(AFFINITY) default(shared)
        for(Nd4jLong i = 0; i < zLen; ++i) {

            Nd4jLong xOffset = xTad.tadOffsets[i];
            Nd4jLong yOffset = yTad.numTads == 1 ? 0 : yTad.tadOffsets[i];            
            auto start = OpType::startingValue(x + xOffset);

            for (int j = 0; j < tadLen; j++) {

                Nd4jLong xOffset2 =  xOffset + j*xTadEws;
                Nd4jLong yOffset2 =  yOffset + j*yTadEws;
                start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2], extraParams), extraParamsVals);
            }            
            z[i] = OpType::postProcess(start, tadLen, extraParamsVals);
        }
    }
    else {
        
        // #pragma omp  parallel for schedule(guided) num_threads(info._numThreads) if (info._numThreads > 1) proc_bind(AFFINITY) default(shared)
        for(Nd4jLong i = 0; i < zLen; ++i) {

            Nd4jLong xOffset = xTad.tadOffsets[i];
            Nd4jLong yOffset = yTad.numTads == 1 ? 0 : yTad.tadOffsets[i];            
            auto start = OpType::startingValue(x + xOffset);

            for (int j = 0; j < tadLen; j++) {
                    
                Nd4jLong xOffset2 =  xOffset + shape::getIndexOffset(j, xTad.tadOnlyShapeInfo, tadLen);
                Nd4jLong yOffset2 =  yOffset + shape::getIndexOffset(j, yTad.tadOnlyShapeInfo, tadLen);                                    
                start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2], extraParams), extraParamsVals);
            }
            z[i] = OpType::postProcess(start, tadLen, extraParamsVals);
        }
    }




//     char xOrder = shape::order(xShapeInfo);
//     char yOrder = shape::order(yShapeInfo);
//     auto zLen = shape::length(zShapeInfo);
//     auto tadLen = shape::tadLen(xShapeInfo,dimension,dimensionLength);
    
//     if(xOrder != yOrder) {
        
//         nd4j::OmpLaunchHelper info(zLen);
//         #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
//         {                
//             auto threadNum = omp_get_thread_num();         
//             auto threadOffset = info.getThreadOffset(threadNum);

//             #pragma omp simd
//             for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
//                 auto xOffset = shape::getIndexOffset(i+threadOffset, xShapeInfo, zLen);
//                 auto yOffset = shape::getIndexOffset(i+threadOffset, yShapeInfo, zLen);
//                 auto zOffset = shape::getIndexOffset(i+threadOffset, zShapeInfo, zLen);
//                 z[zOffset] = OpType::update(z[zOffset], OpType::op(x[xOffset], y[yOffset], extraParamsVals), extraParamsVals);
//             }
//         }
                        
//         auto zEws = shape::elementWiseStride(zShapeInfo);
//         #pragma omp parallel for proc_bind(AFFINITY) default(shared)
//         for(Nd4jLong i = 0; i < zLen; i+=zEws) 
//             z[i] = OpType::postProcess(z[i], tadLen, extraParamsVals);
//     }
//     else {
        
//         auto startingVal = OpType::startingValue(x); 
        
//         shape::TAD xTad(xShapeInfo, dimension, dimensionLength);
//         xTad.createTadOnlyShapeInfo();
//         xTad.createOffsets();

//         shape::TAD yTad(yShapeInfo, dimension, dimensionLength);
//         yTad.createTadOnlyShapeInfo();
//         yTad.createOffsets();

//          // if(!shape::equalsSoft(xTad.tadOnlyShapeInfo, yTad.tadOnlyShapeInfo))
//          //        throw std::runtime_error("NDArray::applyReduce3 cuda method: arrays tads are inconsistent !");

//         /**
//         * The element wise stride belong longs to a reduction index.
//         * When used out of order, we can get rid of the data
//         * dependencies and rely on using the max dimension
//         * specified for stride instead.
//         * Say we take the sum(0,1) along long arr
//         * we can use arr.stride(1) as a representation
//         * along long which to iterate.
//         */
//         int largerElementWiseStride;
//         int smallerElementWiseStride;
//         auto xEws = shape::elementWiseStride(xTad.tadOnlyShapeInfo);
//         auto yEws = shape::elementWiseStride(yTad.tadOnlyShapeInfo);
//         int tadLen;
//         Nd4jLong xModLength;
//         Nd4jLong yModLength;
//         Nd4jLong *iterationTadInfo;
//         bool xTadBigger;
        
//         if(shape::length(xShapeInfo) > shape::length(yShapeInfo)) {
//             tadLen = shape::length(xTad.tadOnlyShapeInfo);
//             iterationTadInfo = xTad.tadOnlyShapeInfo;
//             largerElementWiseStride = shape::elementWiseStride(xShapeInfo);
//             smallerElementWiseStride = shape::elementWiseStride(yShapeInfo);
//             xModLength = 1;
//             yModLength = tadLen;
//             xTadBigger = true;
//         }
//         else {
//             tadLen = shape::length(yTad.tadOnlyShapeInfo);
//             iterationTadInfo = yTad.tadOnlyShapeInfo;
//             largerElementWiseStride = shape::elementWiseStride(yShapeInfo);
//             smallerElementWiseStride = shape::elementWiseStride(xShapeInfo);
//             xModLength = tadLen;
//             yModLength = 1;
//             xTadBigger = false;
//         }
        
//         if (largerElementWiseStride >= 1 && smallerElementWiseStride >= 1 && xEws >= 1 && yEws >= 1) {

//             if(shape::length(xShapeInfo) == shape::length(yShapeInfo)) {
                
//                 //#pragma omp parallel for proc_bind(AFFINITY) default(shared)
//                 for (Nd4jLong i = 0; i < zLen; i++) {
                    
//                     Z *localExtraParams = nullptr;
                    
//                     if (OpType::extraParamsLen > 0)
//                         localExtraParams = new Z[OpType::extraParamsLen];
                    
//                     for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
//                         localExtraParams[extraParamsIdx] = startingVal;
                                
//                     Nd4jLong offset = xTad.tadOffsets[i];
//                     Nd4jLong yOffset = yTad.tadOffsets[i];
//                     z[i] = OpType::op(x[offset], y[yOffset], localExtraParams);
                    
//                     for (int j = 1; j < tadLen; j++) {
//                         int xIdx = (offset + xEws * j);
//                         int yIdx = (yOffset + yEws * j);
//                         z[i] = OpType::update(z[i], OpType::op(x[xIdx],y[yIdx],localExtraParams), localExtraParams);
//                     }

//                     z[i] = OpType::postProcess(z[i], tadLen, localExtraParams);

//                     if (localExtraParams != nullptr)
//                         delete[] localExtraParams;
//                 }
//             }
//             else {
                
//                 int tadsPerThread = zLen / TAD_THRESHOLD;
//                 int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
//                 num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

// //#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
//                 for (int i = 0; i < zLen; i++) {
                
//                     Nd4jLong xOffset = xTadBigger ? xTad.tadOffsets[i] : 0;
//                     Nd4jLong yOffset = !xTadBigger ? yTad.tadOffsets[i] : 0;
//                     auto xShapeInf = xTadBigger ? xTad.tadOnlyShapeInfo : xShapeInfo;
//                     auto yShapeInf = !xTadBigger ? yTad.tadOnlyShapeInfo : yShapeInfo;
//                     auto start = OpType::startingValue(x);

//                     for (int j = 0; j < tadLen; j++) {
                    
//                         int xOffset2 =  xOffset + shape::getIndexOffset(j, xShapeInf, tadLen);
//                         int yOffset2 =  yOffset + shape::getIndexOffset(j, yShapeInf, tadLen);                                    
//                         start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2],extraParams), extraParamsVals);
//                     }

//                     z[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
//                 }   
//             }
//         } 
//         else {
             
//             int tadsPerThread = zLen / TAD_THRESHOLD;
//             int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
//             num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

// //#pragma omp  parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
//             for (int i = 0; i < zLen; i++) {
                
//                 Nd4jLong xOffset = xTad.tadOffsets[i];
//                 Nd4jLong yOffset = yTad.tadOffsets[i];
//                 auto start = OpType::startingValue(x + xOffset);
                
//                 for (int j = 0; j < tadLen; j++) {
//                     Nd4jLong xOffset2 = xOffset + shape::getIndexOffset(j, xTad.tadOnlyShapeInfo, tadLen);
//                     Nd4jLong yOffset2 = yOffset + shape::getIndexOffset(j, yTad.tadOnlyShapeInfo, tadLen);
//                     start = OpType::update(start, OpType::op(x[xOffset2], y[yOffset2],extraParamsVals), extraParamsVals);
//                 }

//                 z[i] = OpType::postProcess(start, shape::length(iterationTadInfo), extraParamsVals);
//             }
//         }
//     }
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

    auto tadLen = shape::tadLength(xShapeInfo, dimension, dimensionLength);
    auto tads = shape::length(xShapeInfo) / tadLen;

//#pragma  omp parallel for proc_bind(AFFINITY) default(shared)
    for (Nd4jLong r = 0; r < tads; r++) {
        
        Nd4jLong offset = tadOffsets[r];
        Z *localExtraParams = nullptr;

        if (OpType::extraParamsLen > 0)
            localExtraParams = new Z[OpType::extraParamsLen];

        for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
            localExtraParams[extraParamsIdx] = startingVal;

        for (Nd4jLong f = 0; f < tadLen; f++) {

            auto xOffset = offset + shape::getIndexOffset(f, tadShapeInfo, tadLen);
            auto yOffset = shape::getIndexOffset(f, yShapeInfo, tadLen);
            z[r] = OpType::update(z[r], OpType::op(x[xOffset], y[yOffset], localExtraParams), localExtraParams);
        }

        z[r] = OpType::postProcess(z[r], tadLen, localExtraParams);

        if (localExtraParams != nullptr)
            delete[] localExtraParams;
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

    #pragma  omp parallel for proc_bind(AFFINITY) default(shared)
    for (Nd4jLong r = 0; r < xTads; r++) {
    
        Nd4jLong xOffset = xOffsets[r];
        auto lX = x + xOffset;

        for (Nd4jLong g = 0; g < yTads; g++) {
        
            auto yOffset = yOffsets[g];
            auto lY = y + yOffset;
            auto ri = (r * yTads) + g;

            Z *localExtraParams = nullptr;
            if (OpType::extraParamsLen > 0)
                localExtraParams = new Z[OpType::extraParamsLen];

            for (int extraParamsIdx = 0; extraParamsIdx < OpType::extraParamsLen; extraParamsIdx++) 
                localExtraParams[extraParamsIdx] = startingVal;

            for (int f = 0; f < xTadLength; f++) {                            
                auto xO = shape::getIndexOffset(f, xTadShapeInfo, xTadLength);
                auto yO = shape::getIndexOffset(f, yTadShapeInfo, xTadLength);
                z[ri] = OpType::update(z[ri], OpType::op(lX[xO], lY[yO], localExtraParams), localExtraParams);
            }

            z[ri] = OpType::postProcess(z[ri], xTadLength, localExtraParams);

            if (localExtraParams != nullptr)
                delete[] localExtraParams;
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