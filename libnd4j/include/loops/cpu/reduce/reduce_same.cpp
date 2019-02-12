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
//  @author raver119@gmail.com
//  @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <types/types.h>
#include <op_boilerplate.h>
#include <loops/reduce_same.h>
#include <loops/legacy_ops.h>
#include <OmpLaunchHelper.h>

using namespace simdOps;

namespace functions {
    namespace reduce {
        template <typename X>
        template <typename OpType>
        void _CUDA_H ReduceSameFunction<X>::execScalar(void *vx,
                                                Nd4jLong *xShapeInfo,
                                                void *vextraParams,
                                                void *vz,
                                                Nd4jLong *zShapeInfo) {
            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<X *>(vz);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            const Nd4jLong length = shape::length(xShapeInfo);
            auto xEws = shape::elementWiseStride(xShapeInfo);

            if (xEws >= 1) {
                z[0] = execScalar<OpType>(x, xEws, length, extraParams);
            }
            else {

                X start = OpType::startingValue(x);

                for(Nd4jLong i = 0; i < length; ++i)                     
                    start = OpType::update(start, OpType::op(x[shape::getIndexOffset(i, xShapeInfo, length)], extraParams), extraParams);                

                z[0] = OpType::postProcess(start, shape::length(xShapeInfo), extraParams);
            }
        }


        template <typename X>
        template <typename OpType>
            X _CUDA_H ReduceSameFunction<X>::execScalar(void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vextraParams) {
                auto x = reinterpret_cast<X *>(vx);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                const Nd4jLong length = shape::length(xShapeInfo);
                auto xEws = shape::elementWiseStride(xShapeInfo);

                if (xEws >= 1) {
                    return execScalar<OpType>(x, xEws, length, extraParams);
                }
                else {

                    X start = OpType::startingValue(x);

                    for(Nd4jLong i = 0; i < length; ++i)                     
                        start = OpType::update(start, OpType::op(x[shape::getIndexOffset(i, xShapeInfo, length)], extraParams), extraParams);                                                    
                    
                    return OpType::postProcess(start, shape::length(xShapeInfo), extraParams);
                }
            }

        template <typename X>
        X ReduceSameFunction<X>::execScalar(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *extraParams) {
                RETURNING_DISPATCH_BY_OPNUM_T(execScalar, PARAMS(x, xShapeInfo, extraParams), REDUCE_SAME_OPS);
        }

        template <typename X>
        void ReduceSameFunction<X>::execScalar(const int opNum,
                                        void *x,
                                        Nd4jLong *xShapeInfo,
                                        void *extraParams,
                                        void *z,
                                        Nd4jLong *zShapeInfo) {
            DISPATCH_BY_OPNUM_T(execScalar, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo), REDUCE_SAME_OPS);
        }

        template <typename X>
        void ReduceSameFunction<X>::exec(const int opNum,
                             void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *z,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {
                DISPATCH_BY_OPNUM_T(exec, PARAMS(x,
                                               xShapeInfo,
                                               extraParams,
                                               z,
                                               zShapeInfo,
                                               dimension,
                                               dimensionLength,
                                               tadShapeInfo,
                                               tadOffset),
                                  REDUCE_SAME_OPS);
        }

        template <typename X>
        template <typename OpType>
        void _CUDA_H ReduceSameFunction<X>::exec(void *vx,
                             Nd4jLong *xShapeInfo,
                             void *vextraParams,
                             void *vz,
                             Nd4jLong *zShapeInfo,
                             int *dimension,
                             int dimensionLength,
                             Nd4jLong *tadShapeInfo,
                             Nd4jLong *tadOffset) {

                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<X *>(vz);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                auto zLength = shape::length(zShapeInfo);

                //pre squeezed: this is for keeping the pointer to the original
                //shape information for tad offset
                //the squeezed information doesn't render the right strides for
                //tad offset
                // || tad.wholeThing
                if (zLength == 1 || dimension == nullptr || dimensionLength == shape::rank(xShapeInfo)) {
                    z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
                    return;
                }

                if (OpType::requiresSpecialAccumulation) {
                    OpType::execSpecial(x, xShapeInfo, extraParams, z, zShapeInfo, dimension, dimensionLength, tadShapeInfo, tadOffset);
                    return;
                }

                auto tadOnlyShapeInfo = tadShapeInfo;
                auto tadOffsets = tadOffset;
                shape::TAD *tad = nullptr;

                if (tadOnlyShapeInfo == nullptr || tadOffsets == nullptr) {
                    tad = new shape::TAD(xShapeInfo, dimension, dimensionLength);
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    if (tad->dimensionLength < 1) {
                        delete tad;
                        return;
                    }

                    tadOnlyShapeInfo = tad->tadOnlyShapeInfo;
                    tadOffsets = tad->tadOffsets;
                }

                const auto tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
                auto numTads = shape::length(xShapeInfo) / tadLength;
                auto tadEWS = shape::elementWiseStride(tadOnlyShapeInfo);

                int tadsPerThread = zLength / TAD_THRESHOLD;
                //int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
                //num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                if (tadEWS > 0 && (numTads == 1 || shape::isVector(tadOnlyShapeInfo) || shape::isScalar(tadOnlyShapeInfo))) {

#pragma omp parallel for schedule(static, TAD_THRESHOLD) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < zLength; i++) {
                        auto iter = x + tadOffsets[i];
                        auto start = OpType::startingValue(iter);
                        if (tadEWS == 1) {

// FIXME: proper reduction should be used here
                            for (int j = 0; j < tadLength; j++) 
                                start = OpType::update(start, OpType::op(iter[j], extraParams), extraParams);                            
                        }
                        else {
// FIXME: proper reduction to be used here
                            for (int j = 0; j < tadLength; j++) 
                                start = OpType::update(start, OpType::op(iter[j * tadEWS], extraParams), extraParams);                            
                        }
                        z[i] = OpType::postProcess(start, tadLength, extraParams);
                    }
                }
                else {                                     

#pragma omp  parallel for schedule(static, TAD_THRESHOLD) proc_bind(AFFINITY) default(shared)
                    for (int i = 0; i < zLength; i++) {
                        
                        auto offset = tadOffsets[i];                        
                        auto start = OpType::startingValue(x + offset);

                        for (int j = 0; j < tadLength; j++) {                            
                            auto xOffset = offset + shape::getIndexOffset(j, tadOnlyShapeInfo, tadLength);
                            start = OpType::update(start, OpType::op(x[xOffset], extraParams), extraParams);
                        }
                        z[i] = OpType::postProcess(start, tadLength, extraParams);;
                    }
                }

                if (tad != nullptr)
                    delete tad;
            }


        template <typename X>
        template<typename OpType>
        void _CUDA_H ReduceSameFunction<X>::exec(void *x,
                             Nd4jLong *xShapeInfo,
                             void *extraParams,
                             void *vz,
                             Nd4jLong *zShapeInfo) {
                // FIXME: wtf???
                auto z = reinterpret_cast<X*>(vz);
                z[0] = execScalar<OpType>(x, xShapeInfo, extraParams);
        }

        template <typename X>
        template <typename OpType>
        X _CUDA_H ReduceSameFunction<X>::execScalar(void *vx, Nd4jLong xEws, Nd4jLong length, void *vextraParams) {

                auto x = reinterpret_cast<X *>(vx);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                auto startingVal = OpType::startingValue(x);
                nd4j::OmpLaunchHelper info(length);

                if (xEws == 1) {
                                           
                    #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
                    {                
                        auto local = OpType::startingValue(x);
                        auto threadNum = omp_get_thread_num();                    
                        Nd4jLong threadOffset = info.getThreadOffset(threadNum);
                        auto xi = x + threadOffset;

                        #pragma omp simd
                        for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++)                                
                            local = OpType::update(local, OpType::op(xi[i], extraParams), extraParams);
                            
                        #pragma omp critical
                        startingVal = OpType::update(startingVal, local, extraParams);        
                    }
                }
                else {

                    #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
                    {                
                        auto local = OpType::startingValue(x);
                        auto threadNum = omp_get_thread_num();                    
                        Nd4jLong threadOffset = info.getThreadOffset(threadNum);
                        auto xi = x + xEws*threadOffset;

                        #pragma omp simd
                        for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++)                                
                            local = OpType::update(local, OpType::op(xi[i*xEws], extraParams), extraParams);
                            
                        #pragma omp critical
                        startingVal = OpType::update(startingVal, local, extraParams);        
                    }                    
                }
                return OpType::postProcess(startingVal, length, extraParams);
            }


        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT ReduceSameFunction, , LIBND4J_TYPES);
    }
}