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
// Created by raver119 on 08.10.2017.
//

#include "../scalar.h"
#include <op_boilerplate.h>
#include <types/types.h>

#include "../legacy_ops.h"

using namespace simdOps;

namespace functions {
namespace scalar    {


        template<typename X, typename Y, typename Z>
        template<typename OpType>
        void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong *xShapeInfo, void *vextraParams, void *vz, Nd4jLong *zShapeInfo, void *vscalars, int *dimension, int dimensionLength, Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets, Nd4jLong *tadShapeInfoZ, Nd4jLong *tadOffsetsZ) {
            auto x = reinterpret_cast<X *>(vx);
            auto z = reinterpret_cast<Z *>(vz);
            auto scalars = reinterpret_cast<Y *>(vscalars);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

            if (tadShapeInfoZ == nullptr) {
                tadShapeInfoZ = tadShapeInfo;
                tadOffsetsZ = tadOffsets;
            }

            // tad preparation
            int tadEWS = shape::elementWiseStride(tadShapeInfo);
            int zEWS = shape::elementWiseStride(tadShapeInfo);
            //int tadRank = shape::rank(tadShapeInfo);
            int tadLength = shape::tadLength(xShapeInfo, dimension, dimensionLength);
            int numTads =shape::length(xShapeInfo) / tadLength;

            int tadsPerThread = numTads / TAD_THRESHOLD;
            int num_threads = nd4j::math::nd4j_max<int>(1, tadsPerThread);
            num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

            // main loop, rolling along tads
#pragma omp parallel for schedule(guided) num_threads(num_threads) if (num_threads > 1) proc_bind(AFFINITY) default(shared)
            for (int r = 0; r < numTads; r++) {

                auto offset = tadOffsets[r];
                auto offsetZ = tadOffsetsZ[r];
                auto scalar = scalars[r];

                if (tadEWS >= 1 && zEWS >= 1) {
                    auto oZ = z + offsetZ;
                    auto oX = x + offset;

                    if (tadEWS == 1 && zEWS == 1) {

#pragma omp simd
                        for (int f = 0; f < tadLength; f++) {
                            oZ[f] = OpType::op(oX[f], scalar, extraParams);
                        }
                    } else {

// TODO: nested loop should be used here probably, instead of simd
#pragma omp simd
                        for (int f = 0; f < tadLength; f++) {
                            oZ[f * zEWS] = OpType::op(oX[f * tadEWS], scalar, extraParams);
                        }
                    }

                } else {
                    // ind2sub loop
                    printf("Super-bad loop visited. Shouldn't ever happen\n");
                }
            }
        }

        template<typename X, typename Y, typename Z>
        void ScalarTransform<X,Y,Z>::transform(int opNum,
                              void *x,
                              Nd4jLong *xShapeInfo,
                              void *extraParams,
                              void *z,
                              Nd4jLong *zShapeInfo,
                              void *scalars,
                              int *dimension,
                              int dimensionLength,
                              Nd4jLong *tadShapeInfo,
                              Nd4jLong *tadOffsets,
                              Nd4jLong *tadShapeInfoZ,
                              Nd4jLong *tadOffsetsZ) {
            DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, extraParams, z, zShapeInfo, scalars, dimension, dimensionLength, tadShapeInfo, tadOffsets, tadShapeInfoZ, tadOffsetsZ), SCALAR_OPS);
        }


        template<typename X, typename Y, typename Z>
        void ScalarTransform<X, Y, Z>::transform(const int opNum,
                void *x,
                Nd4jLong xStride,
                void *z,
                Nd4jLong zStride,
                void *scalar,
                void *extraParams,
                const Nd4jLong n) {
            DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xStride, z, zStride, scalar, extraParams, n), SCALAR_OPS);
        }

        template<typename X, typename Y, typename Z>
        void ScalarTransform<X, Y, Z>::transform(const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *z,
                Nd4jLong *zShapeInfo,
                void *scalar,
                void *extraParams) {
            DISPATCH_BY_OPNUM_TTT(transform, PARAMS(x, xShapeInfo, z, zShapeInfo, scalar, extraParams), SCALAR_OPS);
        }

////////////////////////////////////////////////////////////////////////
template<typename X, typename Y, typename Z>
template<typename OpType>
void ScalarTransform<X, Y, Z>::transform(void *vx, Nd4jLong *xShapeInfo,
                                        void *vz, Nd4jLong *zShapeInfo,
                                        void *vscalar,
                                        void *vextraParams) {

    auto x = reinterpret_cast<X *>(vx);
    auto z = reinterpret_cast<Z *>(vz);
    auto scalar = reinterpret_cast<Y *>(vscalar)[0];
    auto extraParams = reinterpret_cast<Z *>(vextraParams);

    const auto len = shape::length(xShapeInfo);
    const auto xEws = shape::elementWiseStride(xShapeInfo);
    const auto zEws = shape::elementWiseStride(zShapeInfo);
    const auto xOrder = shape::order(xShapeInfo);
    const auto zOrder = shape::order(zShapeInfo);

    if(xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
        transform<OpType>(x, xEws, z, zEws, vscalar, extraParams, len);              
    }
    else {
                
        const bool xSimpe = shape::isStrideSimple(xShapeInfo);
        const bool zSimpe = shape::isStrideSimple(zShapeInfo);
                   
        if(xSimpe) {
                        
            if(xEws == 1) {
                #pragma omp parallel for schedule(guided) if (len > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                for(Nd4jLong i = 0; i < len; ++i)
                    z[shape::getIndexOffset(i, zShapeInfo, len)] = OpType::op(x[i], scalar, extraParams);                       
            }
            else {
                #pragma omp parallel for schedule(guided) if (len > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                for(Nd4jLong i = 0; i < len; ++i)
                    z[shape::getIndexOffset(i, zShapeInfo, len)] = OpType::op(x[i*xEws], scalar, extraParams);
            }
        }
        else if(zSimpe) {

            if(zEws == 1) {
                #pragma omp parallel for schedule(guided) if (len > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                for(Nd4jLong i = 0; i < len; ++i)                                
                    z[i] = OpType::op(x[shape::getIndexOffset(i, xShapeInfo, len)], scalar, extraParams);         
            }
            else {
                #pragma omp parallel for schedule(guided) if (len > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
                for(Nd4jLong i = 0; i < len; ++i)                                
                    z[i*zEws] = OpType::op(x[shape::getIndexOffset(i, xShapeInfo, len)], scalar, extraParams);
            }
        }
        else {
            #pragma omp parallel for schedule(guided) if (len > ELEMENT_THRESHOLD) proc_bind(AFFINITY) default(shared)
            for(Nd4jLong i = 0; i < len; ++i) 
                z[shape::getIndexOffset(i, zShapeInfo, len)] = OpType::op(x[shape::getIndexOffset(i, xShapeInfo, len)], scalar, extraParams);
        }
    }                        
}


            template<typename X, typename Y, typename Z>
            template<typename OpType>
            void ScalarTransform<X, Y, Z>::transform(void *vx,
                    Nd4jLong xEws,
                    void *vz,
                    Nd4jLong zEws,
                    void *vscalar,
                    void *vextraParams,
                    const Nd4jLong len) {
               
                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<Z *>(vz);
                auto scalar = reinterpret_cast<Y *>(vscalar)[0];
                auto extraParams = reinterpret_cast<Z *>(vextraParams);
/*
                Nd4jLong elementsPerThread = len / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());
*/
                if (len < ELEMENT_THRESHOLD) {
                    for (Nd4jLong i = 0; i < len; i++) 
                        z[i * zEws] = OpType::op(x[i * xEws], scalar, extraParams);
                    return;
                }
         
                BlockInformation info(len, ELEMENT_THRESHOLD);
                #pragma omp parallel num_threads(info.threads) if (info.threads > 1) default(shared)
                {                
                    auto i = omp_get_thread_num();            
                    Nd4jLong itemsToLoop = (i < info.threads-1) ? info.items : info.items + info.remainder;
                    Nd4jLong index = i * info.items;
                    auto xi = x + xEws * index;
                    auto zi = z + zEws * index;
                    #pragma omp simd
                    for (Nd4jLong j = 0; j < itemsToLoop; j++) 
                        zi[j * zEws] = OpType::op(xi[j * xEws], scalar, extraParams);
                }
            }
}
}
