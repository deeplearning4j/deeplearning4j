/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// Created by remote on 2018-09-20.
//

#include <loops/pairwise_bool.h>
#include <types/types.h>
#include <LoopKind.h>
#include <OmpLaunchHelper.h>

using namespace simdOps;

namespace functions {
    namespace pairwise_transforms {

        template <typename X, typename Y>
        void PairWiseBoolTransform<X, Y>::exec(
                const int opNum,
                void *x,
                Nd4jLong xEws,
                void *y,
                Nd4jLong yEws,
                void *z,
                Nd4jLong zEws,
                void *extraParams,
                Nd4jLong n) {
            DISPATCH_BY_OPNUM_TT(exec, PARAMS(x,
                                               xEws,
                                               y,
                                               yEws,
                                               z,
                                               zEws,
                                               extraParams,
                                               n), PAIRWISE_BOOL_OPS);
        };



        template <typename X, typename Z>
        template <typename OpType>
        void PairWiseBoolTransform<X, Z>::exec(void *vx,
                                              Nd4jLong xEws,
                                              void *vy,
                                              Nd4jLong yEws,
                                              void *vz,
                                              Nd4jLong zEws,
                                              void *vextraParams,
                                              const Nd4jLong n) {
            
            auto x = reinterpret_cast<X *>(vx);
            auto y = reinterpret_cast<X *>(vy);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            nd4j::OmpLaunchHelper info(n);

            if (xEws == 1 && yEws == 1 && zEws == 1) {

                PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                {                
                    auto threadNum = omp_get_thread_num();
                    Nd4jLong threadOffset = info.getThreadOffset(threadNum);        
                    auto xi = x + threadOffset;
                    auto yi = y + threadOffset;
                    auto zi = z + threadOffset;
                    auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                    PRAGMA_OMP_SIMD
                    for (Nd4jLong i = 0; i < ulen; i++)
                        zi[i] = OpType::op(xi[i], yi[i], extraParams);
                }
            }
            else {

                PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                {                
                    auto threadNum = omp_get_thread_num();
                    Nd4jLong threadOffset = info.getThreadOffset(threadNum);        
                    auto xi = x + xEws*threadOffset;
                    auto yi = y + yEws*threadOffset;
                    auto zi = z + zEws*threadOffset;
                    auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                    PRAGMA_OMP_SIMD
                    for (Nd4jLong i = 0; i < ulen; i++)
                        zi[i*zEws] = OpType::op(xi[i*xEws], yi[i*yEws], extraParams);
                }
            }
        }

        template <typename X, typename Y>
        void PairWiseBoolTransform<X, Y>::exec(
                const int opNum,
                void *x,
                Nd4jLong *xShapeInfo,
                void *y,
                Nd4jLong *yShapeInfo,
                void *z,
                Nd4jLong *zShapeInfo,
                void *extraParams) {
            DISPATCH_BY_OPNUM_TT(exec, PARAMS(x,
                                              xShapeInfo,
                                              y,
                                              yShapeInfo,
                                              z,
                                              zShapeInfo,
                                              extraParams),
                                 PAIRWISE_BOOL_OPS);
        };


        template <typename X, typename Z>
        template <typename OpType>
        void PairWiseBoolTransform<X, Z>::exec(void *vx, Nd4jLong* xShapeInfo,
                                            void *vy, Nd4jLong* yShapeInfo,
                                            void *vz, Nd4jLong* zShapeInfo,
                                            void *vextraParams) {

            auto x = reinterpret_cast<X *>(vx);
            auto y = reinterpret_cast<X *>(vy);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            auto n = shape::length(xShapeInfo);
            auto xEws = shape::elementWiseStride(xShapeInfo);
            auto yEws = shape::elementWiseStride(yShapeInfo);
            auto zEws = shape::elementWiseStride(zShapeInfo);

            nd4j::OmpLaunchHelper info(n);

            if (shape::isScalar(yShapeInfo)) {

               uint xShapeInfoCast[MAX_RANK];
               const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {

                    PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                    {                
                        auto threadNum = omp_get_thread_num();
                        auto threadOffset = info.getThreadOffset(threadNum);
                        auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                        PRAGMA_OMP_SIMD
                        for(Nd4jLong i = 0; i < ulen; i++)  {
                            auto offset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, n, canCastX);
                            z[offset] = OpType::op(x[offset], y[0], extraParams);
                        }
                    }
                }
                else {
                    
                    uint zShapeInfoCast[MAX_RANK];
                    const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

                    PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                    {                
                        auto threadNum = omp_get_thread_num();
                        auto threadOffset = info.getThreadOffset(threadNum);
                        auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                        PRAGMA_OMP_SIMD
                        for(Nd4jLong i = 0; i < ulen; i++)  {
                            auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, n, canCastX);
                            auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, n, canCastZ);
                            z[zOffset] = OpType::op(x[xOffset], y[0], extraParams);
                        }
                    }
                }
                return;
            }

            const nd4j::LoopKind::Kind kindOfLoop = nd4j::LoopKind::deduceKindOfLoopXYZ(xShapeInfo, yShapeInfo, zShapeInfo);
            const bool sameShapesXY = shape::shapeEquals(xShapeInfo, yShapeInfo);

            if ((kindOfLoop == nd4j::LoopKind::EWS1 || kindOfLoop == nd4j::LoopKind::EWSNONZERO) && sameShapesXY) {
                exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, n);
            }            
            else if ((kindOfLoop == nd4j::LoopKind::EWS1 || kindOfLoop == nd4j::LoopKind::EWSNONZERO) && !sameShapesXY) { //not same shape
                exec<OpType>(x, xEws, y, yEws, z, zEws, extraParams, shape::length(yShapeInfo));
            }
            else {                

                if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo) && shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {

                    uint xShapeInfoCast[MAX_RANK];
                    const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);

                    PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                    {                
                        auto threadNum = omp_get_thread_num();
                        auto threadOffset = info.getThreadOffset(threadNum);
                        auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                        PRAGMA_OMP_SIMD
                        for (Nd4jLong i = 0; i < ulen; i++)  {
                            auto offset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, n, canCastX);
                            z[offset] = OpType::op(x[offset], y[offset], extraParams);
                        }
                    }
                }
                else if(shape::haveSameShapeAndStrides(xShapeInfo, yShapeInfo)) {
                    
                    uint xShapeInfoCast[MAX_RANK];
                    uint zShapeInfoCast[MAX_RANK];
                    const bool canCastX = nd4j::DataTypeUtils::castShapeInfo(xShapeInfo, xShapeInfoCast);
                    const bool canCastZ = nd4j::DataTypeUtils::castShapeInfo(zShapeInfo, zShapeInfoCast);

                    PRAGMA_OMP_PARALLEL_THREADS(info._numThreads)
                    {                
                        auto threadNum = omp_get_thread_num();
                        auto threadOffset = info.getThreadOffset(threadNum);
                        auto ulen = static_cast<unsigned int>(info.getItersPerThread(threadNum));

                        PRAGMA_OMP_SIMD
                        for (Nd4jLong i = 0; i < ulen; i++)  {
                            auto offset  = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, n, canCastX);                            
                            auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, n, canCastZ);
                            z[zOffset] = OpType::op(x[offset], y[offset], extraParams);
                        }
                    }
                }
                else if(shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo)) {
                    
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
                        for (Nd4jLong i = 0; i < ulen; i++)  {
                            auto offset  = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, n, canCastX);                            
                            auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, n, canCastY);
                            z[offset] = OpType::op(x[offset], y[yOffset], extraParams);
                        }
                    }
                }
                else if(shape::haveSameShapeAndStrides(yShapeInfo, zShapeInfo)) {
                    
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
                        for (Nd4jLong i = 0; i < ulen; i++)  {
                            auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, n, canCastX);                            
                            auto offset  = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, n, canCastY);
                            z[offset] = OpType::op(x[xOffset], y[offset], extraParams);
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
                        for (Nd4jLong i = 0; i < ulen; i++)  {
                            auto xOffset = shape::indexOffset(i + threadOffset, xShapeInfo, xShapeInfoCast, n, canCastX);                            
                            auto yOffset = shape::indexOffset(i + threadOffset, yShapeInfo, yShapeInfoCast, n, canCastY);
                            auto zOffset = shape::indexOffset(i + threadOffset, zShapeInfo, zShapeInfoCast, n, canCastZ);
                            z[zOffset] = OpType::op(x[xOffset], y[yOffset], extraParams);
                        }
                    }
                }
            }
        }

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT PairWiseBoolTransform, , LIBND4J_TYPES, BOOL_TYPES);
    }
}
