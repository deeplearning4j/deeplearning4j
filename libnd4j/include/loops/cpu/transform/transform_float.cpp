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
//  @author  raver119@gmail.com
//

#include <op_boilerplate.h>
#include <types/types.h>
#include <loops/transform_float.h>
#include <loops/legacy_ops.h>

using namespace simdOps;

namespace functions {
    namespace transform {
        template <typename X, typename Y>
        void TransformFloat<X, Y>::exec(
				int opNum,
				void *x,
				Nd4jLong *xShapeInfo,
				void *z,
				Nd4jLong *zShapeInfo,
				void *extraParams,
				Nd4jLong *tadShapeInfo,
				Nd4jLong *tadOffsets) {
                    DISPATCH_BY_OPNUM_TT(exec, PARAMS(x, xShapeInfo, z, zShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_FLOAT_OPS);
		}

        template <typename X, typename Z>
        template<typename OpType>
		void _CUDA_H TransformFloat<X, Z>::exec(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vz,
                    Nd4jLong *zShapeInfo,
                    void *vextraParams,
                    Nd4jLong *tadShapeInfo,
                    Nd4jLong *tadOffsets) {

		        auto x = reinterpret_cast<X *>(vx);
		        auto z = reinterpret_cast<Z *>(vz);
		        auto extraParams = reinterpret_cast<Z *>(vextraParams);

                if(OpType::requiresSpecial) {
                    OpType::execSpecial(x, xShapeInfo, z, zShapeInfo, extraParams, tadShapeInfo, tadOffsets);
                    return;
                }

                const auto len = shape::length(xShapeInfo);
                const auto xEws = shape::elementWiseStride(xShapeInfo);
                const auto zEws = shape::elementWiseStride(zShapeInfo);
                const auto xOrder = shape::order(xShapeInfo);
                const auto zOrder = shape::order(zShapeInfo);

                // loop2ArrsSame<X>(x, xShapeInfo, z, zShapeInfo, extraParams, OpType::op);

                if(xEws >= 1 && zEws >= 1 && xOrder == zOrder) {
                    //exec<OpType>(x,xEws,z,zEws,extraParams,len);
                    nd4j::OmpLaunchHelper info(len);
#pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
                    {
                        auto threadNum = omp_get_thread_num();
                        Nd4jLong threadOffset = info.getThreadOffset(threadNum);
                        auto xi = x + xEws * threadOffset;
                        auto zi = z + zEws * threadOffset;
#pragma omp simd
                        for (Nd4jLong j = 0; j < info.getItersPerThread(threadNum); j++)
                            zi[j*zEws] = OpType::op(xi[j*xEws], extraParams);
                    }
                }
                else {
                            
                    const bool xSimpe = shape::isStrideSimple(xShapeInfo);
                    const bool zSimpe = shape::isStrideSimple(zShapeInfo);

                    nd4j::OmpLaunchHelper info(len);
                   
                    if(xSimpe) {
                        
                        #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
                        {
                            auto threadNum = omp_get_thread_num();                    
                            Nd4jLong threadOffset = info.getThreadOffset(threadNum);                            
                            auto xi = x + xEws * threadOffset;    
                            #pragma omp simd
                            for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
                                Nd4jLong zOffset = shape::getIndexOffset(i+threadOffset, zShapeInfo, len);
                                z[zOffset] = OpType::op(xi[i*xEws], extraParams);
                            }
                        }
                    }
                    else if(zSimpe) {

                        #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
                        {
                            auto threadNum = omp_get_thread_num();                    
                            Nd4jLong threadOffset = info.getThreadOffset(threadNum);                            
                            auto zi = z + zEws * threadOffset;    
                            #pragma omp simd
                            for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
                                Nd4jLong xOffset = shape::getIndexOffset(i+threadOffset, xShapeInfo, len);
                                zi[i*zEws] = OpType::op(x[xOffset], extraParams);
                            }
                        }
                    }
                    else if(shape::equalsStrict(xShapeInfo, zShapeInfo)) {
                        
                        #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
                        {
                            auto threadNum = omp_get_thread_num();                    
                            Nd4jLong threadOffset = info.getThreadOffset(threadNum);                            
                            #pragma omp simd
                            for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
                                Nd4jLong offset = shape::getIndexOffset(i+threadOffset, xShapeInfo, len);
                                z[offset] = OpType::op(x[offset], extraParams);
                            }
                        }
                    }
                    else {
                        #pragma omp parallel num_threads(info._numThreads) if (info._numThreads > 1) default(shared)
                        {
                            auto threadNum = omp_get_thread_num();                    
                            Nd4jLong threadOffset = info.getThreadOffset(threadNum);                            
                            #pragma omp simd
                            for (Nd4jLong i = 0; i < info.getItersPerThread(threadNum); i++) {
                                Nd4jLong xOffset = shape::getIndexOffset(i+threadOffset, xShapeInfo, len);
                                Nd4jLong zOffset = shape::getIndexOffset(i+threadOffset, zShapeInfo, len);
                                z[zOffset] = OpType::op(x[xOffset], extraParams);
                            }
                        }
                    }                                   
                }
        }

        BUILD_DOUBLE_TEMPLATE(template class ND4J_EXPORT TransformFloat, , LIBND4J_TYPES, FLOAT_TYPES);
    }
}