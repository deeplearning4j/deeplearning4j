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
#include <loops/transform_same.h>
#include <loops/legacy_ops.h>

using namespace simdOps;

namespace functions {
    namespace transform {

        template <typename X>
        void TransformSame<X>::exec(int opNum,
                void *x,
                Nd4jLong xStride,
                void *z,
                Nd4jLong zStride,
                void *extraParams,
                const Nd4jLong n) {
            DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xStride, z, zStride, extraParams, n), TRANSFORM_SAME_OPS);
		}

        template <typename X>
        void TransformSame<X>::exec(
				int opNum,
				void *x,
				Nd4jLong *xShapeInfo,
				void *z,
				Nd4jLong *zShapeInfo,
				void *extraParams,
				Nd4jLong *tadShapeInfo,
				Nd4jLong *tadOffsets) {
                    DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xShapeInfo, z, zShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_SAME_OPS);
		}

        template <typename X>
        template<typename OpType>
		void _CUDA_H TransformSame<X>::exec(void *vx, Nd4jLong *xShapeInfo,
                                            void *vz, Nd4jLong *zShapeInfo,
                                            void *vextraParams,
                                            Nd4jLong *tadShapeInfo, Nd4jLong *tadOffsets) {

		        auto x = reinterpret_cast<X *>(vx);
		        auto z = reinterpret_cast<X *>(vz);
		        auto extraParams = reinterpret_cast<X *>(vextraParams);

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
                    exec<OpType>(x,xEws,z,zEws,extraParams,len);
                }
                else {
                            
                    const bool xSimpe = shape::isStrideSimple(xShapeInfo);
                    const bool zSimpe = shape::isStrideSimple(zShapeInfo);
                   
                    if(xSimpe) {
                        
                        if(xEws == 1) {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)
                                z[shape::getIndexOffset(i, zShapeInfo, len)] = OpType::op(x[i], extraParams);                       
                        }
                        else {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)
                                z[shape::getIndexOffset(i, zShapeInfo, len)] = OpType::op(x[i*xEws], extraParams);
                        }
                    }
                    else if(zSimpe) {

                        if(zEws == 1) {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)                                
                                z[i] = OpType::op(x[shape::getIndexOffset(i, xShapeInfo, len)], extraParams);         
                        }
                        else {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)                                
                                z[i*zEws] = OpType::op(x[shape::getIndexOffset(i, xShapeInfo, len)], extraParams);
                        }
                    }
                    else {
                        #pragma omp parallel for schedule(guided)
                        for(Nd4jLong i = 0; i < len; ++i)
                            z[shape::getIndexOffset(i, zShapeInfo, len)] = OpType::op(x[shape::getIndexOffset(i, xShapeInfo, len)], extraParams);
                    }
                }
            }


        template <typename X>
        template <typename OpType>
		void _CUDA_H TransformSame<X>::exec(void *vx,
                             Nd4jLong xStride,
                             void *vz,
                             Nd4jLong zStride,
                             void *vextraParams,
                             const Nd4jLong n) {
                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<X *>(vz);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                int num_threads = n / ELEMENT_THRESHOLD;
                if(num_threads < 1)
                    num_threads = 1;
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                int span = (n / num_threads) + 8;

                if (xStride == 1 && zStride == 1) {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        int tid = omp_get_thread_num();
                        Nd4jLong start = span * tid;
                        Nd4jLong end = span * (tid + 1);
                        if (end > n)
                            end = n;
#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            z[i] = OpType::op(x[i], extraParams);
                        }
                    }
                } else {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        int tid = omp_get_thread_num();
                        Nd4jLong start = span * tid;
                        Nd4jLong end = span * (tid + 1);
                        if (end > n)
                            end = n;

#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            z[i*zStride] = OpType::op(x[i * xStride], extraParams);
                    }
                }
            }
        }

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT TransformSame, , LIBND4J_TYPES);
    }
}