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
                Nd4jLong xEws,
                void *z,
                Nd4jLong zEws,
                void *extraParams,
                const Nd4jLong n) {
            DISPATCH_BY_OPNUM_T(exec, PARAMS(x, xEws, z, zEws, extraParams, n), TRANSFORM_SAME_OPS);
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
                             Nd4jLong xEws,
                             void *vz,
                             Nd4jLong zEws,
                             void *vextraParams,
                             const Nd4jLong len) {
                
                auto x = reinterpret_cast<X *>(vx);
                auto z = reinterpret_cast<X *>(vz);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                if (len < ELEMENT_THRESHOLD) {
// FIXME: proper reduction to be used here
                    for (Nd4jLong i = 0; i < len; i++) {
                        z[i*zEws] = OpType::op(x[i*xEws], extraParams);
                    }
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
                        z[j*zEws] = OpType::op(x[j*xEws], extraParams);
                }

        }

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT TransformSame, , LIBND4J_TYPES);
    }
}