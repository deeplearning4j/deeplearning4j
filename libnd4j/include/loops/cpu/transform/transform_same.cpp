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
		void _CUDA_H TransformSame<X>::exec(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vz,
                    Nd4jLong *zShapeInfo,
                    void *vextraParams,
                    Nd4jLong *tadShapeInfo,
                    Nd4jLong *tadOffsets) {

		        auto x = reinterpret_cast<X *>(vx);
		        auto z = reinterpret_cast<X *>(vz);
		        auto extraParams = reinterpret_cast<X *>(vextraParams);

                if(OpType::requiresSpecial) {
                    OpType::execSpecial(x, xShapeInfo, z,zShapeInfo, extraParams, tadShapeInfo, tadOffsets);
                    return;
                }

                auto len = shape::length(xShapeInfo);
                auto xEws = shape::elementWiseStride(xShapeInfo);
                auto zEws = shape::elementWiseStride(zShapeInfo);

                if(xEws >= 1 && zEws >= 1 && shape::order(xShapeInfo) == shape::order(zShapeInfo)) {
                    exec<OpType>(x,xEws,z,zEws,extraParams,len);
                }
                else {
                    Nd4jLong shapeIter[MAX_RANK];
                    
                    int dim;
                    Nd4jLong xStridesIter[MAX_RANK];
                    Nd4jLong zStridesIter[MAX_RANK];
                    const auto xShape = shape::shapeOf(xShapeInfo);
                    const auto zShape = shape::shapeOf(zShapeInfo);
                    const auto xStride = shape::stride(xShapeInfo);
                    const auto zStride = shape::stride(zShapeInfo);
                    const auto xOrder = shape::order(xShapeInfo);
                    const auto zOrder = shape::order(zShapeInfo);
                    int xRank = shape::rank(xShapeInfo);
                    int zRank = shape::rank(zShapeInfo);

                    if (0) {

                    if(xOrder == zOrder && xOrder == 'c' && xEws >= 1 && zEws >= 1) {
                        if(xEws == 1 && zEws == 1) {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)
                                z[i] = OpType::op(x[i], extraParams);
                        }
                        else if(xEws == 1 && zEws >= 1) {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)
                                z[i*zEws] = OpType::op(x[i], extraParams);   
                        }
                        else if(xEws >= 1 && zEws == 1) {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)
                                z[i] = OpType::op(x[i*xEws], extraParams);   
                        }
                        else {
                            #pragma omp parallel for schedule(guided)
                            for(Nd4jLong i = 0; i < len; ++i)
                                z[i*zEws] = OpType::op(x[i*xEws], extraParams);   
                        }
                    }
                    else if(xOrder == 'c' && xEws >= 1 && (zOrder == 'f' || zEws < 1)) {                        
                        Nd4jLong zCoord[MAX_RANK]; 
                        memset(zCoord, 0, MAX_RANK * sizeof(Nd4jLong));

                        if(xEws == 1) {
                            #pragma omp parallel for schedule(guided) firstprivate(zCoord)
                            for(Nd4jLong i = 0; i < len; ++i) {
                                shape::nextIter(zRank, zShapeInfo, zCoord);
                                Nd4jLong zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);
                                z[zOffset] = OpType::op(x[i], extraParams);                       
                            }
                        }
                        else {
                            #pragma omp parallel for schedule(guided) firstprivate(zCoord)
                            for(Nd4jLong i = 0; i < len; ++i) {
                                shape::nextIter(zRank, zShapeInfo, zCoord);
                                Nd4jLong zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);
                                z[zOffset] = OpType::op(x[i*xEws], extraParams);
                            }
                        }
                    }
                    else if(zOrder == 'c' && zEws >= 1 && (xOrder == 'f' || xEws < 1)) {

                        Nd4jLong xCoord[MAX_RANK]; 
                        memset(xCoord, 0, MAX_RANK * sizeof(Nd4jLong));

                        if(zEws == 1) {
                            printf("!!!!!!!!!\n");
                            #pragma omp parallel for schedule(guided) firstprivate(xCoord)
                            for(Nd4jLong i = 0; i < len; ++i) {
                                shape::nextIter(xRank, xShapeInfo, xCoord);
                                Nd4jLong xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                                z[i] = OpType::op(x[xOffset], extraParams);         
                            }
                        }
                        else {
                            #pragma omp parallel for schedule(guided) firstprivate(xCoord)
                            for(Nd4jLong i = 0; i < len; ++i) {
                                shape::nextIter(xRank, xShapeInfo, xCoord);
                                Nd4jLong xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                                z[i*zEws] = OpType::op(x[xOffset], extraParams);
                            }
                        }
                    }
                    else {

                        Nd4jLong xCoord[MAX_RANK]; 
                        memset(xCoord, 0, MAX_RANK * sizeof(Nd4jLong));
                        Nd4jLong zCoord[MAX_RANK]; 
                        memset(zCoord, 0, MAX_RANK * sizeof(Nd4jLong));

                        #pragma omp parallel for schedule(guided) firstprivate(xCoord, zCoord)
                        for(Nd4jLong i = 0; i < len; ++i) {

                            Nd4jLong xOffset, zOffset;
                            #pragma omp parallel sections 
                            {
                                #pragma omp section 
                                {
                                    shape::nextIter(xRank, xShapeInfo, xCoord);
                                    xOffset = shape::getOffset(0, xShape, xStride, xCoord, xRank);
                                }
                                #pragma omp section 
                                {
                                    shape::nextIter(zRank, zShapeInfo, zCoord);
                                    zOffset = shape::getOffset(0, zShape, zStride, zCoord, zRank);
                                }
                            }    
                            z[zOffset] = OpType::op(x[xOffset], extraParams);
                        }
                    } 

                    }
                    else {
                        Nd4jLong coord[MAX_RANK]; 

                    if(PrepareTwoRawArrayIter<X>(xRank, xShape,
                                                 x, xStride,
                                                 z, zStride,
                                                 &xRank, shapeIter,
                                                 &x, xStridesIter,
                                                 &z, zStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, xRank, coord, shapeIter);
                        {
                            // Process the innermost dimension
                            auto xIter = x;
                            auto resultIter = z;
                            resultIter[0] = OpType::op(xIter[0], extraParams);
                        }
                        ND4J_RAW_ITER_TWO_NEXT(dim,
                                               xRank,
                                               coord,
                                               shapeIter,
                                               x,
                                               xStridesIter,
                                               z,
                                               zStridesIter);

                    }
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

                int elementsPerThread = n / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
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