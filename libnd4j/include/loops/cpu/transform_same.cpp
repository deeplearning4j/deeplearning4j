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
                void *dx,
                Nd4jLong xStride,
                void *result,
                Nd4jLong resultStride,
                void *extraParams,
                const Nd4jLong n) {
            DISPATCH_BY_OPNUM_T(exec, PARAMS(dx, xStride, result, resultStride, extraParams, n), TRANSFORM_SAME_OPS);
		}

        template <typename X>
        void TransformSame<X>::exec(
				int opNum,
				void *dx,
				Nd4jLong *xShapeInfo,
				void *result,
				Nd4jLong *resultShapeInfo,
				void *extraParams,
				Nd4jLong *tadShapeInfo,
				Nd4jLong *tadOffsets) {
                    DISPATCH_BY_OPNUM_T(exec, PARAMS(dx, xShapeInfo, result, resultShapeInfo, extraParams, tadShapeInfo, tadOffsets), TRANSFORM_SAME_OPS);
		}

        template <typename X>
        template<typename OpType>
		void _CUDA_H TransformSame<X>::exec(
                    void *vx,
                    Nd4jLong *xShapeInfo,
                    void *vresult,
                    Nd4jLong *resultShapeInfo,
                    void *vextraParams,
                    Nd4jLong *tadShapeInfo,
                    Nd4jLong *tadOffsets) {

		        auto dx = reinterpret_cast<X *>(vx);
		        auto result = reinterpret_cast<X *>(vresult);
		        auto extraParams = reinterpret_cast<X *>(vextraParams);

                if(OpType::requiresSpecial) {
                    OpType::execSpecial(dx, xShapeInfo, result,resultShapeInfo, extraParams, tadShapeInfo, tadOffsets);
                    return;
                }

                auto n = shape::length(xShapeInfo);
                auto xElementWiseStride = shape::elementWiseStride(xShapeInfo);
                auto resultElementWiseStride = shape::elementWiseStride(resultShapeInfo);

                if(xElementWiseStride >= 1 && resultElementWiseStride >= 1 && shape::order(xShapeInfo) == shape::order(resultShapeInfo)) {
                    exec<OpType>(dx,xElementWiseStride,result,resultElementWiseStride,extraParams,n);
                }
                else {
                    Nd4jLong shapeIter[MAX_RANK];
                    Nd4jLong coord[MAX_RANK];
                    int dim;
                    Nd4jLong xStridesIter[MAX_RANK];
                    Nd4jLong resultStridesIter[MAX_RANK];
                    auto xShape = shape::shapeOf(xShapeInfo);
                    auto xStride = shape::stride(xShapeInfo);
                    auto resultStride = shape::stride(resultShapeInfo);
                    int rank = shape::rank(xShapeInfo);
                    if(PrepareTwoRawArrayIter<X>(rank,
                                                 xShape,
                                                 dx,
                                                 xStride,
                                                 result,
                                                 resultStride,
                                                 &rank,
                                                 shapeIter,
                                                 &dx,
                                                 xStridesIter,
                                                 &result,
                                                 resultStridesIter) >= 0) {
                        ND4J_RAW_ITER_START(dim, rank, coord, shapeIter);
                        {
                            // Process the innermost dimension
                            auto xIter = dx;
                            auto resultIter = result;
                            resultIter[0] = OpType::op(xIter[0], extraParams);
                        }
                        ND4J_RAW_ITER_TWO_NEXT(dim,
                                               rank,
                                               coord,
                                               shapeIter,
                                               dx,
                                               xStridesIter,
                                               result,
                                               resultStridesIter);

                }
            }
        }

        template <typename X>
        template <typename OpType>
		void _CUDA_H TransformSame<X>::exec(void *vx,
                             Nd4jLong xStride,
                             void *vresult,
                             Nd4jLong resultStride,
                             void *vextraParams,
                             const Nd4jLong n) {
                auto dx = reinterpret_cast<X *>(vx);
                auto result = reinterpret_cast<X *>(vresult);
                auto extraParams = reinterpret_cast<X *>(vextraParams);

                int elementsPerThread = n / ELEMENT_THRESHOLD;
                int num_threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
                num_threads = nd4j::math::nd4j_min<int>(num_threads, omp_get_max_threads());

                int span = (n / num_threads) + 8;

                if (xStride == 1 && resultStride == 1) {

#pragma omp parallel num_threads(num_threads) if (num_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        int tid = omp_get_thread_num();
                        Nd4jLong start = span * tid;
                        Nd4jLong end = span * (tid + 1);
                        if (end > n)
                            end = n;

#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            result[i] = OpType::op(dx[i], extraParams);
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
                            result[i*resultStride] = OpType::op(dx[i * xStride], extraParams);
                    }
                }
            }
        }

        BUILD_SINGLE_TEMPLATE(template class ND4J_EXPORT TransformSame, , LIBND4J_TYPES);
    }
}