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
// Created by remote on 2018-09-20.
//

#include <loops/pairwise_transform.h>
#include <types/types.h>

using namespace simdOps;

namespace functions {
    namespace pairwise_transforms {

        template <typename X, typename Y>
        void PairWiseTransform<X, Y>::exec(
                const int opNum,
                void *dx,
                Nd4jLong xStride,
                void *y,
                Nd4jLong yStride,
                void *result,
                Nd4jLong resultStride,
                void *extraParams,
                Nd4jLong n) {
            DISPATCH_BY_OPNUM_TT(exec, PARAMS(dx,
                                              xStride,
                                              y,
                                              yStride,
                                              result,
                                              resultStride,
                                              extraParams,
                                              n), PAIRWISE_TRANSFORM_OPS);
        };



        template <typename X, typename Y>
        template <typename OpType>
        void PairWiseTransform<X, Y>::exec(void *vx,
                  Nd4jLong xStride,
                  void *vy,
                  Nd4jLong yStride,
                  void *vresult,
                  Nd4jLong resultStride,
                  void *vextraParams,
                  const Nd4jLong n) {
            auto dx = reinterpret_cast<X *>(vx);
            auto y = reinterpret_cast<Y *>(vy);
            auto result = reinterpret_cast<X *>(vresult);
            auto extraParams = reinterpret_cast<X *>(vextraParams);

            int elementsPerThread = n / ELEMENT_THRESHOLD;
            int _threads = nd4j::math::nd4j_max<int>(1, elementsPerThread);
            _threads = nd4j::math::nd4j_min<int>(_threads, omp_get_max_threads());

            int span = (n / _threads) + 8;

            if (xStride == 1 && yStride == 1 && resultStride == 1) {
                if (_threads > 1) {
#pragma omp parallel num_threads(_threads) if (_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jLong tid = omp_get_thread_num();
                        Nd4jLong start = span * tid;
                        Nd4jLong end = span * (tid + 1);
                        if (end > n) end = n;
#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            result[i] = OpType::op(dx[i], y[i], extraParams);
                        }
                    }
                } else {
#pragma omp simd
                    for (Nd4jLong i = 0; i < n; i++) {
                        result[i] = OpType::op(dx[i], y[i], extraParams);
                    }
                }
            }
            else {
                if (_threads > 1) {
#pragma omp parallel num_threads(_threads) if (_threads>1) proc_bind(AFFINITY) default(shared)
                    {
                        Nd4jLong tid = omp_get_thread_num();
                        Nd4jLong start = span * tid;
                        Nd4jLong end = span * (tid + 1);
                        if (end > n) end = n;

#pragma omp simd
                        for (Nd4jLong i = start; i < end; i++) {
                            result[i * resultStride] = OpType::op(dx[i * xStride], y[i * yStride], extraParams);
                        }
                    }
                } else {
#pragma omp simd
                    for (Nd4jLong i = 0; i < n; i++) {
                        result[i * resultStride] = OpType::op(dx[i * xStride], y[i * yStride], extraParams);
                    }
                }
            }
        }


        BUILD_PAIRWISE_TEMPLATE(template class ND4J_EXPORT PairWiseTransform, , LIBND4J_TYPES, LIBND4J_TYPES);
    }
}
