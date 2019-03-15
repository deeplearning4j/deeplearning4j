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

#include <ops/ops.h>
#include <loops/pairwise_transform.h>
#include <types/types.h>
#include <templatemath.h>
#include <helpers/shape.h>
#include <op_boilerplate.h>
#include <OmpLaunchHelper.h>

using namespace simdOps;

namespace functions {
    namespace pairwise_transforms {

        template <typename X, typename Y, typename Z>
        void PairWiseTransform<X, Y, Z>::exec(
                const int opNum,
                void *x,
                Nd4jLong xEws,
                void *y,
                Nd4jLong yEws,
                void *z,
                Nd4jLong zEws,
                void *extraParams,
                Nd4jLong n) {
            DISPATCH_BY_OPNUM_TTT(exec, PARAMS(x,
                                              xEws,
                                              y,
                                              yEws,
                                              z,
                                              zEws,
                                              extraParams,
                                              n), PAIRWISE_TRANSFORM_OPS);
        };



        template <typename X, typename Y, typename Z>
        template <typename OpType>
        void PairWiseTransform<X, Y, Z>::exec(void *vx, Nd4jLong xEws,
                                            void *vy, Nd4jLong yEws,
                                            void *vz, Nd4jLong zEws,
                                            void *vextraParams,
                                            const Nd4jLong n) {

            auto x = reinterpret_cast<X *>(vx);
            auto y = reinterpret_cast<Y *>(vy);
            auto z = reinterpret_cast<Z *>(vz);
            auto extraParams = reinterpret_cast<Z *>(vextraParams);

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
    }
}
