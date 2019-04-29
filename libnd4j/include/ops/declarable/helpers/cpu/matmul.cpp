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
// Created by raver119 on 20.12.17.
//

#include <ops/declarable/helpers/matmul.h>

namespace nd4j {
    namespace ops {
        namespace helpers {
            template <typename X, typename Y, typename Z>
            void __matmul(NDArray *vA, NDArray *vB, NDArray *vC, int transA, int transB, double alpha, double beta) {
                CBLAS_TRANSPOSE tA = (CBLAS_TRANSPOSE) transA;
                CBLAS_TRANSPOSE tB = (CBLAS_TRANSPOSE) transB;

                int M = vA->sizeAt(0);
                int N = vB->sizeAt(1);
                int K = vA->sizeAt(1);

                int ldA = transA == CblasNoTrans ? M : K;
                int ldB = transB == CblasNoTrans ? K : N;
                int ldC = M;

                auto A = reinterpret_cast<X *>(vA->buffer());
                auto B = reinterpret_cast<Y *>(vB->buffer());
                auto C = reinterpret_cast<Z *>(vC->buffer());

                PRAGMA_OMP_PARALLEL_FOR_SIMD_COLLAPSE(2)
                for (int m = 0; m < M; ++m) {
                    for (int n = 0; n < N; ++n) {
                        Z c_mnp = 0;

                        for (int k = 0; k < K; ++k)
                            c_mnp += (Z) A[tA == CblasNoTrans ? (m + k * ldA) : (m * ldA + k)] * (Z) B[tB == CblasNoTrans ? (k + n * ldB) : (k * ldB + n)];

                        C[m + n * ldC] = (Z) alpha * (Z) c_mnp + (Z) beta * (Z) C[m + n * ldC];
                    }
                }
            }


            void _matmul(graph::LaunchContext* context, NDArray *vA, NDArray *vB, NDArray *vC, int transA, int transB, double alpha, double beta) {
                BUILD_TRIPLE_SELECTOR(vA->dataType(), vB->dataType(), vC->dataType(), __matmul, (vA, vB, vC, transA, transB, alpha, beta), LIBND4J_TYPES, LIBND4J_TYPES, LIBND4J_TYPES);
            }

            BUILD_TRIPLE_TEMPLATE(template void __matmul, (NDArray *A, NDArray *B, NDArray *C, int transA, int transB, double alpha, double beta), LIBND4J_TYPES, LIBND4J_TYPES, LIBND4J_TYPES);
        }
    }
}
