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
// Created by raver119 on 07.10.2017.
// Modified by GS <sgazeos@gmail.com> on 3/9/2018
//

#include <gemm.h>
#include <types/types.h>

namespace nd4j {
    namespace blas {

        template <typename T>
        void* transpose(int orderSource, int orderTarget, int rows, int cols, void *vsource) {
            auto ret = new T[rows * cols];
            auto source = reinterpret_cast<T *>(vsource);

            // handle transpose in parallel
#pragma omp parallel for proc_bind(close)
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    int zIdx = orderTarget == CblasRowMajor ? linearIndexC(rows, cols, r, c) : linearIndexF(rows, cols, r, c);
                    int xIdx = orderSource == CblasColMajor ? linearIndexF(rows, cols, r, c) : linearIndexC(rows, cols, r, c);

                    ret[zIdx] = source[xIdx];
                }
            }

            return ret;
        }

        template <typename X, typename Y, typename Z>
        void GEMM<X, Y, Z>::op(int Order, int TransA, int TransB,
                       int M, int N, int K,
                       double alpha,
                       void *vA, int lda,
                       void *vB, int ldb,
                       double beta,
                       void *vC, int ldc) {

            auto A = reinterpret_cast<X *>(vA);
            auto B = reinterpret_cast<Y *>(vB);
            auto C = reinterpret_cast<Z *>(vC);

            bool transAFlag = TransA == CblasTrans;
            bool transBFlag = TransB == CblasTrans;

            if (beta == 0.0) {
                int length = M*N;
                if (length <= 8192) {
#pragma omp simd
                    for (int r = 0; r < length; r++)
                        C[r] = static_cast<Z>(0.0f);
                } else {
#pragma omp parallel for simd
                    for (int r = 0; r < length; r++)
                        C[r] = static_cast<Z>(0.0f);
                }
            }


#pragma omp parallel for simd collapse(2) proc_bind(close)
            for (int r = 0; r < M; r++) {
                for (int c = 0; c < N; c++) {
                    int zIdx = linearIndexF(M, N, r, c);

                    Z dot = static_cast<Z>(0.0f);

                    if (alpha != 0.0) {
                        int bIdx; // = linearIndexF(K, N, 0, c);
                        int aIdx;

                        for (int k = 0; k < K; k++) {
                            aIdx = (transAFlag ? linearIndexC(M, K, r, k) : linearIndexF(M, K, r, k));
                            bIdx = (transBFlag ? linearIndexC(K, N, k, c) : linearIndexF(K,N, k, c));
                            dot += static_cast<Z>(alpha) * static_cast<Z>(A[aIdx]) * static_cast<Z>(B[bIdx]);//A[aIdx]nd4j::math::nd4j_dot<T>(aX, bX, K) * alpha;
                        }
                    }

                    if (beta != 0.0) {
                        C[zIdx] = static_cast<Z>(dot + beta * C[zIdx]);
                    } else {
                        C[zIdx] = static_cast<Z>(dot);
                    }
                }
            }
        }


        template<typename X, typename Y, typename Z>
        void GEMV<X, Y, Z>::op(int TRANS, int M, int N,
                               double alpha,
                               void * vX,
                               int lda,
                               void* vY,
                               int incx,
                               double beta,
                               void* vZ,
                               int incy ) {

            auto x = reinterpret_cast<X *>(vX);
            auto y = reinterpret_cast<Y *>(vY);
            auto z = reinterpret_cast<Z *>(vZ);

            auto aT = TRANS == CblasTrans ? reinterpret_cast<X *>(nd4j::blas::transpose(CblasColMajor, CblasRowMajor, M, N, x)) : x;

#pragma omp parallel for proc_bind(close)
            for (int r = 0; r < M; r++) {
                int aIdx = linearIndexC(M, N, r, 0);
                auto aX = aT + aIdx;

                auto dot = nd4j::math::nd4j_dot<X, Y, Z>(aX, y, lda) * alpha;
                z[r] =  beta == 0.0f ? dot : dot + beta * z[r];
            }

            if (TRANS == CblasTrans)
                delete[] aT;
        }

        BUILD_TRIPLE_TEMPLATE(template class ND4J_EXPORT GEMV, , LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        BUILD_TRIPLE_TEMPLATE(template class ND4J_EXPORT GEMM, , LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
}
