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

#include <ops/gemm.h>
#include <types/types.h>
#include <system/Environment.h>
#include <execution/Threads.h>

namespace sd {
    namespace blas {

        template <typename T>
        void* transpose(int orderSource, int orderTarget, int rows, int cols, void *vsource) {
            auto ret = new T[rows * cols];
            auto source = reinterpret_cast<T *>(vsource);

            // handle transpose in parallel
            auto func = PRAGMA_THREADS_FOR {
                for (auto r = start; r < stop; r++) {
                    for (int c = 0; c < cols; c++) {
                        int zIdx = orderTarget == CblasRowMajor ? linearIndexC(rows, cols, r, c) : linearIndexF(rows, cols, r, c);
                        int xIdx = orderSource == CblasColMajor ? linearIndexF(rows, cols, r, c) : linearIndexC(rows, cols, r, c);

                        ret[zIdx] = source[xIdx];
                    }
                }
            };

            sd::Threads::parallel_for(func, 0, rows);

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
                Z z = 0.f;
                int length = M*N;
                if (length <= Environment::getInstance()->elementwiseThreshold()) {
                    for (int r = 0; r < length; r++)
                        C[r] = z;
                } else {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto r = start; r < stop; r++)
                            C[r] = z;
                    };
                    sd::Threads::parallel_for(func, 0, length);
                }
            }


            auto func = PRAGMA_THREADS_FOR_2D {
                for (auto r = start_x; r < stop_x; r += inc_x) {
                    for (auto c = start_y; c < stop_y; c += inc_y) {
                        int zIdx = linearIndexF(M, N, r, c);

                        Z dot = static_cast<Z>(0.0f);

                        if (alpha != 0.0) {
                            int bIdx; // = linearIndexF(K, N, 0, c);
                            int aIdx;

                            for (int k = 0; k < K; k++) {
                                aIdx = (transAFlag ? linearIndexC(M, K, r, k) : linearIndexF(M, K, r, k));
                                bIdx = (transBFlag ? linearIndexC(K, N, k, c) : linearIndexF(K, N, k, c));
                                dot += static_cast<Z>(alpha) * static_cast<Z>(A[aIdx]) * static_cast<Z>(B[bIdx]);//A[aIdx]sd::math::nd4j_dot<T>(aX, bX, K) * alpha;
                            }
                        }

                        if (beta != 0.0) {
                            C[zIdx] = static_cast<Z>(dot + static_cast<Z>(beta) * C[zIdx]);
                        } else {
                            C[zIdx] = static_cast<Z>(dot);
                        }
                    }
                }
            };

            sd::Threads::parallel_for(func, 0, M, 1, 0, N, 1);
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

            auto aT = TRANS == CblasTrans ? reinterpret_cast<X *>(sd::blas::transpose<X>(CblasColMajor, CblasRowMajor, M, N, reinterpret_cast<void *>(x))) : x;

            auto func = PRAGMA_THREADS_FOR {
                for (auto r = start; r < stop; r++) {
                    int aIdx = linearIndexC(M, N, r, 0);
                    auto aX = aT + aIdx;

                    auto dot = sd::math::nd4j_dot<X, Y, Z>(aX, y, lda) * static_cast<Z>(alpha);
                    z[r] = beta == 0.0f ? dot : dot + static_cast<Z>(beta) * z[r];
                }
            };
            sd::Threads::parallel_for(func, 0, M);

            if (TRANS == CblasTrans)
                delete[] aT;
        }

        //BUILD_TRIPLE_TEMPLATE(template class  GEMV, , LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
        //BUILD_TRIPLE_TEMPLATE(template class  GEMM, , LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
}
