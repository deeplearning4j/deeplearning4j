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
// @author raver119@gmail.com
//

#ifndef LIBND4J_GEMM_H
#define LIBND4J_GEMM_H

#include <cblas.h>
#include <templatemath.h>
#include <op_boilerplate.h>


namespace nd4j {
     namespace blas {
         template <typename T>
         static void * transpose(int orderSource, int orderTarget, int rows, int cols, void *source);

         static inline int linearIndexC(int rows, int cols, int r, int c);
         static inline int linearIndexF(int rows, int cols, int r, int c);

         template <typename X, typename Y, typename Z>
         class GEMM {
         protected:
         public:
             static void op(int Order, int TransA, int TransB, int M, int N, int K, double alpha, void *A, int lda, void *B, int ldb, double beta, void *C, int ldc);
         };

         template <typename X, typename Y, typename Z>
         class GEMV : public nd4j::blas::GEMM<X, Y, Z>{
         public:
             static void op(int TRANS, int M, int N, double alpha, void* vA, int lda, void* vX, int incx, double beta, void* vY, int incy );
         };


         int FORCEINLINE nd4j::blas::linearIndexC(int rows, int cols, int r, int c) {
             return (r * cols + c);
         }

         int FORCEINLINE nd4j::blas::linearIndexF(int rows, int cols, int r, int c) {
             return (c * rows + r);
         }

    }
}

#endif //LIBND4J_GEMM_H
