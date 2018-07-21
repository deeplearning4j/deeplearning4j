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
//  @author raver119@gmail.com
//
#ifndef LIBND4J_HEADERS_BLAS_H
#define LIBND4J_HEADERS_BLAS_H

#include <ops/declarable/headers/common.h>

namespace nd4j {
    namespace ops {
        
        /**
         * This op is general matmum implementation. Depending on inputs dimensionality output result might be different.
         * matrix x matrix = BLAS gemm
         * vector x matrix = BLAS gemm
         * vector x vector = BLAS dot
         * vector x scalar = element-wise mul
         * scalar x vector = element-wise mul
         *
         * Optional T arguments:
         * 0: alpha (where applicable)
         * 1: beta (where applicable)
         *
         * Optional Integer arguments:
         * 0: transA (where applicable)
         * 1: transB (where applicable)
         */
        #if NOT_EXCLUDED(OP_matmul)
        DECLARE_CUSTOM_OP(matmul, 2, 1, false, 0, -2);
        #endif

        /**
         * tensorMmul/tensorDot operation
         * takes 2 ndarrays, and 2 sets of axes
         *
         * Integer argumens map:
         * IArgs[0] - number of axes along for first array
         * IArgs[1]... axes values for first array
         * IArgs[] - number of axes along for second array
         * IArgs[1]... axes values for second array
         */
        #if NOT_EXCLUDED(OP_tensormmul)
        DECLARE_CUSTOM_OP(tensormmul, 2, 1, false, 0, -1);   
        #endif

        /**
         * This op is simple implementation of BLAS AXPY method.
         * Math is: y += a * x;
         */
        #if NOT_EXCLUDED(OP_axpy)
        DECLARE_CONFIGURABLE_OP(axpy, 2, 1, false, -2, 0);
        #endif

        /**
         * This operation implements batched matrix multiplication
         * Expected arguments:
         * alpha: vector of T
         * beta: vector of T
         * ...: A, B matrices sequentially. i.e: AAAAABBBBB
         * 
         * Integer arguments:
         * transA, transB, M, N, K, ldA, ldB, ldC - usual BLAS gemm arguments
         * batchCount - number of operations in this batch
         * 
         * PLEASE NOTE: M, N, K, ldA, ldB, ldC should be equal for all matrices within batch.
         */
        #if NOT_EXCLUDED(OP_batched_gemm)
        DECLARE_CUSTOM_OP(batched_gemm, -1, -1, false, 0, 9);
        #endif

        /**
         * performs singular value decomposition (SVD) of one one or more matrices, evaluates the SVD of each inner-most 2D matrix in input array:
         * x[..., :, :] = u[..., :, :] * s[...,:] * transpose(v[..., :, :]) 
         *
         * Input array:
         * x[..., Rows, Cols], the necessary condition is: rank of x >= 2
         * 
         * Outputs arrays:
         * s[..., diagSize] - array with singular values which are stored in decreasing order, diagSize is smaller among Rows and Cols
         * u[..., Rows, Rows] if IArgs[1] is true, else u[..., Rows, diagSize] - array with right singular vectors
         * v[..., Cols, Cols] if IArgs[1] is true, else v[..., Cols, diagSize] - array with left singular vectors
         * 
         * Integer arguments:
         * IArgs[0] - bool, whether to calculate u and v, s is calculated in any case
         * IArgs[1] - bool, whether to calculate full-sized u and v
         * IArgs[2] - the number of cols or rows which determines what algorithm to use. More precisely:
         *            if diagSize < IArgs[2] then Jacobi algorithm is used, in opposite case the Divide-And-Conquer is applied
         *            Recommended value is 16. 
         */
        #if NOT_EXCLUDED(OP_svd)
        DECLARE_CUSTOM_OP(svd, 1, 1, false, 0, 3);   
        #endif
    }
}

#endif