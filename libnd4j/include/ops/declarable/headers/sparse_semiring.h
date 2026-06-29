/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef SAMEDIFF_SPARSE_SEMIRING_H
#define SAMEDIFF_SPARSE_SEMIRING_H
#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

/**
 * CSR sparse matrix-vector product under a semiring: y = A ⊗ x
 *
 * The operation replaces the conventional (plus, times) arithmetic with the
 * algebraic structure selected by the semiring IArg.  Supported semirings:
 *   0 = PLUS_TIMES    (standard: y[i] = Σ A[i,j] * x[j])
 *   1 = MIN_PLUS      (tropical min-plus: y[i] = min_j(A[i,j] + x[j]))
 *   2 = MAX_PLUS      (max-plus:          y[i] = max_j(A[i,j] + x[j]))
 *   3 = OR_AND        (boolean: y[i] = OR_j(A[i,j] != 0 AND x[j] != 0))
 *   4 = MIN_TIMES     (min-times:         y[i] = min_j(A[i,j] * x[j]))
 *
 * Inputs:
 *   [0] values  – 1D [nnz], floating dtype  (non-zero values of A)
 *   [1] colIdx  – 1D [nnz], INT32 or INT64  (column indices)
 *   [2] rowPtr  – 1D [rows+1], same INT     (row pointer array)
 *   [3] x       – dense 1D vector, length = cols
 * IArgs:
 *   [0] rows
 *   [1] cols
 *   [2] semiring  (0..4, see table above)
 * Output:
 *   [0] y dense 1D, length = rows
 */
#if NOT_EXCLUDED(OP_csr_spmv_semiring)
DECLARE_CUSTOM_OP(csr_spmv_semiring, 4, 1, false, 0, 3);
#endif

/**
 * CSR sparse matrix-matrix product under a semiring: C = A ⊗ B
 *
 * The operation replaces the conventional (plus, times) arithmetic with the
 * algebraic structure selected by the semiring IArg.  Supported semirings:
 *   0 = PLUS_TIMES    (standard: C[i,k] = Σ_j A[i,j] * B[j,k])
 *   1 = MIN_PLUS      (tropical min-plus)
 *   2 = MAX_PLUS      (max-plus)
 *   3 = OR_AND        (boolean or-and)
 *   4 = MIN_TIMES     (min-times)
 *
 * Inputs:
 *   [0] values  – 1D [nnz], floating dtype  (non-zero values of A)
 *   [1] colIdx  – 1D [nnz], INT32 or INT64  (column indices)
 *   [2] rowPtr  – 1D [rows+1], same INT     (row pointer array)
 *   [3] B       – dense 2D matrix, shape [cols, n]
 * IArgs:
 *   [0] rows
 *   [1] cols
 *   [2] semiring  (0..4, see table above)
 * Output:
 *   [0] C dense 2D, shape [rows, n]
 */
#if NOT_EXCLUDED(OP_csr_spmm_semiring)
DECLARE_CUSTOM_OP(csr_spmm_semiring, 4, 1, false, 0, 3);
#endif

}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_SEMIRING_H
