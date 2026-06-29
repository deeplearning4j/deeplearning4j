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

#ifndef SAMEDIFF_SPARSE_BLAS_SEMIRING_H
#define SAMEDIFF_SPARSE_BLAS_SEMIRING_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * GraphBLAS-style semiring codes for sparse matrix operations.
 *
 * Each semiring replaces the standard (plus, times, zero) triple with an
 * alternative (add, mul, identity) that defines how values are accumulated
 * during a sparse matrix product.
 */
enum class SemiringCode : int {
  PLUS_TIMES = 0,  ///< Standard ring:        add=+,   mul=*,  identity=0
  MIN_PLUS   = 1,  ///< Tropical min-plus:    add=min, mul=+,  identity=+inf
  MAX_PLUS   = 2,  ///< Max-plus:             add=max, mul=+,  identity=-inf
  OR_AND     = 3,  ///< Boolean OR/AND:       add=||,  mul=&&, identity=0
  MIN_TIMES  = 4,  ///< Min-times:            add=min, mul=*,  identity=+inf
};

/**
 * CSR sparse matrix-vector product under a semiring:
 *   y[r] = semiring_add over k in row r of (semiring_mul(values[k], x[colIdx[k]]))
 *
 * cuSPARSE is NOT used — semirings require custom kernels.
 *
 * @param values    1D [nnz] float array of CSR non-zero values
 * @param colIdx    1D [nnz] int array of column indices
 * @param rowPtr    1D [rows+1] int array of row pointers
 * @param x         Dense 1D input vector
 * @param y         Dense 1D output vector
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 * @param semiring  Semiring code (0–4)
 */
SD_LIB_HIDDEN void csr_spmv_semiring(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                      NDArray& x, NDArray& y,
                                      sd::LongType rows, sd::LongType cols, int semiring);

/**
 * CSR sparse matrix-matrix product under a semiring:
 *   C[r,j] = semiring_add over k in row r of (semiring_mul(values[k], B[colIdx[k],j]))
 *
 * cuSPARSE is NOT used — semirings require custom kernels.
 *
 * @param values    1D [nnz] float array of CSR non-zero values
 * @param colIdx    1D [nnz] int array of column indices
 * @param rowPtr    1D [rows+1] int array of row pointers
 * @param B         Dense 2D input matrix
 * @param C         Dense 2D output matrix
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 * @param semiring  Semiring code (0–4)
 */
SD_LIB_HIDDEN void csr_spmm_semiring(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                      NDArray& B, NDArray& C,
                                      sd::LongType rows, sd::LongType cols, int semiring);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_BLAS_SEMIRING_H
