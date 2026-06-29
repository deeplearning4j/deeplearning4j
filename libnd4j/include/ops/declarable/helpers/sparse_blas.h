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

#ifndef SAMEDIFF_SPARSE_BLAS_H
#define SAMEDIFF_SPARSE_BLAS_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * CSR sparse matrix-vector product: y = op(A) * x
 *
 * @param values    1D [nnz] float array of CSR non-zero values
 * @param colIdx    1D [nnz] int array of column indices
 * @param rowPtr    1D [rows+1] int array of row pointers
 * @param x         Dense 1D input vector
 * @param y         Dense 1D output vector (pre-zeroed by OUTPUT_NULLIFIED)
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 * @param transposeA  0 = y = A*x, 1 = y = A^T*x
 */
SD_LIB_HIDDEN void csr_spmv(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                              NDArray& x, NDArray& y,
                              sd::LongType rows, sd::LongType cols, int transposeA);

/**
 * CSR sparse matrix-matrix product: C = op(A) * B
 *
 * @param values    1D [nnz] float array of CSR non-zero values
 * @param colIdx    1D [nnz] int array of column indices
 * @param rowPtr    1D [rows+1] int array of row pointers
 * @param B         Dense 2D input matrix
 * @param C         Dense 2D output matrix (pre-zeroed by OUTPUT_NULLIFIED)
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 * @param transposeA  0 = C = A*B, 1 = C = A^T*B
 */
SD_LIB_HIDDEN void csr_spmm(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                              NDArray& B, NDArray& C,
                              sd::LongType rows, sd::LongType cols, int transposeA);

/**
 * SDDMM (sampled dense-dense matrix multiplication).
 * Computes sparse output values at the CSR sparsity pattern:
 *   outValues[k] = sum_{l=0}^{p-1} D1[i,l] * D2[j,l]
 * where i is the row of nonzero k and j = colIdx[k].
 *
 * @param rowPtr    1D [rows+1] int array of row pointers
 * @param colIdx    1D [nnz] int array of column indices
 * @param D1        Dense [rows, p] matrix
 * @param D2        Dense [cols, p] matrix
 * @param outValues 1D [nnz] float output values
 * @param rows      Number of rows
 * @param cols      Number of columns
 */
SD_LIB_HIDDEN void sddmm(NDArray& rowPtr, NDArray& colIdx,
                          NDArray& D1, NDArray& D2, NDArray& outValues,
                          sd::LongType rows, sd::LongType cols);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_BLAS_H
