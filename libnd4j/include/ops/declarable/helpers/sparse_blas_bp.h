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

#ifndef SAMEDIFF_SPARSE_BLAS_BP_H
#define SAMEDIFF_SPARSE_BLAS_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for CSR sparse matrix-vector product: y = op(A) * x
 *
 * Given upstream gradient gradY (same shape as y), computes:
 *   dAValues[k at (i,j)] = gradY[i]*x[j]   (transposeA=0)
 *                         = gradY[j]*x[i]   (transposeA=1)
 *   dx = op_{!transposeA}(A) * gradY
 *        i.e. A^T*gradY when transposeA=0, A*gradY when transposeA=1
 *
 * @param values    1D [nnz]     float CSR non-zero values of A
 * @param colIdx    1D [nnz]     int   column indices
 * @param rowPtr    1D [rows+1]  int   row pointers
 * @param x         Dense input vector to the forward op
 * @param gradY     Upstream gradient (same shape as y)
 * @param dAValues  Output: gradient w.r.t. values [nnz], pre-zeroed
 * @param dx        Output: gradient w.r.t. x, pre-zeroed
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 * @param transposeA  0 = forward was A*x, 1 = forward was A^T*x
 */
SD_LIB_HIDDEN void csr_spmv_bp(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                 NDArray& x, NDArray& gradY,
                                 NDArray& dAValues, NDArray& dx,
                                 sd::LongType rows, sd::LongType cols, int transposeA);

/**
 * Backward pass for CSR sparse matrix-matrix product: C = op(A) * B
 *
 * Given upstream gradient gradC (same shape as C), computes:
 *   dAValues = SDDMM(rowPtr, colIdx, gradC, B)  (transposeA=0)
 *            = SDDMM(rowPtr, colIdx, B, gradC)  (transposeA=1)
 *   dB = op_{!transposeA}(A) * gradC
 *
 * @param values    1D [nnz]     float CSR non-zero values of A
 * @param colIdx    1D [nnz]     int   column indices
 * @param rowPtr    1D [rows+1]  int   row pointers
 * @param B         Dense input matrix to the forward op
 * @param gradC     Upstream gradient (same shape as C)
 * @param dAValues  Output: gradient w.r.t. values [nnz], pre-zeroed
 * @param dB        Output: gradient w.r.t. B, pre-zeroed
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 * @param transposeA  0 = forward was A*B, 1 = forward was A^T*B
 */
SD_LIB_HIDDEN void csr_spmm_bp(NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                                 NDArray& B, NDArray& gradC,
                                 NDArray& dAValues, NDArray& dB,
                                 sd::LongType rows, sd::LongType cols, int transposeA);

/**
 * Backward pass for SDDMM (sampled dense-dense matrix multiplication).
 *
 * Forward: out[k at (i,j)] = D1[i,:] · D2[j,:]
 *
 * Given upstream gradient gradOut [nnz], computes:
 *   dD1[i,:] += sum_{k: row=i} gradOut[k] * D2[j_k,:]
 *   dD2[j,:] += sum_{k: col=j} gradOut[k] * D1[i_k,:]
 *
 * @param rowPtr   1D [rows+1] int row pointers (CSR sparsity pattern)
 * @param colIdx   1D [nnz]   int column indices
 * @param D1       Dense [rows, p] left factor
 * @param D2       Dense [cols, p] right factor
 * @param gradOut  1D [nnz]  upstream gradient
 * @param dD1      Output: gradient w.r.t. D1 [rows, p], pre-zeroed
 * @param dD2      Output: gradient w.r.t. D2 [cols, p], pre-zeroed
 * @param rows     Number of rows
 * @param cols     Number of columns
 */
SD_LIB_HIDDEN void sddmm_bp(NDArray& rowPtr, NDArray& colIdx,
                              NDArray& D1, NDArray& D2, NDArray& gradOut,
                              NDArray& dD1, NDArray& dD2,
                              sd::LongType rows, sd::LongType cols);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_BLAS_BP_H
