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

#ifndef SAMEDIFF_SPARSE_BSR_H
#define SAMEDIFF_SPARSE_BSR_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Convert a CSR sparse matrix to BSR (block-sparse-row) format.
 *
 * @param csrValues   1D [nnz], non-zero values of A (float dtype)
 * @param csrColIdx   1D [nnz], column indices of CSR (INT32 or INT64)
 * @param csrRowPtr   1D [rows+1], row pointers of CSR (same INT dtype)
 * @param bsrValues   1D [nnzb*blockDim*blockDim], output BSR block values (same float dtype)
 * @param bsrColIdx   1D [nnzb], INT32 — block column indices
 * @param bsrRowPtr   1D [mb+1], INT32 — block row pointers  (mb = rows/blockDim)
 * @param rows        number of rows in the matrix (must be divisible by blockDim)
 * @param cols        number of columns in the matrix (must be divisible by blockDim)
 * @param blockDim    block size
 */
SD_LIB_HIDDEN void csr_to_bsr(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                               NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                               LongType rows, LongType cols, LongType blockDim);

/**
 * Convert a BSR sparse matrix to a dense NDArray.
 *
 * @param bsrValues   1D [nnzb*blockDim*blockDim], BSR block values (float dtype)
 * @param bsrColIdx   1D [nnzb], INT32 or INT64 — block column indices
 * @param bsrRowPtr   1D [mb+1], same INT dtype — block row pointers
 * @param output      Dense output [rows, cols] — must be pre-zeroed (OUTPUT_NULLIFIED)
 * @param rows        number of rows in the logical matrix
 * @param cols        number of columns in the logical matrix
 * @param blockDim    block size
 */
SD_LIB_HIDDEN void bsr_to_dense(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                                 NDArray& output, LongType rows, LongType cols, LongType blockDim);

/**
 * BSR sparse matrix-matrix multiply: C = A_bsr * B
 *
 * A_bsr is [rows, cols] in BSR format. B is dense [cols, n]. C is dense [rows, n].
 *
 * @param bsrValues   1D [nnzb*blockDim*blockDim], BSR block values (float dtype)
 * @param bsrColIdx   1D [nnzb], INT32 or INT64 — block column indices
 * @param bsrRowPtr   1D [mb+1], same INT dtype — block row pointers
 * @param B           Dense matrix [cols, n]
 * @param C           Dense output [rows, n] — must be pre-zeroed (OUTPUT_NULLIFIED)
 * @param rows        logical rows of A
 * @param cols        logical cols of A (= rows of B)
 * @param blockDim    block size
 */
SD_LIB_HIDDEN void bsr_spmm(NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
                              NDArray& B, NDArray& C,
                              LongType rows, LongType cols, LongType blockDim);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_BSR_H
