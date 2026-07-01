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

#ifndef SAMEDIFF_SPARSE_BSR_BP_H
#define SAMEDIFF_SPARSE_BSR_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for bsr_to_dense.
 *
 * The forward scatter places block k (block-row br, block-col bc = bsrColIdx[k])
 * into dense rows [br*bd, (br+1)*bd) and cols [bc*bd, (bc+1)*bd).
 * The gradient flows back as a gather from the same region:
 *   dBsrValues[block_k*bd*bd + lr*bd + lc] = gradDense[br*bd + lr, bc*bd + lc]
 *
 * Pure gather — no atomics.
 *
 * @param bsrColIdx  1D [nnzb]            INT — BSR block column indices
 * @param bsrRowPtr  1D [mb+1]            INT — BSR block row pointers  (mb = rows/blockDim)
 * @param gradDense  2D [rows, cols]       float — upstream gradient w.r.t. dense output
 * @param dBsrValues 1D [nnzb*bd*bd]      float — OUTPUT gradient w.r.t. BSR values
 * @param rows       total number of rows  (= mb * blockDim)
 * @param cols       total number of columns (= nb * blockDim)
 * @param blockDim   block size (square blocks of size blockDim × blockDim)
 */
SD_LIB_HIDDEN void bsr_to_dense_bp(
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradDense, NDArray& dBsrValues,
    LongType rows, LongType cols, LongType blockDim);

/**
 * Backward pass for bsr_spmm: C = A_bsr * B
 *
 * Given upstream gradient gradC [rows, n], computes:
 *   dBsrValues[k, lr, lc] = sum_{j=0}^{n-1} gradC[br*bd + lr, j] * B[bc*bd + lc, j]
 *                          = gradC[br*bd+lr, :] · B[bc*bd+lc, :]
 *   dB = A_bsr^T * gradC  (sparse transpose multiply)
 *
 * dBsrValues and dB must be pre-zeroed (OUTPUT_NULLIFIED).
 *
 * @param bsrValues  1D [nnzb*bd*bd] float — BSR non-zero block values of A
 * @param bsrColIdx  1D [nnzb]       INT   — BSR block column indices
 * @param bsrRowPtr  1D [mb+1]       INT   — BSR block row pointers
 * @param B          2D [cols, n]    float — dense right factor
 * @param gradC      2D [rows, n]    float — upstream gradient w.r.t. C
 * @param dBsrValues 1D [nnzb*bd*bd] float — OUTPUT gradient w.r.t. BSR values, pre-zeroed
 * @param dB         2D [cols, n]    float — OUTPUT gradient w.r.t. B, pre-zeroed
 * @param rows       total row count of A (= mb * blockDim)
 * @param cols       total col count of A (= nb * blockDim)
 * @param blockDim   block size
 */
SD_LIB_HIDDEN void bsr_spmm_bp(
    NDArray& bsrValues, NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& B, NDArray& gradC,
    NDArray& dBsrValues, NDArray& dB,
    LongType rows, LongType cols, LongType blockDim);

/**
 * Backward pass for csr_to_bsr.
 *
 * The forward pass groups CSR entry k (at global row r, col c) into BSR block
 * (br = r/bd, bc = c/bd) with local position (lr = r%bd, lc = c%bd).
 * The gradient flows back as:
 *   dCsrValues[k] = gradBsrValues[block_k * bd*bd + lr*bd + lc]
 * where block_k is the BSR block index for (br, bc).
 *
 * Pure gather — each CSR entry maps to exactly one BSR slot.
 *
 * @param csrColIdx      1D [nnz]         INT — CSR column indices
 * @param csrRowPtr      1D [rows+1]      INT — CSR row pointers
 * @param bsrColIdx      1D [nnzb]        INT — BSR block column indices (forward output)
 * @param bsrRowPtr      1D [mb+1]        INT — BSR block row pointers (forward output)
 * @param gradBsrValues  1D [nnzb*bd*bd]  float — upstream gradient w.r.t. BSR values
 * @param dCsrValues     1D [nnz]         float — OUTPUT gradient w.r.t. CSR values
 * @param rows           total number of rows in the CSR matrix
 * @param cols           total number of columns
 * @param blockDim       block size
 */
SD_LIB_HIDDEN void csr_to_bsr_bp(
    NDArray& csrColIdx, NDArray& csrRowPtr,
    NDArray& bsrColIdx, NDArray& bsrRowPtr,
    NDArray& gradBsrValues, NDArray& dCsrValues,
    LongType rows, LongType cols, LongType blockDim);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_BSR_BP_H
