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

#ifndef SAMEDIFF_SPARSE_CONVERT_BP_H
#define SAMEDIFF_SPARSE_CONVERT_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for csr_to_dense.
 *
 * For each non-zero entry e (0..nnz-1):
 *   dValues[e] = gradDense[i, colIdx[e]]
 * where i is the row that contains entry e (found by scanning rowPtr).
 *
 * This is a pure gather — one input element per output — so no atomics are needed.
 *
 * @param colIdx    1D [nnz], INT32/INT64 — CSR column indices
 * @param rowPtr    1D [rows+1], same INT dtype — CSR row pointers
 * @param gradDense 2D [rows, cols], float dtype — upstream gradient w.r.t. dense output
 * @param dValues   1D [nnz], same float dtype — OUTPUT gradient w.r.t. CSR values
 * @param rows      number of rows
 * @param cols      number of columns
 */
SD_LIB_HIDDEN void csr_to_dense_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& gradDense,
                                    NDArray& dValues, LongType rows, LongType cols);

/**
 * Backward pass for dense_to_csr.
 *
 * dDense is pre-zeroed (OUTPUT_NULLIFIED). For each entry e (0..nnz-1):
 *   dDense[i, colIdx[e]] = gradValues[e]
 * where i is the row of entry e (from rowPtr). In a valid CSR pattern every (i,j)
 * pair is unique, so this scatter needs no atomics.
 *
 * @param colIdx     1D [nnz], INT32/INT64 — CSR column indices
 * @param rowPtr     1D [rows+1], same INT dtype — CSR row pointers
 * @param gradValues 1D [nnz], float dtype — upstream gradient w.r.t. CSR values output
 * @param dDense     2D [rows, cols], same float dtype — OUTPUT gradient w.r.t. dense input
 * @param rows       number of rows
 * @param cols       number of columns
 */
SD_LIB_HIDDEN void dense_to_csr_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& gradValues,
                                    NDArray& dDense, LongType rows, LongType cols);

/**
 * Backward pass for csr_to_csc.
 *
 * The forward pass applies a value-permutation P: CSR(A) → CSC(A).
 * The gradient flows back through the inverse permutation P⁻¹.
 *
 * Implementation: treat (gradCscValues, cscRowIdx, cscColPtr) as the CSR representation
 * of A^T (a matrix of shape [cols, rows]), then apply csr_to_csc with dimensions
 * (cols, rows) to get values in the original CSR(A) row-major order.  Two temporary
 * INT32 index arrays are allocated and discarded internally — no raw cudaMalloc.
 *
 * @param cscRowIdx     1D [nnz], INT32 — CSC row indices (forward output[1])
 * @param cscColPtr     1D [cols+1], INT32 — CSC column pointers (forward output[2])
 * @param gradCscValues 1D [nnz], float dtype — upstream gradient w.r.t. cscValues output
 * @param dAValues      1D [nnz], same float dtype — OUTPUT gradient w.r.t. csrValues input
 * @param rows          number of rows in the original matrix A
 * @param cols          number of columns in the original matrix A
 */
SD_LIB_HIDDEN void csr_to_csc_bp(NDArray& cscRowIdx, NDArray& cscColPtr,
                                   NDArray& gradCscValues, NDArray& dAValues,
                                   LongType rows, LongType cols);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_CONVERT_BP_H
