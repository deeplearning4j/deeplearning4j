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

#ifndef SAMEDIFF_SPARSE_ADD_BP_H
#define SAMEDIFF_SPARSE_ADD_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for csr_add: C = A + B (CSR union sparsity).
 *
 * For each non-zero entry k in A at column aColIdx[k] in row r:
 *   dAValues[k] = gradCValues[pos_C(r, aColIdx[k])]
 * For each non-zero entry k in B at column bColIdx[k] in row r:
 *   dBValues[k] = gradCValues[pos_C(r, bColIdx[k])]
 *
 * Positions in C are found via binary search in C's sorted per-row column list.
 * This is a pure gather — no atomics required (each A/B entry maps to exactly
 * one C entry by construction of the sorted-merge union sparsity).
 *
 * @param aColIdx     1D [nnzA] INT — column indices of A (sorted per row)
 * @param aRowPtr     1D [m+1]  INT — row pointers of A
 * @param bColIdx     1D [nnzB] INT — column indices of B (sorted per row)
 * @param bRowPtr     1D [m+1]  INT — row pointers of B
 * @param cColIdx     1D [nnzC] INT — column indices of C (forward output)
 * @param cRowPtr     1D [m+1]  INT — row pointers of C (forward output)
 * @param gradCValues 1D [nnzC] float — upstream gradient w.r.t. C values
 * @param dAValues    1D [nnzA] float — OUTPUT gradient w.r.t. A values
 * @param dBValues    1D [nnzB] float — OUTPUT gradient w.r.t. B values
 * @param m           number of rows
 * @param n           number of columns
 */
SD_LIB_HIDDEN void csr_add_bp(
    NDArray& aColIdx, NDArray& aRowPtr,
    NDArray& bColIdx, NDArray& bRowPtr,
    NDArray& cColIdx, NDArray& cRowPtr,
    NDArray& gradCValues,
    NDArray& dAValues, NDArray& dBValues,
    sd::LongType m, sd::LongType n);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_ADD_BP_H
