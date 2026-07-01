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

#ifndef SAMEDIFF_SPARSE_SDDMM_SPARSE_BP_H
#define SAMEDIFF_SPARSE_SDDMM_SPARSE_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for csr_sddmm_sparse.
 *
 * Forward: for each target nonzero t at (row p_t, col q_t = targetColIdx[t]):
 *   outValues[t] = dot(L row p_t, M row q_t)
 *                = sum_{c: c in L-row-p_t AND c in M-row-q_t} Lvalues[k_L] * Mvalues[k_M]
 *
 * Given upstream gradient gradOut [tnnz], the backward pass computes:
 *   dLvalues[k at (p, c)] += sum_{t: row_t=p, c in M-row-q_t}
 *                              gradOut[t] * Mvalues[k'@(q_t, c)]
 *   dMvalues[k at (q, c)] += sum_{t: col_t=q, c in L-row-p_t}
 *                              gradOut[t] * Lvalues[k'@(p_t, c)]
 *
 * Multiple target positions can contribute to the same L or M value entry,
 * so sd_atomicAdd is used for accumulation.
 *
 * dLvalues and dMvalues must be pre-zeroed (OUTPUT_NULLIFIED).
 *
 * @param targetRowPtr  1D [P+1]   INT — target sparsity row pointers
 * @param targetColIdx  1D [tnnz]  INT — target sparsity column indices
 * @param LcolIdx       1D [Lnnz]  INT — column indices of L [P, R]
 * @param LrowPtr       1D [P+1]   INT — row pointers of L
 * @param McolIdx       1D [Mnnz]  INT — column indices of M [Q, R]
 * @param MrowPtr       1D [Q+1]   INT — row pointers of M
 * @param Lvalues       1D [Lnnz]  float — non-zero values of L
 * @param Mvalues       1D [Mnnz]  float — non-zero values of M
 * @param gradOut       1D [tnnz]  float — upstream gradient w.r.t. outValues
 * @param dLvalues      1D [Lnnz]  float — OUTPUT gradient w.r.t. Lvalues, pre-zeroed
 * @param dMvalues      1D [Mnnz]  float — OUTPUT gradient w.r.t. Mvalues, pre-zeroed
 * @param P             number of rows in L and in the target sparsity pattern
 * @param Q             number of rows in M (= number of cols in target pattern)
 * @param R             inner dimension (shared column space of L and M)
 */
SD_LIB_HIDDEN void csr_sddmm_sparse_bp(
    NDArray& targetRowPtr, NDArray& targetColIdx,
    NDArray& LcolIdx, NDArray& LrowPtr,
    NDArray& McolIdx, NDArray& MrowPtr,
    NDArray& Lvalues, NDArray& Mvalues,
    NDArray& gradOut,
    NDArray& dLvalues, NDArray& dMvalues,
    sd::LongType P, sd::LongType Q, sd::LongType R);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_SDDMM_SPARSE_BP_H
