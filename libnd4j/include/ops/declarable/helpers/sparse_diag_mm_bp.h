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

#ifndef SAMEDIFF_SPARSE_DIAG_MM_BP_H
#define SAMEDIFF_SPARSE_DIAG_MM_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Exact backward pass for csr_diag_mm.
 *
 * Forward: out[e] = dl[i_e] * aValues[e] * dr[j_e]
 *
 * Backward:
 *   dAValues[e]  = gradOut[e] * dl[i_e] * dr[j_e]
 *   ddl[i]       = sum_{e in row i}    gradOut[e] * aValues[e] * dr[j_e]
 *   ddr[j]       = sum_{e: col j_e==j} gradOut[e] * aValues[e] * dl[i_e]
 *
 * @param aValues   1D [nnz]     float — original nonzero values of A
 * @param aColIdx   1D [nnz]     int   — column indices of A
 * @param aRowPtr   1D [rows+1]  int   — row pointers of A
 * @param dl        1D [rows]    float — left diagonal
 * @param dr        1D [cols]    float — right diagonal
 * @param gradOut   1D [nnz]     float — upstream gradient w.r.t. outValues
 * @param dAValues  1D [nnz]     float — gradient w.r.t. aValues (output)
 * @param ddl       1D [rows]    float — gradient w.r.t. dl (output)
 * @param ddr       1D [cols]    float — gradient w.r.t. dr (output)
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 */
SD_LIB_HIDDEN void csr_diag_mm_bp(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                                   NDArray& dl,      NDArray& dr,      NDArray& gradOut,
                                   NDArray& dAValues, NDArray& ddl,    NDArray& ddr,
                                   sd::LongType rows, sd::LongType cols);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_DIAG_MM_BP_H
