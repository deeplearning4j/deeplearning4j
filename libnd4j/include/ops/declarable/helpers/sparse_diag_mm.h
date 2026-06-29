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

#ifndef SAMEDIFF_SPARSE_DIAG_MM_H
#define SAMEDIFF_SPARSE_DIAG_MM_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Two-sided diagonal scaling of a CSR matrix: outValues[e] = dl[i] * aValues[e] * dr[j]
 *
 * For each stored nonzero entry e in the CSR matrix A:
 *   - i is the row containing e  (determined from aRowPtr)
 *   - j = aColIdx[e]             (column of the entry)
 *   - outValues[e] = dl[i] * aValues[e] * dr[j]
 *
 * Only the value array is modified; the sparsity structure (aColIdx, aRowPtr)
 * is unchanged and is NOT an output.  The caller reuses A's structural arrays.
 *
 * This is the GCN normalization D^{-1/2} A D^{-1/2}.
 *
 * @param aValues   1D [nnz]     float — stored nonzero values of A
 * @param aColIdx   1D [nnz]     int   — column indices of A (sorted per row)
 * @param aRowPtr   1D [rows+1]  int   — row pointers of A
 * @param dl        1D [rows]    float — left diagonal
 * @param dr        1D [cols]    float — right diagonal
 * @param outValues 1D [nnz]     float — output values (same dtype as aValues)
 * @param rows      Number of rows in A
 * @param cols      Number of columns in A
 */
SD_LIB_HIDDEN void csr_diag_mm(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                                NDArray& dl,      NDArray& dr,
                                NDArray& outValues,
                                sd::LongType rows, sd::LongType cols);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_DIAG_MM_H
