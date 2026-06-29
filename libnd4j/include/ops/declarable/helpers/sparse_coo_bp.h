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

#ifndef SAMEDIFF_SPARSE_COO_BP_H
#define SAMEDIFF_SPARSE_COO_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for dense_to_coo.
 *
 * Forward: dense[rows, cols] → (indices[nnz, 2], values[nnz])
 *
 * Backward: given the upstream gradient gradValues[nnz] and the forward's
 * output indices[nnz, 2], scatter gradients back into the dense gradient:
 *
 *   dDense is ZERO-INITIALIZED, then for each k in [0, nnz):
 *     dDense[ indices[k,0], indices[k,1] ] = gradValues[k]
 *
 * No atomics are needed because dense_to_coo emits unique (row, col) pairs.
 *
 * @param context      Launch context (used by CUDA path for stream/device)
 * @param indices      [nnz, 2] INT64 — row/col index pairs from the forward pass
 * @param gradValues   [nnz] float — upstream gradient w.r.t. the COO values output
 * @param dDense       [rows, cols] float — OUTPUT gradient w.r.t. the dense input
 * @param rows         number of rows in the dense matrix
 * @param cols         number of columns in the dense matrix
 */
SD_LIB_HIDDEN void denseToCoo_bp(sd::LaunchContext* context,
                                  NDArray* indices,
                                  NDArray* gradValues,
                                  NDArray* dDense,
                                  sd::LongType rows,
                                  sd::LongType cols);

/**
 * Backward pass for coo_to_csr.
 *
 * Forward: (cooIndices[coo_nnz, 2], cooValues[coo_nnz])
 *        → (csrValues[csr_nnz], csrColIdx[csr_nnz], csrRowPtr[rows+1])
 * The forward op may coalesce duplicate (row, col) entries by summing their
 * values (csr_nnz ≤ coo_nnz).  The gradient of a coalescing SUM distributes
 * (copies) to every COO entry that contributed to a given CSR slot.
 *
 * For each COO entry k with (i, j) = cooIndices[k]:
 *   binary-search csrColIdx[ csrRowPtr[i] .. csrRowPtr[i+1] ) for column j
 *   → dCooValues[k] = gradCsrValues[foundPos]
 *
 * No atomics required (each COO entry maps to exactly one CSR slot).
 *
 * @param context        Launch context (CUDA path: stream/device)
 * @param cooIndices     [coo_nnz, 2] INT — row/col index pairs of the original COO input
 * @param csrColIdx      [csr_nnz] INT32 — column indices of the CSR forward output
 * @param csrRowPtr      [rows+1] INT32  — row pointers of the CSR forward output
 * @param gradCsrValues  [csr_nnz] float — upstream gradient w.r.t. the CSR values
 * @param dCooValues     [coo_nnz] float — OUTPUT gradient w.r.t. the COO values input
 * @param rows           number of rows in the logical sparse matrix
 * @param cols           number of columns in the logical sparse matrix
 */
SD_LIB_HIDDEN void cooToCsr_bp(sd::LaunchContext* context,
                                NDArray* cooIndices,
                                NDArray* csrColIdx,
                                NDArray* csrRowPtr,
                                NDArray* gradCsrValues,
                                NDArray* dCooValues,
                                sd::LongType rows,
                                sd::LongType cols);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_COO_BP_H
