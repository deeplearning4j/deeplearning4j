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

#ifndef SAMEDIFF_SPARSE_SDDMM_SPARSE_H
#define SAMEDIFF_SPARSE_SDDMM_SPARSE_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * CSR sparse-SDDMM: sampled sparse×sparseᵀ dot product.
 *
 * For each target nonzero t in the sparsity pattern of [P, Q], whose row p
 * satisfies targetRowPtr[p] <= t < targetRowPtr[p+1] and whose column is
 * q = targetColIdx[t], computes:
 *
 *   outValues[t] = dot(L row p, M row q)
 *                = sum over columns c in BOTH L-row-p AND M-row-q of
 *                  Lvalues[lk] * Mvalues[mk]   (merge-intersect of sorted rows)
 *
 * Equivalently: out = (L · Mᵀ) sampled at (targetRowPtr, targetColIdx).
 *
 * L is CSR [P, R], M is CSR [Q, R].  Both CSR row column-index arrays must
 * be sorted ascending within each row (standard CSR invariant).
 *
 * @param targetRowPtr  1D [P+1]   int — target sparsity row pointers
 * @param targetColIdx  1D [tnnz]  int — target sparsity column indices
 * @param Lvalues       1D [Lnnz]  float — non-zero values of L
 * @param LcolIdx       1D [Lnnz]  int   — column indices of L
 * @param LrowPtr       1D [P+1]   int   — row pointers of L
 * @param Mvalues       1D [Mnnz]  float — non-zero values of M
 * @param McolIdx       1D [Mnnz]  int   — column indices of M
 * @param MrowPtr       1D [Q+1]   int   — row pointers of M
 * @param outValues     1D [tnnz]  float output (same dtype as Lvalues)
 * @param P             Number of rows in L / rows in target
 * @param Q             Number of rows in M / cols in target
 * @param R             Number of columns in both L and M (inner dimension)
 */
SD_LIB_HIDDEN void csr_sddmm_sparse(NDArray& targetRowPtr, NDArray& targetColIdx,
                                      NDArray& Lvalues, NDArray& LcolIdx, NDArray& LrowPtr,
                                      NDArray& Mvalues, NDArray& McolIdx, NDArray& MrowPtr,
                                      NDArray& outValues,
                                      sd::LongType P, sd::LongType Q, sd::LongType R);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_SDDMM_SPARSE_H
