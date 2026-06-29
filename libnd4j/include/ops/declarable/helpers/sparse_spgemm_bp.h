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

#ifndef SAMEDIFF_SPARSE_SPGEMM_BP_H
#define SAMEDIFF_SPARSE_SPGEMM_BP_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Backward pass for CSR sparse matrix–matrix product (SpGEMM): C = A · B.
 *
 * Given the upstream gradient dC (represented by gradCValues, cColIdx, cRowPtr)
 * and the forward-pass inputs A and B, computes:
 *
 *   dA[i,j] = dot(dC row i, B row j)   sampled at A's sparsity pattern
 *           → csr_sddmm_sparse(target=A pattern,
 *                              L=dC[m,n], M=B[k,n], P=m, Q=k, R=n)
 *
 *   dB[j,l] = dot(Aᵀ row j, dCᵀ row l) sampled at B's sparsity pattern
 *           → csr_to_csc(A, m, k) → Aᵀ [k, m] (values reinterpreted as CSR)
 *             csr_to_csc(dC, m, n) → dCᵀ [n, m]
 *             csr_sddmm_sparse(target=B pattern,
 *                              L=Aᵀ[k,m], M=dCᵀ[n,m], P=k, Q=n, R=m)
 *
 * Temporary CSC buffers for Aᵀ and dCᵀ are allocated internally:
 *  – CPU path: via NDArrayFactory::create (host memory).
 *  – CUDA path: via NDArrayFactory::create with a LaunchContext* (device memory,
 *               no raw cudaMalloc).
 *
 * @param aValues     1D [annz]  float – non-zero values of A
 * @param aColIdx     1D [annz]  int   – column indices of A
 * @param aRowPtr     1D [m+1]   int   – row pointers of A
 * @param bValues     1D [bnnz]  float – non-zero values of B
 * @param bColIdx     1D [bnnz]  int   – column indices of B
 * @param bRowPtr     1D [k+1]   int   – row pointers of B
 * @param cColIdx     1D [cnnz]  int   – column indices of C (from forward pass)
 * @param cRowPtr     1D [m+1]   int   – row pointers of C (from forward pass)
 * @param gradCValues 1D [cnnz]  float – upstream gradient w.r.t. C values
 * @param dAValues    1D [annz]  float – OUTPUT gradient w.r.t. aValues
 * @param dBValues    1D [bnnz]  float – OUTPUT gradient w.r.t. bValues
 * @param m           rows of A (= rows of C)
 * @param k           cols of A (= rows of B)
 * @param n           cols of B (= cols of C)
 */
SD_LIB_HIDDEN void csr_spgemm_bp(NDArray& aValues,     NDArray& aColIdx,     NDArray& aRowPtr,
                                   NDArray& bValues,     NDArray& bColIdx,     NDArray& bRowPtr,
                                   NDArray& cColIdx,     NDArray& cRowPtr,
                                   NDArray& gradCValues,
                                   NDArray& dAValues,    NDArray& dBValues,
                                   LongType m, LongType k, LongType n);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // SAMEDIFF_SPARSE_SPGEMM_BP_H
