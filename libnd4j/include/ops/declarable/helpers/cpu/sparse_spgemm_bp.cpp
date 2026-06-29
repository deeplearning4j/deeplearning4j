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

//
// CPU implementation of the SpGEMM backward pass.
//
// Strategy: orchestrate the already-verified helpers csr_to_csc and
// csr_sddmm_sparse; no arithmetic is re-implemented here.
//
//   dAValues = csr_sddmm_sparse(target=A pattern,
//                               L=dC[m,n], M=B[k,n], P=m, Q=k, R=n)
//
//   dBValues = csr_sddmm_sparse(target=B pattern,
//                               L=Aᵀ[k,m], M=dCᵀ[n,m], P=k, Q=n, R=m)
//   where Aᵀ  = csr_to_csc(A,  m, k)
//         dCᵀ = csr_to_csc(dC, m, n)  (dC = (gradCValues, cColIdx, cRowPtr))
//
// Temporary NDArrays for Aᵀ and dCᵀ are stack-allocated here; they are
// created without a CUDA context so they live entirely in host memory.
// preparePrimaryUse / registerPrimaryUse bracket the whole operation to
// satisfy the coherence contract for the final outputs.
//

#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/sparse_csc.h>
#include <ops/declarable/helpers/sparse_sddmm_sparse.h>
#include <ops/declarable/helpers/sparse_spgemm_bp.h>
#include <system/op_boilerplate.h>

#include <memory>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

void csr_spgemm_bp(NDArray& aValues,     NDArray& aColIdx,     NDArray& aRowPtr,
                   NDArray& bValues,     NDArray& bColIdx,     NDArray& bRowPtr,
                   NDArray& cColIdx,     NDArray& cRowPtr,
                   NDArray& gradCValues,
                   NDArray& dAValues,    NDArray& dBValues,
                   LongType m, LongType k, LongType n) {
  // ── Coherence: bring all inputs to host; mark outputs as "about to be written" ──
  NDArray::preparePrimaryUse({&dAValues, &dBValues},
                             {&aValues,     &aColIdx, &aRowPtr,
                              &bValues,     &bColIdx, &bRowPtr,
                              &cColIdx,     &cRowPtr,
                              &gradCValues});

  const LongType annz = aValues.lengthOf();
  const LongType cnnz = gradCValues.lengthOf();
  const auto floatType = aValues.dataType();
  const auto idxType   = DataType::INT32;

  // ── Build Aᵀ [k, m] = csr_to_csc(A, m, k) ──────────────────────────────
  // The CSC outputs of csr_to_csc are reinterpreted as CSR of Aᵀ:
  //   cscValues  → Aᵀ values
  //   cscRowIdx  → Aᵀ colIdx (as CSR)
  //   cscColPtr  → Aᵀ rowPtr (as CSR, length k+1)
  std::unique_ptr<NDArray> atVals  (NDArrayFactory::create('c', std::vector<LongType>{annz},   floatType));
  std::unique_ptr<NDArray> atRowIdx(NDArrayFactory::create('c', std::vector<LongType>{annz},   idxType));
  std::unique_ptr<NDArray> atColPtr(NDArrayFactory::create('c', std::vector<LongType>{k + 1},  idxType));

  csr_to_csc(aValues, aColIdx, aRowPtr,
             *atVals, *atRowIdx, *atColPtr,
             m, k);

  // ── Build dCᵀ [n, m] = csr_to_csc(dC, m, n) ────────────────────────────
  std::unique_ptr<NDArray> dctVals  (NDArrayFactory::create('c', std::vector<LongType>{cnnz},  floatType));
  std::unique_ptr<NDArray> dctRowIdx(NDArrayFactory::create('c', std::vector<LongType>{cnnz},  idxType));
  std::unique_ptr<NDArray> dctColPtr(NDArrayFactory::create('c', std::vector<LongType>{n + 1}, idxType));

  csr_to_csc(gradCValues, cColIdx, cRowPtr,
             *dctVals, *dctRowIdx, *dctColPtr,
             m, n);

  // ── dAValues = sddmm(target=A, L=dC[m,n], M=B[k,n], P=m, Q=k, R=n) ────
  csr_sddmm_sparse(aRowPtr, aColIdx,
                   gradCValues, cColIdx, cRowPtr,
                   bValues,     bColIdx, bRowPtr,
                   dAValues,
                   m, k, n);

  // ── dBValues = sddmm(target=B, L=Aᵀ[k,m], M=dCᵀ[n,m], P=k, Q=n, R=m) ─
  csr_sddmm_sparse(bRowPtr, bColIdx,
                   *atVals,  *atRowIdx, *atColPtr,
                   *dctVals, *dctRowIdx, *dctColPtr,
                   dBValues,
                   k, n, m);

  // ── Coherence: mark final outputs as written on host ─────────────────────
  NDArray::registerPrimaryUse({&dAValues, &dBValues},
                              {&aValues,     &aColIdx, &aRowPtr,
                               &bValues,     &bColIdx, &bRowPtr,
                               &cColIdx,     &cRowPtr,
                               &gradCValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
