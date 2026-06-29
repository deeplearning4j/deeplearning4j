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
// CUDA implementation of the SpGEMM backward pass.
//
// Delegates entirely to the existing CUDA helpers csr_to_csc (cuSPARSE-backed)
// and csr_sddmm_sparse (custom kernel).  No arithmetic is implemented here;
// no raw cudaMalloc is used — all temporary device arrays go through
// NDArrayFactory::create(... ctx) which uses the pool-backed allocator.
//
// Execution flow
// ──────────────
//   1. prepareSpecialUse  — ensure forward inputs + dC are on device.
//   2. Aᵀ [k,m] = csr_to_csc(A, m, k)          (device-side; manages its own prepare/register)
//   3. dCᵀ[n,m] = csr_to_csc(dC, m, n)          (idem)
//   4. dAValues = csr_sddmm_sparse(...)           (device kernel; manages its own prepare/register)
//   5. dBValues = csr_sddmm_sparse(...)           (idem)
//   6. registerSpecialUse — mark final outputs as device-written.
//

#include <cuda_runtime.h>
#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/sparse_csc.h>
#include <ops/declarable/helpers/sparse_sddmm_sparse.h>
#include <ops/declarable/helpers/sparse_spgemm_bp.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

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
  // ── Coherence: ensure all inputs are on device; mark outputs for writing ──
  NDArray::prepareSpecialUse({&dAValues, &dBValues},
                             {&aValues,     &aColIdx, &aRowPtr,
                              &bValues,     &bColIdx, &bRowPtr,
                              &cColIdx,     &cRowPtr,
                              &gradCValues});

  // Acquire launch context from an output array (standard CUDA helper pattern).
  auto* ctx = dAValues.getContext();

  const LongType annz = aValues.lengthOf();
  const LongType cnnz = gradCValues.lengthOf();
  const auto floatType = aValues.dataType();
  const auto idxType   = DataType::INT32;

  // ── Build Aᵀ [k, m] = csr_to_csc(A, m, k) ──────────────────────────────
  // Temporary device NDArrays created through NDArrayFactory (pool-backed,
  // no raw cudaMalloc).  CSC outputs are reinterpreted as CSR of Aᵀ:
  //   cscValues  → Aᵀ values
  //   cscRowIdx  → Aᵀ colIdx (as CSR)
  //   cscColPtr  → Aᵀ rowPtr (as CSR, length k+1)
  std::unique_ptr<NDArray> atVals  (NDArrayFactory::create('c', std::vector<LongType>{annz},   floatType, ctx));
  std::unique_ptr<NDArray> atRowIdx(NDArrayFactory::create('c', std::vector<LongType>{annz},   idxType,   ctx));
  std::unique_ptr<NDArray> atColPtr(NDArrayFactory::create('c', std::vector<LongType>{k + 1},  idxType,   ctx));

  csr_to_csc(aValues, aColIdx, aRowPtr,
             *atVals, *atRowIdx, *atColPtr,
             m, k);

  // ── Build dCᵀ [n, m] = csr_to_csc(dC, m, n) ────────────────────────────
  std::unique_ptr<NDArray> dctVals  (NDArrayFactory::create('c', std::vector<LongType>{cnnz},   floatType, ctx));
  std::unique_ptr<NDArray> dctRowIdx(NDArrayFactory::create('c', std::vector<LongType>{cnnz},   idxType,   ctx));
  std::unique_ptr<NDArray> dctColPtr(NDArrayFactory::create('c', std::vector<LongType>{n + 1},  idxType,   ctx));

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
                   *atVals,   *atRowIdx,  *atColPtr,
                   *dctVals,  *dctRowIdx, *dctColPtr,
                   dBValues,
                   k, n, m);

  // ── Coherence: mark final outputs as written on device ───────────────────
  NDArray::registerSpecialUse({&dAValues, &dBValues},
                              {&aValues,     &aColIdx, &aRowPtr,
                               &bValues,     &bColIdx, &bRowPtr,
                               &cColIdx,     &cRowPtr,
                               &gradCValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
