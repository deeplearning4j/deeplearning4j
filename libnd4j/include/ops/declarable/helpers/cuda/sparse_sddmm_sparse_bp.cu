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
// CUDA backward-pass helper for csr_sddmm_sparse.
//
// Forward: for each target nonzero t at (row p_t, col q_t):
//   outValues[t] = dot(L[p_t, :], M[q_t, :])   — inner dimension R, both sparse.
//
// Backward (one thread per target entry t):
//   1. Find p_t via binary-search in targetRowPtr.
//   2. q_t = targetColIdx[t].
//   3. Merge-intersect L row p_t and M row q_t (both sorted column lists):
//      for each shared inner column c at (L position lk, M position mk):
//        sd_atomicAdd(&dLvalues[lk],  gradOut[t] * Mvalues[mk])
//        sd_atomicAdd(&dMvalues[mk],  gradOut[t] * Lvalues[lk])
//
// Multiple target entries can map to the same L or M value → atomicAdd required.
// dLvalues and dMvalues must be pre-zeroed (OUTPUT_NULLIFIED) before the kernel.
//

#include <cuda_runtime.h>
#include <math/templatemath.h>
#include <ops/declarable/helpers/sparse_sddmm_sparse_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ─── device utility ──────────────────────────────────────────────────────────

// Binary-search rowPtr[0..P] to find row p such that rowPtr[p] <= k < rowPtr[p+1].
template <typename I>
static SD_INLINE SD_DEVICE LongType sddmmSparseFindRow(const I* rowPtr, LongType P, LongType k) {
  LongType lo = 0, hi = P - 1, row = 0;
  while (lo <= hi) {
    const LongType mid = (lo + hi) / 2;
    if (static_cast<LongType>(rowPtr[mid]) <= k &&
        k < static_cast<LongType>(rowPtr[mid + 1])) {
      row = mid;
      break;
    } else if (static_cast<LongType>(rowPtr[mid]) > k) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }
  return row;
}

// ─── Kernel ──────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrSddmmSparseBpKernel(
    const I* targetRowPtr, const I* targetColIdx,
    const I* LcolIdx,      const I* LrowPtr,
    const I* McolIdx,      const I* MrowPtr,
    const X* Lvalues, const X* Mvalues, const X* gradOut,
    X* dLvalues, X* dMvalues,
    LongType tnnz, LongType P, LongType Q) {
  const LongType t = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= tnnz) return;

  // Recover p_t: row of target entry t.
  const LongType p = sddmmSparseFindRow(targetRowPtr, P, t);
  const LongType q = static_cast<LongType>(targetColIdx[t]);

  const X grad = gradOut[t];

  // Merge-intersect sorted column lists of L[p, :] and M[q, :].
  I lk   = LrowPtr[p],  lend = LrowPtr[p + 1];
  I mk   = MrowPtr[q],  mend = MrowPtr[q + 1];

  while (lk < lend && mk < mend) {
    const I lc = LcolIdx[lk];
    const I mc = McolIdx[mk];

    if (lc == mc) {
      // Shared inner dimension: both L and M contribute.
      sd::math::atomics::sd_atomicAdd(
          &dLvalues[lk],
          static_cast<X>(grad * Mvalues[mk]));
      sd::math::atomics::sd_atomicAdd(
          &dMvalues[mk],
          static_cast<X>(grad * Lvalues[lk]));
      ++lk;
      ++mk;
    } else if (lc < mc) {
      ++lk;
    } else {
      ++mk;
    }
  }
}

// ─── typed launcher ──────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrSddmmSparseBpCuda_(
    NDArray& targetRowPtr, NDArray& targetColIdx,
    NDArray& LcolIdx, NDArray& LrowPtr,
    NDArray& McolIdx, NDArray& MrowPtr,
    NDArray& Lvalues, NDArray& Mvalues,
    NDArray& gradOut,
    NDArray& dLvalues, NDArray& dMvalues,
    sd::LongType P, sd::LongType Q, sd::LongType R) {
  const LongType tnnz = targetColIdx.lengthOf();

  const I* tRP   = reinterpret_cast<const I*>(targetRowPtr.specialBuffer());
  const I* tCI   = reinterpret_cast<const I*>(targetColIdx.specialBuffer());
  const I* lCI   = reinterpret_cast<const I*>(LcolIdx.specialBuffer());
  const I* lRP   = reinterpret_cast<const I*>(LrowPtr.specialBuffer());
  const I* mCI   = reinterpret_cast<const I*>(McolIdx.specialBuffer());
  const I* mRP   = reinterpret_cast<const I*>(MrowPtr.specialBuffer());
  const X* lV    = reinterpret_cast<const X*>(Lvalues.specialBuffer());
  const X* mV    = reinterpret_cast<const X*>(Mvalues.specialBuffer());
  const X* gO    = reinterpret_cast<const X*>(gradOut.specialBuffer());
  X*       dLV   = reinterpret_cast<X*>(dLvalues.specialBuffer());
  X*       dMV   = reinterpret_cast<X*>(dMvalues.specialBuffer());

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((tnnz + blockSize - 1) / blockSize);

  auto* stream = dLvalues.getContext()->getCudaStream();
  csrSddmmSparseBpKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      tRP, tCI, lCI, lRP, mCI, mRP,
      lV, mV, gO, dLV, dMV,
      tnnz, P, Q);
}

// ─── public entry point ──────────────────────────────────────────────────────

void csr_sddmm_sparse_bp(
    NDArray& targetRowPtr, NDArray& targetColIdx,
    NDArray& LcolIdx, NDArray& LrowPtr,
    NDArray& McolIdx, NDArray& MrowPtr,
    NDArray& Lvalues, NDArray& Mvalues,
    NDArray& gradOut,
    NDArray& dLvalues, NDArray& dMvalues,
    sd::LongType P, sd::LongType Q, sd::LongType R) {
  NDArray::prepareSpecialUse(
      {&dLvalues, &dMvalues},
      {&targetRowPtr, &targetColIdx,
       &LcolIdx, &LrowPtr, &McolIdx, &MrowPtr,
       &Lvalues, &Mvalues, &gradOut});

  BUILD_DOUBLE_SELECTOR(Lvalues.dataType(), LcolIdx.dataType(), csrSddmmSparseBpCuda_,
                        (targetRowPtr, targetColIdx,
                         LcolIdx, LrowPtr, McolIdx, MrowPtr,
                         Lvalues, Mvalues, gradOut,
                         dLvalues, dMvalues, P, Q, R),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse(
      {&dLvalues, &dMvalues},
      {&targetRowPtr, &targetColIdx,
       &LcolIdx, &LrowPtr, &McolIdx, &MrowPtr,
       &Lvalues, &Mvalues, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
