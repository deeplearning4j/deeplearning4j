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
// CSR sparse-SDDMM — CUDA implementation.
//
// There is no cuSPARSE primitive for sampled sparse×sparseᵀ (cuSPARSE's
// cusparseSDDMM computes sampled dense×dense, not sparse×sparse).  This file
// therefore provides a fully device-side custom kernel.
//
// Kernel design
// ─────────────
// Grid = 1-D, one CUDA thread per target nonzero t in [0, tnnz).
//
//   Step 1 — find the row p that owns position t:
//             binary-search targetRowPtr[0..P] for the unique p satisfying
//             targetRowPtr[p] <= t < targetRowPtr[p+1].
//             (This mirrors the pattern in sddmmKernel in sparse_blas.cu.)
//
//   Step 2 — q = targetColIdx[t]
//
//   Step 3 — two-pointer merge-intersect of the two sorted column-index ranges:
//               LcolIdx[LrowPtr[p] .. LrowPtr[p+1])
//               McolIdx[MrowPtr[q] .. MrowPtr[q+1])
//             summing Lvalues[li]*Mvalues[mi] where LcolIdx[li] == McolIdx[mi].
//
//   Step 4 — write outValues[t] = acc  (no atomics: t is unique per thread).
//
// Constraints satisfied
// ─────────────────────
// • No raw cudaMalloc — workspace allocation (none needed here) would use
//   CudaMemoryPool, but this op requires no scratch beyond registers.
// • No host compute shim — all arithmetic executes on the GPU kernel.
// • No atomics for the output — one thread owns each output element.
// • prepareSpecialUse / registerSpecialUse bracket the kernel launch.
//

#include <cuda_runtime.h>
#include <ops/declarable/helpers/sparse_sddmm_sparse.h>
#include <system/op_boilerplate.h>
#include <types/bfloat16.h>
#include <types/float16.h>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Device kernel: one thread per target nonzero
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static SD_KERNEL void csrSddmmSparseKernel(
    // Target sparsity pattern
    const I* targetRowPtr,   // [P+1]
    const I* targetColIdx,   // [tnnz]
    // L CSR [P, R]
    const X* Lvalues,        // [Lnnz]
    const I* LcolIdx,        // [Lnnz]
    const I* LrowPtr,        // [P+1]
    // M CSR [Q, R]
    const X* Mvalues,        // [Mnnz]
    const I* McolIdx,        // [Mnnz]
    const I* MrowPtr,        // [Q+1]
    // Output
    X*       outValues,      // [tnnz]
    // Dimensions
    LongType P,
    LongType tnnz)
{
  const LongType t = static_cast<LongType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= tnnz) return;

  // ── Step 1: binary-search targetRowPtr for the row p owning position t ──
  //
  // Invariant: targetRowPtr[p] <= t < targetRowPtr[p+1]
  // P > 0 and tnnz > 0 are guaranteed by the op (empty => trivially done).
  LongType lo = 0, hi = P - 1, p = 0;
  while (lo <= hi) {
    const LongType mid = lo + (hi - lo) / 2;
    const LongType rpMid     = static_cast<LongType>(targetRowPtr[mid]);
    const LongType rpMidNext = static_cast<LongType>(targetRowPtr[mid + 1]);
    if (rpMid <= t && t < rpMidNext) {
      p = mid;
      break;
    } else if (rpMid > t) {
      hi = mid - 1;
    } else {
      lo = mid + 1;
    }
  }

  // ── Step 2: column of target nonzero t ──
  const LongType q = static_cast<LongType>(targetColIdx[t]);

  // ── Step 3: two-pointer intersect of L row p and M row q ──
  LongType lStart = static_cast<LongType>(LrowPtr[p]);
  LongType lEnd   = static_cast<LongType>(LrowPtr[p + 1]);
  LongType mStart = static_cast<LongType>(MrowPtr[q]);
  LongType mEnd   = static_cast<LongType>(MrowPtr[q + 1]);

  X acc = static_cast<X>(0);
  LongType li = lStart;
  LongType mi = mStart;
  while (li < lEnd && mi < mEnd) {
    const LongType lc = static_cast<LongType>(LcolIdx[li]);
    const LongType mc = static_cast<LongType>(McolIdx[mi]);
    if (lc == mc) {
      acc += Lvalues[li] * Mvalues[mi];
      ++li;
      ++mi;
    } else if (lc < mc) {
      ++li;
    } else {
      ++mi;
    }
  }

  // ── Step 4: write result — no atomics, each t is owned by exactly one thread ──
  outValues[t] = acc;
}

// ────────────────────────────────────────────────────────────────────────────
// Device-dispatch wrapper (called via BUILD_DOUBLE_SELECTOR)
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrSddmmSparseCuda_(NDArray& targetRowPtr, NDArray& targetColIdx,
                                 NDArray& Lvalues,      NDArray& LcolIdx, NDArray& LrowPtr,
                                 NDArray& Mvalues,      NDArray& McolIdx, NDArray& MrowPtr,
                                 NDArray& outValues,
                                 sd::LongType P, sd::LongType /*Q*/, sd::LongType /*R*/) {
  auto* stream = outValues.getContext()->getCudaStream();

  const I* tRpBuf  = reinterpret_cast<const I*>(targetRowPtr.specialBuffer());
  const I* tCiBuf  = reinterpret_cast<const I*>(targetColIdx.specialBuffer());

  const X* lvBuf   = reinterpret_cast<const X*>(Lvalues.specialBuffer());
  const I* lciBuf  = reinterpret_cast<const I*>(LcolIdx.specialBuffer());
  const I* lrpBuf  = reinterpret_cast<const I*>(LrowPtr.specialBuffer());

  const X* mvBuf   = reinterpret_cast<const X*>(Mvalues.specialBuffer());
  const I* mciBuf  = reinterpret_cast<const I*>(McolIdx.specialBuffer());
  const I* mrpBuf  = reinterpret_cast<const I*>(MrowPtr.specialBuffer());

  X* oBuf = reinterpret_cast<X*>(outValues.specialBuffer());

  const LongType tnnz = outValues.lengthOf();
  if (tnnz == 0) return;  // nothing to do for a zero-nnz target pattern

  const int blockSize = 256;
  const int gridSize  = static_cast<int>((tnnz + blockSize - 1) / blockSize);

  csrSddmmSparseKernel<X, I><<<gridSize, blockSize, 0, *stream>>>(
      tRpBuf, tCiBuf,
      lvBuf, lciBuf, lrpBuf,
      mvBuf, mciBuf, mrpBuf,
      oBuf,
      P, tnnz);
}

// ────────────────────────────────────────────────────────────────────────────
// Public entry point
// ────────────────────────────────────────────────────────────────────────────

void csr_sddmm_sparse(NDArray& targetRowPtr, NDArray& targetColIdx,
                       NDArray& Lvalues,      NDArray& LcolIdx, NDArray& LrowPtr,
                       NDArray& Mvalues,      NDArray& McolIdx, NDArray& MrowPtr,
                       NDArray& outValues,
                       sd::LongType P, sd::LongType Q, sd::LongType R) {
  NDArray::prepareSpecialUse({&outValues},
                             {&targetRowPtr, &targetColIdx,
                              &Lvalues, &LcolIdx, &LrowPtr,
                              &Mvalues, &McolIdx, &MrowPtr});

  BUILD_DOUBLE_SELECTOR(Lvalues.dataType(), LcolIdx.dataType(), csrSddmmSparseCuda_,
                        (targetRowPtr, targetColIdx,
                         Lvalues, LcolIdx, LrowPtr,
                         Mvalues, McolIdx, MrowPtr,
                         outValues, P, Q, R),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&outValues},
                              {&targetRowPtr, &targetColIdx,
                               &Lvalues, &LcolIdx, &LrowPtr,
                               &Mvalues, &McolIdx, &MrowPtr});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
