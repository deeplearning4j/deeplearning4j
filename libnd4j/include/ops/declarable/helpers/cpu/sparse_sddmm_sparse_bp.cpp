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
// CPU backward-pass helper for csr_sddmm_sparse.
//
// Forward: for each target nonzero t at (row p_t, col q_t = targetColIdx[t]):
//   outValues[t] = dot(L row p_t, M row q_t)
//               = sum_{c: c in L-row-p_t AND c in M-row-q_t} Lvalues[li] * Mvalues[mi]
//
// Backward (gradOut [tnnz] given):
//   dLvalues[li] += gradOut[t] * Mvalues[mi]   for every matching (li, mi, t) triple
//   dMvalues[mi] += gradOut[t] * Lvalues[li]
//
// The merge-intersect of sorted per-row column lists mirrors the forward two-pointer
// walk, accumulating into the pre-zeroed output arrays.  Multiple target entries can
// reference the same L or M value position (e.g., two targets in the same L-row but
// different M-rows that both share a column with L); += handles this correctly on CPU
// without atomics since the implementation is serial.
//

#include <ops/declarable/helpers/sparse_sddmm_sparse_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// csrSddmmSparseBp_ — templated core
//
// Outer loop: rows p of the target sparsity (= rows of L).
// For each row p, walk the target entries t in [tRpBuf[p], tRpBuf[p+1]):
//   q  = targetColIdx[t]  (row of M / column of target)
//   g  = gradOut[t]
//   Two-pointer merge over L row p and M row q — when a shared column c is
//   found at positions li (in L) and mi (in M):
//     dLvalues[li] += g * Mvalues[mi]
//     dMvalues[mi] += g * Lvalues[li]
// ---------------------------------------------------------------------------

template <typename X, typename I>
static void csrSddmmSparseBp_(
    NDArray& targetRowPtr, NDArray& targetColIdx,
    NDArray& LcolIdx,      NDArray& LrowPtr,
    NDArray& McolIdx,      NDArray& MrowPtr,
    NDArray& Lvalues,      NDArray& Mvalues,
    NDArray& gradOut,
    NDArray& dLvalues,     NDArray& dMvalues,
    sd::LongType P) {
  const I* tRpBuf  = targetRowPtr.bufferAsT<I>();
  const I* tCiBuf  = targetColIdx.bufferAsT<I>();

  const X* lvBuf   = Lvalues.bufferAsT<X>();
  const I* lciBuf  = LcolIdx.bufferAsT<I>();
  const I* lrpBuf  = LrowPtr.bufferAsT<I>();

  const X* mvBuf   = Mvalues.bufferAsT<X>();
  const I* mciBuf  = McolIdx.bufferAsT<I>();
  const I* mrpBuf  = MrowPtr.bufferAsT<I>();

  const X* gBuf    = gradOut.bufferAsT<X>();

  X* dlBuf  = dLvalues.bufferAsT<X>();
  X* dmBuf  = dMvalues.bufferAsT<X>();

  // Outer loop: each row p of the target/L
  for (I p = 0; p < static_cast<I>(P); ++p) {
    const I tStart = tRpBuf[p];
    const I tEnd   = tRpBuf[p + 1];

    // L row p span — the same for all targets in row p
    const I lStart = lrpBuf[p];
    const I lEnd   = lrpBuf[p + 1];

    // Each target nonzero in this row
    for (I t = tStart; t < tEnd; ++t) {
      const I q = tCiBuf[t];
      const X g = gBuf[t];

      // M row q span
      const I mStart = mrpBuf[q];
      const I mEnd   = mrpBuf[q + 1];

      // Two-pointer merge-intersect of sorted column-index ranges for L row p
      // and M row q.  When a shared column is found, accumulate gradients.
      I li = lStart;
      I mi = mStart;
      while (li < lEnd && mi < mEnd) {
        const I lc = lciBuf[li];
        const I mc = mciBuf[mi];
        if (lc == mc) {
          // Shared inner dimension c: both partial derivatives get a contribution.
          dlBuf[li] += g * mvBuf[mi];
          dmBuf[mi] += g * lvBuf[li];
          ++li;
          ++mi;
        } else if (lc < mc) {
          ++li;
        } else {
          ++mi;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

void csr_sddmm_sparse_bp(
    NDArray& targetRowPtr, NDArray& targetColIdx,
    NDArray& LcolIdx,      NDArray& LrowPtr,
    NDArray& McolIdx,      NDArray& MrowPtr,
    NDArray& Lvalues,      NDArray& Mvalues,
    NDArray& gradOut,
    NDArray& dLvalues,     NDArray& dMvalues,
    sd::LongType P, sd::LongType /*Q*/, sd::LongType /*R*/) {
  NDArray::preparePrimaryUse({&dLvalues, &dMvalues},
      {&targetRowPtr, &targetColIdx,
       &LcolIdx, &LrowPtr,
       &McolIdx, &MrowPtr,
       &Lvalues, &Mvalues, &gradOut});

  BUILD_DOUBLE_SELECTOR(Lvalues.dataType(), LcolIdx.dataType(), csrSddmmSparseBp_,
      (targetRowPtr, targetColIdx,
       LcolIdx, LrowPtr,
       McolIdx, MrowPtr,
       Lvalues, Mvalues,
       gradOut, dLvalues, dMvalues, P),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dLvalues, &dMvalues},
      {&targetRowPtr, &targetColIdx,
       &LcolIdx, &LrowPtr,
       &McolIdx, &MrowPtr,
       &Lvalues, &Mvalues, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
