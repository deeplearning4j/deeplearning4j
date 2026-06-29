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
// CSR sparse-SDDMM — CPU implementation.
//
// For each target nonzero t (belonging to row p, column q = targetColIdx[t]),
// computes the inner product of the p-th row of L with the q-th row of M:
//
//   outValues[t] = sum_{c in BOTH L-row-p ∩ M-row-q} Lvalues[li] * Mvalues[mi]
//
// The intersection is found by a two-pointer merge of the two sorted CSR
// column-index ranges.  Complexity per nonzero: O(nnz_row_L + nnz_row_M).
// Total: O(tnnz * avg_row_nnz).
//
// Template parameters:
//   X — float dtype (Lvalues, Mvalues, outValues)
//   I — index dtype (all CSR structural arrays, must be the same)
//

#include <ops/declarable/helpers/sparse_sddmm_sparse.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename I>
static void csrSddmmSparse_(NDArray& targetRowPtr, NDArray& targetColIdx,
                             NDArray& Lvalues,      NDArray& LcolIdx, NDArray& LrowPtr,
                             NDArray& Mvalues,      NDArray& McolIdx, NDArray& MrowPtr,
                             NDArray& outValues,
                             sd::LongType P) {
  // Raw host buffers — all arrays are 1D and contiguous after preparePrimaryUse.
  const I* tRpBuf  = targetRowPtr.bufferAsT<I>();
  const I* tCiBuf  = targetColIdx.bufferAsT<I>();

  const X* lvBuf   = Lvalues.bufferAsT<X>();
  const I* lciBuf  = LcolIdx.bufferAsT<I>();
  const I* lrpBuf  = LrowPtr.bufferAsT<I>();

  const X* mvBuf   = Mvalues.bufferAsT<X>();
  const I* mciBuf  = McolIdx.bufferAsT<I>();
  const I* mrpBuf  = MrowPtr.bufferAsT<I>();

  X* oBuf = outValues.bufferAsT<X>();

  // Outer loop: rows of the target matrix (= rows of L).
  for (I p = 0; p < static_cast<I>(P); ++p) {
    const I tStart = tRpBuf[p];
    const I tEnd   = tRpBuf[p + 1];

    // L row p span.
    const I lStart = lrpBuf[p];
    const I lEnd   = lrpBuf[p + 1];

    // Inner loop: each target nonzero in row p.
    for (I t = tStart; t < tEnd; ++t) {
      const I q = tCiBuf[t];

      // M row q span.
      const I mStart = mrpBuf[q];
      const I mEnd   = mrpBuf[q + 1];

      // Two-pointer merge-intersect of the two sorted column-index ranges.
      // Both LcolIdx[lStart..lEnd) and McolIdx[mStart..mEnd) are ascending
      // (standard CSR invariant).
      X acc = static_cast<X>(0);
      I li = lStart;
      I mi = mStart;
      while (li < lEnd && mi < mEnd) {
        const I lc = lciBuf[li];
        const I mc = mciBuf[mi];
        if (lc == mc) {
          acc += lvBuf[li] * mvBuf[mi];
          ++li;
          ++mi;
        } else if (lc < mc) {
          ++li;
        } else {
          ++mi;
        }
      }
      oBuf[t] = acc;
    }
  }
}

void csr_sddmm_sparse(NDArray& targetRowPtr, NDArray& targetColIdx,
                       NDArray& Lvalues,      NDArray& LcolIdx, NDArray& LrowPtr,
                       NDArray& Mvalues,      NDArray& McolIdx, NDArray& MrowPtr,
                       NDArray& outValues,
                       sd::LongType P, sd::LongType Q, sd::LongType R) {
  NDArray::preparePrimaryUse({&outValues},
                             {&targetRowPtr, &targetColIdx,
                              &Lvalues, &LcolIdx, &LrowPtr,
                              &Mvalues, &McolIdx, &MrowPtr});

  BUILD_DOUBLE_SELECTOR(Lvalues.dataType(), LcolIdx.dataType(), csrSddmmSparse_,
                        (targetRowPtr, targetColIdx,
                         Lvalues, LcolIdx, LrowPtr,
                         Mvalues, McolIdx, MrowPtr,
                         outValues, P),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&outValues},
                              {&targetRowPtr, &targetColIdx,
                               &Lvalues, &LcolIdx, &LrowPtr,
                               &Mvalues, &McolIdx, &MrowPtr});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
