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
// CPU backward-pass helper for csr_add: C = A + B (CSR union sparsity).
//
// csr_add_bp: for each non-zero in A (resp. B) at (row r, col aColIdx[k]),
//   look up its position in C's sorted per-row column list via binary search,
//   then copy gradCValues at that position into dAValues[k] (resp. dBValues[k]).
//
// This is a pure gather — no atomics are required because the CSR union
// sparsity ensures each A/B entry maps to exactly one C entry.
//

#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/sparse_add_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// csrAddBp_ — templated core
//
// For each row r:
//   cRowPtr[r]..cRowPtr[r+1] is the sorted list of C's column indices for row r.
//   For each A entry k at aColIdx[k] in row r:
//     binary-search cColIdx[c_start..c_end) for aColIdx[k] → position lo
//     dAValues[k] = gradCValues[lo]
//   Same gather for B entries.
// ---------------------------------------------------------------------------

template <typename X, typename I>
static void csrAddBp_(
    NDArray& aColIdx, NDArray& aRowPtr,
    NDArray& bColIdx, NDArray& bRowPtr,
    NDArray& cColIdx, NDArray& cRowPtr,
    NDArray& gradCValues,
    NDArray& dAValues, NDArray& dBValues,
    LongType m) {
  const auto* aci = aColIdx.bufferAsT<I>();
  const auto* arp = aRowPtr.bufferAsT<I>();
  const auto* bci = bColIdx.bufferAsT<I>();
  const auto* brp = bRowPtr.bufferAsT<I>();
  const auto* cci = cColIdx.bufferAsT<I>();
  const auto* crp = cRowPtr.bufferAsT<I>();
  const auto* gv  = gradCValues.bufferAsT<X>();
  auto*       daV = dAValues.bufferAsT<X>();
  auto*       dbV = dBValues.bufferAsT<X>();

  for (LongType r = 0; r < m; ++r) {
    const I c_start = crp[r];
    const I c_end   = crp[r + 1];

    // Gather into dAValues: each A entry at (r, col) maps to a unique C entry
    for (I k = arp[r]; k < arp[r + 1]; ++k) {
      const I col = aci[k];
      // Binary search for col in cci[c_start..c_end) — guaranteed to be found
      // because C's sparsity is the union of A and B sparsities.
      I lo = c_start, hi = c_end;
      while (lo < hi) {
        const I mid = lo + (hi - lo) / 2;
        if (cci[mid] < col) lo = mid + 1; else hi = mid;
      }
      daV[k] = gv[lo];
    }

    // Gather into dBValues: same approach for B entries in the same row
    for (I k = brp[r]; k < brp[r + 1]; ++k) {
      const I col = bci[k];
      I lo = c_start, hi = c_end;
      while (lo < hi) {
        const I mid = lo + (hi - lo) / 2;
        if (cci[mid] < col) lo = mid + 1; else hi = mid;
      }
      dbV[k] = gv[lo];
    }
  }
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

void csr_add_bp(
    NDArray& aColIdx, NDArray& aRowPtr,
    NDArray& bColIdx, NDArray& bRowPtr,
    NDArray& cColIdx, NDArray& cRowPtr,
    NDArray& gradCValues,
    NDArray& dAValues, NDArray& dBValues,
    sd::LongType m, sd::LongType /*n*/) {
  NDArray::preparePrimaryUse({&dAValues, &dBValues},
      {&aColIdx, &aRowPtr, &bColIdx, &bRowPtr, &cColIdx, &cRowPtr, &gradCValues});

  BUILD_DOUBLE_SELECTOR(gradCValues.dataType(), aColIdx.dataType(), csrAddBp_,
      (aColIdx, aRowPtr, bColIdx, bRowPtr, cColIdx, cRowPtr, gradCValues, dAValues, dBValues, m),
      SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dAValues, &dBValues},
      {&aColIdx, &aRowPtr, &bColIdx, &bRowPtr, &cColIdx, &cRowPtr, &gradCValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
