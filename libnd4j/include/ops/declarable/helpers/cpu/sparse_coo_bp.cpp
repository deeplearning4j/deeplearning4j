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

#include <algorithm>
#include <cstring>

#include <ops/declarable/helpers/sparse_coo_bp.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// denseToCoo_bp — CPU
//
// Scatter upstream gradient gradValues[k] into dDense at the position
// (indices[k,0], indices[k,1]).  dDense is zeroed first, then each of the
// nnz non-zero slots is scatter-written.  No atomics: dense_to_coo emits
// unique indices, so each cell of dDense is touched at most once.
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void denseToCoo_bp_(NDArray* indices, NDArray* gradValues, NDArray* dDense,
                           sd::LongType rows, sd::LongType cols) {
  const sd::LongType nnz = gradValues->lengthOf();

  const auto* idxBuf  = indices->bufferAsT<I>();    // [nnz, 2] row-major
  const auto* gvBuf   = gradValues->bufferAsT<X>(); // [nnz]
        auto* dBuf    = dDense->bufferAsT<X>();      // [rows, cols]
  const auto* dStride = dDense->stridesOf();         // [cols, 1] for 'c' order

  // Zero-initialise the dense gradient: positions not in the COO set remain 0.
  std::fill(dBuf, dBuf + rows * dStride[0], static_cast<X>(0));

  for (sd::LongType k = 0; k < nnz; ++k) {
    const auto r = static_cast<sd::LongType>(idxBuf[k * 2 + 0]);
    const auto c = static_cast<sd::LongType>(idxBuf[k * 2 + 1]);
    dBuf[r * dStride[0] + c * dStride[1]] = gvBuf[k];
  }
}

void denseToCoo_bp(sd::LaunchContext* context,
                   NDArray* indices, NDArray* gradValues, NDArray* dDense,
                   sd::LongType rows, sd::LongType cols) {
  NDArray::preparePrimaryUse({dDense}, {indices, gradValues});

  BUILD_DOUBLE_SELECTOR(gradValues->dataType(), indices->dataType(), denseToCoo_bp_,
                        (indices, gradValues, dDense, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({dDense}, {indices, gradValues});
}

// ---------------------------------------------------------------------------
// cooToCsr_bp — CPU
//
// For each COO entry k:
//   row i = cooIndices[k, 0]
//   col j = cooIndices[k, 1]
//   Binary-search csrColIdx[ csrRowPtr[i] .. csrRowPtr[i+1] ) for column j.
//   dCooValues[k] = gradCsrValues[found position]
//
// No atomics: each COO entry maps to exactly one CSR slot.
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void cooToCsr_bp_(NDArray* cooIndices, NDArray* csrColIdx, NDArray* csrRowPtr,
                         NDArray* gradCsrValues, NDArray* dCooValues,
                         sd::LongType rows, sd::LongType /*cols*/) {
  const sd::LongType cooNnz  = dCooValues->lengthOf();

  const auto* idxBuf   = cooIndices->bufferAsT<I>();           // [coo_nnz, 2]
  const auto* colBuf   = csrColIdx->bufferAsT<int32_t>();      // [csr_nnz]
  const auto* rowBuf   = csrRowPtr->bufferAsT<int32_t>();      // [rows + 1]
  const auto* gradBuf  = gradCsrValues->bufferAsT<X>();        // [csr_nnz]
        auto* dCBuf    = dCooValues->bufferAsT<X>();            // [coo_nnz]

  for (sd::LongType k = 0; k < cooNnz; ++k) {
    const int32_t row = static_cast<int32_t>(idxBuf[k * 2 + 0]);
    const int32_t col = static_cast<int32_t>(idxBuf[k * 2 + 1]);

    // Search window: [rowBegin, rowEnd) in the CSR column index array.
    const int32_t rowBegin = rowBuf[row];
    const int32_t rowEnd   = rowBuf[row + 1];

    // Binary search for column j within [rowBegin, rowEnd).
    int32_t lo = rowBegin, hi = rowEnd;
    int32_t foundPos = -1;
    while (lo < hi) {
      const int32_t mid = lo + (hi - lo) / 2;
      if (colBuf[mid] == col) { foundPos = mid; break; }
      else if (colBuf[mid] < col) lo = mid + 1;
      else                         hi = mid;
    }

    // Copy the upstream gradient for the CSR slot back to this COO entry.
    dCBuf[k] = (foundPos >= 0) ? gradBuf[foundPos] : static_cast<X>(0);
  }
}

void cooToCsr_bp(sd::LaunchContext* context,
                 NDArray* cooIndices, NDArray* csrColIdx, NDArray* csrRowPtr,
                 NDArray* gradCsrValues, NDArray* dCooValues,
                 sd::LongType rows, sd::LongType cols) {
  NDArray::preparePrimaryUse({dCooValues},
                             {cooIndices, csrColIdx, csrRowPtr, gradCsrValues});

  BUILD_DOUBLE_SELECTOR(gradCsrValues->dataType(), cooIndices->dataType(), cooToCsr_bp_,
                        (cooIndices, csrColIdx, csrRowPtr, gradCsrValues, dCooValues, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({dCooValues},
                              {cooIndices, csrColIdx, csrRowPtr, gradCsrValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
