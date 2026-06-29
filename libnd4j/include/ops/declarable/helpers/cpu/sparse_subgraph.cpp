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
// csr_subgraph_extract / csr_subgraph_extract_bp — CPU implementation.
//
// Forward — 2-pass algorithm:
//   Pass 1: for each selected row s (in nodeIdx order), scan its original edges
//           [rowPtr[nodeIdx[s]], rowPtr[nodeIdx[s]+1]). Binary-search colIdx[e]
//           in nodeIdx to decide if the destination is also selected. Count kept
//           edges per selected row → fill newRowPtr via exclusive prefix-sum.
//   Pass 2: scatter kept edge values and remapped column ids into newValues /
//           newColIdx using the now-known offsets in newRowPtr.
//
// Backward — traverse the same kept-edge set; scatter dNewValues[e'] → dValues[e].
//   dValues is zeroed first (nullify() from the op + std::memset guard here
//   for absolute safety).
//
// All buffers accessed via bufferAsT<T>()[raw-index] (NOT e<T>()/p() on
// sub-views — that double-counts _offset; this footgun caused embeddingBag bug).
//

#include <ops/declarable/helpers/sparse_subgraph.h>
#include <system/op_boilerplate.h>

#include <algorithm>
#include <cstring>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Helpers: binary search of `key` in sorted int array niBuf[0..K-1].
// Returns the 0-based position (rank) if found, or -1.
// ────────────────────────────────────────────────────────────────────────────

template <typename I>
static sd::LongType bsearchRank(const I* niBuf, sd::LongType K, I key) {
  sd::LongType lo = 0, hi = K - 1;
  while (lo <= hi) {
    sd::LongType mid = (lo + hi) >> 1;
    I mv = niBuf[mid];
    if (mv == key)       return mid;
    else if (mv < key)   lo = mid + 1;
    else                 hi = mid - 1;
  }
  return static_cast<sd::LongType>(-1);
}

// ────────────────────────────────────────────────────────────────────────────
// Forward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrSubgraphExtract_(NDArray& values,    NDArray& colIdx,    NDArray& rowPtr,
                                 NDArray& nodeIdx,   NDArray& newValues, NDArray& newColIdx,
                                 NDArray& newRowPtr, sd::LongType N,     sd::LongType K) {
  const X* valBuf  = values.bufferAsT<X>();
  const I* ciBuf   = colIdx.bufferAsT<I>();
  const I* rpBuf   = rowPtr.bufferAsT<I>();
  const I* niBuf   = nodeIdx.bufferAsT<I>();
  X*       nvBuf   = newValues.bufferAsT<X>();
  int*     ncisBuf = newColIdx.bufferAsT<int>();
  int*     nrpBuf  = newRowPtr.bufferAsT<int>();

  // ---- Pass 1: build newRowPtr (counts per selected row → exclusive prefix-sum) ----
  // newRowPtr has K+1 slots; nrpBuf[0] = 0.
  nrpBuf[0] = 0;
  for (sd::LongType s = 0; s < K; ++s) {
    const I  origRow = niBuf[s];
    const I  eStart  = rpBuf[static_cast<sd::LongType>(origRow)];
    const I  eEnd    = rpBuf[static_cast<sd::LongType>(origRow) + 1];
    int count = 0;
    for (I e = eStart; e < eEnd; ++e) {
      I col = ciBuf[static_cast<sd::LongType>(e)];
      if (bsearchRank(niBuf, K, col) >= 0) ++count;
    }
    nrpBuf[s + 1] = nrpBuf[s] + count;
  }

  // ---- Pass 2: fill newValues and newColIdx ----
  for (sd::LongType s = 0; s < K; ++s) {
    const I  origRow = niBuf[s];
    const I  eStart  = rpBuf[static_cast<sd::LongType>(origRow)];
    const I  eEnd    = rpBuf[static_cast<sd::LongType>(origRow) + 1];
    int writePos     = nrpBuf[s];
    for (I e = eStart; e < eEnd; ++e) {
      I col = ciBuf[static_cast<sd::LongType>(e)];
      sd::LongType rank = bsearchRank(niBuf, K, col);
      if (rank >= 0) {
        nvBuf[writePos]  = valBuf[static_cast<sd::LongType>(e)];
        ncisBuf[writePos] = static_cast<int>(rank);
        ++writePos;
      }
    }
  }
}

void csr_subgraph_extract(LaunchContext* ctx,
                           NDArray& values,    NDArray& colIdx,    NDArray& rowPtr,
                           NDArray& nodeIdx,   NDArray& newValues, NDArray& newColIdx,
                           NDArray& newRowPtr, sd::LongType N,     sd::LongType K) {
  NDArray::preparePrimaryUse({&newValues, &newColIdx, &newRowPtr},
                             {&values, &colIdx, &rowPtr, &nodeIdx});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSubgraphExtract_,
                        (values, colIdx, rowPtr, nodeIdx, newValues, newColIdx, newRowPtr, N, K),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&newValues, &newColIdx, &newRowPtr},
                              {&values, &colIdx, &rowPtr, &nodeIdx});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrSubgraphExtractBp_(NDArray& values,     NDArray& colIdx,    NDArray& rowPtr,
                                   NDArray& nodeIdx,    NDArray& dNewValues,NDArray& dValues,
                                   sd::LongType N,      sd::LongType K) {
  const I* ciBuf   = colIdx.bufferAsT<I>();
  const I* rpBuf   = rowPtr.bufferAsT<I>();
  const I* niBuf   = nodeIdx.bufferAsT<I>();
  const X* dnvBuf  = dNewValues.bufferAsT<X>();
  X*       dvBuf   = dValues.bufferAsT<X>();

  // Zero dValues (OUTPUT_NULLIFIED already zeros it; belt-and-suspenders).
  std::memset(dvBuf, 0, sizeof(X) * static_cast<size_t>(dValues.lengthOf()));

  // Walk the same pass-2 traversal as the forward to determine the e → e' mapping.
  // We need to reconstruct the original edge index e for each kept edge.
  sd::LongType ePrime = 0;  // running position in extracted CSR (= e')
  for (sd::LongType s = 0; s < K; ++s) {
    const I  origRow = niBuf[s];
    const I  eStart  = rpBuf[static_cast<sd::LongType>(origRow)];
    const I  eEnd    = rpBuf[static_cast<sd::LongType>(origRow) + 1];
    for (I e = eStart; e < eEnd; ++e) {
      I col = ciBuf[static_cast<sd::LongType>(e)];
      if (bsearchRank(niBuf, K, col) >= 0) {
        dvBuf[static_cast<sd::LongType>(e)] = dnvBuf[ePrime];
        ++ePrime;
      }
    }
  }
}

void csr_subgraph_extract_bp(LaunchContext* ctx,
                               NDArray& values,     NDArray& colIdx,     NDArray& rowPtr,
                               NDArray& nodeIdx,    NDArray& dNewValues, NDArray& dValues,
                               sd::LongType N,      sd::LongType K) {
  NDArray::preparePrimaryUse({&dValues},
                             {&values, &colIdx, &rowPtr, &nodeIdx, &dNewValues});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrSubgraphExtractBp_,
                        (values, colIdx, rowPtr, nodeIdx, dNewValues, dValues, N, K),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dValues},
                              {&values, &colIdx, &rowPtr, &nodeIdx, &dNewValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
