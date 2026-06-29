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
#include <vector>

#include <ops/declarable/helpers/sparse_coo.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// denseToCoo — CPU
// ---------------------------------------------------------------------------
template <typename X>
static void denseToCoo_(NDArray* dense, NDArray* indices, NDArray* values, double threshold) {
  const auto rows = dense->sizeAt(0);
  const auto cols = dense->sizeAt(1);
  const X    thr  = static_cast<X>(threshold);

  const auto* inStride = dense->stridesOf();
  const auto* iBuf     = dense->bufferAsT<X>();
  auto*       vBuf     = values->bufferAsT<X>();
  // indices has shape [nnz, 2] with row-major strides [2, 1]
  auto*       idxBuf   = indices->bufferAsT<sd::LongType>();

  sd::LongType nnzCount = 0;
  for (sd::LongType r = 0; r < rows; ++r) {
    for (sd::LongType c = 0; c < cols; ++c) {
      const X val    = iBuf[r * inStride[0] + c * inStride[1]];
      const X absVal = val < static_cast<X>(0) ? -val : val;
      if (absVal > thr) {
        idxBuf[nnzCount * 2 + 0] = r;
        idxBuf[nnzCount * 2 + 1] = c;
        vBuf[nnzCount]            = val;
        ++nnzCount;
      }
    }
  }
}

void denseToCoo(sd::LaunchContext* context, NDArray* dense, NDArray* indices, NDArray* values,
                double threshold) {
  NDArray::preparePrimaryUse({indices, values}, {dense});

  BUILD_SINGLE_SELECTOR(dense->dataType(), denseToCoo_,
                        (dense, indices, values, threshold),
                        SD_FLOAT_TYPES);

  NDArray::registerPrimaryUse({indices, values}, {dense});
}

// ---------------------------------------------------------------------------
// cooToCsr — CPU
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void cooToCsr_(NDArray* cooIndices, NDArray* cooValues, NDArray* csrValues,
                      NDArray* colIdx, NDArray* rowPtr, sd::LongType rows) {
  const sd::LongType nnz = cooValues->lengthOf();

  const auto* idxBuf = cooIndices->bufferAsT<I>();
  const auto* vBuf   = cooValues->bufferAsT<X>();
  auto*       cvBuf  = csrValues->bufferAsT<X>();
  auto*       ciBuf  = colIdx->bufferAsT<int32_t>();
  auto*       rpBuf  = rowPtr->bufferAsT<int32_t>();

  // Collect (row, col, val) triples from COO inputs.
  // cooIndices has shape [nnz, 2] with row-major strides [2, 1].
  struct Triple {
    I row, col;
    X val;
  };
  std::vector<Triple> triples(static_cast<size_t>(nnz));
  for (sd::LongType k = 0; k < nnz; ++k) {
    triples[k].row = idxBuf[k * 2 + 0];
    triples[k].col = idxBuf[k * 2 + 1];
    triples[k].val = vBuf[k];
  }

  // Sort by (row, then col) so the CSR output is in canonical order.
  std::sort(triples.begin(), triples.end(), [](const Triple& a, const Triple& b) {
    if (a.row != b.row) return a.row < b.row;
    return a.col < b.col;
  });

  // Coalesce: merge-sum entries with the same (row, col) key.
  // csrValues and colIdx are already sized to coalescedNnz by DECLARE_SHAPE_FN.
  for (sd::LongType r = 0; r <= rows; ++r) rpBuf[r] = 0;

  sd::LongType out = 0;
  for (sd::LongType k = 0; k < nnz; ) {
    X  sumVal = triples[k].val;
    I  curRow = triples[k].row;
    I  curCol = triples[k].col;
    sd::LongType j = k + 1;
    while (j < nnz && triples[j].row == curRow && triples[j].col == curCol) {
      sumVal += triples[j].val;
      ++j;
    }
    cvBuf[out] = sumVal;
    ciBuf[out] = static_cast<int32_t>(curCol);
    rpBuf[static_cast<sd::LongType>(curRow) + 1]++;
    ++out;
    k = j;
  }

  // Prefix-sum rowPtr counts into row pointers.
  for (sd::LongType r = 1; r <= rows; ++r) {
    rpBuf[r] += rpBuf[r - 1];
  }
}

void cooToCsr(sd::LaunchContext* context, NDArray* cooIndices, NDArray* cooValues,
              NDArray* csrValues, NDArray* colIdx, NDArray* rowPtr, sd::LongType rows) {
  NDArray::preparePrimaryUse({csrValues, colIdx, rowPtr}, {cooIndices, cooValues});

  BUILD_DOUBLE_SELECTOR(cooValues->dataType(), cooIndices->dataType(), cooToCsr_,
                        (cooIndices, cooValues, csrValues, colIdx, rowPtr, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({csrValues, colIdx, rowPtr}, {cooIndices, cooValues});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
