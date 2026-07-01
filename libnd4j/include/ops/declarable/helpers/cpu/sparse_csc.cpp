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

#include <ops/declarable/helpers/sparse_csc.h>
#include <system/op_boilerplate.h>

#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// csr_to_csc — CPU
//
// Counting-sort transpose algorithm:
//   1. Count nnz per column → temporary colCount[].
//   2. Exclusive prefix sum of colCount → cscColPtr[0..cols].
//   3. Scatter pass (row-major scan of CSR): for each row r and each entry k
//      in [rowPtr[r], rowPtr[r+1]), write:
//        cscValues[cursor[c]]  = csrValues[k]
//        cscRowIdx [cursor[c]] = r
//      then advance cursor[c]++.
//
// Template params:
//   X — float type (values)
//   I — CSR input integer type (INT32 or INT64)
//
// Output index arrays (cscRowIdx, cscColPtr) are always INT32 per the op
// contract emitted by DECLARE_SHAPE_FN, so they are written as int32_t.
// ---------------------------------------------------------------------------

template <typename X, typename I>
static void csrToCsc_(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                      NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                      LongType rows, LongType cols) {
  const LongType nnz = csrValues.lengthOf();

  const auto* vBuf  = csrValues.bufferAsT<X>();
  const auto* ciBuf = csrColIdx.bufferAsT<I>();
  const auto* rpBuf = csrRowPtr.bufferAsT<I>();

  auto*       cscVBuf  = cscValues.bufferAsT<X>();
  auto*       cscRiBuf = cscRowIdx.bufferAsT<int32_t>();  // always INT32
  auto*       cscCpBuf = cscColPtr.bufferAsT<int32_t>();  // always INT32

  // Step 1: count nnz per column
  std::vector<int32_t> colCount(static_cast<size_t>(cols), 0);
  for (LongType k = 0; k < nnz; ++k) {
    ++colCount[static_cast<size_t>(ciBuf[k])];
  }

  // Step 2: exclusive prefix sum → cscColPtr
  cscCpBuf[0] = 0;
  for (LongType c = 0; c < cols; ++c) {
    cscCpBuf[c + 1] = cscCpBuf[c] + colCount[static_cast<size_t>(c)];
  }

  // Step 3: scatter — use a write-cursor initialised from cscColPtr[0..cols-1]
  std::vector<int32_t> cursor(cscCpBuf, cscCpBuf + static_cast<size_t>(cols));

  for (LongType r = 0; r < rows; ++r) {
    const I start = rpBuf[r];
    const I end   = rpBuf[r + 1];
    for (I k = start; k < end; ++k) {
      const int32_t c    = static_cast<int32_t>(ciBuf[k]);
      const int32_t slot = cursor[static_cast<size_t>(c)]++;
      cscRiBuf[slot] = static_cast<int32_t>(r);
      cscVBuf [slot] = vBuf[k];
    }
  }
}

void csr_to_csc(NDArray& csrValues, NDArray& csrColIdx, NDArray& csrRowPtr,
                NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                LongType rows, LongType cols) {
  NDArray::preparePrimaryUse({&cscValues, &cscRowIdx, &cscColPtr},
                             {&csrValues, &csrColIdx, &csrRowPtr});

  BUILD_DOUBLE_SELECTOR(csrValues.dataType(), csrColIdx.dataType(), csrToCsc_,
                        (csrValues, csrColIdx, csrRowPtr,
                         cscValues, cscRowIdx, cscColPtr, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&cscValues, &cscRowIdx, &cscColPtr},
                              {&csrValues, &csrColIdx, &csrRowPtr});
}

// ---------------------------------------------------------------------------
// csc_to_dense — CPU
//
// For each column j, scatter entries in cscValues[cscColPtr[j] ..
// cscColPtr[j+1]) into output[cscRowIdx[k], j].
// output is OUTPUT_NULLIFIED (pre-zeroed); we only write the non-zeros.
//
// I is the integer type of the index inputs (always INT32 from op contract).
// ---------------------------------------------------------------------------

template <typename X, typename I>
static void cscToDense_(NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                        NDArray& output) {
  const LongType cols = output.sizeAt(1);

  const auto* vBuf   = cscValues.bufferAsT<X>();
  const auto* riBuf  = cscRowIdx.bufferAsT<I>();
  const auto* cpBuf  = cscColPtr.bufferAsT<I>();
  auto*       oBuf   = output.bufferAsT<X>();

  const auto* oStride = output.stridesOf();  // honour arbitrary strides

  for (LongType j = 0; j < cols; ++j) {
    const I start = cpBuf[j];
    const I end   = cpBuf[j + 1];
    for (I k = start; k < end; ++k) {
      const LongType r = static_cast<LongType>(riBuf[k]);
      oBuf[r * oStride[0] + j * oStride[1]] = vBuf[k];
    }
  }
}

void csc_to_dense(NDArray& cscValues, NDArray& cscRowIdx, NDArray& cscColPtr,
                  NDArray& output) {
  NDArray::preparePrimaryUse({&output}, {&cscValues, &cscRowIdx, &cscColPtr});

  BUILD_DOUBLE_SELECTOR(cscValues.dataType(), cscRowIdx.dataType(), cscToDense_,
                        (cscValues, cscRowIdx, cscColPtr, output),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&output}, {&cscValues, &cscRowIdx, &cscColPtr});
}

// ---------------------------------------------------------------------------
// dense_to_csc — CPU
//
// Column-major scan: for c in 0..cols-1, then r in 0..rows-1.
// Emits an entry whenever |x| > threshold.
// cscColPtr[c+1] is filled as a running count (dense_to_csr row-pointer style).
//
// I — output integer type (always INT32 per DECLARE_SHAPE_FN).
// ---------------------------------------------------------------------------

template <typename X, typename I>
static void denseToCsc_(NDArray& input, NDArray& cscValues, NDArray& cscRowIdx,
                        NDArray& cscColPtr, double threshold) {
  const LongType rows = input.sizeAt(0);
  const LongType cols = input.sizeAt(1);
  const X        thr  = static_cast<X>(threshold);

  const auto* inStride = input.stridesOf();
  const auto* iBuf     = input.bufferAsT<X>();
  auto*       vBuf     = cscValues.bufferAsT<X>();
  auto*       riBuf    = cscRowIdx.bufferAsT<int32_t>();  // always INT32
  auto*       cpBuf    = cscColPtr.bufferAsT<int32_t>();  // always INT32

  cpBuf[0] = 0;
  int32_t nnzCount = 0;

  for (LongType c = 0; c < cols; ++c) {
    for (LongType r = 0; r < rows; ++r) {
      const X val    = iBuf[r * inStride[0] + c * inStride[1]];
      const X absVal = val < static_cast<X>(0) ? -val : val;
      if (absVal > thr) {
        vBuf [nnzCount] = val;
        riBuf[nnzCount] = static_cast<int32_t>(r);
        ++nnzCount;
      }
    }
    cpBuf[c + 1] = nnzCount;
  }
}

void dense_to_csc(sd::LaunchContext* /*context*/, NDArray& input, NDArray& cscValues,
                  NDArray& cscRowIdx, NDArray& cscColPtr, double threshold) {
  NDArray::preparePrimaryUse({&cscValues, &cscRowIdx, &cscColPtr}, {&input});

  BUILD_DOUBLE_SELECTOR(input.dataType(), cscRowIdx.dataType(), denseToCsc_,
                        (input, cscValues, cscRowIdx, cscColPtr, threshold),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&cscValues, &cscRowIdx, &cscColPtr}, {&input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
