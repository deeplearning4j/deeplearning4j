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
// CSR SpGEMM — CPU implementation using Gustavson's row-by-row algorithm.
//
// For each row i of A:
//   1. Scatter: for each (ka, kb) pair from A-row i and B-rows, accumulate
//      into a dense vector accum[n] and record touched columns.
//   2. Sort touched columns (ascending) to produce row-sorted CSR output.
//   3. Write cRowPtr, cColIdx, cValues.
//   4. Reset accum and mask for touched entries only (O(nnz_row) reset).
//
// Template parameters:
//   X  — float dtype (aValues, bValues, cValues)
//   I  — index dtype of the input structure arrays (aColIdx, aRowPtr,
//         bColIdx, bRowPtr);  INT32 or INT64.
//
// The output cColIdx and cRowPtr are ALWAYS written as int32_t, matching the
// INT32 shapes produced by DECLARE_SHAPE_FN.
//

#include <ops/declarable/helpers/sparse_spgemm.h>
#include <system/op_boilerplate.h>

#include <algorithm>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename I>
static void csrSpGEMM_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                       NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
                       NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
                       sd::LongType m, sd::LongType /*k*/, sd::LongType n) {
  const auto* avBuf  = aValues.bufferAsT<X>();
  const auto* aciBuf = aColIdx.bufferAsT<I>();
  const auto* arpBuf = aRowPtr.bufferAsT<I>();
  const auto* bvBuf  = bValues.bufferAsT<X>();
  const auto* bciBuf = bColIdx.bufferAsT<I>();
  const auto* brpBuf = bRowPtr.bufferAsT<I>();

  // Outputs: cValues is X, but cColIdx/cRowPtr are always INT32
  auto*  cvBuf  = cValues.bufferAsT<X>();
  auto*  cciOut = cColIdx.bufferAsT<int32_t>();
  auto*  crpOut = cRowPtr.bufferAsT<int32_t>();

  // Dense accumulator and visited-sentinel for Gustavson's algorithm.
  // Sentinel value: -1  (no valid row index equals -1).
  std::vector<X>        accum(static_cast<size_t>(n), static_cast<X>(0));
  std::vector<LongType> mask(static_cast<size_t>(n), static_cast<LongType>(-1));
  std::vector<LongType> touched;

  crpOut[0] = 0;
  LongType pos = 0;  // next write position in cValues / cColIdx

  for (LongType i = 0; i < m; ++i) {
    touched.clear();

    const I aStart = arpBuf[static_cast<size_t>(i)];
    const I aEnd   = arpBuf[static_cast<size_t>(i + 1)];

    for (I ka = aStart; ka < aEnd; ++ka) {
      const I  kcol  = aciBuf[static_cast<size_t>(ka)];
      const X  aVal  = avBuf[static_cast<size_t>(ka)];

      const I bStart = brpBuf[static_cast<size_t>(kcol)];
      const I bEnd   = brpBuf[static_cast<size_t>(kcol + 1)];

      for (I kb = bStart; kb < bEnd; ++kb) {
        const auto bCol = static_cast<LongType>(bciBuf[static_cast<size_t>(kb)]);
        accum[static_cast<size_t>(bCol)] += aVal * bvBuf[static_cast<size_t>(kb)];
        if (mask[static_cast<size_t>(bCol)] == static_cast<LongType>(-1)) {
          mask[static_cast<size_t>(bCol)] = i;  // mark visited for row i
          touched.push_back(bCol);
        }
      }
    }

    // Sort columns for CSR row-order convention
    std::sort(touched.begin(), touched.end());

    // Flush touched entries to output and reset scratch arrays
    for (LongType c : touched) {
      cvBuf[static_cast<size_t>(pos)]  = accum[static_cast<size_t>(c)];
      cciOut[static_cast<size_t>(pos)] = static_cast<int32_t>(c);
      ++pos;
      accum[static_cast<size_t>(c)] = static_cast<X>(0);
      mask[static_cast<size_t>(c)]  = static_cast<LongType>(-1);
    }

    crpOut[static_cast<size_t>(i + 1)] = static_cast<int32_t>(pos);
  }
}

void csr_spgemm(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
                NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
                sd::LongType m, sd::LongType k, sd::LongType n) {
  NDArray::preparePrimaryUse({&cValues, &cColIdx, &cRowPtr},
                             {&aValues, &aColIdx, &aRowPtr,
                              &bValues, &bColIdx, &bRowPtr});

  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrSpGEMM_,
                        (aValues, aColIdx, aRowPtr,
                         bValues, bColIdx, bRowPtr,
                         cValues, cColIdx, cRowPtr,
                         m, k, n),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&cValues, &cColIdx, &cRowPtr},
                              {&aValues, &aColIdx, &aRowPtr,
                               &bValues, &bColIdx, &bRowPtr});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
