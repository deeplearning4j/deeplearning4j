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
// CSR sparse-sparse elementwise addition — CPU implementation.
//
// C = A + B, A and B both CSR [m, n] with the SAME logical shape.
// Output C is CSR with pattern = column-set union of A and B per row;
// overlapping entries are summed.
//
// Algorithm: per row, perform a standard sorted two-pointer merge over the
// two sorted (colIdx, value) sequences of A and B.  Both input per-row colIdx
// sequences are sorted ascending (standard CSR invariant).
//
//   while both pointers valid:
//     col_a < col_b  → emit (col_a, val_a),  advance A
//     col_a > col_b  → emit (col_b, val_b),  advance B
//     col_a == col_b → emit (col_a, val_a + val_b), advance both
//   flush remaining A / B tail entries
//
// Template parameters:
//   X  — float dtype (aValues, bValues, cValues)
//   I  — integer dtype of the input structure arrays (INT32 or INT64)
//
// Output cColIdx and cRowPtr are ALWAYS written as int32_t, matching the
// INT32 shapes produced by DECLARE_SHAPE_FN.
//

#include <ops/declarable/helpers/sparse_add.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename I>
static void csrAdd_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                    NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
                    NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
                    sd::LongType m, sd::LongType /*n*/) {
  const auto* avBuf  = aValues.bufferAsT<X>();
  const auto* aciBuf = aColIdx.bufferAsT<I>();
  const auto* arpBuf = aRowPtr.bufferAsT<I>();
  const auto* bvBuf  = bValues.bufferAsT<X>();
  const auto* bciBuf = bColIdx.bufferAsT<I>();
  const auto* brpBuf = bRowPtr.bufferAsT<I>();

  // Outputs: cValues has type X, but cColIdx/cRowPtr are always INT32
  auto*  cvBuf  = cValues.bufferAsT<X>();
  auto*  cciOut = cColIdx.bufferAsT<int32_t>();
  auto*  crpOut = cRowPtr.bufferAsT<int32_t>();

  crpOut[0] = 0;
  LongType pos = 0;  // next write position in cValues / cColIdx

  for (LongType i = 0; i < m; ++i) {
    const I aStart = arpBuf[static_cast<size_t>(i)];
    const I aEnd   = arpBuf[static_cast<size_t>(i + 1)];
    const I bStart = brpBuf[static_cast<size_t>(i)];
    const I bEnd   = brpBuf[static_cast<size_t>(i + 1)];

    I a = aStart, b = bStart;

    // Sorted two-pointer merge
    while (a < aEnd && b < bEnd) {
      const auto colA = static_cast<LongType>(aciBuf[static_cast<size_t>(a)]);
      const auto colB = static_cast<LongType>(bciBuf[static_cast<size_t>(b)]);
      const X    valA = avBuf[static_cast<size_t>(a)];
      const X    valB = bvBuf[static_cast<size_t>(b)];

      X    emitVal;
      LongType emitCol;

      if (colA < colB) {
        emitCol = colA;
        emitVal = valA;
        ++a;
      } else if (colA > colB) {
        emitCol = colB;
        emitVal = valB;
        ++b;
      } else {
        // Equal column: sum the two values
        emitCol = colA;
        emitVal = valA + valB;
        ++a;
        ++b;
      }

      cvBuf[static_cast<size_t>(pos)]  = emitVal;
      cciOut[static_cast<size_t>(pos)] = static_cast<int32_t>(emitCol);
      ++pos;
    }

    // Flush remaining A tail
    while (a < aEnd) {
      cvBuf[static_cast<size_t>(pos)]  = avBuf[static_cast<size_t>(a)];
      cciOut[static_cast<size_t>(pos)] =
          static_cast<int32_t>(aciBuf[static_cast<size_t>(a)]);
      ++pos;
      ++a;
    }

    // Flush remaining B tail
    while (b < bEnd) {
      cvBuf[static_cast<size_t>(pos)]  = bvBuf[static_cast<size_t>(b)];
      cciOut[static_cast<size_t>(pos)] =
          static_cast<int32_t>(bciBuf[static_cast<size_t>(b)]);
      ++pos;
      ++b;
    }

    crpOut[static_cast<size_t>(i + 1)] = static_cast<int32_t>(pos);
  }
}

void csr_add(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
             NDArray& bValues, NDArray& bColIdx, NDArray& bRowPtr,
             NDArray& cValues, NDArray& cColIdx, NDArray& cRowPtr,
             sd::LongType m, sd::LongType n) {
  NDArray::preparePrimaryUse({&cValues, &cColIdx, &cRowPtr},
                             {&aValues, &aColIdx, &aRowPtr,
                              &bValues, &bColIdx, &bRowPtr});

  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrAdd_,
                        (aValues, aColIdx, aRowPtr,
                         bValues, bColIdx, bRowPtr,
                         cValues, cColIdx, cRowPtr,
                         m, n),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&cValues, &cColIdx, &cRowPtr},
                              {&aValues, &aColIdx, &aRowPtr,
                               &bValues, &bColIdx, &bRowPtr});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
