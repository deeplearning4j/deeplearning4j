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
// csr_segment_max / csr_segment_max_bp — CPU implementation.
//
// Forward — outer loop over rows i, inner loop over features f:
//   For each (i, f): iterate neighbors k in [rowPtr[i], rowPtr[i+1]);
//     take max of X[colIdx[k], f].
//   Empty row → 0 in out[i, f].
//   X and out are accessed stride-aware (arbitrary layout from SameDiff).
//
// Backward — outer loop over (i, f):
//   Recompute argmax k*; accumulate gradOut[i, f] into dX[colIdx[k*], f].
//   CPU is sequential so no atomics needed here; dX is zeroed via
//   std::memset before the loop.
//

#include <ops/declarable/helpers/sparse_segment_max.h>
#include <system/op_boilerplate.h>

#include <cstring>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrSegmentMax_(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr,
                            NDArray& out, sd::LongType rows) {
  const I* ciBuf  = colIdx.bufferAsT<I>();
  const I* rpBuf  = rowPtr.bufferAsT<I>();
  const X* xBuf   = xArr.bufferAsT<X>();
  X*       outBuf = out.bufferAsT<X>();

  const sd::LongType f    = xArr.sizeAt(1);
  const sd::LongType xs0  = xArr.strideAt(0);
  const sd::LongType xs1  = xArr.strideAt(1);
  const sd::LongType os0  = out.strideAt(0);
  const sd::LongType os1  = out.strideAt(1);

  for (sd::LongType i = 0; i < rows; ++i) {
    const I eStart = rpBuf[i];
    const I eEnd   = rpBuf[i + 1];

    for (sd::LongType feat = 0; feat < f; ++feat) {
      X*        o = outBuf + i * os0 + feat * os1;
      if (eStart == eEnd) {
        *o = static_cast<X>(0);
        continue;
      }

      // Initialise to the first neighbor's value, then scan the rest
      I firstNode = ciBuf[eStart];
      X m = xBuf[static_cast<sd::LongType>(firstNode) * xs0 + feat * xs1];

      for (I e = eStart + 1; e < eEnd; ++e) {
        I node = ciBuf[e];
        X v    = xBuf[static_cast<sd::LongType>(node) * xs0 + feat * xs1];
        if (v > m) m = v;
      }
      *o = m;
    }
  }
}

void csr_segment_max(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr, NDArray& out,
                     sd::LongType rows) {
  NDArray::preparePrimaryUse({&out}, {&colIdx, &rowPtr, &xArr});

  BUILD_DOUBLE_SELECTOR(xArr.dataType(), colIdx.dataType(), csrSegmentMax_,
                        (colIdx, rowPtr, xArr, out, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&out}, {&colIdx, &rowPtr, &xArr});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrSegmentMaxBp_(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr,
                              NDArray& gradOut, NDArray& dX, sd::LongType rows) {
  const I* ciBuf  = colIdx.bufferAsT<I>();
  const I* rpBuf  = rowPtr.bufferAsT<I>();
  const X* xBuf   = xArr.bufferAsT<X>();
  const X* goBuf  = gradOut.bufferAsT<X>();
  X*       dxBuf  = dX.bufferAsT<X>();

  const sd::LongType f    = xArr.sizeAt(1);
  const sd::LongType xs0  = xArr.strideAt(0);
  const sd::LongType xs1  = xArr.strideAt(1);
  const sd::LongType gs0  = gradOut.strideAt(0);
  const sd::LongType gs1  = gradOut.strideAt(1);
  const sd::LongType ds0  = dX.strideAt(0);
  const sd::LongType ds1  = dX.strideAt(1);

  // Zero-initialise dX (scatter target)
  std::memset(dxBuf, 0, sizeof(X) * static_cast<size_t>(dX.lengthOf()));

  for (sd::LongType i = 0; i < rows; ++i) {
    const I eStart = rpBuf[i];
    const I eEnd   = rpBuf[i + 1];
    if (eStart == eEnd) continue;  // empty row — no contribution

    for (sd::LongType feat = 0; feat < f; ++feat) {
      // Recompute argmax k*
      I argmaxNode = ciBuf[eStart];
      X maxVal     = xBuf[static_cast<sd::LongType>(argmaxNode) * xs0 + feat * xs1];

      for (I e = eStart + 1; e < eEnd; ++e) {
        I node = ciBuf[e];
        X v    = xBuf[static_cast<sd::LongType>(node) * xs0 + feat * xs1];
        if (v > maxVal) {
          maxVal     = v;
          argmaxNode = node;
        }
      }

      // Accumulate gradient into dX at the argmax node
      dxBuf[static_cast<sd::LongType>(argmaxNode) * ds0 + feat * ds1] +=
          goBuf[i * gs0 + feat * gs1];
    }
  }
}

void csr_segment_max_bp(NDArray& colIdx, NDArray& rowPtr, NDArray& xArr,
                         NDArray& gradOut, NDArray& dX, sd::LongType rows) {
  NDArray::preparePrimaryUse({&dX}, {&colIdx, &rowPtr, &xArr, &gradOut});

  BUILD_DOUBLE_SELECTOR(xArr.dataType(), colIdx.dataType(), csrSegmentMaxBp_,
                        (colIdx, rowPtr, xArr, gradOut, dX, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dX}, {&colIdx, &rowPtr, &xArr, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
