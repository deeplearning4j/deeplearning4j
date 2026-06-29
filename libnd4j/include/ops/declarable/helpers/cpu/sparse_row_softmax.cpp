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
// csr_row_softmax / csr_row_softmax_bp — CPU implementation.
//
// Forward (per row i):
//   1. m = max over k in [rowPtr[i], rowPtr[i+1]) of values[k]
//   2. alpha[k] = exp(values[k] - m)
//   3. s = sum of alpha[k] over the row
//   4. alpha[k] /= s
//   Empty row → nothing written.
//
// Backward (per row i):
//   dot = sum_k alpha[k] * gradOut[k]
//   dValues[k] = alpha[k] * (gradOut[k] - dot)
//

#include <ops/declarable/helpers/sparse_row_softmax.h>
#include <system/op_boilerplate.h>

#include <cmath>
#include <cstring>

namespace sd {
namespace ops {
namespace helpers {

// ────────────────────────────────────────────────────────────────────────────
// Forward kernel
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrRowSoftmax_(NDArray& values, NDArray& rowPtr, NDArray& alpha,
                            sd::LongType rows) {
  const X* vBuf  = values.bufferAsT<X>();
  const I* rpBuf = rowPtr.bufferAsT<I>();
  X*       aBuf  = alpha.bufferAsT<X>();

  for (sd::LongType i = 0; i < rows; ++i) {
    const I eStart = rpBuf[i];
    const I eEnd   = rpBuf[i + 1];
    if (eStart == eEnd) continue;  // empty row — nothing to do

    // Step 1: find row max for numerical stability
    X m = vBuf[eStart];
    for (I e = eStart + 1; e < eEnd; ++e) {
      if (vBuf[e] > m) m = vBuf[e];
    }

    // Step 2: shifted exp
    X s = static_cast<X>(0);
    for (I e = eStart; e < eEnd; ++e) {
      X ex = static_cast<X>(std::exp(static_cast<double>(vBuf[e] - m)));
      aBuf[e] = ex;
      s += ex;
    }

    // Step 3: normalise
    for (I e = eStart; e < eEnd; ++e) {
      aBuf[e] /= s;
    }
  }
}

void csr_row_softmax(NDArray& values, NDArray& rowPtr, NDArray& alpha,
                     sd::LongType rows) {
  NDArray::preparePrimaryUse({&alpha}, {&values, &rowPtr});

  BUILD_DOUBLE_SELECTOR(values.dataType(), rowPtr.dataType(), csrRowSoftmax_,
                        (values, rowPtr, alpha, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&alpha}, {&values, &rowPtr});
}

// ────────────────────────────────────────────────────────────────────────────
// Backward kernel
// ────────────────────────────────────────────────────────────────────────────

template <typename X, typename I>
static void csrRowSoftmaxBp_(NDArray& alpha, NDArray& rowPtr, NDArray& gradOut,
                              NDArray& dValues, sd::LongType rows) {
  const X* aBuf  = alpha.bufferAsT<X>();
  const I* rpBuf = rowPtr.bufferAsT<I>();
  const X* goBuf = gradOut.bufferAsT<X>();
  X*       dvBuf = dValues.bufferAsT<X>();

  for (sd::LongType i = 0; i < rows; ++i) {
    const I eStart = rpBuf[i];
    const I eEnd   = rpBuf[i + 1];
    if (eStart == eEnd) continue;  // empty row

    // dot_i = sum_k alpha[k] * gradOut[k]
    X dot = static_cast<X>(0);
    for (I e = eStart; e < eEnd; ++e) {
      dot += aBuf[e] * goBuf[e];
    }

    // dValues[k] = alpha[k] * (gradOut[k] - dot_i)
    for (I e = eStart; e < eEnd; ++e) {
      dvBuf[e] = aBuf[e] * (goBuf[e] - dot);
    }
  }
}

void csr_row_softmax_bp(NDArray& alpha, NDArray& rowPtr, NDArray& gradOut,
                         NDArray& dValues, sd::LongType rows) {
  NDArray::preparePrimaryUse({&dValues}, {&alpha, &rowPtr, &gradOut});

  BUILD_DOUBLE_SELECTOR(alpha.dataType(), rowPtr.dataType(), csrRowSoftmaxBp_,
                        (alpha, rowPtr, gradOut, dValues, rows),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dValues}, {&alpha, &rowPtr, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
