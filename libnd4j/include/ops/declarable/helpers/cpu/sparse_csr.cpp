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

#include <ops/declarable/helpers/sparse_csr.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

// ---------------------------------------------------------------------------
// csr_to_dense — CPU
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void csrToDense_(NDArray& values, NDArray& colIdx, NDArray& rowPtr, NDArray& output) {
  const auto rows = static_cast<I>(output.sizeAt(0));

  const auto* vBuf  = values.bufferAsT<X>();
  const auto* ciBuf = colIdx.bufferAsT<I>();
  const auto* rpBuf = rowPtr.bufferAsT<I>();
  auto*       oBuf  = output.bufferAsT<X>();

  // output is OUTPUT_NULLIFIED — already zeroed
  const auto* oStride = output.stridesOf();

  for (I r = 0; r < rows; ++r) {
    const I start = rpBuf[r];
    const I end   = rpBuf[r + 1];
    for (I k = start; k < end; ++k) {
      const I c = ciBuf[k];
      oBuf[r * oStride[0] + c * oStride[1]] = vBuf[k];
    }
  }
}

void csr_to_dense(NDArray& values, NDArray& colIdx, NDArray& rowPtr, NDArray& output) {
  NDArray::preparePrimaryUse({&output}, {&values, &colIdx, &rowPtr});

  BUILD_DOUBLE_SELECTOR(values.dataType(), colIdx.dataType(), csrToDense_,
                        (values, colIdx, rowPtr, output),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&output}, {&values, &colIdx, &rowPtr});
}

// ---------------------------------------------------------------------------
// dense_to_csr — CPU
// ---------------------------------------------------------------------------
template <typename X, typename I>
static void denseToCsr_(NDArray& input, NDArray& values, NDArray& colIdx, NDArray& rowPtr, double threshold) {
  const auto rows = static_cast<I>(input.sizeAt(0));
  const auto cols = static_cast<I>(input.sizeAt(1));
  const X    thr  = static_cast<X>(threshold);

  const auto* inStride = input.stridesOf();
  const auto* iBuf     = input.bufferAsT<X>();
  auto*       vBuf     = values.bufferAsT<X>();
  auto*       ciBuf    = colIdx.bufferAsT<I>();
  auto*       rpBuf    = rowPtr.bufferAsT<I>();

  rpBuf[0] = static_cast<I>(0);
  I nnzCount = 0;

  for (I r = 0; r < rows; ++r) {
    for (I c = 0; c < cols; ++c) {
      const X val = iBuf[r * inStride[0] + c * inStride[1]];
      const X absVal = val < static_cast<X>(0) ? -val : val;
      if (absVal > thr) {
        vBuf[nnzCount]  = val;
        ciBuf[nnzCount] = c;
        ++nnzCount;
      }
    }
    rpBuf[r + 1] = nnzCount;
  }
}

void dense_to_csr(NDArray& input, NDArray& values, NDArray& colIdx, NDArray& rowPtr, double threshold) {
  NDArray::preparePrimaryUse({&values, &colIdx, &rowPtr}, {&input});

  BUILD_DOUBLE_SELECTOR(input.dataType(), colIdx.dataType(), denseToCsr_,
                        (input, values, colIdx, rowPtr, threshold),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&values, &colIdx, &rowPtr}, {&input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
