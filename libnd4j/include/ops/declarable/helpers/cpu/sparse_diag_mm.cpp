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
// csr_diag_mm — CPU implementation.
//
// Outer loop over rows i; inner loop over entries e in row i's span.
// For each entry e at column j = aColIdx[e]:
//   outValues[e] = dl[i] * aValues[e] * dr[j]
//
// Complexity: O(nnz).  No allocations beyond the output buffer.
//

#include <ops/declarable/helpers/sparse_diag_mm.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename I>
static void csrDiagMm_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                        NDArray& dl,      NDArray& dr,
                        NDArray& outValues,
                        sd::LongType rows, sd::LongType /*cols*/) {
  const X* avBuf = aValues.bufferAsT<X>();
  const I* ciBuf = aColIdx.bufferAsT<I>();
  const I* rpBuf = aRowPtr.bufferAsT<I>();
  const X* dlBuf = dl.bufferAsT<X>();
  const X* drBuf = dr.bufferAsT<X>();
  X*       oBuf  = outValues.bufferAsT<X>();

  for (sd::LongType i = 0; i < rows; ++i) {
    const I eStart = rpBuf[i];
    const I eEnd   = rpBuf[i + 1];
    const X dli    = dlBuf[i];

    for (I e = eStart; e < eEnd; ++e) {
      const I j = ciBuf[e];
      oBuf[e] = dli * avBuf[e] * drBuf[j];
    }
  }
}

void csr_diag_mm(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                 NDArray& dl,      NDArray& dr,
                 NDArray& outValues,
                 sd::LongType rows, sd::LongType cols) {
  NDArray::preparePrimaryUse({&outValues},
                             {&aValues, &aColIdx, &aRowPtr, &dl, &dr});

  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrDiagMm_,
                        (aValues, aColIdx, aRowPtr, dl, dr, outValues, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&outValues},
                              {&aValues, &aColIdx, &aRowPtr, &dl, &dr});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
