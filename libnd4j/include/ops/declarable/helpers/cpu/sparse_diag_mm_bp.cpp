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
// csr_diag_mm_bp — CPU implementation.
//
// Outer loop over rows i; inner loop over entries e in [aRowPtr[i], aRowPtr[i+1]).
// For each entry e at column j = aColIdx[e]:
//   t          = gradOut[e] * aValues[e]
//   dAValues[e] = gradOut[e] * dl[i] * dr[j]
//   ddl[i]     += t * dr[j]
//   ddr[j]     += t * dl[i]
//
// ddl and ddr are zero-initialised before the loop.
// Single-threaded per row: ddl[i] accumulation within a row is clean (no race).
// ddr[j] can be updated by multiple rows but CPU is sequential here.
//

#include <ops/declarable/helpers/sparse_diag_mm_bp.h>
#include <system/op_boilerplate.h>

#include <cstring>  // memset

namespace sd {
namespace ops {
namespace helpers {

template <typename X, typename I>
static void csrDiagMmBp_(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                          NDArray& dl,      NDArray& dr,      NDArray& gradOut,
                          NDArray& dAValues, NDArray& ddl,    NDArray& ddr,
                          sd::LongType rows, sd::LongType /*cols*/) {
  const X* avBuf  = aValues.bufferAsT<X>();
  const I* ciBuf  = aColIdx.bufferAsT<I>();
  const I* rpBuf  = aRowPtr.bufferAsT<I>();
  const X* dlBuf  = dl.bufferAsT<X>();
  const X* drBuf  = dr.bufferAsT<X>();
  const X* goBuf  = gradOut.bufferAsT<X>();
  X*       daBuf  = dAValues.bufferAsT<X>();
  X*       ddlBuf = ddl.bufferAsT<X>();
  X*       ddrBuf = ddr.bufferAsT<X>();

  // Zero-initialise the scatter-accumulation outputs
  std::memset(ddlBuf, 0, sizeof(X) * static_cast<size_t>(ddl.lengthOf()));
  std::memset(ddrBuf, 0, sizeof(X) * static_cast<size_t>(ddr.lengthOf()));

  for (sd::LongType i = 0; i < rows; ++i) {
    const I eStart = rpBuf[i];
    const I eEnd   = rpBuf[i + 1];
    const X dli    = dlBuf[i];

    for (I e = eStart; e < eEnd; ++e) {
      const I j = ciBuf[e];
      const X t = goBuf[e] * avBuf[e];  // gradOut[e] * aValues[e]

      daBuf[e]   = goBuf[e] * dli * drBuf[j];  // exact: gradOut * dl[i] * dr[j]
      ddlBuf[i] += t * drBuf[j];               // accumulate within row i
      ddrBuf[j] += t * dli;                    // scatter to column j
    }
  }
}

void csr_diag_mm_bp(NDArray& aValues, NDArray& aColIdx, NDArray& aRowPtr,
                    NDArray& dl,      NDArray& dr,      NDArray& gradOut,
                    NDArray& dAValues, NDArray& ddl,    NDArray& ddr,
                    sd::LongType rows, sd::LongType cols) {
  NDArray::preparePrimaryUse({&dAValues, &ddl, &ddr},
                             {&aValues, &aColIdx, &aRowPtr, &dl, &dr, &gradOut});

  BUILD_DOUBLE_SELECTOR(aValues.dataType(), aColIdx.dataType(), csrDiagMmBp_,
                        (aValues, aColIdx, aRowPtr, dl, dr, gradOut,
                         dAValues, ddl, ddr, rows, cols),
                        SD_FLOAT_TYPES, SD_INDEXING_TYPES);

  NDArray::registerPrimaryUse({&dAValues, &ddl, &ddr},
                              {&aValues, &aColIdx, &aRowPtr, &dl, &dr, &gradOut});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
