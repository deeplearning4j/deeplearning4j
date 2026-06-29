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
// spdiags — CPU implementation.
//
// Trivial diagonal CSR construction: values[i]=diag[i], colIdx[i]=i, rowPtr[i]=i.
// No complex logic, no data-dependent shapes.
//

#include <ops/declarable/helpers/sparse_spdiags.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename X>
static void spdiags_(NDArray& diag, NDArray& values, NDArray& colIdx, NDArray& rowPtr,
                     sd::LongType n) {
  const X* dBuf  = diag.bufferAsT<X>();
  X*       vBuf  = values.bufferAsT<X>();
  auto*    ciBuf = colIdx.bufferAsT<int32_t>();
  auto*    rpBuf = rowPtr.bufferAsT<int32_t>();

  for (sd::LongType i = 0; i < n; ++i) {
    vBuf[i]  = dBuf[i];
    ciBuf[i] = static_cast<int32_t>(i);
    rpBuf[i] = static_cast<int32_t>(i);
  }
  rpBuf[n] = static_cast<int32_t>(n);
}

void spdiags(sd::LaunchContext* context, NDArray& diag, NDArray& values,
             NDArray& colIdx, NDArray& rowPtr, sd::LongType n) {
  NDArray::preparePrimaryUse({&values, &colIdx, &rowPtr}, {&diag});

  BUILD_SINGLE_SELECTOR(diag.dataType(), spdiags_,
                        (diag, values, colIdx, rowPtr, n),
                        SD_FLOAT_TYPES);

  NDArray::registerPrimaryUse({&values, &colIdx, &rowPtr}, {&diag});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
