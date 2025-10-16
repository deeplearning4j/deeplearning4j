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
// @author Yurii Shyrma, created on 14.02.2018
//

#ifndef LIBND4J_LSTM_H
#define LIBND4J_LSTM_H
#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray sigmoid(NDArray& arr) {
  NDArray* result = (const_cast<NDArray&>(arr)).transform(transform::Sigmoid);
  NDArray copy = *result;
  delete result;
  return copy;
}

static SD_INLINE void sigmoidInplace(NDArray& arr) {
  arr.applyTransform(transform::Sigmoid, &arr);
}

//////////////////////////////////////////////////////////////////////////
static SD_INLINE NDArray tanh(NDArray& arr) {
  NDArray* result = (const_cast<NDArray&>(arr)).transform(transform::Tanh);
  NDArray copy = *result;
  delete result;
  return copy;
}

static SD_INLINE void tanhInplace(NDArray& arr) {
  arr.applyTransform(transform::Tanh, &arr);
}

//////////////////////////////////////////////////////////////////////////
static NDArray* timeSubset(NDArray* arr, const int t, const int dataFormat) {
  if (dataFormat == 0) {  // TNS: shape [timeLength, numExamples, inOutSize]
    return (*arr)({t, t + 1, 0, 0, 0, 0});
  } else if (dataFormat == 1) {  // NST: shape [numExamples, inOutSize, timeLength]
    return (*arr)({0, 0, 0, 0, t, t + 1});
  } else {  // NTS: shape [numExamples, timeLength, inOutSize] - TF "time_major=false" layout
    return (*arr)({0, 0, t, t + 1, 0, 0});
  }
}

SD_LIB_HIDDEN void lstmCell(LaunchContext* context, NDArray* xt, NDArray* ht_1, NDArray* ct_1,
                            NDArray* Wx, NDArray* Wh, NDArray* Wc, NDArray* Wp,
                            NDArray* b, NDArray* ht, NDArray* ct, const std::vector<double>& params);

SD_LIB_HIDDEN void lstmTimeLoop(LaunchContext* context, NDArray* x, NDArray* h0, NDArray* c0,
                                NDArray* Wx, NDArray* Wh, NDArray* Wc, NDArray* Wp,
                                NDArray* b, NDArray* h, NDArray* c, const std::vector<double>& params);

SD_LIB_HIDDEN void lstmBlockCell(NDArray* xt, NDArray* cLast, NDArray* yLast, NDArray* W,
                                 NDArray* Wci, NDArray* Wcf, NDArray* Wco, NDArray* b,
                                 NDArray* i, NDArray* c, NDArray* f, NDArray* o, NDArray* z, NDArray* h, NDArray* y,
                                 const std::vector<double>& params);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_LSTM_H
