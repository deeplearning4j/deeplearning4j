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
//  @author GS <sgazeos@gmail.com>
//
#include <execution/Threads.h>
#include <ops/declarable/helpers/confusion.h>
#if NOT_EXCLUDED(OP_confusion_matrix)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void _confusionFunctor(NDArray* labels, NDArray* predictions, NDArray* weights, NDArray* output) {
  // Initialize output to zero
  output->nullify();

  int lLen = labels->lengthOf();
  // Pre-fetch labels and predictions (may be INT32 or INT64)
  std::vector<sd::LongType> labelsVec(lLen), predsVec(lLen);
  for (sd::LongType j = 0; j < lLen; j++) {
    labelsVec[j] = labels->e<sd::LongType>(j);
    predsVec[j] = predictions->e<sd::LongType>(j);
  }
  auto outputBuf = output->bufferAsT<T>();
  auto weightsBuf = (weights != nullptr) ? weights->bufferAsT<T>() : nullptr;
  // Get strides for offset calculation
  auto stride0 = output->strideAt(0);
  auto stride1 = output->strideAt(1);

  // Sequential loop to avoid race conditions when updating same cell
  for (sd::LongType j = 0; j < lLen; j++) {
    auto label = labelsVec[j];
    auto pred = predsVec[j];
    T value = (weightsBuf == nullptr ? (T)1.0f : weightsBuf[j]);
    auto offset = label * stride0 + pred * stride1;
    outputBuf[offset] += value;
  }
  output->tickWriteHost();
  output->syncToDevice();
}

void confusionFunctor(sd::LaunchContext* context, NDArray* labels, NDArray* predictions, NDArray* weights,
                      NDArray* output) {
  auto xType = output->dataType();  // weights can be null

  BUILD_SINGLE_SELECTOR(xType, _confusionFunctor, (labels, predictions, weights, output), SD_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE( void _confusionFunctor,
                      (NDArray * labels, NDArray* predictions, NDArray* weights, NDArray* output);
                      , SD_NUMERIC_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif