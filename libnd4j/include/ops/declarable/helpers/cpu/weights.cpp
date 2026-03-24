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
//  @author sgazeos@gmail.com
//
#include <ops/declarable/helpers/weights.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void adjustWeights_(NDArray* input, NDArray* weights, NDArray* output, int minLength, int maxLength) {
  // Pre-fetch input values as int (input may be INT32 or INT64)
  std::vector<int> inputVals(input->lengthOf());
  for (sd::LongType i = 0; i < input->lengthOf(); i++) inputVals[i] = input->e<int>(i);
  auto outputBuf = output->bufferAsT<T>();
  auto weightsBuf = (weights != nullptr) ? weights->bufferAsT<T>() : nullptr;

  for (sd::LongType e = 0; e < input->lengthOf(); e++) {
    int val = inputVals[e];
    if (val < maxLength) {
      if (weightsBuf != nullptr)
        outputBuf[val] += weightsBuf[e];
      else
        outputBuf[val] += static_cast<T>(1);
    }
  }
  output->tickWriteHost();
  output->syncToDevice();
}

void adjustWeights(sd::LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
                   int maxLength) {
  BUILD_SINGLE_SELECTOR(output->dataType(), adjustWeights_, (input, weights, output, minLength, maxLength),
                        SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE( void adjustWeights_,
                      (NDArray * input, NDArray* weights, NDArray* output, int minLength, int maxLength),
                      SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
