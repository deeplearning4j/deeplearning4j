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
#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_bincount)
#include <ops/declarable/helpers/weights.h>
#include <ops/op_types.h>

#include <type_traits>
#include <vector>

namespace sd {
namespace ops {
namespace helpers {

// output[v] += weights[i] (or 1) for each value v = input[i] in [0, output length); other values add nothing.
// input holds INT64 values and weights, when given, the output's type and the input's shape; every array is read
// through its own strides.
template <typename T>
static void adjustWeights_(NDArray* input, NDArray* weights, NDArray* output) {
  // Sums in a wider type than the bins hold: 64-bit integers, or FLOAT for HALF and BFLOAT16
  using AccT = typename std::conditional<std::is_integral<T>::value, LongType,
                                         typename simdOps::AggregateType<T>::type>::type;
  const LongType bins = output->lengthOf();
  const LongType length = input->lengthOf();
  if (bins == 0) return;

  NDArray::preparePrimaryUse({output}, {input, weights});
  const auto values = input->bufferAsT<LongType>();
  const T* w = weights != nullptr ? weights->bufferAsT<T>() : nullptr;
  auto out = output->bufferAsT<T>();
  const LongType rank = input->rankOf();
  const LongType* shape = input->shapeOf();
  const LongType* valueStrides = input->stridesOf();
  const LongType* weightStrides = weights != nullptr ? weights->stridesOf() : nullptr;

  std::vector<AccT> sums(bins, static_cast<AccT>(0));
  LongType coords[SD_MAX_RANK];
  for (LongType i = 0; i < length; i++) {
    INDEX2COORDS(i, rank, shape, coords);
    LongType valueOffset;
    COORDS2INDEX(rank, valueStrides, coords, valueOffset);
    const LongType v = values[valueOffset];
    if (v < 0 || v >= bins) continue;
    if (w != nullptr) {
      LongType weightOffset;
      COORDS2INDEX(rank, weightStrides, coords, weightOffset);
      sums[v] += static_cast<AccT>(w[weightOffset]);
    } else {
      sums[v] += static_cast<AccT>(1);
    }
  }

  const LongType outStride = output->strideAt(0);
  for (LongType b = 0; b < bins; b++) out[b * outStride] = static_cast<T>(sums[b]);
  NDArray::registerPrimaryUse({output}, {input, weights});
}

void adjustWeights(sd::LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
                   int maxLength) {
  BUILD_SINGLE_SELECTOR(output->dataType(), adjustWeights_, (input, weights, output), SD_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE(void adjustWeights_, (NDArray * input, NDArray* weights, NDArray* output), SD_NUMERIC_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
