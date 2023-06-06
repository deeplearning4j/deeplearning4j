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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 17.05.2018
//
#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <ops/declarable/helpers/percentile.h>
#if NOT_EXCLUDED(OP_percentile)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void _percentile(const NDArray& input, NDArray& output, std::vector<LongType>& axises, const float q,
                        const int interpolation) {
  const int inputRank = input.rankOf();

  if (axises.empty())
    for (int i = 0; i < inputRank; ++i) axises.push_back(i);
  else
    shape::checkDimensions(inputRank, &axises);  // check, sort dimensions and remove duplicates if they are present

  auto listOfSubArrs = input.allTensorsAlongDimension(axises);

  std::vector<sd::LongType> shapeOfSubArr(listOfSubArrs.at(0)->rankOf());
  for (int i = 0; i < shapeOfSubArr.size(); ++i) shapeOfSubArr[i] = listOfSubArrs.at(0)->shapeOf()[i];

  auto flattenedArr = NDArrayFactory::create('c', shapeOfSubArr, input.dataType(), input.getContext());
  const int len = flattenedArr.lengthOf();

  const float fraction = 1.f - q / 100.;
  sd::LongType position = 0;

  switch (interpolation) {
    case 0:  // lower
      position = static_cast<sd::LongType>(math::sd_ceil<float, T>((len - 1) * fraction));
      break;
    case 1:  // higher
      position = static_cast<sd::LongType>(math::sd_floor<float, T>((len - 1) * fraction));
      break;
    case 2:  // nearest
      position = static_cast<sd::LongType>(math::sd_round<float, T>((len - 1) * fraction));
      break;
  }
  position = len - position - 1;

  // FIXME: our sort impl should be used instead, so this operation might be implemented as generic
  // FIXME: parallelism !
  for (int i = 0; i < listOfSubArrs.size(); ++i) {
    auto buff = reinterpret_cast<T*>(flattenedArr.buffer());
    flattenedArr.assign(listOfSubArrs.at(i));
    std::sort(buff, buff + len);
    output.p(i, flattenedArr.e<T>(position));
  }
}

void percentile(sd::LaunchContext* context, const NDArray& input, NDArray& output, std::vector<LongType>& axises,
                const float q, const int interpolation) {
  BUILD_SINGLE_SELECTOR(input.dataType(), _percentile, (input, output, axises, q, interpolation), SD_COMMON_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void _percentile,
                      (const NDArray& input, NDArray& output, std::vector<LongType>& axises, const float q,
                       const int interpolation),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif