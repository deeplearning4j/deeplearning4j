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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.08.2018
//
#include <ops/declarable/helpers/histogramFixedWidth.h>
#if NOT_EXCLUDED(OP_histogram_fixed_width)
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
void histogramFixedWidth_(NDArray& input, NDArray& range, NDArray& output) {
  const int nbins = output.lengthOf();

  // firstly initialize output with zeros
  output.nullify();

  const T leftEdge = static_cast<T>(range.e<double>(0));
  const T rightEdge = static_cast<T>(range.e<double>(1));

  const T binWidth = (rightEdge - leftEdge) / nbins;
  const T secondEdge = leftEdge + binWidth;
  const T lastButOneEdge = rightEdge - binWidth;

  sd::LongType inputLength = input.lengthOf();

  // FIXME: make this one parallel without CRITICAL section
  for (sd::LongType i = 0; i < inputLength; ++i) {
    const T value = input.e<T>(i);

    if (value < secondEdge) {
      output.p<sd::LongType>(0, output.e<sd::LongType>(0) + 1);
    } else if (value >= lastButOneEdge) {
      output.p<sd::LongType>(nbins - 1, output.e<sd::LongType>(nbins - 1) + 1);
    } else {
      sd::LongType currInd = static_cast<sd::LongType>((value - leftEdge) / binWidth);
      output.p<sd::LongType>(currInd, output.e<sd::LongType>(currInd) + 1);
    }
  }
}

void histogramFixedWidth(sd::LaunchContext* context, NDArray& input, NDArray& range, NDArray& output) {
  BUILD_SINGLE_SELECTOR(input.dataType(), histogramFixedWidth_, (input, range, output), SD_COMMON_TYPES);
}
BUILD_SINGLE_TEMPLATE( void histogramFixedWidth_, (NDArray& input, NDArray& range, NDArray& output),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif