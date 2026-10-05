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
// @author raver119@gmail.com
//

#ifndef LIBND4J_HISTOGRAM_H
#define LIBND4J_HISTOGRAM_H
#include <array/NDArray.h>
#include <system/common.h>

namespace sd {
namespace ops {
namespace helpers {
SD_LIB_HIDDEN void histogramHelper(LaunchContext *context, NDArray &input, NDArray &output);

// The histogram of the input has numBins bins of equal width between its smallest (low) and largest (high) value.
// The width is taken in DOUBLE whatever the input type: an integer input would truncate it (0..10 over 4 bins is 2.5
// wide, not 2).
SD_HOST_DEVICE SD_INLINE double histogramBinWidth(double low, double high, LongType numBins) {
  return (high - low) / static_cast<double>(numBins);
}

// The bin of a value: floor((value - low) / binWidth), kept within [0, numBins - 1] (the largest value closes the last
// bin). When every value is equal the width is 0 and everything falls in bin 0, as does NaN. CPU and CUDA both bin
// through this function, so they agree on every value.
SD_HOST_DEVICE SD_INLINE LongType histogramBin(double value, double low, double binWidth, LongType numBins) {
  if (!(binWidth > 0.0)) return 0;
  const double position = (value - low) / binWidth;
  if (!(position > 0.0)) return 0;
  if (position >= static_cast<double>(numBins - 1)) return numBins - 1;
  return static_cast<LongType>(position);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTS_HISTOGRAM_H
