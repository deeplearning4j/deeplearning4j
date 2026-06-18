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
#include <execution/Threads.h>
#include <ops/declarable/helpers/hamming.h>
#include <ops/declarable/helpers/helpers.h>
#include <system/selective_rendering.h>
namespace sd {
namespace ops {
namespace helpers {

static sd::LongType hamming_distance(unsigned long long x, unsigned long long y) {
  sd::LongType dist = 0;

  for (unsigned long long val = x ^ y; val > 0; val /= 2) {
    if (val & 1) dist++;
  }
  return dist;
}

template <typename X, typename Z>
static void _hamming(LaunchContext *context, NDArray &x, NDArray &y, NDArray &z) {
  auto xBuffer = x.bufferAsT<X>();
  auto yBuffer = y.bufferAsT<X>();

  sd::LongType distance = 0;
  auto lengthOf = x.lengthOf();
  int maxThreads = sd::math::sd_min(256, omp_get_max_threads());
  sd::LongType intermediate[256];

  // nullify temp values
  for (int e = 0; e < maxThreads; e++) intermediate[e] = 0;

  auto xRank = x.rankOf();
  auto yRank = y.rankOf();
  auto xShape = x.shapeOf();
  auto yShape = y.shapeOf();
  auto xStride = x.stridesOf();
  auto yStride = y.stridesOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto e = start; e < stop; e++) {
      sd::LongType coords[SD_MAX_RANK];
      INDEX2COORDS(e, xRank, xShape, coords);
      sd::LongType xOffset, yOffset;
      COORDS2INDEX(xRank, xStride, coords, xOffset);
      COORDS2INDEX(yRank, yStride, coords, yOffset);

      auto _x = static_cast<unsigned long long>(xBuffer[xOffset]);
      auto _y = static_cast<unsigned long long>(yBuffer[yOffset]);

      intermediate[thread_id] += hamming_distance(_x, _y);
    }
  };

  maxThreads = samediff::Threads::parallel_for(func, 0, lengthOf);

  // accumulate intermediate variables into output array
  for (int e = 0; e < maxThreads; e++) distance += intermediate[e];

  z.p(0, distance);
}

void hamming(LaunchContext *context, NDArray &x, NDArray &y, NDArray &output) {
  auto xDType = x.dataType();
  auto outputDType = output.dataType();
  BUILD_DOUBLE_SELECTOR(x.dataType(), output.dataType(), _hamming, (context, x, y, output), SD_INTEGER_TYPES, SD_INTEGER_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
