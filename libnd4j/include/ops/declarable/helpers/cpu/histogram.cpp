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
#include <ops/declarable/helpers/histogram.h>
#include <system/selective_rendering.h>
#if NOT_EXCLUDED(OP_histogram)
namespace sd {
namespace ops {
namespace helpers {
template <typename X, typename Z>
static void histogram_(void const *xBuffer, sd::LongType const *xShapeInfo, void *zBuffer,
                       sd::LongType const *zShapeInfo, sd::LongType numBins, double min_val, double max_val) {
  auto dx = reinterpret_cast<X const *>(xBuffer);
  auto result = reinterpret_cast<Z *>(zBuffer);

  int length = shape::length(xShapeInfo);

  X binSize = static_cast<X>((max_val - min_val) / (numBins));

  // FIXME: this op should be parallelized
  {
    int *bins = new int[numBins];
    std::memset(bins, 0, sizeof(int) * numBins);

    PRAGMA_OMP_SIMD
    for (int x = 0; x < length; x++) {
      int idx = (int)((dx[x] - min_val) / binSize);
      if (idx < 0)
        idx = 0;
      else if (idx >= numBins)
        idx = numBins - 1;

      bins[idx]++;
    }

    PRAGMA_OMP_SIMD
    for (sd::LongType x = 0; x < numBins; x++) {
      result[x] += bins[x];
    }

    delete[] bins;
  }
}

void histogramHelper(sd::LaunchContext *context, NDArray &input, NDArray &output) {
  sd::LongType numBins = output.lengthOf();
  double min_val = input.reduceNumber(reduce::SameOps::Min).e<double>(0);
  double max_val = input.reduceNumber(reduce::SameOps::Max).e<double>(0);
#if SD_IS_PAIR_TYPE_COMPILED(input.DataType(),output.dataType())
  BUILD_DOUBLE_SELECTOR(
      input.dataType(), output.dataType(), histogram_,
      (input.buffer(), input.shapeInfo(), output.buffer(), output.shapeInfo(), numBins, min_val, max_val),
      SD_COMMON_TYPES, SD_INDEXING_TYPES);
#endif
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif