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
#include <ops/declarable/helpers/histogram.h>
#include <system/Environment.h>
#include <system/selective_rendering.h>

#include <algorithm>
#include <vector>
#if NOT_EXCLUDED(OP_histogram)
namespace sd {
namespace ops {
namespace helpers {

// Each worker counts a contiguous share of the elements into a histogram of its own, so no two workers touch a bin;
// the histograms are added up afterwards. Workers are kept to this many counters in all, so a histogram with very many
// bins is counted by few workers (one, from this many bins on) instead of by a table of partial histograms.
static constexpr LongType HISTOGRAM_MAX_PARTIAL_COUNTERS = static_cast<LongType>(1) << 22;

// counts[b] += the number of elements of the input in bin b; counts is dense and holds numBins counters
template <typename X>
static void histogram_(NDArray &input, LongType numBins, double low, double high, LongType *counts) {
  const LongType length = input.lengthOf();
  const double binWidth = histogramBinWidth(low, high, numBins);

  const X *x = input.bufferAsT<X>();
  auto xShapeInfo = input.shapeInfo();
  const LongType rank = shape::rank(xShapeInfo);
  const LongType *xShape = shape::shapeOf(xShapeInfo);
  const LongType *xStride = shape::stride(xShapeInfo);
  // packed elements (C or F order, a view of them too) are the buffer's first length entries: the order a histogram
  // counts them in does not matter. Any other layout is walked through its strides.
  const bool packed = shape::strideDescendingCAscendingF(xShapeInfo);

  int workers = samediff::ThreadsHelper::numberOfThreads(sd::Environment::getInstance().maxMasterThreads(), length);
  workers = static_cast<int>(
      std::min<LongType>(workers, std::max<LongType>(1, HISTOGRAM_MAX_PARTIAL_COUNTERS / numBins)));

  // worker 0 counts into counts itself, the others into their rows of partial
  std::vector<LongType> partial(static_cast<size_t>(workers - 1) * static_cast<size_t>(numBins), 0);

  auto func = PRAGMA_THREADS_DO {
    LongType *mine = thread_id == 0 ? counts : partial.data() + (thread_id - 1) * numBins;
    const LongType from = length * thread_id / numThreads;
    const LongType to = length * (thread_id + 1) / numThreads;

    if (packed) {
      for (LongType i = from; i < to; i++) mine[histogramBin(static_cast<double>(x[i]), low, binWidth, numBins)]++;
    } else {
      LongType coords[SD_MAX_RANK];
      for (LongType i = from; i < to; i++) {
        LongType xOffset;
        INDEX2COORDS(i, rank, xShape, coords);
        COORDS2INDEX(rank, xStride, coords, xOffset);
        mine[histogramBin(static_cast<double>(x[xOffset]), low, binWidth, numBins)]++;
      }
    }
  };
  samediff::Threads::parallel_do(func, workers);

  for (int w = 1; w < workers; w++) {
    const LongType *row = partial.data() + static_cast<size_t>(w - 1) * static_cast<size_t>(numBins);
    for (LongType b = 0; b < numBins; b++) counts[b] += row[b];
  }
}

// The counts go to the output through assign: it casts them to the output's integer type and stores each through the
// output's own strides.
void histogramHelper(sd::LaunchContext *context, NDArray &input, NDArray &output) {
  const LongType numBins = output.lengthOf();
  if (numBins == 0) return;
  if (input.lengthOf() == 0) {
    output.nullify();
    return;
  }

  NDArray *min = input.reduceNumber(reduce::SameOps::Min);
  NDArray *max = input.reduceNumber(reduce::SameOps::Max);
  const double minValue = min->e<double>(0);
  const double maxValue = max->e<double>(0);
  delete min;
  delete max;

  // counts has the output's shape, so assign pairs the two element by element
  std::vector<LongType> countsShape(shape::shapeOf(output.shapeInfo()),
                                    shape::shapeOf(output.shapeInfo()) + shape::rank(output.shapeInfo()));
  NDArray counts('c', countsShape, DataType::INT64, context);
  counts.nullify();

  BUILD_SINGLE_SELECTOR(input.dataType(), histogram_,
                        (input, numBins, minValue, maxValue, counts.bufferAsT<LongType>()), SD_NUMERIC_TYPES);
  output.assign(&counts);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
