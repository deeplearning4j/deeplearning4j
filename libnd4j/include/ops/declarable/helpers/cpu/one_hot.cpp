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
#include "../one_hot.h"

#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/TAD.h>
#if NOT_EXCLUDED(OP_onehot)
namespace sd {
namespace ops {
namespace helpers {
template <typename Z, typename I>
static void onehot_(void* voutput, sd::LongType const* zShapeInfo, void const* vindices, sd::LongType const* iShapeInfo,
                    int axis, double on, double off) {
  auto output = reinterpret_cast<Z*>(voutput);
  auto indices = reinterpret_cast<I const*>(vindices);

  // Get TAD pack once
  auto tadPack = sd::ConstantTadHelper::getInstance().tadForDimensions(zShapeInfo, {axis});

  // Cache TAD shape information
  const auto tadShapeInfo = tadPack->primaryShapeInfo();
  const int tadRank = shape::rank(tadShapeInfo);
  const sd::LongType* tadShape = shape::shapeOf(tadShapeInfo);
  const sd::LongType* tadStride = shape::stride(tadShapeInfo);

  // Cache lengths and counts
  const auto iLen = static_cast<sd::LongType>(shape::length(iShapeInfo));
  const auto tLen = static_cast<sd::LongType>(shape::length(tadShapeInfo));
  const auto numTads = static_cast<unsigned int>(tadPack->numberOfTads());

  // Get TAD offsets pointer
  const auto tadOffsets = tadPack->primaryOffsets();

  if (iLen != numTads)
    THROW_EXCEPTION("OneHot: number of TADs should be equal to number of indices");

  // Cache constant values
  const Z zero = static_cast<Z>(off);
  const Z one = static_cast<Z>(on);

  auto func = PRAGMA_THREADS_FOR {
    // Pre-allocate coords array outside the loop
    sd::LongType coords[SD_MAX_RANK];

    for (auto e = start; e < stop; e++) {
      auto cO = output + tadOffsets[e];
      auto idx = static_cast<sd::LongType>(indices[e]);

      if (idx < 0 || idx >= tLen) {
        // Fill with zeros using cached shape data
        PRAGMA_OMP_SIMD
        for (sd::LongType t = 0; t < tLen; t++) {
          INDEX2COORDS(t, tadRank, tadShape, coords);
          LongType offset;
          COORDS2INDEX(tadRank, tadStride, coords, offset);
          cO[offset] = zero;
        }
      } else {
        // One-hot encode using cached shape data
        PRAGMA_OMP_SIMD
        for (sd::LongType t = 0; t < tLen; t++) {
          INDEX2COORDS(t, tadRank, tadShape, coords);
          LongType offset;
          COORDS2INDEX(tadRank, tadStride, coords, offset);
          cO[offset] = idx == t ? one : zero;
        }
      }
    }
  };

  samediff::Threads::parallel_tad(func, 0, numTads);
}
void onehot(const sd::LaunchContext* context, NDArray* indices, NDArray* output, const sd::LongType axis,
            const sd::LongType depth, const double on, const double off) {
  auto zType = output->dataType();
  auto iType = indices->dataType();

  BUILD_DOUBLE_SELECTOR(zType, iType, helpers::onehot_,
                        (output->buffer(), output->shapeInfo(), indices->buffer(), indices->shapeInfo(), axis, on, off),
                        SD_COMMON_TYPES, SD_COMMON_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
}
#endif