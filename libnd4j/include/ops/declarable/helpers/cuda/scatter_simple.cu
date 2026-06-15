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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//

#include <array/NDArrayFactory.h>
#include <array/ResultSet.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

// Use the same host-side TAD logic as the CPU implementation.
// This is called only from reduce_min_bp / reduce_max_bp / reduce_norm_max_bp
// where the scatter size equals the number of TADs (typically small).
template <typename T>
static void scatterSimpleCopy(LaunchContext* context, NDArray& input, NDArray& updates,
                              NDArray& indices, const std::vector<LongType>& tadDimensions) {
  if (tadDimensions.empty()) {
    THROW_EXCEPTION("helpers::scatterSimple: TAD dimensions must not be empty for copy operation");
  }

  auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(
      input.shapeInfo(), const_cast<LongType*>(tadDimensions.data()), tadDimensions.size());
  auto* tadShapeInfo = tadPack->primaryShapeInfo();
  auto* tadOffsets = tadPack->primaryOffsets();
  const auto numTads = tadPack->numberOfTads();
  const auto tadLen = shape::length(tadShapeInfo);
  const auto tadRank = shape::rank(tadShapeInfo);
  const auto* tadShape = shape::shapeOf(tadShapeInfo);
  const auto* tadStride = shape::stride(tadShapeInfo);

  if (indices.lengthOf() != updates.lengthOf()) {
    THROW_EXCEPTION("helpers::scatterSimple: indices and updates must have the same length");
  }
  if (static_cast<LongType>(indices.lengthOf()) != numTads) {
    std::string msg = "scatterSimple CUDA: numTads=" + std::to_string(numTads) +
        " but indices.length=" + std::to_string(indices.lengthOf()) +
        ", input shape=" + ShapeUtils::shapeAsString(&input) +
        ", dims={";
    for (size_t d = 0; d < tadDimensions.size(); d++) {
      if (d > 0) msg += ",";
      msg += std::to_string(tadDimensions[d]);
    }
    msg += "}";
    THROW_EXCEPTION(msg.c_str());
  }

  // Sync to host for element access — same pattern as CPU implementation
  input.syncToHost();
  updates.syncToHost();
  indices.syncToHost();

  auto* inputBuffer = input.bufferAsT<T>();

  for (LongType t = 0; t < numTads; t++) {
    const auto localIndex = indices.t<LongType>(t);
    if (localIndex < 0 || localIndex >= tadLen) {
      std::string msg = "scatterSimple: index " + std::to_string(localIndex) +
          " out of bounds for TAD length " + std::to_string(tadLen);
      THROW_EXCEPTION(msg.c_str());
    }

    LongType coords[SD_MAX_RANK];
    LongType localOffset = 0;
    INDEX2COORDS(localIndex, tadRank, tadShape, coords);
    COORDS2INDEX(tadRank, tadStride, coords, localOffset);

    inputBuffer[tadOffsets[t] + localOffset] = updates.e<T>(t);
  }

  // Mark host as written so next device access triggers H2D sync
  input.tickWriteHost();
}

void scatterSimple(LaunchContext* context, const int opId, NDArray& input, NDArray& updates,
                   NDArray& indices, const std::vector<LongType>& dimensions) {
  if (opId != 6) THROW_EXCEPTION("scatterSimple: only copy op is supported");

  BUILD_SINGLE_SELECTOR(input.dataType(), scatterSimpleCopy,
                        (context, input, updates, indices, dimensions), SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
