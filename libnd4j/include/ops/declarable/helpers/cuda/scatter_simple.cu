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
#include <helpers/DebugHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>

#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////
// One thread per TAD: read indices[t], compute offset within the TAD
// using the TAD shape/stride, then scatter updates[t] into input.
template <typename T>
SD_KERNEL static void scatterSimpleCopyCuda(
    void* vInput,            // device buffer of input  (write target)
    const void* vUpdates,    // device buffer of updates (read source)
    const LongType* indicesBuffer,   // device buffer of indices (1-D, length = numTads)
    const LongType* tadShapeInfo,    // device TAD shape info (shape + stride embedded)
    const LongType* tadOffsets,      // device TAD offsets into input
    const LongType numTads,
    const LongType tadLen) {

  __shared__ LongType tadRank;
  __shared__ const LongType* tadShape;
  __shared__ const LongType* tadStride;

  if (threadIdx.x == 0) {
    tadRank   = shape::rank(tadShapeInfo);
    tadShape  = shape::shapeOf(tadShapeInfo);
    tadStride = shape::stride(tadShapeInfo);
  }
  __syncthreads();

  auto* inputBuffer   = reinterpret_cast<T*>(vInput);
  const auto* updatesBuffer = reinterpret_cast<const T*>(vUpdates);

  for (LongType t = blockIdx.x * blockDim.x + threadIdx.x; t < numTads; t += gridDim.x * blockDim.x) {
    const LongType localIndex = indicesBuffer[t];

    // Bounds check — skip invalid indices rather than hard-fault on device
    if (localIndex < 0 || localIndex >= tadLen) continue;

    // Replicate the host-side INDEX2COORDS / COORDS2INDEX math on device
    LongType coords[SD_MAX_RANK];
    INDEX2COORDS(localIndex, tadRank, tadShape, coords);

    LongType localOffset = 0;
    COORDS2INDEX(tadRank, tadStride, coords, localOffset);

    inputBuffer[tadOffsets[t] + localOffset] = updatesBuffer[t];
  }
}

///////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST static void scatterSimpleCopyCudaLauncher(
    const cudaStream_t* stream,
    void* vInput,
    const void* vUpdates,
    const LongType* indicesBuffer,
    const LongType* tadShapeInfo,
    const LongType* tadOffsets,
    const LongType numTads,
    const LongType tadLen) {

  dim3 launchDims = getLaunchDims("scatter_simple");
  scatterSimpleCopyCuda<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
      vInput, vUpdates, indicesBuffer, tadShapeInfo, tadOffsets, numTads, tadLen);
  sd::DebugHelper::checkErrorCode(const_cast<cudaStream_t*>(stream), "scatterSimpleCopy");
}

///////////////////////////////////////////////////////////////////
template <typename T>
static void scatterSimpleCopy(LaunchContext* context, NDArray& input, NDArray& updates,
                              NDArray& indices, const std::vector<LongType>& tadDimensions) {
  if (tadDimensions.empty()) {
    THROW_EXCEPTION("helpers::scatterSimple: TAD dimensions must not be empty for copy operation");
  }

  auto tadPack = ConstantTadHelper::getInstance().tadForDimensions(
      input.shapeInfo(), const_cast<LongType*>(tadDimensions.data()), tadDimensions.size());

  const LongType numTads = tadPack->numberOfTads();
  const LongType tadLen  = shape::length(tadPack->primaryShapeInfo());

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

  // Ensure device buffers of updates and indices are up-to-date;
  // input is write-only so it goes in the write list.
  NDArray::prepareSpecialUse({&input}, {&updates, &indices});

  scatterSimpleCopyCudaLauncher<T>(
      context->getCudaStream(),
      input.specialBuffer(),
      updates.specialBuffer(),
      reinterpret_cast<const LongType*>(indices.specialBuffer()),
      tadPack->platformShapeInfo(),
      tadPack->platformOffsets(),
      numTads,
      tadLen);

  NDArray::registerSpecialUse({&input}, {&updates, &indices});
}

///////////////////////////////////////////////////////////////////
void scatterSimple(LaunchContext* context, const int opId, NDArray& input, NDArray& updates,
                   NDArray& indices, const std::vector<LongType>& dimensions) {
  if (opId != 6) THROW_EXCEPTION("scatterSimple: only copy op is supported");

  BUILD_SINGLE_SELECTOR(input.dataType(), scatterSimpleCopy,
                        (context, input, updates, indices, dimensions), SD_COMMON_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
