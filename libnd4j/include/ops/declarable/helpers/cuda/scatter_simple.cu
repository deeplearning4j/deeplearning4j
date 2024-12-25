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
#include <exceptions/cuda_exception.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <helpers/TAD.h>
#include <ops/declarable/helpers/transforms.h>

#include <numeric>

#include "execution/cuda/LaunchDims.h"


namespace sd {
namespace ops {
namespace helpers {
template <typename X, typename Y>
static SD_KERNEL void scatterSimpleKernel(void* vx, const LongType* xTadShape, const LongType* xTadOffsets,
                                          LongType xLength, LongType numTads, const void* vi,
                                          const LongType* iShapeInfo, LongType iLength, const void* vu,
                                          const LongType* uShapeInfo, LongType uLength) {
  // Shared memory caching for shape and stride information
  __shared__ LongType iRank, xTadRank, uRank;
  __shared__ const LongType* iShape;
  __shared__ const LongType* xTadShapePtr;
  __shared__ const LongType* uShape;
  __shared__ const LongType* iStride;
  __shared__ const LongType* xTadStride;
  __shared__ const LongType* uStride;

  // Initialize shared memory
  if (threadIdx.x == 0) {
    iRank = shape::rank(iShapeInfo);
    xTadRank = shape::rank(xTadShape);
    uRank = shape::rank(uShapeInfo);

    iShape = shape::shapeOf(iShapeInfo);
    xTadShapePtr = shape::shapeOf(xTadShape);
    uShape = shape::shapeOf(uShapeInfo);

    iStride = shape::stride(iShapeInfo);
    xTadStride = shape::stride(xTadShape);
    uStride = shape::stride(uShapeInfo);
  }
  __syncthreads();

  // Cast input pointers
  const X* u = reinterpret_cast<const X*>(vu);
  const Y* indices = reinterpret_cast<const Y*>(vi);

  // Calculate thread ID
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  // Iterate over the indices
  for (int i = tid; i < iLength; i += blockDim.x * gridDim.x) {
    // Offset for `x`
    auto x = reinterpret_cast<X*>(vx) + xTadOffsets[i];

    // Compute coordinates and offsets for index tensor
    LongType idxCoords[SD_MAX_RANK];
    LongType idxOffset;
    INDEX2COORDS(i, iRank, iShape, idxCoords);
    COORDS2INDEX(iRank, iStride, idxCoords, idxOffset);
    auto idx = indices[idxOffset];

    // Compute coordinates and offsets for x
    LongType xCoords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(idx, xTadRank, xTadShapePtr, xCoords);
    COORDS2INDEX(xTadRank, xTadStride, xCoords, xOffset);

    // Compute coordinates and offsets for u
    LongType uCoords[SD_MAX_RANK];
    LongType uOffset;
    INDEX2COORDS(i, uRank, uShape, uCoords);
    COORDS2INDEX(uRank, uStride, uCoords, uOffset);

    // Perform the scatter update
    x[xOffset] = u[uOffset];
  }
}


template <typename X, typename Y>
void scatterSimple_(LaunchContext* context, const int opId, NDArray& input, NDArray& updates,
                    NDArray& indices, const std::vector<LongType>& dimensions) {
  auto dims = ShapeUtils::evalDimsToExclude(input.rankOf(),dimensions.size(),dimensions.data());
  auto packX = ConstantTadHelper::getInstance().tadForDimensions(input.shapeInfo(), dims);

  auto xLength = shape::length(packX->primaryShapeInfo());
  auto iLength = indices.lengthOf();
  auto uLength = updates.lengthOf();

  dim3 launchDims = getLaunchDims("scatter_simple");
  scatterSimpleKernel<X, Y><<<launchDims.y, launchDims.x, launchDims.z, *context->getCudaStream()>>>(
      input.specialBuffer(), packX->platformShapeInfo(), packX->platformOffsets(), xLength, packX->numberOfTads(),
      indices.specialBuffer(), indices.specialShapeInfo(), iLength, updates.specialBuffer(), updates.specialShapeInfo(),
      uLength);
  sd::DebugHelper::checkErrorCode(context->getCudaStream(), "scatterUpdateCuda failed");


}

void scatterSimple(LaunchContext* context, const int opId, NDArray& input, NDArray& updates,
                   NDArray& indices, const std::vector<LongType>& dimensions) {
  auto xType = input.dataType();
  auto yType = indices.dataType();

  if (opId != 6) THROW_EXCEPTION("scatterSimple: only copy op is supported");

  NDArray::prepareSpecialUse({&input}, {&updates, &indices});

  BUILD_DOUBLE_SELECTOR(xType, yType, scatterSimple_, (context, opId, input, updates, indices, dimensions),
                        SD_COMMON_TYPES, SD_INDEXING_TYPES);

  NDArray::registerSpecialUse({&input}, {&updates, &indices});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
