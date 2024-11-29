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
  auto u = reinterpret_cast<const X*>(vu);
  auto indices = reinterpret_cast<const Y*>(vi);

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < iLength; i += blockDim.x * gridDim.x) {
    auto x = reinterpret_cast<X*>(vx) + xTadOffsets[i];
    LongType idxCoords[SD_MAX_RANK];
    LongType idxOffset;
    INDEX2COORDS(i, shape::rank(iShapeInfo), iShapeInfo, idxCoords);
    COORDS2INDEX(shape::rank(iShapeInfo), shape::shapeOf(iShapeInfo), idxCoords, idxOffset);
    auto idx = indices[idxOffset];

    LongType xCoords[SD_MAX_RANK];
    LongType xOffset;
    INDEX2COORDS(idx, shape::rank(xTadShape), xTadShape, xCoords);
    COORDS2INDEX(shape::rank(xTadShape), shape::shapeOf(xTadShape), xCoords, xOffset);

    LongType uCoords[SD_MAX_RANK];
    LongType uOffset;
    INDEX2COORDS(i, shape::rank(uShapeInfo), uShapeInfo, uCoords);
    COORDS2INDEX(shape::rank(uShapeInfo), shape::shapeOf(uShapeInfo), uCoords, uOffset);

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
