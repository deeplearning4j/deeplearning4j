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
//  @author raver119@gmail.com
//
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/flatten.h>

#include "execution/cuda/LaunchDims.h"
#include "helpers/DebugHelper.h"


namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static void SD_KERNEL flattenKernel(void **xBuffers, LongType **xShapeInfos, LongType *offsets, LongType numInputs, void *zBuffer, const LongType *zShapeInfo, char order) {
  int xCoord[SD_MAX_RANK];

  // each block of threads works on 1 input array
  for (LongType e = blockIdx.x; e < numInputs; e += gridDim.x) {
    auto z = reinterpret_cast<T *>(zBuffer) + offsets[e];

    auto xBuffer = reinterpret_cast<T *>(xBuffers[e]);
    auto xShapeInfo = xShapeInfos[e];
    auto xLength = shape::length(xShapeInfo);

    // each element of this input array has own place within common output array
    for (LongType i = threadIdx.x; i < xLength; i += blockDim.x) {
      LongType xOffset;
      sd::LongType xCoords[SD_MAX_RANK];
      INDEX2COORDS(i, shape::rank(xShapeInfo), shape::shapeOf(xShapeInfo), xCoords);
      COORDS2INDEX(shape::rank(xShapeInfo), shape::stride(xShapeInfo), xCoords, xOffset);
      z[i] = xBuffer[xOffset];
    }
  }
}

template <typename T>
static void flatten_(LaunchContext *context, std::vector<NDArray *> &inputs, NDArray *output, char order) {
  PointersManager pm(context, "flatten");

  std::vector<const void *> hdBuffers(inputs.size());
  std::vector<LongType> hOffsets(inputs.size());
  std::vector<const LongType *> hdShapes(inputs.size());
  LongType cOffset = 0;

  // calculating offsets in output
  for (int e = 0; e < inputs.size(); e++) {
    hOffsets[e] = cOffset;
    cOffset += inputs[e]->lengthOf();

    hdBuffers[e] = inputs[e]->specialBuffer();
    hdShapes[e] = inputs[e]->specialShapeInfo();
  }

  // copying pointers to device
  auto dBuffers = (void **)pm.replicatePointer(hdBuffers.data(), inputs.size() * sizeof(void *));
  auto dShapes = (LongType **)pm.replicatePointer(hdShapes.data(), inputs.size() * sizeof(LongType *));
  auto dOffsets = (LongType *)pm.replicatePointer(hOffsets.data(), inputs.size() * sizeof(LongType));
  dim3 launchDims = getLaunchDims("flatten");
  flattenKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
      dBuffers, dShapes, dOffsets, inputs.size(), output->specialBuffer(), output->specialShapeInfo(), order);
  DebugHelper::checkErrorCode(context->getCudaStream(),"flattenKernel failed");

  pm.synchronize();
}

void flatten(LaunchContext *context, std::vector<NDArray *> &inputs, NDArray *output, char order) {
  // FIXME: we want NDArrayFactory::prepareSpecialUse here eventually
  const std::vector<NDArray *> v(inputs.begin(), inputs.end());
  //prepareSpecialUse requires const
  NDArray::prepareSpecialUse({output}, v, {});
  BUILD_SINGLE_SELECTOR(output->dataType(), flatten_, (context, inputs, output, order), SD_COMMON_TYPES);
  NDArray::registerSpecialUse({output}, {});
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
