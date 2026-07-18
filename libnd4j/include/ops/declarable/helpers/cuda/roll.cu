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
#include <ops/declarable/helpers/roll.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {

static LongType normalizedShift(LongType shift, LongType dimension) {
  if (dimension <= 1) return 0;
  shift %= dimension;
  return shift < 0 ? shift + dimension : shift;
}

template <typename T>
static SD_KERNEL void rollLinearKernel(const void* inputBuffer, const LongType* inputShapeInfo, void* outputBuffer,
                                       const LongType* outputShapeInfo, LongType length, LongType shift) {
  const auto input = reinterpret_cast<const T*>(inputBuffer);
  auto output = reinterpret_cast<T*>(outputBuffer);
  const auto inputRank = shape::rank(inputShapeInfo);
  const auto outputRank = shape::rank(outputShapeInfo);
  const auto inputShape = shape::shapeOf(inputShapeInfo);
  const auto outputShape = shape::shapeOf(outputShapeInfo);
  const auto inputStrides = shape::stride(inputShapeInfo);
  const auto outputStrides = shape::stride(outputShapeInfo);

  for (LongType outputIndex = blockIdx.x * blockDim.x + threadIdx.x; outputIndex < length;
       outputIndex += blockDim.x * gridDim.x) {
    const auto inputIndex = (outputIndex + length - shift) % length;
    LongType inputCoords[SD_MAX_RANK];
    LongType outputCoords[SD_MAX_RANK];
    LongType inputOffset;
    LongType outputOffset;
    INDEX2COORDS(inputIndex, inputRank, inputShape, inputCoords);
    INDEX2COORDS(outputIndex, outputRank, outputShape, outputCoords);
    COORDS2INDEX(inputRank, inputStrides, inputCoords, inputOffset);
    COORDS2INDEX(outputRank, outputStrides, outputCoords, outputOffset);
    output[outputOffset] = input[inputOffset];
  }
}

template <typename T>
static SD_KERNEL void rollAxesKernel(const void* inputBuffer, const LongType* inputShapeInfo, void* outputBuffer,
                                     const LongType* outputShapeInfo, LongType length, const LongType* shifts,
                                     const LongType* axes, LongType numAxes) {
  const auto input = reinterpret_cast<const T*>(inputBuffer);
  auto output = reinterpret_cast<T*>(outputBuffer);
  const auto rank = shape::rank(outputShapeInfo);
  const auto inputStrides = shape::stride(inputShapeInfo);
  const auto outputShape = shape::shapeOf(outputShapeInfo);
  const auto outputStrides = shape::stride(outputShapeInfo);

  for (LongType outputIndex = blockIdx.x * blockDim.x + threadIdx.x; outputIndex < length;
       outputIndex += blockDim.x * gridDim.x) {
    LongType sourceCoords[SD_MAX_RANK];
    LongType outputCoords[SD_MAX_RANK];
    INDEX2COORDS(outputIndex, rank, outputShape, outputCoords);
    for (LongType dimension = 0; dimension < rank; ++dimension) sourceCoords[dimension] = outputCoords[dimension];

    for (LongType i = 0; i < numAxes; ++i) {
      const auto axis = axes[i];
      const auto dimension = outputShape[axis];
      auto shift = shifts[i] % dimension;
      if (shift < 0) shift += dimension;
      sourceCoords[axis] = (sourceCoords[axis] + dimension - shift) % dimension;
    }

    LongType inputOffset;
    LongType outputOffset;
    COORDS2INDEX(rank, inputStrides, sourceCoords, inputOffset);
    COORDS2INDEX(rank, outputStrides, outputCoords, outputOffset);
    output[outputOffset] = input[inputOffset];
  }
}

template <typename T>
static void launchRollLinear(LaunchContext* context, NDArray* input, NDArray* output, LongType shift) {
  const auto launchDims = getLaunchDims("roll");
  rollLinearKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
      input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
      input->lengthOf(), shift);
  DebugHelper::checkErrorCode(context->getCudaStream(), "rollLinearKernel failed");
}

template <typename T>
static void launchRollAxes(LaunchContext* context, NDArray* input, NDArray* output, const LongType* shifts,
                           const LongType* axes, LongType numAxes) {
  const auto launchDims = getLaunchDims("roll");
  rollAxesKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
      input->specialBuffer(), input->specialShapeInfo(), output->specialBuffer(), output->specialShapeInfo(),
      input->lengthOf(), shifts, axes, numAxes);
  DebugHelper::checkErrorCode(context->getCudaStream(), "rollAxesKernel failed");
}

void rollFunctorLinear(LaunchContext* context, NDArray* input, NDArray* output, LongType shift, bool inplace) {
  const auto length = input->lengthOf();
  if (length <= 1) {
    if (!inplace) output->assign(input);
    return;
  }

  const auto actualShift = normalizedShift(shift, length);
  if (actualShift == 0) {
    if (!inplace) output->assign(input);
    return;
  }

  NDArray* snapshot = inplace ? input->dup() : nullptr;
  NDArray* source = snapshot == nullptr ? input : snapshot;
  PointersManager manager(context, "roll");
  NDArray::prepareSpecialUse({output}, {source});

  BUILD_SINGLE_SELECTOR(input->dataType(), launchRollLinear, (context, source, output, actualShift), SD_COMMON_TYPES);

  NDArray::registerSpecialUse({output}, {source});
  manager.synchronize();
  delete snapshot;
}

void rollFunctorFull(LaunchContext* context, NDArray* input, NDArray* output, const std::vector<LongType>& shifts,
                     const std::vector<LongType>& axes, bool inplace) {
  const auto length = input->lengthOf();
  if (length <= 1 || axes.empty()) {
    if (!inplace) output->assign(input);
    return;
  }

  bool noOp = true;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (normalizedShift(shifts[i], input->sizeAt(axes[i])) != 0) {
      noOp = false;
      break;
    }
  }
  if (noOp) {
    if (!inplace) output->assign(input);
    return;
  }

  NDArray* snapshot = inplace ? input->dup() : nullptr;
  NDArray* source = snapshot == nullptr ? input : snapshot;
  PointersManager manager(context, "roll");
  auto deviceShifts = reinterpret_cast<LongType*>(
      manager.replicatePointer(shifts.data(), static_cast<LongType>(shifts.size() * sizeof(LongType))));
  auto deviceAxes = reinterpret_cast<LongType*>(
      manager.replicatePointer(axes.data(), static_cast<LongType>(axes.size() * sizeof(LongType))));

  NDArray::prepareSpecialUse({output}, {source});
  BUILD_SINGLE_SELECTOR(input->dataType(), launchRollAxes,
                        (context, source, output, deviceShifts, deviceAxes, static_cast<LongType>(axes.size())),
                        SD_COMMON_TYPES);
  NDArray::registerSpecialUse({output}, {source});

  manager.synchronize();
  delete snapshot;
}

BUILD_SINGLE_TEMPLATE(void launchRollLinear, (LaunchContext* context, NDArray* input, NDArray* output, LongType shift),
                      SD_COMMON_TYPES);
BUILD_SINGLE_TEMPLATE(void launchRollAxes,
                      (LaunchContext* context, NDArray* input, NDArray* output, const LongType* shifts,
                       const LongType* axes, LongType numAxes),
                      SD_COMMON_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd
