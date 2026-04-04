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
// Squared ReLU CUDA implementation
// Forward:  out = max(0, x)^2
// Backward: dIn = dOut * 2 * x * (x > 0)
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <helpers/shape.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <types/float16.h>
#include <ops/declarable/helpers/squared_relu.h>

namespace sd {
namespace ops {
namespace helpers {

///////////////////////////////////////////////////////////////////////////////
// Forward kernel: out = max(0, x)^2
///////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void squaredReluKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const LongType length,
    const LongType rank,
    const LongType* xShape,
    const LongType* xStrides,
    const LongType* zShape,
    const LongType* zStrides) {

  // Grid-stride loop
  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x;
       i < length;
       i += gridDim.x * blockDim.x) {

    LongType xCoords[SD_MAX_RANK];
    LongType zCoords[SD_MAX_RANK];
    LongType xOffset = 0;
    LongType zOffset = 0;

    INDEX2COORDS(i, rank, xShape, xCoords);
    COORDS2INDEX(rank, xStrides, xCoords, xOffset);

    INDEX2COORDS(i, rank, zShape, zCoords);
    COORDS2INDEX(rank, zStrides, zCoords, zOffset);

    T val = input[xOffset];
    T relu = val > static_cast<T>(0) ? val : static_cast<T>(0);
    output[zOffset] = relu * relu;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Backward kernel: dIn = dOut * 2 * x * (x > 0)
///////////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void squaredReluBpKernel(
    const T* __restrict__ input,
    const T* __restrict__ gradOut,
    T* __restrict__ gradIn,
    const LongType length,
    const LongType rank,
    const LongType* xShape,
    const LongType* xStrides,
    const LongType* goShape,
    const LongType* goStrides,
    const LongType* giShape,
    const LongType* giStrides) {

  // Grid-stride loop
  for (LongType i = blockIdx.x * blockDim.x + threadIdx.x;
       i < length;
       i += gridDim.x * blockDim.x) {

    LongType xCoords[SD_MAX_RANK];
    LongType goCoords[SD_MAX_RANK];
    LongType giCoords[SD_MAX_RANK];
    LongType xOffset = 0;
    LongType goOffset = 0;
    LongType giOffset = 0;

    INDEX2COORDS(i, rank, xShape, xCoords);
    COORDS2INDEX(rank, xStrides, xCoords, xOffset);

    INDEX2COORDS(i, rank, goShape, goCoords);
    COORDS2INDEX(rank, goStrides, goCoords, goOffset);

    INDEX2COORDS(i, rank, giShape, giCoords);
    COORDS2INDEX(rank, giStrides, giCoords, giOffset);

    T val = input[xOffset];
    gradIn[giOffset] = val > static_cast<T>(0)
                            ? gradOut[goOffset] * static_cast<T>(2) * val
                            : static_cast<T>(0);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Forward launcher (typed)
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static void squaredRelu_(LaunchContext* context, NDArray* input, NDArray* output) {
  auto stream = context->getCudaStream();
  auto length = input->lengthOf();
  auto rank = input->rankOf();

  const T* xBuf = reinterpret_cast<const T*>(input->specialBuffer());
  T* zBuf = reinterpret_cast<T*>(output->specialBuffer());

  // Device-side shape info: shapeInfo layout is [rank, shape..., strides..., ...]
  const LongType* xShapeDevice = input->specialShapeInfo() + 1;
  const LongType* xStridesDevice = input->specialShapeInfo() + 1 + rank;
  const LongType* zShapeDevice = output->specialShapeInfo() + 1;
  const LongType* zStridesDevice = output->specialShapeInfo() + 1 + rank;

  dim3 launchDims = getLaunchDims("squared_relu");
  int threadsPerBlock = launchDims.y;
  int blocksPerGrid = static_cast<int>((length + threadsPerBlock - 1) / threadsPerBlock);
  if (blocksPerGrid > static_cast<int>(launchDims.x)) {
    blocksPerGrid = static_cast<int>(launchDims.x);
  }

  squaredReluKernel<T><<<blocksPerGrid, threadsPerBlock, launchDims.z, *stream>>>(
      xBuf, zBuf, length, rank, xShapeDevice, xStridesDevice, zShapeDevice, zStridesDevice);

  DebugHelper::checkGlobalErrorCode("squaredReluKernel failed");
}

///////////////////////////////////////////////////////////////////////////////
// Backward launcher (typed)
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static void squaredReluBp_(LaunchContext* context, NDArray* input, NDArray* dLdO, NDArray* dLdI) {
  auto stream = context->getCudaStream();
  auto length = input->lengthOf();
  auto rank = input->rankOf();

  const T* xBuf = reinterpret_cast<const T*>(input->specialBuffer());
  const T* goBuf = reinterpret_cast<const T*>(dLdO->specialBuffer());
  T* giBuf = reinterpret_cast<T*>(dLdI->specialBuffer());

  const LongType* xShapeDevice = input->specialShapeInfo() + 1;
  const LongType* xStridesDevice = input->specialShapeInfo() + 1 + rank;
  const LongType* goShapeDevice = dLdO->specialShapeInfo() + 1;
  const LongType* goStridesDevice = dLdO->specialShapeInfo() + 1 + rank;
  const LongType* giShapeDevice = dLdI->specialShapeInfo() + 1;
  const LongType* giStridesDevice = dLdI->specialShapeInfo() + 1 + rank;

  dim3 launchDims = getLaunchDims("squared_relu");
  int threadsPerBlock = launchDims.y;
  int blocksPerGrid = static_cast<int>((length + threadsPerBlock - 1) / threadsPerBlock);
  if (blocksPerGrid > static_cast<int>(launchDims.x)) {
    blocksPerGrid = static_cast<int>(launchDims.x);
  }

  squaredReluBpKernel<T><<<blocksPerGrid, threadsPerBlock, launchDims.z, *stream>>>(
      xBuf, goBuf, giBuf, length, rank,
      xShapeDevice, xStridesDevice,
      goShapeDevice, goStridesDevice,
      giShapeDevice, giStridesDevice);

  DebugHelper::checkGlobalErrorCode("squaredReluBpKernel failed");
}

///////////////////////////////////////////////////////////////////////////////
// Public interface -- forward
///////////////////////////////////////////////////////////////////////////////
void squaredRelu(LaunchContext* context, NDArray* input, NDArray* output) {
  NDArray::prepareSpecialUse({output}, {input});

  BUILD_SINGLE_SELECTOR(input->dataType(), squaredRelu_, (context, input, output), SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({output}, {input});
}

///////////////////////////////////////////////////////////////////////////////
// Public interface -- backward
///////////////////////////////////////////////////////////////////////////////
void squaredReluBp(LaunchContext* context, NDArray* input, NDArray* dLdO, NDArray* dLdI) {
  NDArray::prepareSpecialUse({dLdI}, {input, dLdO});

  BUILD_SINGLE_SELECTOR(input->dataType(), squaredReluBp_, (context, input, dLdO, dLdI), SD_FLOAT_TYPES);

  NDArray::registerSpecialUse({dLdI}, {input, dLdO});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
