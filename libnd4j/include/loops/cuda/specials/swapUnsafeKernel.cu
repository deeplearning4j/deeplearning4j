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
// @author GS <sgazeos@gmail.com>, created on 25.01.2019
//
#include <loops/special_kernels.h>

#include "execution/cuda/LaunchDims.h"


namespace sd {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// kernel to swap two NDArrays vals as linear sequences
// input - theSecondBuffer/Shape from input NDArray
// output - theFirstBuffer/Shape from input NDArray
template <typename T>
static SD_KERNEL void swapUnsafeKernel(void* theFirstBuffer, LongType const* theFirstShape, void* theSecondBuffer,
                                       LongType const* theSecondShape) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalThreads = gridDim.x * blockDim.x;

  __shared__ LongType resultLength;
  __shared__ bool sameOffsets, sameOrders;
  __shared__ T* input;
  __shared__ T* output;

  if (0 == threadIdx.x) {
    resultLength = shape::length(theFirstShape);
    input = reinterpret_cast<T*>(theSecondBuffer);
    output = reinterpret_cast<T*>(theFirstBuffer);

    sameOffsets = shape::haveSameShapeAndStrides(theFirstShape, theSecondShape);
    sameOrders = shape::order(theFirstShape) == shape::order(theSecondShape);
  }
  __syncthreads();

  for (int i = tid; i < resultLength; i += totalThreads) {
    sd::LongType xCoords[SD_MAX_RANK];
    sd::LongType yCoords[SD_MAX_RANK];
    sd::LongType xOffset;
    sd::LongType yOffset;

    INDEX2COORDS(i, shape::rank(theFirstShape), theFirstShape, xCoords);
    COORDS2INDEX(shape::rank(theFirstShape), shape::shapeOf(theFirstShape), xCoords, xOffset);
    INDEX2COORDS(i, shape::rank(theSecondShape), theSecondShape, yCoords);
    COORDS2INDEX(shape::rank(theSecondShape), shape::shapeOf(theSecondShape), yCoords, yOffset);

    if (sameOrders && xOffset >= 0 && yOffset >= 0) {
      math::sd_swap(output[xOffset], input[yOffset]);
    } else if (sameOffsets) {
      math::sd_swap(output[xOffset], input[xOffset]);
    } else {
      math::sd_swap(output[xOffset], input[yOffset]);
    }
  }
}
BUILD_SINGLE_TEMPLATE(template SD_KERNEL void swapUnsafeKernel,
                      (void* theFirstBuffer, sd::LongType const* theFirstShape, void* theSecondBuffer,
                       sd::LongType const* theSecondShape),
                      SD_COMMON_TYPES);

template <typename T>
void templatedSwapUnsafe(void* theFirstBuffer, LongType const* theFirstShape, void* theSecondBuffer,
                         LongType const* theSecondShape, cudaStream_t* theStream) {
  dim3 launchDims = getLaunchDims("swap_unsafe");
  swapUnsafeKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *theStream>>>(theFirstBuffer, theFirstShape,
                                                                                theSecondBuffer, theSecondShape);
  DebugHelper::checkGlobalErrorCode("templatedSwapUnsafe(...) failed");

}
BUILD_SINGLE_TEMPLATE(template void templatedSwapUnsafe,
                      (void* theFirstBuffer, sd::LongType const* theFirstShape, void* theSecondBuffer,
                       sd::LongType const* theSecondShape, cudaStream_t* theStream),
                      SD_COMMON_TYPES);

}  // namespace sd
