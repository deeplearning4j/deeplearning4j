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
// @author GS <sgazeos@gmail.com>, created on 16.01.2019
//
#include <loops/special_kernels.h>

#include <execution/cuda/LaunchDims.h>


namespace sd {


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  tileKernel:
//  input: (inputBuffer and inputShape) - NDArray buffer and shape to tile
//  output: (outputBuffer and outputShape) - NDArray to tile input
//  resultLength - length for output array
template <typename T>
static SD_KERNEL void tileKernel(void const* inputBuffer, LongType const* inputShape, void* outputBuffer,
                                 LongType const* outputShape, LongType resultLength) {
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;  // copy linear sequence of elements, so one-level threading
  int totalThreads = gridDim.x * blockDim.x;

  if (shape::order(outputShape) == 'c') {
    for (int i = tid; i < resultLength; i += totalThreads) {
      sd::LongType yCoords[SD_MAX_RANK];
      sd::LongType yOffset;

      INDEX2COORDS(i, shape::rank(outputShape), outputShape, yCoords);
      COORDS2INDEX(shape::rank(outputShape), shape::shapeOf(outputShape), yCoords, yOffset);

      *(reinterpret_cast<T*>(outputBuffer) + i) = *(reinterpret_cast<T const*>(inputBuffer) + yOffset);
    }
  } else {
    for (int i = tid; i < resultLength; i += totalThreads) {
      sd::LongType xCoords[SD_MAX_RANK];
      sd::LongType yCoords[SD_MAX_RANK];
      sd::LongType xOffset;
      sd::LongType yOffset;

      INDEX2COORDS(i, shape::rank(outputShape), outputShape, xCoords);
      COORDS2INDEX(shape::rank(outputShape), shape::shapeOf(outputShape), xCoords, xOffset);
      INDEX2COORDS(i, shape::rank(inputShape), inputShape, yCoords);
      COORDS2INDEX(shape::rank(inputShape), shape::shapeOf(inputShape), yCoords, yOffset);

      *(reinterpret_cast<T*>(outputBuffer) + xOffset) = *(reinterpret_cast<T const*>(inputBuffer) + yOffset);
    }
  }
}

BUILD_SINGLE_TEMPLATE(template SD_KERNEL void tileKernel,
                      (void const* inputBuffer, sd::LongType const* inputShape, void* outputBuffer,
                       sd::LongType const* outputShape, sd::LongType resultLength),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
void tileKernelH(void const* inputBuffer, LongType const* inputShape, void* outputBuffer, LongType const* outputShape,
                 LongType resultLength, cudaStream_t* stream) {
  dim3 launchDims = getLaunchDims("tile");
  tileKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuffer, inputShape, outputBuffer,
                                                                       outputShape, resultLength);
  sd::DebugHelper::checkErrorCode(stream, "tileKernel  failed");


}

BUILD_SINGLE_TEMPLATE(template void tileKernelH,
                      (void const* inputBuffer, sd::LongType const* inputShape, void* outputBuffer,
                       sd::LongType const* outputShape, sd::LongType resultLength, cudaStream_t* stream),
                      SD_COMMON_TYPES);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// enhancement for tileKernel to different input and output data types: X - output type, Y - input type
template <typename X, typename Y>
static SD_KERNEL void tileKernelDouble(void const* inputBuffer, LongType const* inputShape, void* outputBuffer,
                                       LongType const* outputShape, LongType resultLength) {
  char ordering = shape::order(outputShape);
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int totalThreads = gridDim.x * blockDim.x;

  if (ordering == 'c') {
    for (int i = tid; i < resultLength; i += totalThreads) {
      sd::LongType yCoords[SD_MAX_RANK];
      sd::LongType yOffset;

      INDEX2COORDS(i, shape::rank(outputShape), outputShape, yCoords);
      COORDS2INDEX(shape::rank(outputShape), shape::shapeOf(outputShape), yCoords, yOffset);

      *(reinterpret_cast<X*>(outputBuffer) + i) = static_cast<X>(*(reinterpret_cast<Y const*>(inputBuffer) + yOffset));
    }
  } else {
    for (int i = tid; i < resultLength; i += totalThreads) {
      sd::LongType xCoords[SD_MAX_RANK];
      sd::LongType yCoords[SD_MAX_RANK];
      sd::LongType xOffset;
      sd::LongType yOffset;

      INDEX2COORDS(i, shape::rank(outputShape), outputShape, xCoords);
      COORDS2INDEX(shape::rank(outputShape), shape::shapeOf(outputShape), xCoords, xOffset);
      INDEX2COORDS(i, shape::rank(inputShape), inputShape, yCoords);
      COORDS2INDEX(shape::rank(inputShape), shape::shapeOf(inputShape), yCoords, yOffset);

      *(reinterpret_cast<X*>(outputBuffer) + xOffset) = static_cast<X>(*(reinterpret_cast<Y const*>(inputBuffer) + yOffset));
    }
  }
}
BUILD_SINGLE_TEMPLATE_TWICE(template SD_KERNEL void tileKernelDouble,
                            (void const* inputBuffer, sd::LongType const* inputShape, void* outputBuffer,
                             sd::LongType const* outputShape, sd::LongType resultLength, sd::LongType ews),
                            SD_COMMON_TYPES);

template <typename X, typename Y>
void tileKernelHH(void const* inputBuffer, LongType const* inputShape, void* outputBuffer, LongType const* outputShape,
                  LongType resultLength, LongType ews, cudaStream_t* stream) {
  dim3 launchDims = getLaunchDims("tile");
  tileKernelDouble<X, Y><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuffer, inputShape, outputBuffer,
                                                                                outputShape, resultLength, ews);

  DebugHelper::checkErrorCode(stream,"templatedSwapUnsafe(...) failed");

}

BUILD_SINGLE_TEMPLATE_TWICE(template void tileKernelHH,
                            (void const* inputBuffer, sd::LongType const* inputShape, void* outputBuffer,
                             sd::LongType const* outputShape, sd::LongType resultLength, sd::LongType ews,
                             cudaStream_t* stream),
                            SD_COMMON_TYPES);
}  // namespace sd
