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
//  @author sgazeos@gmail.com
//
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/weights.h>


#include "helpers/DebugHelper.h"
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_DEVICE void adjustWeightsKernelD(void* inputBuffer, LongType const* inputShape, void* weightsBuffer,
                                           LongType const* weightsShape, void* outputBuffer, LongType inputLength,
                                           LongType outputLength, int val) {
  if(inputBuffer == nullptr || outputBuffer == nullptr) return;

  auto tid = threadIdx.x;

  // Cache shape and stride information
  const sd::LongType inputRank = shape::rank(inputShape);
  const sd::LongType* inputShapePtr = shape::shapeOf(inputShape);
  const sd::LongType* inputStridePtr = shape::stride(inputShape);

  // Cache weights shape and stride if weightsBuffer exists
  const sd::LongType weightsRank = weightsBuffer != nullptr ? shape::rank(weightsShape) : 0;
  const sd::LongType* weightsShapePtr = weightsBuffer != nullptr ? shape::shapeOf(weightsShape) : nullptr;
  const sd::LongType* weightsStridePtr = weightsBuffer != nullptr ? shape::stride(weightsShape) : nullptr;

  LongType xCoords[SD_MAX_RANK];
  LongType yCoords[SD_MAX_RANK];
  LongType xOffset;
  LongType yOffset;

  for (LongType e = tid; e < inputLength; e += blockDim.x) {
    INDEX2COORDS(e, inputRank, inputShapePtr, xCoords);
    COORDS2INDEX(inputRank, inputStridePtr, xCoords, xOffset);

    if (xOffset >= inputLength) return;

    LongType current = *(reinterpret_cast<LongType*>(inputBuffer) + xOffset);
    if (current == val) {
      if (weightsBuffer != nullptr) {
        INDEX2COORDS(e, weightsRank, weightsShapePtr, yCoords);
        COORDS2INDEX(weightsRank, weightsStridePtr, yCoords, yOffset);
        math::atomics::sd_atomicAdd(
            reinterpret_cast<T*>(outputBuffer),
            reinterpret_cast<T*>(weightsBuffer)[yOffset]);
      } else {
        math::atomics::sd_atomicAdd(reinterpret_cast<T*>(outputBuffer), T(1));
      }
    }
  }
}

template <typename T>
static SD_KERNEL void adjustWeightsKernel(void* inputBuffer, LongType const* inputShape, void* weightsBuffer,
                                          LongType const* weightsShape, void* outputBuffer, LongType const* outputShape,
                                          int minLength, int maxLength) {
  // Shared variables for shape information
  __shared__ sd::LongType inputLen;
  __shared__ sd::LongType outputLen;
  __shared__ sd::LongType outputRank;
  __shared__ const sd::LongType* outputShapePtr;
  __shared__ const sd::LongType* outputStridePtr;

  // Cache shape information in thread 0
  if (threadIdx.x == 0) {
    inputLen = shape::length(inputShape);
    outputLen = shape::length(outputShape);
    outputRank = shape::rank(outputShape);
    outputShapePtr = shape::shapeOf(outputShape);
    outputStridePtr = shape::stride(outputShape);
  }
  __syncthreads();

  int threadCount = gridDim.x * blockDim.x;
  LongType borderLen = 1;

  LongType zCoords[SD_MAX_RANK];
  LongType zOffset;

  for (LongType e = blockIdx.x; e < outputLen; e += threadCount) {
    INDEX2COORDS(e, outputRank, outputShapePtr, zCoords);
    COORDS2INDEX(outputRank, outputStridePtr, zCoords, zOffset);

    T* outputBufferZ = reinterpret_cast<T*>(outputBuffer) + zOffset;
    adjustWeightsKernelD<T>(inputBuffer, inputShape, weightsBuffer, weightsShape,
                            (void*)outputBufferZ, inputLen, outputLen, (int)zOffset);
  }
}
template <typename T>
static void adjustWeights_(LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
                          int maxLength) {
 dim3 launchDims = getLaunchDims("adjustWeights");
 auto stream = context->getCudaStream();
 adjustWeightsKernel<T><<<launchDims.y, launchDims.x, launchDims.z, *stream>>>(
     input->specialBuffer(), input->specialShapeInfo(), weights ? weights->specialBuffer() : nullptr,
     weights ? weights->specialShapeInfo() : nullptr, output->specialBuffer(), output->specialShapeInfo(), minLength,
     maxLength);
 sd::DebugHelper::checkErrorCode(stream, "adjustWeightsKernel failed");

}

void adjustWeights(LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
                  int maxLength) {
 BUILD_SINGLE_SELECTOR(output->dataType(), adjustWeights_, (context, input, weights, output, minLength, maxLength),
                       SD_GENERIC_NUMERIC_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void adjustWeights_,
                     (sd::LaunchContext * context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
                      int maxLength),
                     SD_GENERIC_NUMERIC_TYPES);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
