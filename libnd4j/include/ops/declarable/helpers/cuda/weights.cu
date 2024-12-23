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

Optimized Adjust Weights CUDA Device Function

    template <typename T>
    static SD_DEVICE void adjustWeightsKernelD(void* inputBuffer, LongType const* inputShape,
                                               const LongType inputRank, const LongType* inShape, const LongType* inStride,
                                               void* weightsBuffer, LongType const* weightsShape,
                                               const LongType weightsRank, const LongType* wShape, const LongType* wStride,
                                               void* outputBuffer, LongType inputLength,
                                               LongType outputLength, int val, LongType* coords) {
  if(inputBuffer == nullptr || outputBuffer == nullptr) return;
  auto tid = threadIdx.x;
  LongType xOffset, yOffset;

  for (LongType e = tid; e < inputLength; e += blockDim.x) {
    INDEX2COORDS(e, inputRank, inShape, coords);
    COORDS2INDEX(inputRank, inStride, coords, xOffset);

    if (xOffset >= inputLength) return;

    LongType current = *(reinterpret_cast<LongType*>(inputBuffer) + xOffset);
    if (current == val) {
      if (weightsBuffer != nullptr) {
        INDEX2COORDS(e, weightsRank, wShape, coords);
        COORDS2INDEX(weightsRank, wStride, coords, yOffset);
        math::atomics::sd_atomicAdd(
            reinterpret_cast<T*>(outputBuffer),
            reinterpret_cast<T*>(weightsBuffer)[yOffset]);
      } else {
        math::atomics::sd_atomicAdd(reinterpret_cast<T*>(outputBuffer),
                                    T(1));
      }
    }
  }

template <typename T>
static SD_KERNEL void adjustWeightsKernel(void* inputBuffer, LongType const* inputShape, void* weightsBuffer,
                                          LongType const* weightsShape, void* outputBuffer, LongType const* outputShape,
                                          int minLength, int maxLength) {
  __shared__ LongType outputLength, outputRank;
  __shared__ LongType *sharedMem;
  __shared__ const LongType *outShape, *outStride;

  if (threadIdx.x == 0) {
    extern __shared__ unsigned char shmem[];
    sharedMem = reinterpret_cast<LongType*>(shmem);

    outputLength = shape::length(outputShape);
    outputRank = shape::rank(outputShape);
    outShape = shape::shapeOf(outputShape);
    outStride = shape::stride(outputShape);
  }
  __syncthreads();

  int threadCount = gridDim.x * blockDim.x;
  LongType inputLength = shape::length(inputShape);
  LongType borderLen = 1;

  // Get thread-local coordinate array from shared memory
  auto coords = sharedMem + threadIdx.x * SD_MAX_RANK;

  for (LongType e = blockIdx.x; e < outputLength; e += threadCount) {
    LongType zOffset;
    INDEX2COORDS(e, outputRank, outShape, coords);
    COORDS2INDEX(outputRank, outStride, coords, zOffset);

    T* outputBufferZ = reinterpret_cast<T*>(outputBuffer) + zOffset;
    adjustWeightsKernelD<T>(inputBuffer, inputShape, weightsBuffer, weightsShape, (void*)outputBufferZ,
                            inputLength, outputLength, (int)zOffset);
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
