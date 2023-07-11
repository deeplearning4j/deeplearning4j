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
#include <ops/declarable/helpers/weights.h>
#include <execution/cuda/LaunchDims.h>
namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static SD_DEVICE void adjustWeightsKernelD(void* inputBuffer, sd::LongType const* inputShape, void* weightsBuffer,
                                           sd::LongType const* weightsShape, void* outputBuffer,
                                           sd::LongType inputLength, sd::LongType outputLength, int val) {
  auto tid = threadIdx.x;
  __shared__ T* outputPart;
  __shared__ sd::LongType offset;
  for (sd::LongType e = tid; e < inputLength; e += blockDim.x) {
    sd::LongType xOffset = shape::getIndexOffset(e, inputShape);
    sd::LongType current = *(reinterpret_cast<sd::LongType*>(inputBuffer) + xOffset);
    if (current == val) {
      if (weightsBuffer != nullptr) {
        sd::LongType yOffset = shape::getIndexOffset(e, weightsShape);
        // atomicAdd();
        sd::math::atomics::sd_atomicAdd(
            reinterpret_cast<T*>(outputBuffer),
            reinterpret_cast<T*>(weightsBuffer)[yOffset]);
      } else {
        sd::math::atomics::sd_atomicAdd(reinterpret_cast<T*>(outputBuffer),
                                        T(1));

      }
    }
  }
}

template <typename T>
static SD_KERNEL void adjustWeightsKernel(void* inputBuffer, sd::LongType const* inputShape, void* weightsBuffer,
                                          sd::LongType const* weightsShape, void* outputBuffer,
                                          sd::LongType const* outputShape, int minLength, int maxLength) {
  int threadCount = gridDim.x * blockDim.x;
  sd::LongType inputLength = shape::length(inputShape);

  sd::LongType outputLength = shape::length(outputShape);
  sd::LongType borderLen = 1;

  for (sd::LongType e = blockIdx.x; e < outputLength; e += threadCount) {
    sd::LongType zOffset = shape::getIndexOffset(e, outputShape);
    T* outputBufferZ = reinterpret_cast<T*>(outputBuffer) + zOffset;
    adjustWeightsKernelD<T>(inputBuffer, inputShape, weightsBuffer, weightsShape, (void*)outputBufferZ, inputLength,
                            outputLength, (int)zOffset);
  }
}

template <typename T>
static void adjustWeights_(sd::LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
                           int maxLength) {
  dim3 launchDims = getLaunchDims("adjustWeights");
  auto stream = context->getCudaStream();
  adjustWeightsKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(
      input->specialBuffer(), input->specialShapeInfo(), weights ? weights->specialBuffer() : nullptr,
      weights ? weights->specialShapeInfo() : nullptr, output->specialBuffer(), output->specialShapeInfo(), minLength,
      maxLength);
}

void adjustWeights(sd::LaunchContext* context, NDArray* input, NDArray* weights, NDArray* output, int minLength,
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
