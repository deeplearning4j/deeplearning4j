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
#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/fake_quantization.h>

#include "execution/cuda/LaunchDims.h"

namespace sd {
namespace ops {
namespace helpers {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// fakeQuantWithMinMaxVars_
// input - input tensor
// min - min scalar tensor
// max - max scalar tensor
// numBits - (default 16bit)
// narrowed - shrink is true
// output - output tensor
//
template <typename T>
static SD_HOST_DEVICE void nudge(T min, T max, int quantMin, int quantMax, T* scale, T* nudgedMin, T* nudgedMax) {
  T quantMaxF = static_cast<T>(quantMax);
  T quantMinF = static_cast<T>(quantMin);
  *scale = (max - min) / (quantMaxF - quantMinF);
  auto zeroPointFromMin = quantMinF - min / *scale;
  uint16_t const nudgedZeroPoint = [zeroPointFromMin, quantMin, quantMax, quantMaxF, quantMinF] {
    if (zeroPointFromMin < quantMinF) {
      return static_cast<uint16_t>(quantMin);
    }
    if (zeroPointFromMin > quantMaxF) {
      return static_cast<uint16_t>(quantMax);
    }
    return sd::math::sd_round<T, uint16_t>(zeroPointFromMin);
  }();
  *nudgedMax = (quantMaxF - static_cast<T>(nudgedZeroPoint)) * (*scale);
  *nudgedMin = (quantMinF - static_cast<T>(nudgedZeroPoint)) * (*scale);
}

template <typename T>
void fakeQuantWithMinMaxVars_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
  int lowIntBound = narrowed ? 1 : 0;
  int upperIntBound = (1 << numBits) - 1;
  min->syncToHost();  // these are scalars, so nothing much happened
  max->syncToHost();
  T scale, nudgedMin, nudgedMax;
  nudge(min->t<T>(0), max->t<T>(0), lowIntBound, upperIntBound, &scale, &nudgedMin, &nudgedMax);

  auto wiseMinMaxAndSoOn = LAMBDA_T(x, nudgedMin, nudgedMax, scale) {
    T val = x;
    if (x < nudgedMin) {
      val = nudgedMin;
    } else if (x > nudgedMax) {
      val = nudgedMax;
    } else
      val = x;
    return (math::sd_floor<T, T>((val - nudgedMin) / scale + T(0.5)) * scale + nudgedMin);
  };

  input->applyLambda(wiseMinMaxAndSoOn, *output);
}

template <typename T>
static SD_KERNEL void fakeQuantWithMinMaxKernel(const T* input, const sd::LongType* inputShape, T* min, T* max,
                                                int lowIntBound, int upperIntBound, sd::LongType channels, T* output,
                                                const sd::LongType* outputShape, sd::LongType length) {
  __shared__ int block;
  if (threadIdx.x == 0) {
    block = length / channels;  // to loop with last dimension as block
  }
  __syncthreads();

  for (auto i = blockIdx.x; i < (int)channels; i += gridDim.x) {
    T scale, nudgedMin, nudgedMax;
    nudge(min[i], max[i], lowIntBound, upperIntBound, &scale, &nudgedMin, &nudgedMax);
    // loop over blocks to quantization between nudged min and max
    for (auto b = threadIdx.x; b < block; b += blockDim.x) {
      T val = input[shape::getIndexOffset(b * channels + i, inputShape)];
      if (val < nudgedMin) {
        val = nudgedMin;
      } else if (val > nudgedMax) {
        val = nudgedMax;
      }
      output[shape::getIndexOffset(b * channels + i, outputShape)] =
          (math::sd_floor<T, T>((val - nudgedMin) / scale + T(0.5f)) * scale + nudgedMin);
    };
  }
}

template <typename T>
void fakeQuantWithMinMaxVarsPerChannel_(LaunchContext* context, NDArray* input, NDArray* min, NDArray* max, int numBits,
                                        bool narrowed, NDArray* output) {
  int lowIntBound = narrowed ? 1 : 0;
  int upperIntBound = (1 << numBits) - 1;
  auto channels = min->lengthOf();
  auto length = input->lengthOf();
  NDArray::prepareSpecialUse({output}, {min, max, input});
  auto stream = context->getCudaStream();
  T* inputBuf = input->dataBuffer()->specialAsT<T>();
  T* outputBuf = output->dataBuffer()->specialAsT<T>();
  T* minBuf = min->dataBuffer()->specialAsT<T>();
  T* maxBuf = max->dataBuffer()->specialAsT<T>();
  dim3 launchDims = getLaunchDims("fake_quantization");
  fakeQuantWithMinMaxKernel<<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(inputBuf, input->specialShapeInfo(), minBuf, maxBuf,
                                                        lowIntBound, upperIntBound, channels, outputBuf,
                                                        output->specialShapeInfo(), length);
  NDArray::registerSpecialUse({output}, {min, max, input});
}

void fakeQuantWithMinMaxVars(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVars_, (input, min, max, numBits, narrowed, output),
                        SD_FLOAT_TYPES);
}
void fakeQuantWithMinMaxVarsPerChannel(LaunchContext* context, NDArray* input, NDArray* min, NDArray* max, int numBits,
                                       bool narrowed, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVarsPerChannel_,
                        (context, input, min, max, numBits, narrowed, output), SD_FLOAT_TYPES);
}
}  // namespace helpers
}  // namespace ops
}  // namespace sd
