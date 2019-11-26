/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <ops/declarable/helpers/fake_quantization.h>
#include <NDArrayFactory.h>

namespace nd4j {
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
    static __host__ __device__ void
    nudge(T min, T max, int quantMin, int quantMax, T* scale, T* nudgedMin, T* nudgedMax) {
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
            return nd4j::math::nd4j_round<T,uint16_t>(zeroPointFromMin);
        }();
        *nudgedMin = (quantMinF - nudgedZeroPoint) * (*scale);
        *nudgedMax = (quantMaxF - nudgedZeroPoint) * (*scale);
    }

    template <typename T>
    void fakeQuantWithMinMaxVars_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed?1:0;
        int upperIntBound = (1 << numBits) - 1;
        min->syncToHost(); // these are scalars, so nothing much happened
        max->syncToHost();
        T scale, nudgedMin, nudgedMax;
        nudge(min->t<T>(0), max->t<T>(0), lowIntBound, upperIntBound, &scale, &nudgedMin, &nudgedMax);

        auto wiseMinMaxAndSoOn = LAMBDA_T(x, nudgedMin, nudgedMax, scale) {
            T val = x;
            if (x < nudgedMin) {
                val = nudgedMin;
            }
            else if (x > nudgedMax) {
                val = nudgedMax;
            }
            else
                val = x;
            return (math::nd4j_floor<T,T>((val - nudgedMin) / scale + T(0.5)) * scale + nudgedMin);
        };

        input->applyLambda(wiseMinMaxAndSoOn, output);
    }

    template <typename T>
    static __global__ void fakeQuantWithMinMaxKernel(T* input, Nd4jLong* inputShape, T* min, T* max,
            int lowIntBound, int upperIntBound, Nd4jLong channels,
            T* output, Nd4jLong* outputShape, Nd4jLong length) {
        __shared__ int block;
        if (threadIdx.x == 0) {
            block = length / channels; // to loop with last dimension as block
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
                        (math::nd4j_floor<T, T>((val - nudgedMin) / scale + T(0.5f)) * scale + nudgedMin);
            };
        }
    }

    template <typename T>
    void fakeQuantWithMinMaxVarsPerChannel_(LaunchContext* context, NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed?1:0;
        int upperIntBound = (1 << numBits) - 1;
        auto channels = min->lengthOf();
        auto length = input->lengthOf();
        NDArray::prepareSpecialUse({output}, {min, max, input});
        auto stream = context->getCudaStream();
        T* inputBuf = input->dataBuffer()->specialAsT<T>();
        T* outputBuf = output->dataBuffer()->specialAsT<T>();
        T* minBuf = min->dataBuffer()->specialAsT<T>();
        T* maxBuf = max->dataBuffer()->specialAsT<T>();
        fakeQuantWithMinMaxKernel<<<128, 256, 256, *stream>>>(inputBuf, input->specialShapeInfo(),
                minBuf, maxBuf, lowIntBound, upperIntBound, channels, outputBuf, output->specialShapeInfo(), length);
        NDArray::registerSpecialUse({output}, {min, max, input});

    }

    void fakeQuantWithMinMaxVars(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVars_, (input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }
    void fakeQuantWithMinMaxVarsPerChannel(LaunchContext* context, NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVarsPerChannel_, (context, input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void fakeQuantWithMinMaxVars_, (NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output), FLOAT_TYPES);
    BUILD_SINGLE_TEMPLATE(template void fakeQuantWithMinMaxVarsPerChannel_, (LaunchContext* context, NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output), FLOAT_TYPES);

}
}
}
