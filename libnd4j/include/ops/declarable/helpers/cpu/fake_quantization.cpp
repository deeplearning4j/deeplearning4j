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

    //
    // nudge - nudged min max over scale
    // scale = (Max - Min) / (quantMax - quantMin)
    // quantMin = 0 or 1, quantMax = 2^b - 1 == (1 << b) - 1
    //
    template <typename T>
    static void nudge(T min, T max, int quantMin, int quantMax, T* scale, T* nudgedMin, T* nudgedMax) {
        // floating point instead integers
        T quantMaxF = static_cast<T>(quantMax);
        T quantMinF = static_cast<T>(quantMin);
        // compute scale
        *scale = (max - min) / (quantMaxF - quantMinF);
        // compute left bound point
        auto zeroPointFromMin = quantMinF - min / *scale;
        // bound zero point to conform with range [0 or 1, 2^b - 1]
        uint16_t const nudged_zero_point = [zeroPointFromMin, quantMin, quantMax, quantMaxF, quantMinF] {
                if (zeroPointFromMin < quantMinF) {
                    return static_cast<uint16_t>(quantMin);
                }
                if (zeroPointFromMin > quantMaxF) {
                    return static_cast<uint16_t>(quantMax);
                }
                return nd4j::math::nd4j_round<T,uint16_t>(zeroPointFromMin);
        }();
        // compute nudged min and max with computed nudged zero point
        *nudgedMin = (quantMinF - nudged_zero_point) * (*scale);
        *nudgedMax = (quantMaxF - nudged_zero_point) * (*scale);
    }

    template <typename T>
    void fakeQuantWithMinMaxVarsPerChannel_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed ? 1 : 0; // 0 or 1
        int upperIntBound = (1 << numBits) - 1; // 2^b - 1
        auto channels = input->sizeAt(-1); // last dimension

        PRAGMA_OMP_PARALLEL_FOR
        for (auto i = 0; i < channels; i++) {
            T scale, nudged_min, nudged_max;
            // nudge min and max first, with scale computing
            nudge<T>(min->t<T>(i), max->t<T>(i), lowIntBound, upperIntBound, &scale, &nudged_min, &nudged_max);
            // slide using last dimension and process all for given channel
            for (auto e = 0; e < input->lengthOf(); e += channels) {
                T val = input->t<T>(e + i);
                if ( val <= nudged_min)
                    val = nudged_min;
                else if (val >= nudged_max)
                    val = nudged_max;
                // quantization itself
                output->t<T>(e + i) = math::nd4j_floor<T,T>((val - nudged_min)/scale + T(0.5)) * scale + nudged_min;
            }
        }
    }

    template <typename T>
    void fakeQuantWithMinMaxVars_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed ? 1 : 0;
        int upperIntBound = (1 << numBits) - 1;

        T nudgedMin, nudgedMax, scale;
        // nudge with given min and max and compute scale and nudged min and max
        nudge<T>(min->t<T>(0), max->t<T>(0), lowIntBound, upperIntBound, &scale, &nudgedMin, &nudgedMax);
        // quantization as one
        auto fakeQuantizationWithMinMax = LAMBDA_T(x, nudgedMin, nudgedMax, scale) {
            T val = x; // boundign value between nudged min and max
            if (val < nudgedMin) {
                val = nudgedMin;
            }
            else if (val > nudgedMax)
                val = nudgedMax;
            // converse value with scale and shifted with nudged min
            return (nd4j::math::nd4j_floor<T,T>((val - nudgedMin)/scale + T(0.5)) * scale + nudgedMin);
        };

        input->applyLambda<T>(fakeQuantizationWithMinMax, output);
    }

    void fakeQuantWithMinMaxVars(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVars_, (input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }
    void fakeQuantWithMinMaxVarsPerChannel(LaunchContext* context, NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVarsPerChannel_, (input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void fakeQuantWithMinMaxVars_, (NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output), FLOAT_TYPES);

}
}
}
