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

    template <typename T>
    static void nudge(T min, T max, int quant_min, int quant_max, T* scale, T* nudged_min, T* nudged_max) {
        T quant_max_float = static_cast<T>(quant_max);
        T quant_min_float = static_cast<T>(quant_min);
        *scale = (max - min) / (quant_max_float - quant_min_float);
        auto zero_point_from_min = quant_min_float - min / *scale;
        uint16_t const nudged_zero_point = [zero_point_from_min, quant_min, quant_max, quant_max_float, quant_min_float] {
                if (zero_point_from_min < quant_min_float) {
                    return static_cast<uint16_t>(quant_min);
                }
                if (zero_point_from_min > quant_max_float) {
                    return static_cast<uint16_t>(quant_max);
                }
                return nd4j::math::nd4j_round<T,uint16_t>(zero_point_from_min);
            }();
            *nudged_min = (quant_min_float - nudged_zero_point) * (*scale);
            *nudged_max = (quant_max_float - nudged_zero_point) * (*scale);
    }

    template <typename T>
    void fakeQuantWithMinMaxVarsPerChannel_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed ? 1 : 0;
        int upperIntBound = (1 << numBits) - 1;
        auto channels = input->sizeAt(-1);

        PRAGMA_OMP_PARALLEL_FOR
        for (auto i = 0; i < channels; i++) {
            T scale, nudged_min, nudged_max;
            nudge<T>(min->t<T>(i), max->t<T>(i), lowIntBound, upperIntBound, &scale, &nudged_min, &nudged_max);

            for (auto e = 0; e < input->lengthOf(); e += channels) {
                T val = input->t<T>(e + i);
                if ( val <= nudged_min)
                    val = nudged_min;
                else if (val >= nudged_max)
                    val = nudged_max;

                output->t<T>(e + i) = math::nd4j_floor<T,T>((val - nudged_min)/scale + T(0.5)) * scale + nudged_min;
            }
        }
    }

    template <typename T>
    void fakeQuantWithMinMaxVars_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed ? 1 : 0;
        int upperIntBound = (1 << numBits) - 1;

        T nudgedMin, nudgedMax, scale;

        nudge<T>(min->t<T>(0), max->t<T>(0), lowIntBound, upperIntBound, &scale, &nudgedMin, &nudgedMax);

        auto fakeQuantizationWithMinMax = LAMBDA_T(x, nudgedMin, nudgedMax, scale) {
            T val = x;
            if (val < nudgedMin) {
                val = nudgedMin;
            }
            else if (val > nudgedMax)
                val = nudgedMax;
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
