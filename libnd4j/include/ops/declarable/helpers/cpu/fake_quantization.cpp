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
    static void Nudge(T min, T max, T quant_min, T quant_max, T* scale, T* nudged_min, T* nudged_max) {
        *scale = (max - min) / (quant_max - quant_min);
        auto zero_point_from_min = quant_min - min / *scale;
        uint16_t const nudged_zero_point = [zero_point_from_min, quant_min, quant_max] {
                if (zero_point_from_min < quant_min) {
                    return static_cast<uint16_t>(quant_min);
                }
                if (zero_point_from_min > quant_max) {
                    return static_cast<uint16_t>(quant_max);
                }
                return nd4j::math::nd4j_round<T,uint16_t>(zero_point_from_min);
            }();
            *nudged_min = (quant_min - nudged_zero_point) * (*scale);
            *nudged_max = (quant_max - nudged_zero_point) * (*scale);
    }

    template <typename T>
    void fakeQuantWithMinMaxVarsPerChannel_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed ? 1 : 0;
        int upperIntBound = 1 << numBits - 1;

        const float quant_min_float = static_cast<float>(lowIntBound);
        const float quant_max_float = static_cast<float>(upperIntBound);
//        auto scaleTensor(*input); // = NDArrayFactory::create(input->ordering(), input->getShapeAsVector(), input->getWorkspace());
        auto clamped(*input); // = NDArrayFactory::create(input->ordering(), input->getShapeAsVector(), input->getWorkspace());
        for (auto i = 0; i < min->lengthOf(); i++) {
            T scale, nudged_min, nudged_max;
            Nudge<T>(min->t<T>(i), max->t<T>(i), quant_min_float, quant_max_float, &scale, &nudged_min, &nudged_max);
            auto wiseMinMax = LAMBDA_T(x, nudged_min, nudged_max) {
                if (x < nudged_min) {
                    return nudged_min;
                }
                else if (x > nudged_max)
                    return nudged_max;
                return x;
            };
//            scaleTensor.assign(scale);
            input->applyLambda<T>(wiseMinMax, &clamped);
            clamped -= nudged_min;
            // auto nudgedScale = scale;
            clamped /= scale;
            clamped += T(0.5f);
            clamped.applyTransform(transform::Floor, output, nullptr);
            (*output) *= scale;
            (*output) += nudged_min;
        }
    }

    template <typename T>
    static void WiseMinMax(NDArray* input, T min, T max, NDArray* output) {
        auto wiseMinMax = LAMBDA_T(x, min, max) {
            if (x < min) {
                return min;
            }
            else if (x > max)
                return max;
            return x;
        };

        input->applyLambda<T>(wiseMinMax, output);
    }

    template <typename T>
    void fakeQuantWithMinMaxVars_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed ? 1 : 0;
        int upperIntBound = 1 << numBits - 1;

        const float quant_min_float = static_cast<float>(lowIntBound);
        const float quant_max_float = static_cast<float>(upperIntBound);
        T nudged_min, nudged_max, scale;

        Nudge<T>(min->t<T>(0), max->t<T>(0), quant_min_float, quant_max_float, &scale, &nudged_min, &nudged_max);
        WiseMinMax<T>(input, nudged_min, nudged_max, output);
        *output -= nudged_min;
        (*output) /= scale;
        (*output) += T(0.5f);
        output->applyTransform(transform::Floor, nullptr, nullptr);
        (*output) *= scale;
        (*output) += nudged_min;
    }

    void fakeQuantWithMinMaxVars(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVars_, (input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }
    void fakeQuantWithMinMaxVarsPerChannel(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVarsPerChannel_, (input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void fakeQuantWithMinMaxVars_, (NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output), FLOAT_TYPES);

}
}
}
