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
    void fakeQuantWithMinMaxVars_(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        int lowIntBound = narrowed ? 1 : 0;
        int upperIntBound = 1 << numBits - 1;

        const float quant_min_float = static_cast<float>(lowIntBound);
        const float quant_max_float = static_cast<float>(upperIntBound);
        T scale = (max->t<T>(0) - min->t<T>(0)) / (quant_max_float - quant_min_float);
        const T zero_point_from_min = quant_min_float - min->e<T>(0) / scale;
        const uint16_t nudged_zero_point = [zero_point_from_min, lowIntBound,
                                        quant_min_float, upperIntBound,
                                        quant_max_float] {
        if (zero_point_from_min < quant_min_float) {
          return static_cast<uint16_t>(lowIntBound);
        }
        if (zero_point_from_min > quant_max_float) {
          return static_cast<uint16_t>(upperIntBound);
        }
        return static_cast<uint16_t>(roundf(zero_point_from_min));
        }();

        auto nudged_min = (quant_min_float - nudged_zero_point) * (scale);
        auto nudged_max = (quant_max_float - nudged_zero_point) * (scale);
        //input->applyScalar(scalar::CompareAndSet, nudged_max, clamped, nullptr); //.cwiseMin(nudged_max).cwiseMax(nudged_min);
        //input->applyScalar(scalar::CompareAndSet, nudged_min, clamped, nullptr); //.cwiseMin(nudged_max).cwiseMax(nudged_min);
        auto wiseMax = LAMBDA_T(x, nudged_min) {
            if (x < nudged_min) {
                return nudged_min;
            }
            return x;

        };
        auto wiseMin = LAMBDA_T(x, nudged_max) {
            if (x > nudged_max) {
                return nudged_max;
            }
            return x;
        };
        auto scaleTensor(*input); // = NDArrayFactory::create(input->ordering(), input->getShapeAsVector(), input->getWorkspace());
        auto clamped(*input); // = NDArrayFactory::create(input->ordering(), input->getShapeAsVector(), input->getWorkspace());
        scaleTensor.assign(scale);
        input->applyLambda<T>(wiseMin, &clamped);
//        const auto clamped = inputs.cwiseMin(nudged_max).cwiseMax(nudged_min);
        clamped.applyLambda<T>(wiseMax, output);
//        const auto clamped_shifted = clamped - nudged_min;
        *output -= nudged_min;
       // auto nudgedScale = scale;
        (*output) /= scaleTensor;
        (*output) += T(0.5f);
        output->applyTransform(transform::Floor, nullptr, nullptr);
        (*output) *= scaleTensor;
        (*output) += nudged_min;
        //output->printIndexedBuffer("FAKE QUANTED");
        /*
    const auto nudged_scale_repl = inputs.constant(nudged_scale);

    const auto clamped = inputs.cwiseMin(nudged_max).cwiseMax(nudged_min);
    const auto clamped_shifted = clamped - nudged_min;
    *output = (clamped_shifted / nudged_scale_repl + 0.5f).floor() *
                            nudged_scale_repl +
                        nudged_min;
*/

    }

    void fakeQuantWithMinMaxVars(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVars_, (input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }
    void fakeQuantWithMinMaxVarsPerChannel(NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), fakeQuantWithMinMaxVars_, (input, min, max, numBits, narrowed, output), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void fakeQuantWithMinMaxVars_, (NDArray* input, NDArray* min, NDArray* max, int numBits, bool narrowed, NDArray* output), FLOAT_TYPES);

}
}
}
