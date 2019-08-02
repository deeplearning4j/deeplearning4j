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

#include <ops/declarable/helpers/axis.h>
#include <op_boilerplate.h>

namespace nd4j {
namespace ops {
namespace helpers {
template <typename T>
static void applyGradientDescent_(LaunchContext* context, NDArray* input, NDArray* step, double weight, NDArray* output) {
    auto lambda = LAMBDA_TT(_x, _y, weight) {
        return _x - (_y * weight);
    };

    input->applyPairwiseLambda(step, lambda, output);
}

void applyGradientDescent(nd4j::LaunchContext* context, NDArray* input, NDArray* step, double weight, NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), applyGradientDescent_, (context, input, step, weight, output), FLOAT_TYPES);
}
BUILD_SINGLE_TEMPLATE(template void applyGradientDescent_, (LaunchContext* context, NDArray* input, NDArray* step, double weight, NDArray* output), FLOAT_TYPES);
}
}
}
