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

#include <ResultSet.h>
//#include <ops/declarable/helpers/reduce_product.h>
#include <ops/declarable/helpers/legacy_helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void reduceNorm2BP_scalar_(NDArray *input, NDArray *epsilon, NDArray *tempNorm, NDArray *output) {

    }
    BUILD_SINGLE_TEMPLATE(template void reduceNorm2BP_scalar_, (NDArray *input, NDArray *epsilon, NDArray *tempNorm, NDArray *output), FLOAT_TYPES);


    void reduceNorm2BP_scalar(graph::LaunchContext* context, NDArray *input, NDArray *epsilon, NDArray *tempNorm, NDArray *output) {
        auto xType = epsilon->dataType();

        BUILD_SINGLE_SELECTOR(xType, reduceNorm2BP_scalar_, (input, epsilon, tempNorm, output), FLOAT_TYPES);
    }

    void reduceNorm1BP(graph::LaunchContext* context, NDArray* input, NDArray* epsilon, NDArray* tempNorm, NDArray* output, std::vector<int> const& axes, bool keepDims) {

    }

    void reduceNorm2BP(graph::LaunchContext* context, NDArray* input, NDArray* epsilon, NDArray* tempNorm, NDArray* output, std::vector<int> const& axes, bool keepDims) {

    }

    void reduceSquareNormBP(graph::LaunchContext* context, NDArray* input, NDArray* epsilon, NDArray* tempNorm, NDArray* output, std::vector<int> const& axes, bool keepDims) {

    }
}
}
}