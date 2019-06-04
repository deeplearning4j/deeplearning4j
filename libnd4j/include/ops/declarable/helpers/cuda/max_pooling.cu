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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/max_pooling.h>
#include <ops/declarable/helpers/convolutions.h>


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void maxPoolingFunctor_(nd4j::graph::Context& block, NDArray* input, NDArray* values, std::vector<int> const& params, NDArray* indices) {

    }

    void maxPoolingFunctor(nd4j::LaunchContext * context, nd4j::graph::Context& block, NDArray* input, NDArray* values, std::vector<int> const& params, NDArray* indices) {
        BUILD_SINGLE_SELECTOR(input->dataType(), maxPoolingFunctor_, (block, input, values, params, indices), FLOAT_TYPES);
    }


    BUILD_SINGLE_TEMPLATE(template void maxPoolingFunctor_, (nd4j::graph::Context& block, NDArray* input, NDArray* values, std::vector<int> const& params, NDArray* indices), FLOAT_TYPES);

}
}
}