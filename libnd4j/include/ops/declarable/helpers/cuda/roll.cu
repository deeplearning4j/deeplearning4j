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

#include <ops/declarable/helpers/roll.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void rollFunctorLinear_(NDArray* input, NDArray* output, int shift, bool inplace){

    }

    void rollFunctorFull(nd4j::LaunchContext * context, NDArray* input, NDArray* output, int shift, std::vector<int> const& axes, bool inplace){

    }

    void rollFunctorLinear(nd4j::LaunchContext * context, NDArray* input, NDArray* output, int shift, bool inplace){
        BUILD_SINGLE_SELECTOR(input->dataType(), rollFunctorLinear_, (input, output, shift, inplace), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void rollFunctorLinear_, (NDArray* input, NDArray* output, int shift, bool inplace), LIBND4J_TYPES);
}
}
}