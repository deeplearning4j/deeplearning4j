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
#include <ops/declarable/helpers/reduce_norm.h>

namespace nd4j {
namespace ops {
namespace helpers {

    void reduceProductBP(NDArray *input, NDArray *epsilon, NDArray *tempProd, NDArray *output,
                         std::vector<int> const &axes) {
        //
    }

    template<typename T>
    static void reduceProductBPScalar_(NDArray *input, NDArray *epsilon, NDArray *tempProd, NDArray *output) {

    }

    void reduceProductBPScalar(NDArray *input, NDArray *epsilon, NDArray *tempProd, NDArray *output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), reduceProductBPScalar_, (input, epsilon, tempProd, output),
                              FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void reduceProductBPScalar_,
        (NDArray * input, NDArray * epsilon, NDArray * tempProd, NDArray * output), FLOAT_TYPES);
}
}
}