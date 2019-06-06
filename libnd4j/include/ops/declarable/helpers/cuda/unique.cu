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

#include <ops/declarable/helpers/unique.h>
#include <Status.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static Nd4jLong uniqueCount_(NDArray* input) {
        Nd4jLong count = input->lengthOf();
        return count;
    }

    Nd4jLong uniqueCount(nd4j::LaunchContext * context, NDArray* input) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return uniqueCount_, (input), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template Nd4jLong uniqueCount_, (NDArray* input), LIBND4J_TYPES);


    template <typename T>
    static Nd4jStatus uniqueFunctor_(NDArray* input, NDArray* values, NDArray* indices, NDArray* counts) {
        return Status::OK();
    }

    Nd4jStatus uniqueFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* values, NDArray* indices, NDArray* counts) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return uniqueFunctor_,(input, values, indices, counts), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template Nd4jStatus uniqueFunctor_, (NDArray* input, NDArray* values, NDArray* indices, NDArray* counts), LIBND4J_TYPES);

}
}
}