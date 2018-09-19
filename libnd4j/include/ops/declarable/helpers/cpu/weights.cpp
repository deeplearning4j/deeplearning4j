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

#include <ops/declarable/helpers/weights.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void adjustWeights_(NDArray* input, NDArray* weights, NDArray* output, int minLength, int maxLength) {
            for (int e = 0; e < input->lengthOf(); e++) {
                int val = input->getScalar<int>(e);
                if (val < maxLength) {
                    if (weights != nullptr)
                        output->putScalar(val, output->getScalar<T>(val) + weights->getScalar<T>(e));
                    else
                        output->putScalar(val, output->getScalar<T>(val) + 1);
                }
            }
    }

    void adjustWeights(NDArray* input, NDArray* weights, NDArray* output, int minLength, int maxLength) {
        BUILD_SINGLE_SELECTOR(output->dataType(), adjustWeights_, (input, weights, output, minLength, maxLength), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void adjustWeights_, (NDArray* input, NDArray* weights, NDArray* output, int minLength, int maxLength), LIBND4J_TYPES);
}
}
}