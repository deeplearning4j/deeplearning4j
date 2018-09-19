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

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static void _minMaxReduceFunctor(NDArray* input, NDArray* gradOut, NDArray* tempVals, NDArray* output, bool normalize) {
            if (tempVals->isScalar()) {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    T compared = (normalize?nd4j::math::nd4j_abs(input->getScalar<T>(e)):input->getScalar<T>(e));
                    if (nd4j::math::nd4j_abs(tempVals->getScalar<T>(0) - compared) < T(1.E-5f)) { // if input value equals to max
                         output->putScalar<T>(e, (normalize ? gradOut->getScalar<T>(0) * nd4j::math::nd4j_sign(input->getScalar<T>(e)):gradOut->getScalar<T>(0)));
                    }
                }
            }
            else {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    for (Nd4jLong j = 0; j < tempVals->lengthOf(); j++) {
                        T compared = (normalize?nd4j::math::nd4j_abs(input->getScalar<T>(e)):input->getScalar<T>(e));
                        if (nd4j::math::nd4j_abs(tempVals->getScalar<T>(j) - compared) < T(1.E-5f))  // if input value equals to max
                            output->putScalar(e, (normalize ? gradOut->getScalar<T>(j) * nd4j::math::nd4j_sign(input->getScalar<T>(e)): gradOut->getScalar<T>(j)));
                    }
                }
            }

    }

    void minMaxReduceFunctor(NDArray* input, NDArray* gradOut, NDArray* tempVals, NDArray* output, bool normalize) {
        BUILD_SINGLE_SELECTOR(gradOut->dataType(), _minMaxReduceFunctor, (input, gradOut, tempVals, output, normalize), FLOAT_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _minMaxReduceFunctor, (NDArray* input, NDArray* gradOut, NDArray* tempVals, NDArray* output, bool normalize), FLOAT_TYPES);
}
}
}
