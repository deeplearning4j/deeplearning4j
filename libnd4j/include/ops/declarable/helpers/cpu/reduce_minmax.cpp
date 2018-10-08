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
    void minMaxReduceFunctor(NDArray<T>* input, NDArray<T>* gradOut, NDArray<T>* tempVals, NDArray<T>* output, bool normalize) {
            if (tempVals->isScalar()) {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    T compared = (normalize?nd4j::math::nd4j_abs((*input)(e)):(*input)(e));
                    if (nd4j::math::nd4j_abs((*tempVals)(0.) - compared) < T(1.E-5f)) { // if input value equals to max
                         (*output)(e) = (normalize?(*gradOut)(0.) * nd4j::math::nd4j_sign((*input)(e)):(*gradOut)(0.));
                    }
                }
            }
            else {
                for (Nd4jLong e = 0; e < input->lengthOf(); e++) {
                    for (Nd4jLong j = 0; j < tempVals->lengthOf(); j++) {
                        T compared = (normalize?nd4j::math::nd4j_abs((*input)(e)):(*input)(e));
                        if (nd4j::math::nd4j_abs((*tempVals)(j) - compared) < T(1.E-5f))  // if input value equals to max
                            (*output)(e) = (normalize?(*gradOut)(j) * nd4j::math::nd4j_sign((*input)(e)):(*gradOut)(j));
                    }
                }
            }

    }

    template void minMaxReduceFunctor(NDArray<float>* input, NDArray<float>* gradOut, NDArray<float>* tempVals, NDArray<float>* output, bool normalize);
    template void minMaxReduceFunctor(NDArray<float16>* input, NDArray<float16>* gradOut, NDArray<float16>* tempVals, NDArray<float16>*  output, bool normalize);
    template void minMaxReduceFunctor(NDArray<double>* input, NDArray<double>* gradOut, NDArray<double>* tempVals, NDArray<double>* output, bool normalize);
}
}
}
