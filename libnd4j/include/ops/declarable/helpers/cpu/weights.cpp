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
    void adjustWeights(NDArray<T>* input, NDArray<T>* weights, NDArray<T>* output, int minLength, int maxLength) {
            for (int e = 0; e < input->lengthOf(); e++) {
                int val = static_cast<int>((*input)(e));
                if (val < maxLength) {
                    if (weights != nullptr)
                        (*output)(val) += (*weights)(e);
                    else
                        (*output)(val)++;
                }
            }
    }

    template void adjustWeights(NDArray<float>* input, NDArray<float>* weights, NDArray<float>* output, int minLength, int maxLength);
    template void adjustWeights(NDArray<float16>* input, NDArray<float16>* weights, NDArray<float16>* output, int minLength, int maxLength);
    template void adjustWeights(NDArray<double>* input, NDArray<double>* weights, NDArray<double>* output, int minLength, int maxLength);
}
}
}