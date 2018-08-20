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
    void adjustAxis(NDArray<T>* input, NDArray<T>* axisVector, std::vector<int>& output) {
        output.resize(axisVector->lengthOf());

        for (int e = 0; e < axisVector->lengthOf(); e++) {
                    int ca = (int) (*axisVector)(e);
                    if (ca < 0)
                        ca += input->rankOf();

                    output[e] = ca;
            }
    }

    template void adjustAxis(NDArray<float>* input, NDArray<float>* axisVector, std::vector<int>& output);
    template void adjustAxis(NDArray<float16>* input, NDArray<float16>* axisVector, std::vector<int>& output);
    template void adjustAxis(NDArray<double>* input, NDArray<double>* axisVector, std::vector<int>& output);
}
}
}