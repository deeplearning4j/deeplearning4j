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

#include <ops/declarable/helpers/nth_element.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void nthElementFunctor(NDArray<T>* input, int n, NDArray<T>* output) {
        if (input->isVector()) {
            std::vector<T> data(input->lengthOf());
            memcpy(&data[0], input->getBuffer(), sizeof(T) * data.size());
            std::nth_element(data.begin(), data.begin() + n, data.end());
            output->putScalar(0, data[n]);
        }
        else { // rank greater than 1
            std::vector<int> lastDims({input->rankOf() - 1});
            std::unique_ptr<ResultSet<T>> rows(input->allTensorsAlongDimension(lastDims));
            for (Nd4jLong e = 0; e < output->lengthOf(); e++) {
                auto row = rows->at(e);
                std::vector<T> data(row->lengthOf());
                memcpy(&data[0], row->getBuffer(), sizeof(T) * data.size());
                std::nth_element(data.begin(), data.begin() + n, data.end());
                output->putScalar(e, data[n]);
            }
        }
    }

    template void nthElementFunctor(NDArray<float>* input, int n, NDArray<float>* output);
    template void nthElementFunctor(NDArray<float16>* input, int n, NDArray<float16>* output);
    template void nthElementFunctor(NDArray<double>* input, int n, NDArray<double>* output);
}
}
}
