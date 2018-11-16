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
    void nthElementFunctor_(NDArray* input, NDArray* nVal, NDArray* output) {
        Nd4jLong n = nVal->e<Nd4jLong>(0);
        if (input->isVector()) {
            std::vector<T> data(input->lengthOf());
            //memcpy(&data[0], input->getBuffer(), sizeof(T) * data.size());
            size_t l = 0;
            for (size_t l = 0; l < data.size(); ++l)
                data[l] = input->e<T>(l);
            auto nthPos = data.begin();
            nthPos += n;
            std::nth_element(data.begin(), nthPos, data.end());
            output->p<T>(0, data[n]);
        }
        else { // rank greater than 1
            std::vector<int> lastDims({input->rankOf() - 1});
            std::unique_ptr<ResultSet> rows(input->allTensorsAlongDimension(lastDims));
            for (Nd4jLong e = 0; e < output->lengthOf(); e++) {
                auto row = rows->at(e);
                std::vector<T> internalData(row->lengthOf());
                //T* internalP = const_cast<T*>(internalData.data());
                for (size_t l = 0; l < internalData.size(); ++l)
                    internalData[l] = input->e<T>(l);

                //memcpy(&internalData[0], row->getBuffer(), sizeof(T) * internalData.size());
                auto nthPos = internalData.begin();
                nthPos += n;
                std::nth_element(internalData.begin(), nthPos, internalData.end());
                output->p<T>(e, internalData[n]);
            }
        }
    }
    void nthElementFunctor(NDArray* input, NDArray* n, NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), nthElementFunctor_, (input, n, output), LIBND4J_TYPES);

    }
    BUILD_SINGLE_TEMPLATE(template void nthElementFunctor_, (NDArray* input, NDArray* n, NDArray* output), LIBND4J_TYPES);
    
}
}
}
