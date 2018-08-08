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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/confusion.h>


namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void confusionFunctor(NDArray<T>* labels, NDArray<T>* predictions, NDArray<T>* weights, NDArray<T>* output) {
        std::unique_ptr<ResultSet<T>> arrs(output->allTensorsAlongDimension({1}));

#pragma omp parallel for if(labels->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)                    
        for (int j = 0; j < labels->lengthOf(); ++j){
            Nd4jLong label = (*labels)(j);
            Nd4jLong pred = (*predictions)(j);
            T value = (weights == nullptr ? (T)1.0 : (*weights)(j));
            (*arrs->at(label))(pred) = value;
        }
    }

    template void confusionFunctor(NDArray<float>* labels, NDArray<float>* predictions, NDArray<float>* weights, NDArray<float>* output);
    template void confusionFunctor(NDArray<float16>* labels, NDArray<float16>* predictions, NDArray<float16>* weights, NDArray<float16>* output);
    template void confusionFunctor(NDArray<double>* labels, NDArray<double>* predictions, NDArray<double>* weights, NDArray<double>* output);
    template void confusionFunctor(NDArray<int>* labels, NDArray<int>* predictions, NDArray<int>* weights, NDArray<int>* output);
    template void confusionFunctor(NDArray<Nd4jLong>* labels, NDArray<Nd4jLong>* predictions, NDArray<Nd4jLong>* weights, NDArray<Nd4jLong>* output);
}
}
}