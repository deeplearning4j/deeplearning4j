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

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int uniqueCount(NDArray<T>* input) {
        int count = 0;

        std::vector<T> values;

        for (int e = 0; e < input->lengthOf(); e++) {
            T v = (*input)(e);
            if (std::find(values.begin(), values.end(), v) == values.end()) {
                values.push_back(v);
                count++;
            }
        }
        return count;
    }

    template int uniqueCount(NDArray<float>* input);
    template int uniqueCount(NDArray<float16>* input);
    template int uniqueCount(NDArray<double>* input);
    template int uniqueCount(NDArray<int>* input);
    template int uniqueCount(NDArray<Nd4jLong>* input);


    template <typename T>
    int uniqueFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indices, NDArray<T>* counts) { 
    
        std::vector<T> valuesVector;
        std::map<T, int> indicesMap;
        std::map<T, int> countsMap;

        for (int e = 0; e < input->lengthOf(); e++) {
            T v = (*input)(e);
            if (std::find(valuesVector.begin(), valuesVector.end(), v) == valuesVector.end()) {
                valuesVector.push_back(v);
                indicesMap[v] = e;
                countsMap[v] = 1;
            }
            else {
                countsMap[v]++;
            }
        }

#pragma omp parallel for if(values->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
        for (int e = 0; e < values->lengthOf(); e++) {
            (*values)(e) = valuesVector[e];
            if (counts != nullptr) 
                (*counts)(e) = countsMap[valuesVector[e]];
        }

#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
        for (int e = 0; e < indices->lengthOf(); e++) {
            (*indices)(e) = indicesMap[(*input)(e)];
        }

        return ND4J_STATUS_OK;
    }

    template int uniqueFunctor(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indices, NDArray<float>* counts);
    template int uniqueFunctor(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indices, NDArray<float16>* counts);
    template int uniqueFunctor(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indices, NDArray<double>* counts);
    template int uniqueFunctor(NDArray<int>* input, NDArray<int>* values, NDArray<int>* indices, NDArray<int>* counts);
    template int uniqueFunctor(NDArray<Nd4jLong>* input, NDArray<Nd4jLong>* values, NDArray<Nd4jLong>* indices, NDArray<Nd4jLong>* counts);

}
}
}