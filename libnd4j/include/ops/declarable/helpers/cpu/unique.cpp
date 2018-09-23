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
        Nd4jLong count = 0;

        std::vector<T> values;

        for (int e = 0; e < input->lengthOf(); e++) {
            T v = input->getScalar<T>(e);
            if (std::find(values.begin(), values.end(), v) == values.end()) {
                values.push_back(v);
                count++;
            }
        }
        return count;
    }

    Nd4jLong uniqueCount(NDArray* input) {
        BUILD_SINGLE_SELECTOR(input->dataType(), return uniqueCount_, (input), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template Nd4jLong uniqueCount_, (NDArray* input), LIBND4J_TYPES);


    template <typename T>
    static Nd4jLong uniqueFunctor_(NDArray* input, NDArray* values, NDArray* indices, NDArray* counts) {
    
        std::vector<T> valuesVector;
        std::map<T, int> indicesMap;
        std::map<T, int> countsMap;

        for (int e = 0; e < input->lengthOf(); e++) {
            T v = input->getScalar<T>(e);
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
            values->putScalar(e, static_cast<T>(valuesVector[e]));
            if (counts != nullptr) 
                counts->putScalar(e, countsMap[valuesVector[e]]);
        }

//#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
        for (int e = 0; e < indices->lengthOf(); e++) {
            auto posI = std::find(valuesVector.begin(), valuesVector.end(), input->getScalar<T>(e));
            auto dist = std::distance(valuesVector.begin(), posI);
            indices->putScalar(e, Nd4jLong(dist));//indicesMap[(*input)(e)];
        }

        return Status::OK();
    }

    Nd4jLong uniqueFunctor(NDArray* input, NDArray* values, NDArray* indices, NDArray* counts) {
        BUILD_SINGLE_SELECTOR(input->dataType(), uniqueFunctor_,(input, values, indices, counts), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template Nd4jLong uniqueFunctor_, (NDArray* input, NDArray* values, NDArray* indices, NDArray* counts), LIBND4J_TYPES);
}
}
}