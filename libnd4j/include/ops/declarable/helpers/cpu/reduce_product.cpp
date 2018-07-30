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

#include <ResultSet.h>
#include <ops/declarable/helpers/reduce_norm.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceProductBP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempProd, NDArray<T>* output, std::vector<int> const& axes) {
        std::vector<int> dimensions; //(input->rankOf() - axes.size());

//#pragma omp parallel for if (input->rankOf() >  Environment::getInstance()->elementwiseThreshold()) schedule(static)        
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet<T>> inList(input->allTensorsAlongDimension(dimensions));
//#pragma omp parallel for if (outList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static) 
        for (Nd4jLong e = 0; e < outList->size(); ++e) {
            outList->at(e)->assign(epsilon);
            outList->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(tempProd, nullptr);
            outList->at(e)->template applyPairwiseTransform<simdOps::Divide<T>>(inList->at(e), nullptr);
        }
    }

    template void reduceProductBP(NDArray<float>* input, NDArray<float>* epsilon, NDArray<float>* tempProd, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceProductBP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* tempProd, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceProductBP(NDArray<double>* input, NDArray<double>* epsilon, NDArray<double>* tempProd, NDArray<double>* output, std::vector<int> const& axes);
}
}
}