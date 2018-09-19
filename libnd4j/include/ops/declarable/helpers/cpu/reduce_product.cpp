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

    void reduceProductBP(NDArray* input, NDArray* epsilon, NDArray* tempProd, NDArray* output, std::vector<int> const& axes) {
        std::vector<int> dimensions; //(input->rankOf() - axes.size());

//#pragma omp parallel for if (input->rankOf() >  Environment::getInstance()->elementwiseThreshold()) schedule(static)        
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet> inList(input->allTensorsAlongDimension(dimensions));
//#pragma omp parallel for if (outList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static) 
        for (Nd4jLong e = 0; e < outList->size(); ++e) {
            outList->at(e)->assign(epsilon);
            outList->at(e)->applyPairwiseTransform(pairwise::Multiply, tempProd, nullptr);
            outList->at(e)->applyPairwiseTransform(pairwise::Divide, inList->at(e), nullptr);
        }
    }

}
}
}