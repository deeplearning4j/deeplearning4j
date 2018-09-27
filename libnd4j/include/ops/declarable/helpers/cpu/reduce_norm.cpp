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
//#include <ops/declarable/helpers/reduce_product.h>
#include <ops/declarable/helpers/legacy_helpers.h>

namespace nd4j {
namespace ops {
namespace helpers {

    void reduceNorm1BP(NDArray* input, NDArray* epsilon, NDArray* tempNorm, NDArray* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet> inList(input->allTensorsAlongDimension(dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            //inList->at(e)->applyPairwiseTransform(pairwise::ReduceNorm1E, epsilon, outList->at(e), nullptr);
            helpers::reduceNorm1(inList->at(e), epsilon, outList->at(e));
        }
    }

    void reduceNorm2BP(NDArray* input, NDArray* epsilon, NDArray* tempNorm, NDArray* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet> inList(input->allTensorsAlongDimension(dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            epsilon->applyPairwiseTransform(pairwise::Multiply, inList->at(e), outList->at(e), nullptr);
            outList->at(e)->applyPairwiseTransform(pairwise::Divide, tempNorm, outList->at(e), nullptr);
        }
    }

    void reduceSquareNormBP(NDArray* input, NDArray* epsilon, NDArray* tempNorm, NDArray* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet> inList(input->allTensorsAlongDimension(dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            outList->at(e)->assign(2.f);
            outList->at(e)->applyPairwiseTransform(pairwise::Multiply, epsilon, outList->at(e), nullptr);
            outList->at(e)->applyPairwiseTransform(pairwise::MinPairwise, inList->at(e), outList->at(e), nullptr);
        }
    }
}
}
}