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
#include <ops/declarable/helpers/reduce_dot.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceDotBP(NDArray<T>* inputX, NDArray<T>* inputY, NDArray<T>* epsilon, NDArray<T>* outputX, NDArray<T>* outputY, std::vector<int> const& axes) {
//                std::unique_ptr<ResultSet<T>> outList(output->allTensorsAlongDimension(dimensions));
                std::vector<int> dimensions; //(input->rankOf() - axes.size());
                for (Nd4jLong e = 0; e < inputX->rankOf(); e++) {
                    if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                        dimensions.emplace_back(e);
                    }
                }
                std::unique_ptr<ResultSet<T>> outListX(outputX->allTensorsAlongDimension(dimensions));
                std::unique_ptr<ResultSet<T>> outListY(outputY->allTensorsAlongDimension(dimensions));
                std::unique_ptr<ResultSet<T>> yList(inputY->allTensorsAlongDimension(dimensions));
                std::unique_ptr<ResultSet<T>> xList(inputX->allTensorsAlongDimension(dimensions));
                //output->
#pragma omp parallel for if (outListX->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (Nd4jLong e = 0; e < outListX->size(); ++e) {
                    outListX->at(e)->assign(epsilon);
                    outListY->at(e)->assign(epsilon);
                    outListX->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(yList->at(e), outListX->at(e), nullptr);
                    outListY->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(xList->at(e), outListY->at(e), nullptr);
                }

    }

    template void reduceDotBP(NDArray<float>* inputX,  NDArray<float>* inputY,   NDArray<float>* epsilon, NDArray<float>* outputX, NDArray<float>* outputY, std::vector<int> const& axes);
    template void reduceDotBP(NDArray<float16>* inputX,NDArray<float16>* inputY, NDArray<float16>* epsilon, NDArray<float16>* outputX, NDArray<float16>* outputY, std::vector<int> const& axes);
    template void reduceDotBP(NDArray<double>* inputX, NDArray<double>* inputY,  NDArray<double>* epsilon, NDArray<double>* outputX, NDArray<double>* outputY, std::vector<int> const& axes);
}
}
}