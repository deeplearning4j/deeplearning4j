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
#include <ops/declarable/helpers/reduce_product.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    void reduceNorm1BP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet<T>> inList(input->allTensorsAlongDimension(dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            auto norm1Backprop = LAMBDA_TT(_x, _e) {
                return (_x >= T(0.f) ?_e:-_e);
            };
            inList->at(e)->applyPairwiseLambda(epsilon, norm1Backprop, outList->at(e));
        }
    }

    template <typename T>
    void reduceNorm2BP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet<T>> inList(input->allTensorsAlongDimension(dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(inList->at(e), outList->at(e), nullptr);
            outList->at(e)->template applyPairwiseTransform<simdOps::Divide<T>>(tempNorm, outList->at(e), nullptr);
        }
    }

    template <typename T>
    void reduceSquareNormBP(NDArray<T>* input, NDArray<T>* epsilon, NDArray<T>* tempNorm, NDArray<T>* output, std::vector<int> const& axes) {

        std::vector<int> dimensions; //(input->rankOf() - axes.size());
        for (Nd4jLong e = 0; e < input->rankOf(); e++) {
            if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                dimensions.emplace_back(e);
            }
        }
        std::unique_ptr<ResultSet<T>> outList(output->allTensorsAlongDimension(dimensions));
        std::unique_ptr<ResultSet<T>> inList(input->allTensorsAlongDimension(dimensions));
        for (int e = 0; e < outList->size(); ++e) {
            outList->at(e)->assign(T(2.f));
            outList->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(epsilon, outList->at(e), nullptr);
            outList->at(e)->template applyPairwiseTransform<simdOps::Multiply<T>>(inList->at(e), outList->at(e), nullptr);
        }
    }

    template void reduceNorm1BP(NDArray<float>* input, NDArray<float>* epsilon, NDArray<float>* tempNorm, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceNorm1BP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* tempNorm, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceNorm1BP(NDArray<double>* input, NDArray<double>* epsilon, NDArray<double>* tempNorm, NDArray<double>* output, std::vector<int> const& axes);
    template void reduceNorm1BP(NDArray<int>* input, NDArray<int>* epsilon, NDArray<int>* tempNorm, NDArray<int>* output, std::vector<int> const& axes);
    template void reduceNorm1BP(NDArray<Nd4jLong>* input, NDArray<Nd4jLong>* epsilon, NDArray<Nd4jLong>* tempNorm, NDArray<Nd4jLong>* output, std::vector<int> const& axes);

    template void reduceNorm2BP(NDArray<float>* input, NDArray<float>* epsilon, NDArray<float>* tempNorm, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceNorm2BP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* tempNorm, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceNorm2BP(NDArray<double>* input, NDArray<double>* epsilon, NDArray<double>* tempNorm, NDArray<double>* output, std::vector<int> const& axes);
    template void reduceNorm2BP(NDArray<int>* input, NDArray<int>* epsilon, NDArray<int>* tempNorm, NDArray<int>* output, std::vector<int> const& axes);
    template void reduceNorm2BP(NDArray<Nd4jLong>* input, NDArray<Nd4jLong>* epsilon, NDArray<Nd4jLong>* tempNorm, NDArray<Nd4jLong>* output, std::vector<int> const& axes);

    template void reduceSquareNormBP(NDArray<float>* input, NDArray<float>* epsilon, NDArray<float>* tempNorm, NDArray<float>* output, std::vector<int> const& axes);
    template void reduceSquareNormBP(NDArray<float16>* input, NDArray<float16>* epsilon, NDArray<float16>* tempNorm, NDArray<float16>* output, std::vector<int> const& axes);
    template void reduceSquareNormBP(NDArray<double>* input, NDArray<double>* epsilon, NDArray<double>* tempNorm, NDArray<double>* output, std::vector<int> const& axes);
    template void reduceSquareNormBP(NDArray<int>* input, NDArray<int>* epsilon, NDArray<int>* tempNorm, NDArray<int>* output, std::vector<int> const& axes);
    template void reduceSquareNormBP(NDArray<Nd4jLong>* input, NDArray<Nd4jLong>* epsilon, NDArray<Nd4jLong>* tempNorm, NDArray<Nd4jLong>* output, std::vector<int> const& axes);

}
}
}