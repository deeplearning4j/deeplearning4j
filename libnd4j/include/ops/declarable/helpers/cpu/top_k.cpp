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
//  @author raver119@gmail.com
//

#include <ops/declarable/helpers/top_k.h>
#include <ops/declarable/headers/parity_ops.h>
#include <NDArrayFactory.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    static int topKFunctor_(NDArray* input, NDArray* values, NDArray* indeces, int k, bool needSort) {
        int width = input->sizeAt(-1);
        std::unique_ptr<ResultSet> lastDimList(input->allTensorsAlongDimension({input->rankOf() - 1}));

// ----------------------------------------------------------------------------------------------- //
// this assumption is right:
//        if (values->lengthOf() != k * lastDimList->size()) {
//            nd4j_printf("top_k: something is wrong. %i expected, but %i given.\n",
//                values->lengthOf(), k * lastDimList->size());
//        }
// ----------------------------------------------------------------------------------------------- //

            if (k == 1) {
                int pos = 0;
#pragma omp parallel for if(lastDimList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (int e = 0; e < lastDimList->size(); ++e) {
                    int maxPos = lastDimList->at(e)->argMax();
                    indeces->putScalar(e, maxPos); //topIndex;
                    values->putScalar(e, lastDimList->at(e)->e<T>(maxPos));
                }
            }
            else { 
                int nextPos = 0;

//#pragma omp parallel for if(lastDimList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (int e = 0; e < lastDimList->size(); ++e) {
                    auto trial = lastDimList->at(e); // a vector to be search

                    std::vector<int> topIndices(k);
                    std::vector<T> topValues(k);

                    // fill up the first k elements
                    for (int pos = 0; pos < k; ++pos) {
                        topIndices[pos] = pos;
                        topValues[pos] = trial->e<T>(pos);
                    }
                    std::vector<T> sortedVals(topValues);
                    std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                    
                    for (int i = k; i < width; ++i) {
                        T val = trial->e<T>(i);
                        if (sortedVals[0] < val) { // value should be inserted to top k
                            // only if it is not contained in 
                            if (sortedVals.end() == std::find(sortedVals.begin(), sortedVals.end(), val)) {    
                                // exchangePos - a distance between begin and minimal existed to be suppressed by val
                                auto exchangePos = std::distance(topValues.begin(), std::find(topValues.begin(), topValues.end(), sortedVals[0]));
                                topValues[exchangePos] = val;
                                sortedVals[0] = val; // suppress in sorted
                                std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                            }
                        }
                    }

                    if (needSort) {
                        std::sort(topValues.begin(), topValues.end(), [](T a, T b) {
                            return a > b;   
                        });
                    }

                    for (int j = 0; j < width; j++)
                        for (int pos = 0; pos < k; ++pos)
                            if (topValues[pos] == trial->e<T>(j))
                                topIndices[pos] = j;

                    for (int pos = 0; pos < k; ++pos, ++nextPos) {
                        if (values != nullptr)
                            values->putScalar(nextPos, topValues[pos]);

                        indeces->putScalar(nextPos, topIndices[pos]);
                    }
                }
        }
        return Status::OK();
    }
// ----------------------------------------------------------------------------------------------- //

    template <typename T>
    static int inTopKFunctor_(NDArray* input, NDArray* target, NDArray* result, int k) {

            std::vector<Nd4jLong> shapeV(input->rankOf() + 1);
            for (int i = 0; i < input->rankOf(); i++)
                shapeV[i] = input->sizeAt(i);
            shapeV[input->rankOf()] = k;
            std::unique_ptr<NDArray> indices(NDArrayFactory::create<T>(input->ordering(), shapeV));
            NDArray* values = nullptr;
            int status = topKFunctor(input, values, indices.get(), k, true);

            if (status == ND4J_STATUS_OK) {
#pragma omp parallel for if(target->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (int e = 0; e < target->lengthOf(); e++) {
                    bool found = false;
                    for (int j = 0; j < k; j++) {
                        if (target->e<T>(e) == indices->e<T>(e * k + j)) {
                            found = true;
                            break;
                        }
                    }
                    if (found)
                        result->putScalar<T>(e, (T)1);
                }
            }
            return status; 

    }

        int topKFunctor(NDArray* input, NDArray* values, NDArray* indeces, int k, bool needSort) {
            BUILD_SINGLE_SELECTOR(input->dataType(), return topKFunctor_, (input, values, indeces, k, needSort), NUMERIC_TYPES);
        }

        int inTopKFunctor(NDArray* input, NDArray* target, NDArray* result, int k) {
            BUILD_SINGLE_SELECTOR(input->dataType(), return inTopKFunctor_, (input, target, result, k), NUMERIC_TYPES);
        }

        BUILD_SINGLE_TEMPLATE(template int topKFunctor_, (NDArray* input, NDArray* values, NDArray* indeces, int k, bool needSort), NUMERIC_TYPES);
        BUILD_SINGLE_TEMPLATE(template int inTopKFunctor_, (NDArray* input, NDArray* target, NDArray* result, int k), NUMERIC_TYPES);
}
}
}