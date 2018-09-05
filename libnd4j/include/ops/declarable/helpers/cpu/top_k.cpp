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
namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    int topKFunctor(NDArray<T>* input, NDArray<T>* values, NDArray<T>* indeces, int k, bool needSort) {
        int width = input->sizeAt(-1); // last dim of input
        std::unique_ptr<ResultSet<T>> lastDimList(input->allTensorsAlongDimension({input->rankOf() - 1}));

// ----------------------------------------------------------------------------------------------- //
// this assumption is right:
//        if (values->lengthOf() != k * lastDimList->size()) {
//            nd4j_printf("top_k: something is wrong. %i expected, but %i given.\n",
//                values->lengthOf(), k * lastDimList->size());
//        }
// ----------------------------------------------------------------------------------------------- //

            if (k == 1) {
                int pos = 0;
//#pragma omp parallel for if(lastDimList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (size_t e = 0; e < lastDimList->size(); ++e) {
                    int maxPos = lastDimList->at(e)->argMax();
                    if (indeces)
                        (*indeces)(e) = maxPos; //topIndex;
                    if (values)
                        (*values)(e) = (*lastDimList->at(e))(maxPos);
                }
            }
            else { 
                int nextPos = 0;

//#pragma omp parallel for if(lastDimList->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (int e = 0; e < lastDimList->size(); ++e) {
                    NDArray<T>* trial = lastDimList->at(e); // a vector to be search

                    std::vector<int> topIndices(k);
                    std::vector<T> topValues(k);

                    // fill up the first k elements
                    for (int pos = 0; pos < k; ++pos) {
                        topIndices[pos] = pos;
                        topValues[pos] = (*trial)(pos);
                    }
                    std::vector<T> sortedVals(topValues);
                    std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                    
                    for (int i = k; i < width; ++i) {
                        T val = (*trial)(i);
                        if (sortedVals[0] < val) { // value should be inserted to top k
                            // only if it is not contained in
                            auto itPos = std::find(sortedVals.begin(), sortedVals.end(), val);
                            if (sortedVals.end() == itPos) {
                                //exchangePos - a distance between begin and minimal existed to be suppressed by val
                                auto exchangePos = std::distance(topValues.begin(), std::find(topValues.begin(), topValues.end(), sortedVals[0]));
                                topValues[exchangePos] = val; //*exchangeIt = val;
                                topIndices[exchangePos] = i;
                                sortedVals[0] = val; // suppress in sorted
                                std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                            }
                        }
                    }

                    if (needSort) {
                        std::sort(topValues.begin(), topValues.end(), [](T a, T b) {
                            return a > b;   
                        });

                        for (int j = 0; j < width; j++)
                            for (int pos = 0; pos < k; ++pos)
                                if (topValues[pos] == (*trial)(j))
                                    topIndices[pos] = j;
                    }
                    else { // else sort by indices

                        std::vector<std::pair<int, T>> data(topValues.size());
                        for (size_t e = 0; e < topValues.size(); ++e) {
                            data[e].first = topIndices[e];
                            data[e].second = topValues[e];
                        }

                        std::sort(data.begin(), data.end(), [](std::pair<int, T> const& a, std::pair<int, T> const& b) {
                            return a.first < b.first;
                        });

                        for (size_t e = 0; e < topValues.size(); ++e) {
                            topIndices[e] = data[e].first;
                            topValues[e] = data[e].second;
                        }

                    }

                    for (int pos = 0; pos < k; ++pos, ++nextPos) {
                        if (values != nullptr)
                            (*values)(nextPos) = topValues[pos];

                        (*indeces)(nextPos) = topIndices[pos];
                    }
                }
        }
        return ND4J_STATUS_OK;
    }
// ----------------------------------------------------------------------------------------------- //

    template <typename T>
    int inTopKFunctor(NDArray<T>* input, NDArray<T>* target, NDArray<T>* result, int k) {

            std::vector<Nd4jLong> shapeV(input->rankOf() + 1);
            for (int i = 0; i < input->rankOf(); i++)
                shapeV[i] = input->sizeAt(i);
            shapeV[input->rankOf()] = k;
            std::unique_ptr<NDArray<T>> indices( new NDArray<T>(input->ordering(), shapeV));
            NDArray<T>* values = nullptr;
            int status = topKFunctor(input, values, indices.get(), k, true);

            if (status == ND4J_STATUS_OK) {
#pragma omp parallel for if(target->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                for (int e = 0; e < target->lengthOf(); e++) {
                    bool found = false;
                    for (int j = 0; j < k; j++) {
                        if ((*target)(e) == (*indices)(e * k + j)) {
                            found = true;
                            break;
                        }
                    }
                    if (found)
                        (*result)(e) = (T)1.f;
                }
            }
            return status; 

    }
    template int topKFunctor<float>(NDArray<float>* input, NDArray<float>* values, NDArray<float>* indeces, int k, bool needSort);
    template int topKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* values, NDArray<float16>* indeces, int k, bool needSort);
    template int topKFunctor<double>(NDArray<double>* input, NDArray<double>* values, NDArray<double>* indeces, int k, bool needSort);
    template int inTopKFunctor<float>(NDArray<float>* input, NDArray<float>* target, NDArray<float>* result, int k);
    template int inTopKFunctor<float16>(NDArray<float16>* input, NDArray<float16>* target, NDArray<float16>* result, int k);
    template int inTopKFunctor<double>(NDArray<double>* input, NDArray<double>* target, NDArray<double>* result, int k);

}
}
}