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
#include <array/NDArrayFactory.h>
#include <execution/Threads.h>

namespace sd {
namespace ops {
namespace helpers {

    template <typename T>
    static int topKFunctor_(const NDArray* input, NDArray* values, NDArray* indices, const uint k, bool needSort) {
        Nd4jLong width = input->sizeAt(-1);
        int lastDim = input->rankOf() - 1;
// ----------------------------------------------------------------------------------------------- //
// this assumption is right:
//        if (values->lengthOf() != k * lastDimList->size()) {
//            nd4j_printf("top_k: something is wrong. %i expected, but %i given.\n",
//                values->lengthOf(), k * lastDimList->size());
//        }
// ----------------------------------------------------------------------------------------------- //
        std::vector<int> dimsToExclude(input->rankOf() - 1);
        for (size_t d = 0; d < dimsToExclude.size(); ++d)
            dimsToExclude[d] = d;

        const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(input->shapeInfo(), dimsToExclude);

            if (k == 1) {
                for (Nd4jLong e = 0; e < numOfSubArrs; ++e) {
                    auto trial = (*input)(e, dimsToExclude);
                    //int maxPos = //lastDimList->at(e)->argMax();
                    Nd4jLong maxPos = 0;
                    //trial.printIndexedBuffer("TRIAL:");
                    T maxVal = trial.e<T>(0);
                    for (Nd4jLong pos = 1; pos < trial.lengthOf(); pos++)
                        if (maxVal < trial.e<T>(pos)) {
                            maxPos = pos;
                            maxVal = trial.e<T>(pos);
                        }
                    if (indices)
                        indices->p(e, maxPos); //topIndex;
                    if (values)
                        values->p(e, maxVal);
                }
            }
            else {
                int nextPos = 0;

                for (Nd4jLong e = 0; e < numOfSubArrs; ++e) {
                    auto trial = (*input)(e, dimsToExclude);

                    // fill up the first k elements
                    NDArray topValues = NDArrayFactory::create<T>('c', {k}, input->getContext());
                    NDArray sortedVals = NDArrayFactory::create<T>('c', {k}, input->getContext());
                    NDArray topIndices = NDArrayFactory::create<Nd4jLong>('c', {k}, input->getContext());
                    for (uint pos = 0; pos < k; ++pos) {
                        topIndices.t<Nd4jLong>(pos) = pos;
                        topValues.t<T>(pos) = trial.t<T>(pos);
                    }
                    //std::vector<T> sortedVals(topValues);
                    sortedVals.assign(topValues);// = NDArrayFactory::create<T>('c', {k});
                    //std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                    SpecialMethods<T>::sortGeneric(sortedVals.buffer(), sortedVals.shapeInfo(), false);
                    for (Nd4jLong i = static_cast<Nd4jLong>(k); i < width; ++i) {
                        T val = trial.e<T>(i);
                        T minTopVal = sortedVals.t<T>(0);
                        if (minTopVal < val) { // value should be inserted to top k
                            // only if it is not contained in
                            T* begin = reinterpret_cast<T*>(sortedVals.buffer());
                            T* end = begin + k;
                            bool exists = std::binary_search(begin, end, val);
                            if (!exists) {
                                //exchangePos - a distance between begin and minimal existed to be suppressed by val
                                T* topBegin = reinterpret_cast<T*>(topValues.buffer());
                                T* topEnd = topBegin + k;
                                auto exchangePos = std::distance(topBegin, std::find(topBegin, topEnd, sortedVals.t<T>(0)));
                                topValues.t<T>(exchangePos) = val; //*exchangeIt = val;
                                topIndices.t<Nd4jLong>(exchangePos) = i;
                                sortedVals.t<T>(0) = val; // suppress in sorted
                                //std::sort(sortedVals.begin(), sortedVals.end()); // sorted in ascending order
                                SpecialMethods<T>::sortGeneric(sortedVals.buffer(), sortedVals.shapeInfo(), false);
                            }
                        }
                    }
                    if (needSort) {
                        SpecialMethods<T>::sortGeneric(topValues.buffer(), topValues.shapeInfo(), true);

                        for (Nd4jLong j = 0; j < width; j++)
                            for (uint pos = 0; pos < k; ++pos)
                                if (topValues.t<T>(pos) == trial.t<T>(j))
                                    topIndices.t<Nd4jLong>(pos) = j;
                    }
                    else { // else sort by indices
                        std::map<Nd4jLong, T> sortValsMap;
                        //std::vector<std::pair<int, T>> data(topValues.lengthOf());
                        for (Nd4jLong e = 0; e < topValues.lengthOf(); ++e) {
                            sortValsMap[topIndices.t<Nd4jLong>(e)] = topValues.t<T>(e);
                        }

                        //std::sort(data.begin(), data.end(), [](std::pair<int, T> const& a, std::pair<int, T> const& b) {
                        //    return a.first < b.first;
                        //});
                        Nd4jLong e = 0;
                        for (auto it = sortValsMap.begin(); it != sortValsMap.end(); ++it, e++) {
                            topIndices.t<Nd4jLong>(e) = it->first;
                            topValues.t<T>(e) = it->second;
                        }

                    }
                    if (values)
                    (*values)(e, dimsToExclude).assign(topValues);
                    if (indices)
                    (*indices)(e, dimsToExclude).assign(topIndices);
                }
                //indices->printIndexedBuffer("Indices as is");
        }
        return Status::OK();
    }
// ----------------------------------------------------------------------------------------------- //

    template <typename T>
    static int inTopKFunctor_(sd::LaunchContext* context, const NDArray* input, const NDArray* target, NDArray* result, const uint k) {

            std::vector<Nd4jLong> shapeI(input->rankOf());
            for (int i = 0; i < input->rankOf() - 1; i++)
                shapeI[i] = input->sizeAt(i);
            shapeI[input->rankOf() - 1] = k;
            std::unique_ptr<NDArray> indices(NDArrayFactory::create_<Nd4jLong>(input->ordering(), shapeI, context));
            NDArray* values = nullptr;
            int status = topKFunctor(context, input, values, indices.get(), k, true);
            result->assign(0);
            if (status == ND4J_STATUS_OK) {
                auto func = PRAGMA_THREADS_FOR {
                    for (auto e = start; e < stop; e++) {
                        bool found = false;
                        for (uint j = 0; j < k; j++) {
                            if (target->e<Nd4jLong>(e) == indices->e<Nd4jLong>(e * k + j)) {
                                found = true;
                                break;
                            }
                        }
                        if (found)
                            result->p<bool>(e, true);
                    }
                };

                samediff::Threads::parallel_tad(func, 0, target->lengthOf());
            }
            return status;

    }

        int topKFunctor(sd::LaunchContext * context, const NDArray* input, NDArray* values, NDArray* indices, const uint k, bool needSort) {
            BUILD_SINGLE_SELECTOR(input->dataType(), return topKFunctor_, (input, values, indices, k, needSort), NUMERIC_TYPES);
        }

        int inTopKFunctor(sd::LaunchContext * context, const NDArray* input, const NDArray* target, NDArray* result, const uint k) {
            BUILD_SINGLE_SELECTOR(input->dataType(), return inTopKFunctor_, (context, input, target, result, k), NUMERIC_TYPES);
        }

        BUILD_SINGLE_TEMPLATE(template int topKFunctor_, (const NDArray* input, NDArray* values, NDArray* indices, const uint k, bool needSort), NUMERIC_TYPES);
        BUILD_SINGLE_TEMPLATE(template int inTopKFunctor_, (sd::LaunchContext * context, const NDArray* input, const NDArray* target, NDArray* result, const uint k), NUMERIC_TYPES);
}
}
}
