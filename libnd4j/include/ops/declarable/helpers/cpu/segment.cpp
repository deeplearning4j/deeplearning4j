/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/segment.h>
#include <helpers/ShapeUtils.h>
#include <execution/Threads.h>
#include <unordered_map>

namespace sd {
    namespace ops {
        namespace helpers {

            // segment max
            template <typename T>
            static void segmentMaxFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
                //int numClasses = output->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                Nd4jLong idx = indices->e<Nd4jLong>(0);
                if (input->isVector() || input->isScalar()) {
                    T val = input->e<T>(0);

                    for (Nd4jLong e = 1; e < indices->lengthOf(); e++) {
                        if (idx == indices->e<Nd4jLong>(e)) {
                            // max
                            val = sd::math::nd4j_max<T>(val, input->t<T>(e));
                        }
                        else {
                            idx = indices->e<Nd4jLong>(e);
                            val = input->t<T>(e);
                        }
                        output->r<T>(idx) = val;
                    }
                }
                else {
                    std::vector<int> restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});
                    auto listOfTensors = input->allTensorsAlongDimension(restDims);
                    auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    auto numOfClasses = output->sizeAt(0); // number of classes
                    std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
                    auto maxT = listOfOutTensors.at(idx);

                    //int pos = 0;
                    maxT->assign(listOfTensors.at(0));

                    for (Nd4jLong i = 1; i < indices->lengthOf(); i++) {
                        if (indices->e<int>(i) == idx) {

                            for (Nd4jLong e = 0; e < maxT->lengthOf(); e++) {
                                maxT->r<T>(e) = sd::math::nd4j_max(maxT->t<T>(e), listOfTensors.at(i)->t<T>(e));
                            }
                        }
                        else {
                            idx = indices->e<Nd4jLong>(i);
                            maxT = listOfOutTensors.at(idx);
                            maxT->assign(listOfTensors.at(i));
                        }

                    }
                }
            }

            // segmen min
            template <typename T>
            static void segmentMinFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
                //int numClasses = output->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                Nd4jLong idx = indices->e<Nd4jLong>(0);
                if (input->isVector() || input->isScalar()) {
                    T val = input->e<T>(0);

                    for (Nd4jLong e = 1; e < indices->lengthOf(); e++) {
                        if (idx == indices->e<Nd4jLong>(e)) {
                            // min
                            val = sd::math::nd4j_min<T>(val, input->t<T>(e));
                        }
                        else {
                            idx = indices->e<Nd4jLong>(e);
                            val = input->t<T>(e);
                        }
                        output->r<T>(idx) = val;
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    int numOfClasses = output->sizeAt(0); // number of classes
                    std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
                    auto minT = listOfOutTensors.at(idx);

                    int pos = 0;
                    minT->assign(listOfTensors.at(0));

                    for (Nd4jLong i = 1; i < indices->lengthOf(); i++) {
                        if (indices->e<Nd4jLong>(i) == idx) {

                            for (Nd4jLong e = 0; e < minT->lengthOf(); e++) {
                                minT->p(e, sd::math::nd4j_min(minT->e<T>(e), listOfTensors.at(i)->e<T>(e)));
                            }
                        }
                        else {
                            idx = indices->e<Nd4jLong>(i);
                            minT = listOfOutTensors.at(idx);
                            minT->assign(listOfTensors.at(i));
                        }
                    }
                }
            }

            // segmen mean
            template <typename T>
            static void segmentMeanFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
                int numClasses = output->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                int idx = indices->e<int>(0);
                if (input->isVector() || input->isScalar()) {
                    T val = T(0.f);
                    int count = 0;

                    for (Nd4jLong e = 0; e < indices->lengthOf(); e++) {
                        if (idx == indices->e<int>(e)) {
                            // mean
                            val += input->e<T>(e);
                            count++;
                        }
                        else {
                            output->p<T>(idx, val / count);
                            idx = indices->e<int>(e);
                            val = input->e<T>(e);
                            count = 1;
                        }
                        output->p<T>(idx, val / count);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    auto listOfTensors = input->allTensorsAlongDimension(restDims);
                    auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    int numOfClasses = output->sizeAt(0); // number of classes
                    std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
                    auto meanT = listOfOutTensors.at(idx);
                    int count = 1;
                    auto meanV = meanT->dup();
                    meanV.assign(listOfTensors.at(0));

                    for (Nd4jLong i = 1; i < indices->lengthOf(); i++) {
                        if (indices->e<int>(i) == idx) {
                            auto func = PRAGMA_THREADS_FOR {
                                for (auto e = start; e < stop; e++) {
                                    meanV.p<T>(e, meanV.e<T>(e) + listOfTensors.at(i)->e<T>(e));
                                }
                            };
                            samediff::Threads::parallel_for(func, 0, meanT->lengthOf());

                            count++;
                        }
                        else {
                            //meanT->assign(meanV);
                            meanV.applyScalar(scalar::Divide, count, *meanT);
                            idx = indices->e<int>(i);
                            meanT = listOfOutTensors.at(idx);
                            meanV.assign(listOfTensors.at(i));
                            count = 1;
                        }
                        meanV.applyScalar(scalar::Divide, count, *meanT);
                    }
                }
            }

            template <typename T>
            static void segmentSumFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
                int numClasses = output->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                int idx = indices->e<int>(0);
                if (input->isVector() || input->isScalar()) {
                    T val = T(0.f);
                    int count = 0;
                    for (Nd4jLong e = 0; e < indices->lengthOf(); e++) {
                        if (idx == indices->e<int>(e)) {
                            // sum
                            val += input->t<T>(e);
                        }
                        else {
                            idx = indices->e<int>(e);
                            val = input->t<T>(e);
                        }
                        output->p(idx, val);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    auto listOfTensors = input->allTensorsAlongDimension(restDims);
                    auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    int numOfClasses = output->sizeAt(0); // number of classes
                    std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
                    auto sumT = listOfOutTensors.at(idx);

                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        if (indices->e<int>(i) == idx) {
                            auto func = PRAGMA_THREADS_FOR {
                                for (auto e = start; e < stop; e++) {
                                    sumT->p(e, sumT->e<T>(e) + listOfTensors.at(i)->e<T>(e));
                                }
                            };
                            samediff::Threads::parallel_for(func, 0, sumT->lengthOf());
                        }
                        else {
                            idx = indices->e<int>(i);
                            sumT = listOfOutTensors.at(idx);
                            sumT->assign(listOfTensors.at(i));
                        }
                    }
                }
            }

            template <typename T>
            static void segmentProdFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
                //int numClasses = output->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                int idx = indices->e<int>(0);
                output->assign(1.f);
                if (input->isVector() || input->isScalar()) {
                    T val = input->e<T>(0);
                    int count = 0;

                    for (Nd4jLong e = 1; e < indices->lengthOf(); e++) {
                        if (idx == indices->e<int>(e)) {
                            // sum
                            val *= input->e<T>(e);
                        }
                        else {
                            idx = indices->e<int>(e);
                            val = input->e<T>(e);
                        }
                        output->p(idx, val);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    auto listOfTensors = input->allTensorsAlongDimension(restDims);
                    auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    int numOfClasses = output->sizeAt(0); // number of classes
                    auto sumT = listOfOutTensors.at(idx);
                    sumT->assign(listOfTensors.at(0));
                    for (Nd4jLong i = 1; i < indices->lengthOf(); i++) {
                        if (indices->e<int>(i)  == idx) {
                            auto func = PRAGMA_THREADS_FOR {
                                for (auto e = start; e < stop; e++) {
                                    sumT->p(e, sumT->e<T>(e) * listOfTensors.at(i)->e<T>(e));
                                }
                            };
                            samediff::Threads::parallel_for(func, 0, sumT->lengthOf());
                        }
                        else {
                            idx = indices->e<int>(i);
                            sumT = listOfOutTensors.at(idx);
                            sumT->assign(listOfTensors.at(i));
                        }
                    }
                }
            }

//    template <typename T>
//    static bool segmentIndicesValidate_(NDArray* indices, NDArray& aexpected, NDArray& anOutput) {
//      }

            void segmentMaxFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), segmentMaxFunctor_, (input, indices, output), LIBND4J_TYPES);
            }

            void segmentMinFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), segmentMinFunctor_, (input, indices, output), LIBND4J_TYPES);
            }

            void segmentMeanFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), segmentMeanFunctor_, (input, indices, output), LIBND4J_TYPES);
            }

            void segmentSumFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), segmentSumFunctor_, (input, indices, output), LIBND4J_TYPES);
            }

            void segmentProdFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), segmentProdFunctor_, (input, indices, output), LIBND4J_TYPES);
            }

            bool segmentIndicesValidate(sd::LaunchContext * context, NDArray* indices, NDArray& expected, NDArray& output) {
                auto val = indices->e(0);
                for (Nd4jLong e = 1; e < indices->lengthOf(); e++) {
                    output = indices->e(e);
                    if (val.e<Nd4jLong>(0) > output.e<Nd4jLong>(0))
                        return false;
                    val = indices->e(e);
                }

                return true;
            }

            //BUILD_SINGLE_TEMPLATE(template bool segmentIndicesValidate_, (NDArray*, NDArray&, NDArray&), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void segmentProdFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void segmentSumFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void segmentMeanFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void segmentMinFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
            BUILD_SINGLE_TEMPLATE(template void segmentMaxFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
            // -------------------------------------------------------------------------------------------------------------- //
            // Unsorted segment ops
            // -------------------------------------------------------------------------------------------------------------- //

            bool unsortedSegmentIndicesValidate(sd::LaunchContext * context, NDArray* indices, Nd4jLong expected, Nd4jLong& output) {
                Nd4jLong val = indices->e<Nd4jLong>(0);

                Nd4jLong maxInd = indices->argMax();
                if (indices->e<Nd4jLong>(maxInd) >= expected) {
                    output = val;
                    return false;
                }
                output = expected;
                return true;
            }

            template <typename T>
            static void unsortedSegmentMaxFunctor_(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

                // if input is a vector: (as if in doc sample)
                //int idx = static_cast<int>((*indices)(0.));
                MAP_IMPL<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
                    idxs[indices->e<Nd4jLong>(e)].push_back(e);


                //std::sort(idxs.begin(), idxs.end());

                if (input->isVector() || input->isScalar()) { // 1D case
                    T maxVal = DataTypeUtils::max<T>();
                    output->assign(-maxVal);

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        T val = input->e<T>(fi->second.at(0));
                        for (Nd4jLong idx = 1; idx < static_cast<Nd4jLong>(fi->second.size()); ++idx) {
                            val = sd::math::nd4j_max(val, input->e<T>(fi->second.at(idx)));
                        }
                        output->p(fi->first, val);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    T maxVal = DataTypeUtils::max<T>();
                    output->assign(-maxVal);

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        auto outputT = listOfOutTensors.at(fi->first);
                        outputT->assign(listOfTensors.at(fi->second.at(0)));
                        for (Nd4jLong idx = 0; idx < listOfTensors.size(); ++idx) {
                               if(idx >= fi->second.size() ||  fi->second.size() < 2 ||  fi->second.at(idx) >= listOfTensors.size()) {
                                   continue;
                            }

                            auto maxT = listOfTensors.at(fi->second.at(idx));
                            for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                                T val = sd::math::nd4j_max(maxT->e<T>(e), outputT->e<T>(e));

                                outputT->p(e, val);
                            }
                        }


                    }


                }
            }
            void unsortedSegmentMaxFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMaxFunctor_, (input, indices, numOfClasses, output), NUMERIC_TYPES);
            }
            BUILD_SINGLE_TEMPLATE(template void unsortedSegmentMaxFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

            template <typename T>
            static void unsortedSegmentMinFunctor_(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                // if input is a vector: (as if in doc sample)
                //int idx = static_cast<int>((*indices)(0.));
                MAP_IMPL<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());

                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
                    idxs[indices->e<Nd4jLong>(e)].push_back(e);

                //std::sort(idxs.begin(), idxs.end());

                if (input->isVector() || input->isScalar()) { // 1D case
                    T maxVal = DataTypeUtils::max<T>();
                    output->assign(maxVal);

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        T val = input->t<T>(fi->second.at(0));

                        for (size_t idx = 1; idx < fi->second.size(); ++idx) {
                            val = sd::math::nd4j_min(val, input->t<T>(fi->second.at(idx)));
                        }
                        output->r<T>(fi->first) = val;
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    T maxVal = DataTypeUtils::max<T>();
                    output->assign(maxVal);

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        auto outputT = listOfOutTensors.at(fi->first);
                        outputT->assign(listOfTensors.at(fi->second.at(0)));
                        for (size_t idx = 1; idx < fi->second.size(); ++idx) {
                            auto minT = listOfTensors.at(fi->second.at(idx));

                            for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                                outputT->r<T>(e) = sd::math::nd4j_min(minT->t<T>(e), outputT->t<T>(e));
                            }
                        }
                        //outputT->assign(maxT);
                    }
                }

            }
            void unsortedSegmentMinFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMinFunctor_, (input, indices, numOfClasses, output),
                                      NUMERIC_TYPES);
            }

            BUILD_SINGLE_TEMPLATE(template void unsortedSegmentMinFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

            void unsortedSegmentMeanFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                MAP_IMPL<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
                    idxs[indices->e<Nd4jLong>(e)].push_back(e);

                //std::sort(idxs.begin(), idxs.end());

                if (input->isVector() || input->isScalar()) { // 1D case

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        double sumValue = input->e<double>(fi->second.at(0));
                        int loop_size = fi->second.size();

                        // FIXME: parallelism here?
                        for (size_t idx = 1; idx < loop_size; ++idx) {
                            sumValue += input->e<double>(fi->second.at(idx));
                        }

                        output->p(fi->first, sumValue / fi->second.size());
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    // FIXME: parallelism here?
                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        auto outputT = listOfOutTensors.at(fi->first);
                        outputT->assign(listOfTensors.at(fi->second.at(0)));
                        Nd4jLong loopSize = fi->second.size();

                        for (Nd4jLong idx = 1; idx < loopSize; ++idx) {
                            auto current = listOfTensors.at(fi->second.at(idx));
                            *outputT += *current;
                        }
                        (*outputT) /= double(fi->second.size());
                    }
                }
            }

            void unsortedSegmentSumFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                MAP_IMPL<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
                    idxs[indices->e<Nd4jLong>(e)].push_back(e);

                if (input->isVector() || input->isScalar()) { // 1D case

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        double sumValue = input->e<double>(fi->second.at(0));
                        Nd4jLong loop_size = fi->second.size();

                        // FIXME: parallelism here?
                        for (Nd4jLong idx = 1; idx < loop_size; ++idx) {
                            sumValue += input->e<double>(fi->second.at(idx));
                        }
                        output->p(fi->first, sumValue);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        auto outputT = listOfOutTensors.at(fi->first);
                        outputT->assign(listOfTensors.at(fi->second.at(0)));
                        Nd4jLong loop_size = fi->second.size();

                        // FIXME: parallelism here?
                        for (Nd4jLong idx = 1; idx < loop_size; ++idx) {
                            auto current = listOfTensors.at(fi->second.at(idx));
                            *(outputT) += *current;
                        }
                        //outputT->assign(maxT);
                    }
                }
            }

            template <typename T>
            void unsortedSegmentProdFunctor_(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                MAP_IMPL<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
                    idxs[indices->e<Nd4jLong>(e)].push_back(e);

                //std::sort(idxs.begin(), idxs.end());

                output->assign(1.f);

                if (input->isVector() || input->isScalar()) { // 1D case
                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        T prodValue = input->e<T>(fi->second.at(0));
                        for (size_t idx = 1; idx < fi->second.size(); ++idx) {
                            prodValue *= input->e<T>(fi->second.at(idx));
                        }
                        output->p(fi->first, prodValue);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        auto outputT = listOfOutTensors.at(fi->first);
                        outputT->assign(listOfTensors.at(fi->second.at(0)));
                        for (size_t idx = 1; idx < fi->second.size(); ++idx) {
                            auto current = listOfTensors.at(fi->second.at(idx));

                            *outputT *= *current;
                        }
                    }
                }
            }

            void unsortedSegmentProdFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentProdFunctor_, (input, indices, numOfClasses, output), NUMERIC_TYPES);
            }
            BUILD_SINGLE_TEMPLATE(template void unsortedSegmentProdFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

            void unsortedSegmentSqrtNFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
                MAP_IMPL<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
                    idxs[indices->e<Nd4jLong>(e)].push_back(e);

                //std::sort(idxs.begin(), idxs.end());

                if (input->isVector() || input->isScalar()) { // 1D case
                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        double sumValue = input->e<double>(fi->second.at(0));
                        for (size_t idx = 1; idx < fi->second.size(); ++idx) {
                            sumValue += input->e<double>(fi->second.at(idx));
                        }
                        output->p(fi->first, sumValue / sd::math::nd4j_sqrt<Nd4jLong, double>(fi->second.size()));
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                        auto outputT = listOfOutTensors.at(fi->first);
                        outputT->assign(listOfTensors.at(fi->second.at(0)));
                        for (size_t idx = 1; idx < fi->second.size(); ++idx) {
                            auto current = listOfTensors.at(fi->second.at(idx));
                            *outputT += *current;
                        }
                        //outputT->assign(maxT);
                        (*outputT) /= sd::math::nd4j_sqrt<size_t, double>(fi->second.size());
                    }
                }
            }

            // -------------------------------------------------------------------------------------------------------------- //
            // Backpropagate ops helpers
            // -------------------------------------------------------------------------------------------------------------- //
            // Sorted backpropagate ops
            //
            // segment max
            template <typename T>
            int segmentMaxFunctorBP_(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
                //int numOfClasses = gradOut->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                auto tempRes = gradOut->dup();
                segmentMaxFunctor_<T>(input, indices, &tempRes);
                if (input->isVector() || input->isScalar()) {
                    Nd4jLong loop_size = input->lengthOf();

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto classNum = indices->e<Nd4jLong>(e);
                            if (sd::math::nd4j_abs(tempRes.e<T>(classNum) - input->e<T>(e)) <= T(1.e-6))
                                output->p(e, gradOut->e<T>(classNum));
                        }
                    };
                    samediff::Threads::parallel_for(func, 0, loop_size);
                }
                else {
                    std::vector<int> restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfBPTensors = tempRes.allTensorsAlongDimension(restDims);
                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    //int numOfClasses = tempRes.sizeAt(0); // number of classes
                    //std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto i = start; i < stop; i++) {
                            auto classNum = indices->e<Nd4jLong>(i);
                            auto current = listOfTensors.at(i);
                            auto currentOut = listOfOutTensors.at(i);
                            auto currentGradOut = listOfGradOuts.at(classNum);

                            for (Nd4jLong e = 0; e < current->lengthOf(); e++) {
                                if (sd::math::nd4j_abs(listOfBPTensors.at(classNum)->e<T>(e) - current->e<T>(e)) <= T(1.e-6))
                                    currentOut->p(e, currentGradOut->e<T>(e));
                            }
                        }
                    };

                    samediff::Threads::parallel_tad(func, 0, indices->lengthOf());
                }

                return ND4J_STATUS_OK;
            }

            int segmentMaxFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
                BUILD_SINGLE_SELECTOR(output->dataType(), return segmentMaxFunctorBP_, (context, input, indices, gradOut, output), NUMERIC_TYPES);
            }
            BUILD_SINGLE_TEMPLATE(template int segmentMaxFunctorBP_, (sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output), NUMERIC_TYPES);

            // segmen min
            int segmentMinFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
                NDArray tempRes = gradOut->dup();
                segmentMinFunctor(context, input, indices, &tempRes);
                if (input->isVector() || input->isScalar()) {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto classNum = indices->e<Nd4jLong>(e);
                            if (sd::math::nd4j_abs(tempRes.e<double>(classNum) - input->e<double>(e)) < 1.e-5)
                                output->p(e, gradOut->e<double>(classNum));
                        }
                    };
                    samediff::Threads::parallel_for(func, 0, input->lengthOf());
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfBPTensors = tempRes.allTensorsAlongDimension(restDims);
                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    //int numOfClasses = tempRes.sizeAt(0); // number of classes
                    //std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
                    output->assign(0.);
                    int pos = 0;

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto i = start; i < stop; i++) {
                            auto classNum = indices->e<Nd4jLong>(i);
                            auto current = listOfTensors.at(i);
                            auto currentOut = listOfOutTensors.at(i);
                            auto currentGradOut = listOfGradOuts.at(classNum);

                            for (Nd4jLong e = 0; e < current->lengthOf(); e++) {
                                if (sd::math::nd4j_abs(listOfBPTensors.at(classNum)->e<double>(e) - current->e<double>(e)) <
                                    1.e-5)
                                    currentOut->p(e, currentGradOut->e<double>(e));
                            }
                        }
                    };

                    samediff::Threads::parallel_tad(func, 0, indices->lengthOf());
                }
                return ND4J_STATUS_OK;
            }

            // segmen mean
            int segmentMeanFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
                int numClasses = output->sizeAt(0);
                MAP_IMPL<Nd4jLong, Nd4jLong> classCount;//(numClasses);

                for (Nd4jLong count = 0; count < numClasses; ++count) {
                    classCount[count] = 0;
                }

                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                    classCount[indices->e<Nd4jLong>(e)] ++;
                }

                // if input is a vector: (as if in doc sample)
                if (input->isVector() || input->isScalar()) {
                    for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(e);
                        output->p(e, gradOut->e<double>(classNum) / classCount[classNum]);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);
                    ;

                    int pos = 0;
                    //auto func = [&](uint64_t thread_id, uint64_t start, uint64_t stop, uint64_t increment) -> void {
                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        auto classNum = indices->e<Nd4jLong>(i);
                        auto current = listOfTensors.at(i);
                        auto currentOut = listOfOutTensors.at(i);
                        auto currentGradOut = listOfGradOuts.at(classNum);

                        for (Nd4jLong e = 0; e < current->lengthOf(); e++) {
                            currentOut->p(e, currentGradOut->e<double>(e) / classCount.at(classNum));
                        }
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }
                return ND4J_STATUS_OK;
            }

            int segmentSumFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
//        int numClasses = output->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                Nd4jLong idx = indices->e<Nd4jLong>(0);
                if (input->isVector() || input->isScalar()) {
                    for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(e);
                        output->p(e, gradOut->e<double>(classNum));
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    //auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        auto classNum = indices->e<Nd4jLong>(i);
                        auto current = listOfTensors.at(i);
                        auto currentOut = listOfOutTensors.at(i);
                        auto currentGradOut = listOfGradOuts.at(classNum);

                        currentOut->assign(currentGradOut);
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }
                return Status::OK();
            }

            int segmentProdFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
                auto tempRes = gradOut->dup();
                segmentProdFunctor(context, input, indices, &tempRes);
                if (input->isVector() || input->isScalar()) {
                    for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(e);
                        output->p(e, gradOut->e<double>(classNum) * tempRes.e<double>(classNum)/ input->e<double>(e));
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfBPTensors = tempRes.allTensorsAlongDimension(restDims);
                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    //int numOfClasses = tempRes.sizeAt(0); // number of classes
                    //std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);

                    //auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        auto classNum = indices->e<Nd4jLong>(i);
                        auto current = listOfTensors.at(i);
                        auto currentOut = listOfOutTensors.at(i);
                        auto currentGradOut = listOfGradOuts.at(classNum);
                        auto currentFFOut = listOfBPTensors.at(classNum);

                        currentOut->assign((*currentFFOut) * (*currentGradOut) / (*current));
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }

                return ND4J_STATUS_OK;
            }

            // -------------------------------------------------------------------------------------------------------------- //
            // Unsorted backpropagate segment ops
            // -------------------------------------------------------------------------------------------------------------- //

            template <typename T>
            static int unsortedSegmentMaxFunctorBP_(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
//        int numOfClasses = gradOut->sizeAt(0);
                // if input is a vector: (as if in doc sample)
                auto tempRes = gradOut->dup();
                unsortedSegmentMaxFunctor(context, input, indices, numOfClasses, &tempRes);
                if (input->isVector() || input->isScalar()) {

                    for (Nd4jLong e = 0; e < input->lengthOf(); ++e) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(e);
                        if (sd::math::nd4j_abs(tempRes.e<double>(classNum) - input->e<double>(e)) < 1.e-5)
                            output->p(e, gradOut->e<T>(classNum));
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfBPTensors = tempRes.allTensorsAlongDimension(restDims);
                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(i);
                        NDArray* current = listOfTensors.at(i);
                        NDArray* currentOut = listOfOutTensors.at(i);
                        NDArray* currentGradOut = listOfGradOuts.at(classNum);
                        for (int e = 0; e < current->lengthOf(); e++) {
                            if (sd::math::nd4j_abs(listOfBPTensors.at(classNum)->e<double>(e) - current->e<double>(e)) < 1.e-5)
                                currentOut->p(e, currentGradOut->e<T>(e));
                        }
                    }
                }

                return ND4J_STATUS_OK;
            }

            int unsortedSegmentMaxFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
                BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMaxFunctorBP_, (context, input, indices, gradOut, numOfClasses, output), NUMERIC_TYPES);
            }
            BUILD_SINGLE_TEMPLATE(template int unsortedSegmentMaxFunctorBP_, (sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

            template <typename T>
            static int unsortedSegmentMinFunctorBP_(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
                auto tempRes = gradOut->dup();
                unsortedSegmentMinFunctor(context, input, indices, numOfClasses, &tempRes);
                if (input->isVector() || input->isScalar()) {

                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto classNum = indices->e<Nd4jLong>(e);
                            if (sd::math::nd4j_abs(tempRes.t<T>(classNum) - input->t<T>(e)) < 1.e-6)
                                output->r<T>(e) = gradOut->t<T>(classNum);
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, input->lengthOf());
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfBPTensors = tempRes.allTensorsAlongDimension(restDims);
                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    //auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        auto classNum = indices->e<Nd4jLong>(i);
                        auto current = listOfTensors.at(i);
                        auto currentOut = listOfOutTensors.at(i);
                        auto currentGradOut = listOfGradOuts.at(classNum);

                        for (Nd4jLong e = 0; e < current->lengthOf(); e++) {
                            if (sd::math::nd4j_abs(listOfBPTensors.at(classNum)->t<T>(e) - current->t<T>(e)) < 1.e-6)
                                currentOut->r<T>(e) = currentGradOut->t<T>(e);
                        }
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }

                return ND4J_STATUS_OK;
            }

            int unsortedSegmentMinFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
                BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMinFunctorBP_, (context, input, indices, gradOut, numOfClasses, output), NUMERIC_TYPES);
            }
            BUILD_SINGLE_TEMPLATE(template int unsortedSegmentMinFunctorBP_, (sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

            int unsortedSegmentMeanFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {

                MAP_IMPL<Nd4jLong, Nd4jLong> classCount;//(numClasses);

                for (Nd4jLong count = 0; count < numOfClasses; ++count) {
                    classCount[count] = 0;
                }

                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                    classCount[indices->e<Nd4jLong>(e)]++;
                }

                // if input is a vector: (as if in doc sample)
                if (input->isVector() || input->isScalar()) {
                    for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(e);
                        output->p(e, gradOut->e<double>(classNum) / classCount[classNum]);
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(i);
                        NDArray* current = listOfTensors.at(i);
                        NDArray* currentOut = listOfOutTensors.at(i);
                        NDArray* currentGradOut = listOfGradOuts.at(classNum);
                        currentOut->assign(*currentGradOut / double(classCount[classNum]));
                    }
                }
                return ND4J_STATUS_OK;
            }

            int unsortedSegmentSumFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {

                // if input is a vector: (as if in doc sample)
                Nd4jLong idx = indices->e<Nd4jLong>(0);
                if (input->isVector() || input->isScalar()) {
                    for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                        Nd4jLong classNum = indices->e<Nd4jLong>(e);
                        output->p(e, gradOut->e<double>(classNum));
                    }
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    //auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        auto classNum = indices->e<Nd4jLong>(i);
                        auto currentOut = listOfOutTensors.at(i);
                        auto currentGradOut = listOfGradOuts.at(classNum);

                        currentOut->assign(currentGradOut);
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }
                return Status::OK();
            }

            int unsortedSegmentProdFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {

                auto tempRes = gradOut->dup();
                unsortedSegmentProdFunctor(context, input, indices, numOfClasses, &tempRes);
                if (input->isVector() || input->isScalar()) {
                    auto func = PRAGMA_THREADS_FOR {
                        for (auto e = start; e < stop; e++) {
                            auto classNum = indices->e<Nd4jLong>(e);
                            output->p<double>(e, gradOut->e<double>(classNum) * tempRes.e<double>(classNum) / input->e<double>(e));
                        }
                    };

                    samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfBPTensors = tempRes.allTensorsAlongDimension(restDims);
                    ResultSet listOfGradOuts = gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors = input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors = output->allTensorsAlongDimension(restDims);

                    //auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        auto classNum = indices->e<Nd4jLong>(i);
                        auto current = listOfTensors.at(i);
                        auto currentOut = listOfOutTensors.at(i);
                        auto currentGradOut = listOfGradOuts.at(classNum);
                        auto currentFFOut = listOfBPTensors.at(classNum);

                        currentOut->assign((*currentFFOut) * (*currentGradOut) / (*current));
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }

                return Status::OK();
            }

//    template <typename T>
            int unsortedSegmentSqrtNFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
                MAP_IMPL<Nd4jLong, Nd4jLong> classCount;//(numClasses);

                for (Nd4jLong count = 0; count < numOfClasses; ++count) {
                    classCount[count] = 0;
                }

                for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                    classCount[indices->e<Nd4jLong>(e)]++;
                }

                // if input is a vector: (as if in doc sample)
                if (input->isVector() || input->isScalar()) {
                    //auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong e = 0; e < indices->lengthOf(); e++) {
                        auto classNum = indices->e<Nd4jLong>(e);
                        output->p(e, gradOut->e<double>(classNum) / sd::math::nd4j_sqrt<double, double>(classCount[classNum]));
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }
                else {
                    auto restDims = ShapeUtils::evalDimsToExclude(input->rankOf(), {0});

                    ResultSet listOfGradOuts  =gradOut->allTensorsAlongDimension(restDims);
                    ResultSet listOfTensors  =input->allTensorsAlongDimension(restDims);
                    ResultSet listOfOutTensors  =output->allTensorsAlongDimension(restDims);

                    //auto func = PRAGMA_THREADS_FOR {
                    for (Nd4jLong i = 0; i < indices->lengthOf(); i++) {
                        auto classNum = indices->e<Nd4jLong>(i);
                        auto current = listOfTensors.at(i);
                        auto currentOut = listOfOutTensors.at(i);
                        auto currentGradOut = listOfGradOuts.at(classNum);

                        for (int e = 0; e < current->lengthOf(); e++) {
                            currentOut->p<double>(e, currentGradOut->e<double>(e) / sd::math::nd4j_sqrt<double, double>(classCount[classNum]));
                        }
                    }
                    //};

                    //samediff::Threads::parallel_for(func, 0, indices->lengthOf());
                }
                return Status::OK();
            }

        }
    }
}
