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
// Created by george on 05.04.18.
//
#include <ops/declarable/helpers/dynamic.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            void dynamicPartitionFunctor(NDArray<T> const* input, NDArray<T> const* indices, std::vector<NDArray<T>*>& outputList) {
                std::vector<std::pair<NDArray<T> *, int>> outputs(outputList.size());
                int sourceDimsLen = input->rankOf() - indices->rankOf();
                if (sourceDimsLen) {
                    std::vector<int> sourceDims(sourceDimsLen);

//#pragma omp parallel for if(sourceDims.size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (int i = sourceDimsLen; i > 0; i--)
                        sourceDims[sourceDimsLen - i] = input->rankOf() - i;

                    std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(sourceDims));

//#pragma omp parallel for if(outputList.size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (unsigned int i = 0; i < outputList.size(); i++) {
                        outputs[i].first = outputList[i];
                        std::vector<int> outDims(outputs[i].first->rankOf() - 1);

//#pragma omp parallel for if(outputs[i].first->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                        for (int k = 1; k < outputs[i].first->rankOf(); k++)
                            outDims[k - 1] = k;

                        std::unique_ptr<ResultSet<T>> listOutForCurrent(
                                outputs[i].first->allTensorsAlongDimension(outDims));

                        outputs[i].second = 0;

//#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                        for (int e = 0; e < indices->lengthOf(); ++e)
                            if ((*indices)(e) == T(i))
                                listOutForCurrent->at(outputs[i].second++)->assign(listOfTensors->at(e));
                    }

                } else
//#pragma omp parallel for if(outputList.size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (unsigned int i = 0; i < outputList.size(); i++) {
                        outputs[i].first = outputList[i];
                        outputs[i].second = 0;
                        for (int e = 0; e < indices->lengthOf(); ++e)
                            if ((*indices)(e) == T(i))
                                outputs[i].first->putScalar(outputs[i].second++, (*input)(e));
                    }
            }
            template <typename T>
            int dynamicStitchFunctor(std::vector<NDArray<T>*> const& inputs, std::vector<NDArray<T>*> const& indices, NDArray<T>* output){

                int numOfData = inputs.size();

                if (output->isVector()) {
                    for (int e = 0; e < numOfData; e++) {
                        NDArray<T>* data = inputs[e];
                        NDArray<T>* index = indices[e];
                        for (int i = 0; i < index->lengthOf(); i++) {
                            int pos = (*index)(i);
                            if (pos < 0) {
                                nd4j_printf("dynamic_stitch: Index value should be non-negative. But %i was given", pos);
                                return ND4J_STATUS_VALIDATION;
                            }
                            if (pos >= output->lengthOf()) {
                                nd4j_printf("dynamic_stitch: Index should be less than %i. But %i was given",
                                            output->lengthOf(), pos);
                                return ND4J_STATUS_VALIDATION;
                            }
                            (*output)(pos) = (*data)(i);
                        }
                    }
                }
                else {
                    std::vector<int> restDims(output->rankOf() - 1);
                    for (int i = restDims.size(); i > 0;  i--)
                        restDims[restDims.size() - i] = output->rankOf() - i;

                    std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

                    for (int e = 0; e < numOfData; e++) {
                        NDArray<T>* data = inputs[e];
                        NDArray<T>* index = indices[e];
                        std::vector<int> sourceDims(data->rankOf() - index->rankOf());
                        for (int i = sourceDims.size(); i > 0;  i--)
                            sourceDims[sourceDims.size() - i] = data->rankOf() - i;

                        std::unique_ptr<ResultSet<T>> listOfTensors(data->allTensorsAlongDimension(sourceDims));

                        for (int i = 0; i < index->lengthOf(); i++) {
                            int pos = (*index)(i);
                            if (pos < 0) {
                                nd4j_printf("dynamic_stitch: Index value should be non-negative. But %i was given", pos);
                                return ND4J_STATUS_VALIDATION;
                            }
                            if (pos >= output->lengthOf()) {
                                nd4j_printf("dynamic_stitch: Index should be less than %i. But %i was given",
                                         output->lengthOf(), pos);
                                return ND4J_STATUS_VALIDATION;
                            }

                            listOfOutTensors->at(pos)->assign(listOfTensors->at(i));
                        }
                    }
                }
                return ND4J_STATUS_OK;
            }

            template void dynamicPartitionFunctor(NDArray<float> const* input, NDArray<float> const* indices, std::vector<NDArray<float>*>& outputList);
            template void dynamicPartitionFunctor(NDArray<float16> const* input, NDArray<float16> const* indices, std::vector<NDArray<float16>*>& outputList);
            template void dynamicPartitionFunctor(NDArray<double> const* input, NDArray<double> const* indices, std::vector<NDArray<double>*>& outputList);

            template int dynamicStitchFunctor(std::vector<NDArray<float>*> const& inputs, std::vector<NDArray<float>*> const& indices, NDArray<float>* output);
            template int dynamicStitchFunctor(std::vector<NDArray<float16>*> const& inputs, std::vector<NDArray<float16>*> const& indices, NDArray<float16>* output);
            template int dynamicStitchFunctor(std::vector<NDArray<double>*> const& inputs, std::vector<NDArray<double>*> const& indices, NDArray<double>* output);

            template <typename T>
            void dynamicPartitionFunctorBP(NDArray<T>const* input, NDArray<T>const* indices, std::vector<NDArray<T>*> const& inputGradientList, std::vector<NDArray<T>*>& outputList) {
                std::vector<NDArray<T>*> inputGradientListY(inputGradientList.size());
                for (size_t e = 0; e < inputGradientList.size(); e++) {
                    inputGradientListY[e] = inputGradientList[e]->dup('c');
                    inputGradientListY[e]->printShapeInfo("inputGradientListY");
                }

                dynamicPartitionFunctor(input, indices, inputGradientListY);
                dynamicStitchFunctor(inputGradientList, inputGradientListY, outputList[0]);
                //dynamicStitchFunctor(inputGradientListX, inputGradientListY, outputList[1]);
                for (size_t e = 0; e < inputGradientListY.size(); e++)
                    delete inputGradientListY[e];

            }

            template <typename T>
            int dynamicStitchFunctorBP(std::vector<NDArray<T>*> const& inputs, std::vector<NDArray<T>*> const& indices, NDArray<T> const* gradInput, std::vector<NDArray<T>*>& outputList){

                return ND4J_STATUS_OK;
            }

            template void dynamicPartitionFunctorBP(NDArray<float> const* input, NDArray<float> const* indices, std::vector<NDArray<float>*> const& inputGradientList, std::vector<NDArray<float>*>& outputList);
            template void dynamicPartitionFunctorBP(NDArray<float16> const* input, NDArray<float16> const* indices, std::vector<NDArray<float16>*> const& inputGradientList, std::vector<NDArray<float16>*>& outputList);
            template void dynamicPartitionFunctorBP(NDArray<double> const* input, NDArray<double> const* indices, std::vector<NDArray<double>*> const& inputGradientList, std::vector<NDArray<double>*>& outputList);

            template int dynamicStitchFunctorBP(std::vector<NDArray<float>*> const& inputs, std::vector<NDArray<float>*> const& indices, NDArray<float> const* gradInput, std::vector<NDArray<float>*>& outputList);
            template int dynamicStitchFunctorBP(std::vector<NDArray<float16>*> const& inputs, std::vector<NDArray<float16>*> const& indices, NDArray<float16> const* gradInput, std::vector<NDArray<float16>*>& outputList);
            template int dynamicStitchFunctorBP(std::vector<NDArray<double>*> const& inputs, std::vector<NDArray<double>*> const& indices, NDArray<double> const* gradInput, std::vector<NDArray<double>*>& outputList);

        }
    }
}

