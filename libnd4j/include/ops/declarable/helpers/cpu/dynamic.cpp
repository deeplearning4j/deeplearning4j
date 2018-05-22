//
// Created by george on 05.04.18.
//
#include <ops/declarable/helpers/dynamic.h>
#include <NDArrayFactory.h>
//#include <op_boilerplate.h>

namespace nd4j {
    namespace ops {
        namespace helpers {

            template <typename T>
            void dynamicPartitionFunctor(NDArray<T>* input, NDArray<T>* indices, std::vector<NDArray<T>*>& outputList) {
                std::vector<std::pair<NDArray<T> *, int>> outputs(outputList.size());
                int sourceDimsLen = input->rankOf() - indices->rankOf();
                if (sourceDimsLen) {
                    std::vector<int> sourceDims(sourceDimsLen);

#pragma omp parallel for if(sourceDims.size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (int i = sourceDimsLen; i > 0; i--)
                        sourceDims[sourceDimsLen - i] = input->rankOf() - i;

                    std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(input, sourceDims));

#pragma omp parallel for if(outputList.size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (unsigned int i = 0; i < outputList.size(); i++) {
                        outputs[i].first = outputList[i];
                        std::vector<int> outDims(outputs[i].first->rankOf() - 1);

#pragma omp parallel for if(outputs[i].first->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                        for (int k = 1; k < outputs[i].first->rankOf(); k++)
                            outDims[k - 1] = k;

                        std::unique_ptr<ResultSet<T>> listOutForCurrent(
                                NDArrayFactory<T>::allTensorsAlongDimension(outputs[i].first, outDims));

                        outputs[i].second = 0;

#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                        for (int e = 0; e < indices->lengthOf(); ++e)
                            if ((*indices)(e) == T(i))
                                listOutForCurrent->at(outputs[i].second++)->assign(listOfTensors->at(e));
                    }

                } else
#pragma omp parallel for if(outputList.size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (unsigned int i = 0; i < outputList.size(); i++) {
                        outputs[i].first = outputList[i];
                        outputs[i].second = 0;
                        for (int e = 0; e < indices->lengthOf(); ++e)
                            if ((*indices)(e) == T(i))
                                outputs[i].first->putScalar(outputs[i].second++, (*input)(e));
                    }
            }
            template <typename T>
            int dynamicStitchFunctor(std::vector<NDArray<T>*>& inputs, std::vector<NDArray<T>*>& indices, NDArray<T>* output){

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

                    std::unique_ptr<ResultSet<T>> listOfOutTensors(NDArrayFactory<T>::allTensorsAlongDimension(output, restDims));

                    for (int e = 0; e < numOfData; e++) {
                        NDArray<T>* data = inputs[e];
                        NDArray<T>* index = indices[e];
                        std::vector<int> sourceDims(data->rankOf() - index->rankOf());
                        for (int i = sourceDims.size(); i > 0;  i--)
                            sourceDims[sourceDims.size() - i] = data->rankOf() - i;

                        std::unique_ptr<ResultSet<T>> listOfTensors(NDArrayFactory<T>::allTensorsAlongDimension(data, sourceDims));

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

            template void dynamicPartitionFunctor(NDArray<float>* input, NDArray<float>* indices, std::vector<NDArray<float>*>& outputList);
            template void dynamicPartitionFunctor(NDArray<float16>* input, NDArray<float16>* indices, std::vector<NDArray<float16>*>& outputList);
            template void dynamicPartitionFunctor(NDArray<double>* input, NDArray<double>* indices, std::vector<NDArray<double>*>& outputList);

            template int dynamicStitchFunctor(std::vector<NDArray<float>*>& inputs, std::vector<NDArray<float>*>& indices, NDArray<float>* output);
            template int dynamicStitchFunctor(std::vector<NDArray<float16>*>& inputs, std::vector<NDArray<float16>*>& indices, NDArray<float16>* output);
            template int dynamicStitchFunctor(std::vector<NDArray<double>*>& inputs, std::vector<NDArray<double>*>& indices, NDArray<double>* output);
        }
    }
}

