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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/segment.h>

namespace nd4j {
namespace ops {
namespace helpers {

    // segment max
    template <typename T>
    void segmentMaxFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = (*input)(0.);
//#pragma omp parallel for
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // max 
                   val = nd4j::math::nd4j_max(val, (*input)(e));
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* maxT = listOfOutTensors->at(idx);

            int pos = 0;
            maxT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
#pragma omp parallel for
                    for (int e = 0; e < maxT->lengthOf(); e++) {
                       (*maxT)(e) = nd4j::math::nd4j_max((*maxT)(e), (*listOfTensors->at(i))(e));
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    maxT = listOfOutTensors->at(idx);
                    maxT->assign(listOfTensors->at(i));
                }

            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    // segmen min 
    template <typename T>
    void segmentMinFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = (*input)(0.);
//#pragma omp parallel for
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // min 
                   val = nd4j::math::nd4j_min(val, (*input)(e));
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors( input->allTensorsAlongDimension(restDims) );
            std::unique_ptr<ResultSet<T>> listOfOutTensors( output->allTensorsAlongDimension(restDims) );

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* minT = listOfOutTensors->at(idx);

            int pos = 0;
            minT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
#pragma omp parallel for if(minT->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (int e = 0; e < minT->lengthOf(); e++) {
                       (*minT)(e) = nd4j::math::nd4j_min((*minT)(e), (*listOfTensors->at(i))(e));
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    minT = listOfOutTensors->at(idx);
                    minT->assign(listOfTensors->at(i));
                }
            }
        }
    }

    // segmen mean
    template <typename T>
    void segmentMeanFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // mean 
                   val += (*input)(e);
                   count++;
                }
                else {
                   (*output)(idx) = val / count;
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                    count = 1;
                }
                (*output)(idx) = val / count;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* meanT = listOfOutTensors->at(idx);
            T count = T(1.f);
            NDArray<T>* meanV = meanT->dup();
            meanV->assign(listOfTensors->at(0));
//#pragma omp parallel for
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < meanT->lengthOf(); e++) {
                       (*meanV)(e) += (*listOfTensors->at(i))(e);
                    }
                    count += T(1.f);
                }
                else {
                    //meanT->assign(meanV);
                    meanV->template applyScalar<simdOps::Divide<T>>(count, meanT);
                    idx = static_cast<int>((*indices)(i));
                    meanT = listOfOutTensors->at(idx);
                    meanV->assign(listOfTensors->at(i));
                    count = T(1.f);
                }
                meanV->template applyScalar<simdOps::Divide<T>>(count, meanT);
            }
            delete meanV;
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    void segmentSumFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // sum 
                   val += (*input)(e);
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* sumT = listOfOutTensors->at(idx);

            for (int i = 0; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                       (*sumT)(e) += (*listOfTensors->at(i))(e);
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    void segmentProdFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        output->assign((T)1.);
        if (input->isVector()) {
            T val = (*input)(0.);
            int count = 0;
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                   // sum 
                   val *= (*input)(e);
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            NDArray<T>* sumT = listOfOutTensors->at(idx);
            sumT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                       (*sumT)(e) *= (*listOfTensors->at(i))(e);
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    bool segmentIndicesValidate(NDArray<T>* indices, Nd4jLong& expected, Nd4jLong& output) {
            T val = (*indices)(0.);
            for (int e = 1; e < indices->lengthOf(); e++) {
                output = (*indices)(e);
                if (val > output) 
                    return false;
                val = (*indices)(e);
            }
            return true;
    }

    template bool segmentIndicesValidate(NDArray<float>* indices, Nd4jLong& expected, Nd4jLong& output);
    template bool segmentIndicesValidate(NDArray<float16>* indices, Nd4jLong& expected, Nd4jLong& output);
    template bool segmentIndicesValidate(NDArray<double>* indices, Nd4jLong& expected, Nd4jLong& output);

    template void segmentMaxFunctor<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* output);
    template void segmentMaxFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentMaxFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentMinFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentMinFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentMinFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentMeanFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentMeanFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentMeanFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentSumFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentSumFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentSumFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    template void segmentProdFunctor<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* output);
    template void segmentProdFunctor<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* output);
    template void segmentProdFunctor<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* output);

    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted segment ops
    // -------------------------------------------------------------------------------------------------------------- //
    template <typename T>
    bool unsortedSegmentIndicesValidate(NDArray<T>* indices, Nd4jLong expected, Nd4jLong& output) {
        Nd4jLong val = static_cast<Nd4jLong >((*indices)(0.));
        for (int e = 1; e < indices->lengthOf(); e++) {
            if (val >= expected) {
                output = val;
                return false;
            }
            val = static_cast<Nd4jLong >((*indices)(e));
        }
        output = expected;
        return true;
    }

    template <typename T>
    void unsortedSegmentMaxFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output) {

        // if input is a vector: (as if in doc sample)
        //int idx = static_cast<int>((*indices)(0.));
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
            T maxVal = DataTypeUtils::max<T>();
            output->assign(-maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T val = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    val = nd4j::math::nd4j_max(val, input->getScalar(fi->second.at(idx)));
                }
                (*output)(fi->first) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
            Nd4jLong idx = idxs[0][0];
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
            T maxVal = DataTypeUtils::max<T>();
            output->assign(-maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto maxT = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        T val = nd4j::math::nd4j_max(maxT->getScalar(e), outputT->getScalar(e));

                        (*outputT)(e) = val;
                    }
                }
                //outputT->assign(maxT);
            }
        }
    }

    template <typename T>
    void unsortedSegmentMinFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output) {
        // if input is a vector: (as if in doc sample)
        //int idx = static_cast<int>((*indices)(0.));
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
            T maxVal = DataTypeUtils::max<T>();
            output->assign(maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T val = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    val = nd4j::math::nd4j_min(val, input->getScalar(fi->second.at(idx)));
                }
                (*output)(fi->first) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
            Nd4jLong idx = idxs[0][0];
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
            T maxVal = DataTypeUtils::max<T>();
            output->assign(maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto minT = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        T val = nd4j::math::nd4j_min(minT->getScalar(e), outputT->getScalar(e));

                        (*outputT)(e) = val;
                    }
                }
                //outputT->assign(maxT);
            }
        }

    }

    template <typename T>
    void unsortedSegmentMeanFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T sumValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    sumValue += input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = sumValue / T(fi->second.size());
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) += current->getScalar(e);
                    }
                }
                //outputT->assign(maxT);
                (*outputT) /= T(fi->second.size());
            }
        }
    }

    template <typename T>
    void unsortedSegmentSumFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T sumValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    sumValue += input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = sumValue;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) += current->getScalar(e);
                    }
                }
                //outputT->assign(maxT);
            }
        }
    }

    template <typename T>
    void unsortedSegmentProdFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        output->assign(T(1.));

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T prodValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    prodValue *= input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = prodValue;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) *= current->getScalar(e);
                    }
                }
            }
        }
    }

    template <typename T>
    void unsortedSegmentSqrtNFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T sumValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    sumValue += input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = sumValue / nd4j::math::nd4j_sqrt<T>(fi->second.size());;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) += current->getScalar(e);
                    }
                }
                //outputT->assign(maxT);
                (*outputT) /= nd4j::math::nd4j_sqrt<T>(fi->second.size());
            }
        }
    }

    template void unsortedSegmentMaxFunctor<float>(NDArray<float>* input, NDArray<float>* indices, Nd4jLong numOfClasses, NDArray<float>* output);
    template void unsortedSegmentMaxFunctor<float16>(NDArray<float16>* input, NDArray<float16>* indices, Nd4jLong numOfClasses, NDArray<float16>* output);
    template void unsortedSegmentMaxFunctor<double>(NDArray<double>* input, NDArray<double>* indices, Nd4jLong numOfClasses, NDArray<double>* output);

    template void unsortedSegmentMinFunctor<float>(NDArray<float>* input, NDArray<float>* indices, Nd4jLong numOfClasses, NDArray<float>* output);
    template void unsortedSegmentMinFunctor<float16>(NDArray<float16>* input, NDArray<float16>* indices, Nd4jLong numOfClasses, NDArray<float16>* output);
    template void unsortedSegmentMinFunctor<double>(NDArray<double>* input, NDArray<double>* indices, Nd4jLong numOfClasses, NDArray<double>* output);

    template void unsortedSegmentMeanFunctor<float>(NDArray<float>* input, NDArray<float>* indices, Nd4jLong numOfClasses, NDArray<float>* output);
    template void unsortedSegmentMeanFunctor<float16>(NDArray<float16>* input, NDArray<float16>* indices, Nd4jLong numOfClasses, NDArray<float16>* output);
    template void unsortedSegmentMeanFunctor<double>(NDArray<double>* input, NDArray<double>* indices, Nd4jLong numOfClasses, NDArray<double>* output);

    template void unsortedSegmentSumFunctor<float>(NDArray<float>* input, NDArray<float>* indices, Nd4jLong numOfClasses, NDArray<float>* output);
    template void unsortedSegmentSumFunctor<float16>(NDArray<float16>* input, NDArray<float16>* indices, Nd4jLong numOfClasses, NDArray<float16>* output);
    template void unsortedSegmentSumFunctor<double>(NDArray<double>* input, NDArray<double>* indices, Nd4jLong numOfClasses, NDArray<double>* output);

    template void unsortedSegmentProdFunctor<float>(NDArray<float>* input, NDArray<float>* indices, Nd4jLong numOfClasses, NDArray<float>* output);
    template void unsortedSegmentProdFunctor<float16>(NDArray<float16>* input, NDArray<float16>* indices, Nd4jLong numOfClasses, NDArray<float16>* output);
    template void unsortedSegmentProdFunctor<double>(NDArray<double>* input, NDArray<double>* indices, Nd4jLong numOfClasses, NDArray<double>* output);

    template void unsortedSegmentSqrtNFunctor<float>(NDArray<float>* input, NDArray<float>* indices, Nd4jLong numOfClasses, NDArray<float>* output);
    template void unsortedSegmentSqrtNFunctor<float16>(NDArray<float16>* input, NDArray<float16>* indices, Nd4jLong numOfClasses, NDArray<float16>* output);
    template void unsortedSegmentSqrtNFunctor<double>(NDArray<double>* input, NDArray<double>* indices, Nd4jLong numOfClasses, NDArray<double>* output);

    template bool unsortedSegmentIndicesValidate(NDArray<float>* indices, Nd4jLong expected, Nd4jLong& output);
    template bool unsortedSegmentIndicesValidate(NDArray<float16>* indices, Nd4jLong expected, Nd4jLong& output);
    template bool unsortedSegmentIndicesValidate(NDArray<double>* indices, Nd4jLong expected, Nd4jLong& output);

    // -------------------------------------------------------------------------------------------------------------- //
    // Backpropagate ops helpers
    // -------------------------------------------------------------------------------------------------------------- //
    // Sorted backpropagate ops
    //
    
    // segment max
    template <typename T>
    int segmentMaxFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = (*input)(0.);
//#pragma omp parallel for
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                    // max
                    val = nd4j::math::nd4j_max(val, (*input)(e));
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* maxT = listOfOutTensors->at(idx);

            int pos = 0;
            maxT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
#pragma omp parallel for
                    for (int e = 0; e < maxT->lengthOf(); e++) {
                        (*maxT)(e) = nd4j::math::nd4j_max((*maxT)(e), (*listOfTensors->at(i))(e));
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    maxT = listOfOutTensors->at(idx);
                    maxT->assign(listOfTensors->at(i));
                }

            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
        return ND4J_STATUS_OK;
    }

    // segmen min
    template <typename T>
    int segmentMinFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = (*input)(0.);
//#pragma omp parallel for
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                    // min
                    val = nd4j::math::nd4j_min(val, (*input)(e));
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors( input->allTensorsAlongDimension(restDims) );
            std::unique_ptr<ResultSet<T>> listOfOutTensors( output->allTensorsAlongDimension(restDims) );

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* minT = listOfOutTensors->at(idx);

            int pos = 0;
            minT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
#pragma omp parallel for if(minT->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
                    for (int e = 0; e < minT->lengthOf(); e++) {
                        (*minT)(e) = nd4j::math::nd4j_min((*minT)(e), (*listOfTensors->at(i))(e));
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    minT = listOfOutTensors->at(idx);
                    minT->assign(listOfTensors->at(i));
                }
            }
        }
        return ND4J_STATUS_OK;
    }

    // segmen mean
    template <typename T>
    int segmentMeanFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                    // mean
                    val += (*input)(e);
                    count++;
                }
                else {
                    (*output)(idx) = val / count;
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                    count = 1;
                }
                (*output)(idx) = val / count;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* meanT = listOfOutTensors->at(idx);
            T count = T(1.f);
            NDArray<T>* meanV = meanT->dup();
            meanV->assign(listOfTensors->at(0));
//#pragma omp parallel for
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < meanT->lengthOf(); e++) {
                        (*meanV)(e) += (*listOfTensors->at(i))(e);
                    }
                    count += T(1.f);
                }
                else {
                    //meanT->assign(meanV);
                    meanV->template applyScalar<simdOps::Divide<T>>(count, meanT);
                    idx = static_cast<int>((*indices)(i));
                    meanT = listOfOutTensors->at(idx);
                    meanV->assign(listOfTensors->at(i));
                    count = T(1.f);
                }
                meanV->template applyScalar<simdOps::Divide<T>>(count, meanT);
            }
            delete meanV;
            delete listOfTensors;
            delete listOfOutTensors;
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int segmentSumFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                    // sum
                    val += (*input)(e);
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
            NDArray<T>* sumT = listOfOutTensors->at(idx);

            for (int i = 0; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                        (*sumT)(e) += (*listOfTensors->at(i))(e);
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int segmentProdFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        output->assign((T)1.);
        if (input->isVector()) {
            T val = (*input)(0.);
            int count = 0;
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == static_cast<int>((*indices)(e))) {
                    // sum
                    val *= (*input)(e);
                }
                else {
                    idx = static_cast<int>((*indices)(e));
                    val = (*input)(e);
                }
                (*output)(idx) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;
            ResultSet<T>* listOfTensors = input->allTensorsAlongDimension(restDims);
            ResultSet<T>* listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            NDArray<T>* sumT = listOfOutTensors->at(idx);
            sumT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (static_cast<int>((*indices)(i)) == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                        (*sumT)(e) *= (*listOfTensors->at(i))(e);
                    }
                }
                else {
                    idx = static_cast<int>((*indices)(i));
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
        return ND4J_STATUS_OK;
    }

    template int segmentMaxFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, NDArray<float>* output);
    template int segmentMaxFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, NDArray<float16>* output);
    template int segmentMaxFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, NDArray<double>* output);

    template int segmentMinFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, NDArray<float>* output);
    template int segmentMinFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, NDArray<float16>* output);
    template int segmentMinFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, NDArray<double>* output);

    template int segmentMeanFunctorBP<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* gradOut, NDArray<float>* output);
    template int segmentMeanFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, NDArray<float16>* output);
    template int segmentMeanFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, NDArray<double>* output);

    template int segmentSumFunctorBP<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* gradOut, NDArray<float>* output);
    template int segmentSumFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* gradOut, NDArray<float16>* output);
    template int segmentSumFunctorBP<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* gradOut, NDArray<double>* output);

    template int segmentProdFunctorBP<float>(NDArray<float>* input, NDArray<float>* , NDArray<float>* gradOut, NDArray<float>* output);
    template int segmentProdFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* , NDArray<float16>* gradOut, NDArray<float16>* output);
    template int segmentProdFunctorBP<double>(NDArray<double>* input, NDArray<double>* , NDArray<double>* gradOut, NDArray<double>* output);

    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted backpropagate segment ops
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T>
    int unsortedSegmentMaxFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {

        // if input is a vector: (as if in doc sample)
        //int idx = static_cast<int>((*indices)(0.));
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
            T maxVal = DataTypeUtils::max<T>();
            output->assign(-maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T val = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    val = nd4j::math::nd4j_max(val, input->getScalar(fi->second.at(idx)));
                }
                (*output)(fi->first) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
            Nd4jLong idx = idxs[0][0];
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
            T maxVal = DataTypeUtils::max<T>();
            output->assign(-maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto maxT = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        T val = nd4j::math::nd4j_max(maxT->getScalar(e), outputT->getScalar(e));

                        (*outputT)(e) = val;
                    }
                }
                //outputT->assign(maxT);
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentMinFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        // if input is a vector: (as if in doc sample)
        //int idx = static_cast<int>((*indices)(0.));
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
            T maxVal = DataTypeUtils::max<T>();
            output->assign(maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T val = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    val = nd4j::math::nd4j_min(val, input->getScalar(fi->second.at(idx)));
                }
                (*output)(fi->first) = val;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
            Nd4jLong idx = idxs[0][0];
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
            T maxVal = DataTypeUtils::max<T>();
            output->assign(maxVal);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto minT = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        T val = nd4j::math::nd4j_min(minT->getScalar(e), outputT->getScalar(e));

                        (*outputT)(e) = val;
                    }
                }
                //outputT->assign(maxT);
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentMeanFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T sumValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    sumValue += input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = sumValue / T(fi->second.size());
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) += current->getScalar(e);
                    }
                }
                //outputT->assign(maxT);
                (*outputT) /= T(fi->second.size());
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentSumFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T sumValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    sumValue += input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = sumValue;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) += current->getScalar(e);
                    }
                }
                //outputT->assign(maxT);
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentProdFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        output->assign(T(1.));

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T prodValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    prodValue *= input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = prodValue;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) *= current->getScalar(e);
                    }
                }
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentSqrtNFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, std::vector<Nd4jLong>> idxs;//(indices->lengthOf());
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e)
            idxs[static_cast<Nd4jLong>(indices->getScalar(e))].push_back(e);

        //std::sort(idxs.begin(), idxs.end());

        if (input->isVector()) { // 1D case
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                T sumValue = input->getScalar(fi->second.at(0));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    sumValue += input->getScalar(fi->second.at(idx));
                }
                (*output)(fi->first) = sumValue / nd4j::math::nd4j_sqrt<T>(fi->second.size());;
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

//            int numOfClasses = output->sizeAt(0); // number of classes
//            std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);
//            NDArray<T>* maxT = listOfOutTensors->at(idx);
//#pragma omp parallel for schedule(static)
            for (auto fi = idxs.begin(); fi != idxs.end(); ++fi) {
                auto outputT = listOfOutTensors->at(fi->first);
                outputT->assign(listOfTensors->at(fi->second.at(0)));
                for (Nd4jLong idx = 1; idx < fi->second.size(); ++idx) {
                    auto current = listOfTensors->at(fi->second.at(idx));
                    for (Nd4jLong e = 0; e < outputT->lengthOf(); ++e) {
                        //T val = minT->getScalar(e) + outputT->getScalar(e);

                        (*outputT)(e) += current->getScalar(e);
                    }
                }
                //outputT->assign(maxT);
                (*outputT) /= nd4j::math::nd4j_sqrt<T>(fi->second.size());
            }
        }
        return ND4J_STATUS_OK;
    }

    template int unsortedSegmentMaxFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, Nd4jLong numOfClasses, NDArray<float>* output);
    template int unsortedSegmentMaxFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, Nd4jLong numOfClasses, NDArray<float16>* output);
    template int unsortedSegmentMaxFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, Nd4jLong numOfClasses, NDArray<double>* output);

    template int unsortedSegmentMinFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, Nd4jLong numOfClasses, NDArray<float>* output);
    template int unsortedSegmentMinFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, Nd4jLong numOfClasses, NDArray<float16>* output);
    template int unsortedSegmentMinFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, Nd4jLong numOfClasses, NDArray<double>* output);

    template int unsortedSegmentMeanFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, Nd4jLong numOfClasses, NDArray<float>* output);
    template int unsortedSegmentMeanFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, Nd4jLong numOfClasses, NDArray<float16>* output);
    template int unsortedSegmentMeanFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, Nd4jLong numOfClasses, NDArray<double>* output);

    template int unsortedSegmentSumFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, Nd4jLong numOfClasses, NDArray<float>* output);
    template int unsortedSegmentSumFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, Nd4jLong numOfClasses, NDArray<float16>* output);
    template int unsortedSegmentSumFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, Nd4jLong numOfClasses, NDArray<double>* output);

    template int unsortedSegmentProdFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, Nd4jLong numOfClasses, NDArray<float>* output);
    template int unsortedSegmentProdFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, Nd4jLong numOfClasses, NDArray<float16>* output);
    template int unsortedSegmentProdFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, Nd4jLong numOfClasses, NDArray<double>* output);

    template int unsortedSegmentSqrtNFunctorBP<float>(NDArray<float>* input, NDArray<float>* indices, NDArray<float>* gradOut, Nd4jLong numOfClasses, NDArray<float>* output);
    template int unsortedSegmentSqrtNFunctorBP<float16>(NDArray<float16>* input, NDArray<float16>* indices, NDArray<float16>* gradOut, Nd4jLong numOfClasses, NDArray<float16>* output);
    template int unsortedSegmentSqrtNFunctorBP<double>(NDArray<double>* input, NDArray<double>* indices, NDArray<double>* gradOut, Nd4jLong numOfClasses, NDArray<double>* output);
}
}
}