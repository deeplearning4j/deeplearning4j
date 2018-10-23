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
    static void segmentMaxFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = indices->e<int>(0);
        if (input->isVector()) {
            T val = input->e<T>(0);
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == indices->e<int>(e)) {
                   // max 
                   val = nd4j::math::nd4j_max<T>(val, input->e<T>(e));
                }
                else {
                    idx = indices->e<int>(e);
                    val = input->e<T>(e);
                }
                output->p<T>(idx, val);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            auto listOfTensors = input->allTensorsAlongDimension(restDims);
            auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
            auto maxT = listOfOutTensors->at(idx);

            int pos = 0;
            maxT->assign(listOfTensors->at(0));
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (indices->e<int>(i) == idx) {
                    for (int e = 0; e < maxT->lengthOf(); e++) {
                       maxT->p<T>(e, nd4j::math::nd4j_max(maxT->e<T>(e), listOfTensors->at(i)->e<T>(e)));
                    }
                }
                else {
                    idx = indices->e<int>(i);
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
    static void segmentMinFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = indices->e<int>(0);
        if (input->isVector()) {
            T val = input->e<T>(0);
#pragma omp parallel for if(indices->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < indices->lengthOf(); e++) {
                if (idx == indices->e<int>(e)) {
                   // min 
                   val = nd4j::math::nd4j_min<T>(val, input->e<T>(e));
                }
                else {
                    idx = indices->e<int>(e);
                    val = input->e<T>(e);
                }
                output->p(idx, val);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet> listOfTensors( input->allTensorsAlongDimension(restDims) );
            std::unique_ptr<ResultSet> listOfOutTensors( output->allTensorsAlongDimension(restDims) );

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
            auto minT = listOfOutTensors->at(idx);

            int pos = 0;
            minT->assign(listOfTensors->at(0));
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (indices->e<T>(i) == idx) {
                    for (int e = 0; e < minT->lengthOf(); e++) {
                       minT->p(e, nd4j::math::nd4j_min(minT->e<T>(e), listOfTensors->at(i)->e<T>(e)));
                    }
                }
                else {
                    idx = indices->e<T>(i);
                    minT = listOfOutTensors->at(idx);
                    minT->assign(listOfTensors->at(i));
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
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
//#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 0; e < indices->lengthOf(); e++) {
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
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            auto listOfTensors = input->allTensorsAlongDimension(restDims);
            auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
            auto meanT = listOfOutTensors->at(idx);
            int count = 1;
            auto meanV = meanT->dup();
            meanV->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (indices->e<int>(i) == idx) {
                    for (int e = 0; e < meanT->lengthOf(); e++) {
                       meanV->p<T>(e, meanV->e<T>(e) + listOfTensors->at(i)->e<T>(e));
                    }
                    count++;
                }
                else {
                    //meanT->assign(meanV);
                    meanV->applyScalar(scalar::Divide, count, meanT, nullptr);
                    idx = indices->e<int>(i);
                    meanT = listOfOutTensors->at(idx);
                    meanV->assign(listOfTensors->at(i));
                    count = 1;
                }
                meanV->applyScalar(scalar::Divide, count, meanT, nullptr);
            }
            delete meanV;
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    static void segmentSumFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = indices->e<int>(0);
        if (input->isVector()) {
            T val = T(0.f);
            int count = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                if (idx == indices->e<int>(e)) {
                   // sum 
                   val += input->e<T>(e);
                }
                else {
                    idx = indices->e<int>(e);
                    val = input->e<T>(e);
                }
                output->p(idx, val);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            auto listOfTensors = input->allTensorsAlongDimension(restDims);
            auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
            auto sumT = listOfOutTensors->at(idx);

            for (int i = 0; i < indices->lengthOf(); i++) {
                if (indices->e<int>(i) == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                       sumT->p(e, sumT->e<T>(e) +listOfTensors->at(i)->e<T>(e));
                    }
                }
                else {
                    idx = indices->e<int>(i);
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    static void segmentProdFunctor_(NDArray* input, NDArray* indices, NDArray* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = indices->e<int>(0);
        output->assign((T)1.);
        if (input->isVector()) {
            T val = input->e<T>(0);
            int count = 0;
            for (int e = 1; e < indices->lengthOf(); e++) {
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
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->rankOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            auto listOfTensors = input->allTensorsAlongDimension(restDims);
            auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            auto sumT = listOfOutTensors->at(idx);
            sumT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (indices->e<int>(i)  == idx) {
                    for (int e = 0; e < sumT->lengthOf(); e++) {
                       sumT->p(e, sumT->e<T>(e) * listOfTensors->at(i)->e<T>(e));
                    }
                }
                else {
                    idx = indices->e<int>(i);
                    sumT = listOfOutTensors->at(idx);
                    sumT->assign(listOfTensors->at(i));
                }
            }
            delete listOfTensors;
            delete listOfOutTensors;
        }
    }

    template <typename T>
    static bool segmentIndicesValidate_(NDArray* indices, NDArray& aexpected, NDArray& aoutput) {
            T val = indices->e<T>(0);
            for (int e = 1; e < indices->lengthOf(); e++) {
                aoutput.p<T>(Nd4jLong(0), indices->e<T>(e));
                if (val > aoutput.e<T>(0))
                    return false;
                val = indices->e<T>(e);
            }

            return true;
    }

    void segmentMaxFunctor(NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentMaxFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentMinFunctor(NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentMinFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentMeanFunctor(NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentMeanFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentSumFunctor(NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentSumFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentProdFunctor(NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentProdFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    bool segmentIndicesValidate(NDArray* indices, NDArray& expected, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), return segmentIndicesValidate_, (indices, expected, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template bool segmentIndicesValidate_, (NDArray*, NDArray&, NDArray&), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentProdFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentSumFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentMeanFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentMinFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentMaxFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted segment ops
    // -------------------------------------------------------------------------------------------------------------- //
    /*
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
//#pragma omp parallel for if(idxs.size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
        int numOfClasses = gradOut->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        auto tempRes = gradOut->dup();
        segmentMaxFunctor(input, indices, tempRes);
        if (input->isVector()) {
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (Nd4jLong e = 0; e < input->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                if (nd4j::math::nd4j_abs(tempRes->getScalar(classNum) -(*input)(e)) < T(1.e-5))
                    (*output)(e) = (*gradOut)(classNum);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfBPTensors(tempRes->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);

                for (int e = 0; e < current->lengthOf(); e++) {
                    if (nd4j::math::nd4j_abs(listOfBPTensors->at(classNum)->getScalar(e) - current->getScalar(e)) < T(1.e-5))
                        (*currentOut)(e) = (*currentGradOut)(e);
                }
            }
        }
        delete tempRes;
        return ND4J_STATUS_OK;
    }

    // segmen min
    template <typename T>
    int segmentMinFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        auto tempRes = gradOut->dup();
        segmentMinFunctor(input, indices, tempRes);
        if (input->isVector()) {
            for (Nd4jLong e = 0; e < input->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                if (nd4j::math::nd4j_abs(tempRes->getScalar(classNum) -(*input)(e)) < T(1.e-5))
                    (*output)(e) = (*gradOut)(classNum);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfBPTensors(tempRes->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);
                for (int e = 0; e < current->lengthOf(); e++) {
                    if (nd4j::math::nd4j_abs(listOfBPTensors->at(classNum)->getScalar(e) - current->getScalar(e)) < T(1.e-5))
                        (*currentOut)(e) = (*currentGradOut)(e);
                }
            }
        }
        delete tempRes;
        return ND4J_STATUS_OK;
    }

    // segmen mean
    template <typename T>
    int segmentMeanFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        std::map<Nd4jLong, Nd4jLong> classCount;//(numClasses);

        for (Nd4jLong count = 0; count < numClasses; ++count) {
            classCount[count] = 0;
        }

        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
            classCount[static_cast<Nd4jLong>(indices->getScalar(e))] ++;
        }

        // if input is a vector: (as if in doc sample)
        if (input->isVector()) {
            for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                (*output)(e) = (*gradOut)(classNum) / T(classCount[classNum]);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);
//#pragma omp parallel for
                for (int e = 0; e < current->lengthOf(); e++) {
                    (*currentOut)(e) = (*currentGradOut)(e) / T(classCount[classNum]);
                }
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int segmentSumFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        int numClasses = output->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                (*output)(e) = (*gradOut)(classNum);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);
                for (int e = 0; e < current->lengthOf(); e++) {
                    (*currentOut)(e) = (*currentGradOut)(e);
                }
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int segmentProdFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output) {
        auto tempRes = gradOut->dup();
        segmentProdFunctor(input, indices, tempRes);
        if (input->isVector()) {
            for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                (*output)(e) = (*gradOut)(classNum) * (*tempRes)(classNum)/ (*input)(e);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfBPTensors(tempRes->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);
                auto currentFFOut = listOfBPTensors->at(classNum);
                for (int e = 0; e < current->lengthOf(); e++) {
                    (*currentOut)(e) = (*currentFFOut)(e) * (*currentGradOut)(e) / (*current)(e);
                }
            }
        }
        delete tempRes;
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
//        int numOfClasses = gradOut->sizeAt(0);
        // if input is a vector: (as if in doc sample)
        auto tempRes = gradOut->dup();
        unsortedSegmentMaxFunctor(input, indices, numOfClasses, tempRes);
        if (input->isVector()) {
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (Nd4jLong e = 0; e < input->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                if (nd4j::math::nd4j_abs(tempRes->getScalar(classNum) -(*input)(e)) < T(1.e-5))
                    (*output)(e) = (*gradOut)(classNum);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfBPTensors(tempRes->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);
                for (int e = 0; e < current->lengthOf(); e++) {
                    if (nd4j::math::nd4j_abs(listOfBPTensors->at(classNum)->getScalar(e) - current->getScalar(e)) < T(1.e-5))
                        (*currentOut)(e) = (*currentGradOut)(e);
                }
            }
        }
        delete tempRes;
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentMinFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        auto tempRes = gradOut->dup();
        unsortedSegmentMinFunctor(input, indices, numOfClasses, tempRes);
        if (input->isVector()) {
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (Nd4jLong e = 0; e < input->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                if (nd4j::math::nd4j_abs(tempRes->getScalar(classNum) -(*input)(e)) < T(1.e-5))
                    (*output)(e) = (*gradOut)(classNum);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfBPTensors(tempRes->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);

                for (int e = 0; e < current->lengthOf(); e++) {
                    if (nd4j::math::nd4j_abs(listOfBPTensors->at(classNum)->getScalar(e) - current->getScalar(e)) < T(1.e-5))
                        (*currentOut)(e) = (*currentGradOut)(e);
                }
            }
        }
        delete tempRes;
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentMeanFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {

        std::map<Nd4jLong, Nd4jLong> classCount;//(numClasses);

//#pragma omp parallel for if(numOfClasses > Environment::getInstance()->elementwiseThreshold()) schedule(static)
        for (Nd4jLong count = 0; count < numOfClasses; ++count) {
            classCount[count] = 0;
        }

//#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
            classCount[static_cast<Nd4jLong>(indices->getScalar(e))] ++;
        }

        // if input is a vector: (as if in doc sample)
        if (input->isVector()) {
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                (*output)(e) = (*gradOut)(classNum) / T(classCount[classNum]);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);
                for (int e = 0; e < current->lengthOf(); e++) {
                    (*currentOut)(e) = (*currentGradOut)(e) / T(classCount[classNum]);
                }
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentSumFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {

        // if input is a vector: (as if in doc sample)
        int idx = static_cast<int>((*indices)(0.));
        if (input->isVector()) {
            for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                (*output)(e) = (*gradOut)(classNum);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);

                for (int e = 0; e < current->lengthOf(); e++) {
                    (*currentOut)(e) = (*currentGradOut)(e);
                }
            }
        }
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentProdFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        auto tempRes = gradOut->dup();

        unsortedSegmentProdFunctor(input, indices, numOfClasses, tempRes);
        if (input->isVector()) {
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                (*output)(e) = (*gradOut)(classNum) * (*tempRes)(classNum)/ (*input)(e);
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(input->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfBPTensors(tempRes->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);
                auto currentFFOut = listOfBPTensors->at(classNum);

                for (int e = 0; e < current->lengthOf(); e++) {
                    (*currentOut)(e) = (*currentFFOut)(e) * (*currentGradOut)(e) / (*current)(e);
                }
            }
        }
        delete tempRes;
        return ND4J_STATUS_OK;
    }

    template <typename T>
    int unsortedSegmentSqrtNFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output) {
        std::map<Nd4jLong, Nd4jLong> classCount;//(numClasses);

//#pragma omp parallel for if(numOfClasses > Environment::getInstance()->elementwiseThreshold()) schedule(static)
        for (Nd4jLong count = 0; count < numOfClasses; ++count) {
            classCount[count] = 0;
        }

//#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
        for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
            classCount[static_cast<Nd4jLong>(indices->getScalar(e))] ++;
        }

        // if input is a vector: (as if in doc sample)
        if (input->isVector()) {
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (Nd4jLong e = 0; e < indices->lengthOf(); ++e) {
                Nd4jLong classNum = static_cast<Nd4jLong>(indices->getScalar(e));
                (*output)(e) = (*gradOut)(classNum) / nd4j::math::nd4j_sqrt(T(classCount[classNum]));
            }
        }
        else {
            std::vector<int> restDims(input->rankOf() - 1);
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            std::unique_ptr<ResultSet<T>> listOfGradOuts(gradOut->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfTensors(input->allTensorsAlongDimension(restDims));
            std::unique_ptr<ResultSet<T>> listOfOutTensors(output->allTensorsAlongDimension(restDims));

            //int numOfClasses = tempRes->sizeAt(0); // number of classes
            //std::vector<std::pair<NDArray<T>*, int>> outputs(numOfClasses);

            int pos = 0;
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
            for (int i = 0; i < indices->lengthOf(); i++) {
                Nd4jLong classNum = static_cast<Nd4jLong>((*indices)(i));
                NDArray<T>* current = listOfTensors->at(i);
                NDArray<T>* currentOut = listOfOutTensors->at(i);
                NDArray<T>* currentGradOut = listOfGradOuts->at(classNum);

                for (int e = 0; e < current->lengthOf(); e++) {
                    (*currentOut)(e) = (*currentGradOut)(e) / nd4j::math::nd4j_sqrt(T(classCount[classNum]));
                }
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
     */
}
}
}