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
//#pragma omp parallel for
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
#pragma omp parallel for
            for (int e = 1; e < input->rankOf(); e++)
                restDims[e - 1] = e;

            auto listOfTensors = input->allTensorsAlongDimension(restDims);
            auto listOfOutTensors = output->allTensorsAlongDimension(restDims);

            int numOfClasses = output->sizeAt(0); // number of classes
            std::vector<std::pair<NDArray*, int>> outputs(numOfClasses);
            auto maxT = listOfOutTensors->at(idx);

            int pos = 0;
            maxT->assign(listOfTensors->at(0));
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (indices->e<int>(i) == idx) {
#pragma omp parallel for
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
//#pragma omp parallel for
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
            for (int i = 1; i < indices->lengthOf(); i++) {
                if (indices->e<T>(i) == idx) {
#pragma omp parallel for if(minT->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
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
//#pragma omp parallel for
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
}
}
}