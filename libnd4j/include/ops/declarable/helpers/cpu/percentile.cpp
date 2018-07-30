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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 17.05.2018
//

#include <ops/declarable/helpers/percentile.h>
#include "ResultSet.h"

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////    
template <typename T>
void percentile(const NDArray<T>& input, NDArray<T>& output, std::vector<int>& axises, const T q, const int interpolation) {
    
    const int inputRank = input.rankOf();    

    if(axises.empty())
        for(int i=0; i<inputRank; ++i)
            axises.push_back(i);
    else
        shape::checkDimensions(inputRank, axises);          // check, sort dimensions and remove duplicates if they are present


    ResultSet<T>* listOfSubArrs = input.allTensorsAlongDimension(axises);
    
    std::vector<Nd4jLong> shapeOfSubArr(listOfSubArrs->at(0)->rankOf());
    for(int i=0; i<shapeOfSubArr.size(); ++i)
        shapeOfSubArr[i] = listOfSubArrs->at(0)->shapeOf()[i];

    NDArray<T> flattenedArr('c', shapeOfSubArr, input.getWorkspace());    
    const int len = flattenedArr.lengthOf();
    
    const T fraction = 1. - q / 100.;
    int position = 0;
    
    switch(interpolation) {
        case 0: // lower
            position = math::nd4j_ceil<T>((len - 1) * fraction);
            break;
        case 1: // higher
            position = math::nd4j_floor<T>((len - 1) * fraction);
            break;
        case 2: // nearest
            position = math::nd4j_round<T>((len - 1) * fraction);
            break;
    }
    position = len - position - 1;

#pragma omp parallel for schedule(guided) firstprivate(flattenedArr)
    for(int i=0; i<listOfSubArrs->size(); ++i) {
        
        T* buff = flattenedArr.getBuffer();
        flattenedArr.assign(listOfSubArrs->at(i));
        std::sort(buff, buff + len);
        output(i) = flattenedArr(position);
    }

    delete listOfSubArrs;
}

template void percentile(const NDArray<float>& input, NDArray<float>& output, std::vector<int>& axises, const float q, const int interpolation);
template void percentile(const NDArray<float16>& input, NDArray<float16>& output, std::vector<int>& axises, const float16 q, const int interpolation);
template void percentile(const NDArray<double>& input, NDArray<double>& output, std::vector<int>& axises, const double q, const int interpolation);

}
}
}