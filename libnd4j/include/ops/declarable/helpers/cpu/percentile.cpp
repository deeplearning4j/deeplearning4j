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
#include <NDArrayFactory.h>
#include "ResultSet.h"

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
static void _percentile(const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation) {
    
    const int inputRank = input.rankOf();    

    if(axises.empty())
        for(int i=0; i<inputRank; ++i)
            axises.push_back(i);
    else
        shape::checkDimensions(inputRank, axises);          // check, sort dimensions and remove duplicates if they are present


    auto listOfSubArrs = input.allTensorsAlongDimension(axises);
    
    std::vector<Nd4jLong> shapeOfSubArr(listOfSubArrs->at(0)->rankOf());
    for(int i=0; i<shapeOfSubArr.size(); ++i)
        shapeOfSubArr[i] = listOfSubArrs->at(0)->shapeOf()[i];

    auto flattenedArr = NDArrayFactory::_create('c', shapeOfSubArr, input.dataType(), input.getWorkspace());
    const int len = flattenedArr.lengthOf();
    
    const float fraction = 1.f - q / 100.;
    Nd4jLong position = 0;
    
    switch(interpolation) {
        case 0: // lower
            position = static_cast<Nd4jLong>(math::nd4j_ceil<float>((len - 1) * fraction));
            break;
        case 1: // higher
            position = static_cast<Nd4jLong>(math::nd4j_floor<float>((len - 1) * fraction));
            break;
        case 2: // nearest
            position = static_cast<Nd4jLong>(math::nd4j_round<float>((len - 1) * fraction));
            break;
    }
    position = len - position - 1;

    // FIXME: our sort impl should be used instead, so this operation might be implemented as generic
#pragma omp parallel for schedule(guided) firstprivate(flattenedArr)
    for(int i=0; i<listOfSubArrs->size(); ++i) {
        
        T* buff = reinterpret_cast<T *>(flattenedArr.getBuffer());
        flattenedArr.assign(listOfSubArrs->at(i));
        std::sort(buff, buff + len);
        output.putScalar(i, flattenedArr.getScalar<T>(position));
    }

    delete listOfSubArrs;
}

    void percentile(const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation) {
        BUILD_SINGLE_SELECTOR(input.dataType(), _percentile, (input, output, axises, q, interpolation), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _percentile, (const NDArray& input, NDArray& output, std::vector<int>& axises, const float q, const int interpolation), LIBND4J_TYPES);

}
}
}