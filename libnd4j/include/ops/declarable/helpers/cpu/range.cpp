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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 27.08.2018
//


#include <ops/declarable/helpers/range.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// be careful: outVector must have c-order and ews = 1 !!!
template <typename T>
static void _range(const NDArray& start, const NDArray& delta, NDArray& outVector) {
        
    const Nd4jLong len = outVector.lengthOf();

    auto buff = reinterpret_cast<T *>(outVector.getBuffer());

// #pragma omp parallel for simd if(len > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for simd schedule(guided)
    for(Nd4jLong i = 0; i < len; ++i)
    	buff[i] =  start.e<T>(0) + i * delta.e<T>(0);
        
}

    void range(const NDArray& start, const NDArray& delta, NDArray& outVector) {
        BUILD_SINGLE_SELECTOR(outVector.dataType(), _range, (start, delta, outVector), LIBND4J_TYPES);
    }

BUILD_SINGLE_TEMPLATE(template void _range, (const NDArray& start, const NDArray& delta, NDArray& outVector), LIBND4J_TYPES);


}
}
}