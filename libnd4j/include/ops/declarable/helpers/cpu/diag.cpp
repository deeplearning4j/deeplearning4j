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
// Created by GS <sgazeos@gmail.com> on 4/6/2018.
//

#include "ResultSet.h"
#include <ops/declarable/helpers/diag.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
template <typename T>
void diagFunctor(const NDArray<T>* input, NDArray<T>* output) {

    const int inLength = input->lengthOf();    

#pragma omp parallel for if(inLength > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
    for(int i = 0; i < inLength; ++i)
        (*output)(i * (inLength + 1)) = (*input)(i);
}


template void diagFunctor<float>(const NDArray<float>* input, NDArray<float>* output);
template void diagFunctor<float16>(const NDArray<float16>* input, NDArray<float16>* output);
template void diagFunctor<double>(const NDArray<double>* input, NDArray<double>* output);
template void diagFunctor<int>(const NDArray<int>* input, NDArray<int>* output);
template void diagFunctor<Nd4jLong>(const NDArray<Nd4jLong>* input, NDArray<Nd4jLong>* output);


}
}
}