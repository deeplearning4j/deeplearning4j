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
static void _diagFunctor(const NDArray* input, NDArray* output) {

    const int inLength = input->lengthOf();    

#pragma omp parallel for if(inLength > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
    for(int i = 0; i < inLength; ++i)
        (*output).putScalar<T>(i * (inLength + 1), (*input).getScalar<T>(i));
}

    void _diagFunctor(const NDArray* input, NDArray* output) {
        auto xType = input->dataType();

        BUILD_SINGLE_SELECTOR(xType, _diagFunctor, (input, output), LIBND4J_TYPES);
    }

BUILD_SINGLE_TEMPLATE(template void _diagFunctor, (const NDArray* input, NDArray* output);, LIBND4J_TYPES);


}
}
}