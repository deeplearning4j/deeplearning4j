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
// Created by GS <sgazeos@gmail.com> on 3/21/2018.
//

#include "ResultSet.h"
#include <ops/declarable/helpers/matrix_diag_part.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
template <typename T>
int matrixDiagPart(const NDArray<T>* input, NDArray<T>* output) {

    ResultSet<T>* listOut  = output->allTensorsAlongDimension({output->rankOf() - 1});
    ResultSet<T>* listDiag = input->allTensorsAlongDimension({input->rankOf() - 2, input->rankOf() - 1});

    if (listOut->size() != listDiag->size()) {
        nd4j_printf("matrix_diag_part: Input matrix has wrong shape.", "");
        return ND4J_STATUS_VALIDATION;
    }
    int lastDimension = nd4j::math::nd4j_min(input->sizeAt(-2), input->sizeAt(-1));
    // TODO: tune this properlys
#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
    // condition is hold: listOut->size() == listDiag->size()
    for(int i = 0; i < listOut->size(); ++i)       
        for(int j = 0; j < lastDimension; ++j)
            (*listOut->at(i))(j) = (*listDiag->at(i))(j, j);            
    
    delete listOut;
    delete listDiag;

    return ND4J_STATUS_OK;
}


template int matrixDiagPart<float>(const NDArray<float>* input, NDArray<float>* output);
template int matrixDiagPart<float16>(const NDArray<float16>* input, NDArray<float16>* output);
template int matrixDiagPart<double>(const NDArray<double>* input, NDArray<double>* output);


}
}
}