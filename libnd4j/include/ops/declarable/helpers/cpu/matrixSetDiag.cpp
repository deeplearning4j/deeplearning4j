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
// Created by Yurii Shyrma on 07.12.2017.
//

#include "ResultSet.h"
#include <ops/declarable/helpers/matrixSetDiag.h>

namespace nd4j {
namespace ops {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
// Returns a batched matrix tensor with new batched diagonal values.
// for detailed explanations please take a look on web page: https://www.tensorflow.org/api_docs/python/tf/matrix_set_diag
template <typename T>
static void _matrixSetDiag(const NDArray* input, const NDArray* diagonal, NDArray* output) {

    *output = *input;

//    ResultSet<T>* listOut  = output->allTensorsAlongDimension({output->rankOf()-2, output->rankOf()-1});
//    ResultSet<T>* listDiag = diagonal->allTensorsAlongDimension({diagonal->rankOf()-1});

    // TODO: tune this properlys
//#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(static)
    // condition is hold: listOut->size() == listDiag->size()
//    for(int i = 0; i < listOut->size(); ++i)       
//    	for(int j = 0; j < diagonal->sizeAt(-1); ++j)
//        	(*listOut->at(i))(j,j) = (*listDiag->at(i))(j);            
//    
//    delete listOut;
//    delete listDiag;

            *output = *input;

            const int lastDimSize = input->sizeAt(-1);
            const int last2DimSize = input->sizeAt(-1) * input->sizeAt(-2);
            const int lastSmallDim = diagonal->sizeAt(-1);
            const int batchSize = input->lengthOf()/last2DimSize;
    
// #pragma omp parallel for if(batchSize > Environment::getInstance()->elementwiseThreshold()) schedule(static) 
            for(int i = 0; i < batchSize; ++i )
                for(int j = 0; j < lastSmallDim; ++j) {
                    output->p(i*last2DimSize + j*(lastDimSize + 1), diagonal->e<T>(i*lastSmallDim + j));
                }
             

}

    void matrixSetDiag(const NDArray* input, const NDArray* diagonal, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), _matrixSetDiag, (input, diagonal, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void _matrixSetDiag, (const NDArray* input, const NDArray* diagonal, NDArray* output), LIBND4J_TYPES);

}
}
}