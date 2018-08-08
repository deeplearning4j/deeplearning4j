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
void matrixSetDiag(const NDArray<T>* input, const NDArray<T>* diagonal, NDArray<T>* output) {

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
                    (*output)(i*last2DimSize + j*(lastDimSize + 1)) = (*diagonal)(i*lastSmallDim + j);            
                }
             

}



template void matrixSetDiag<float>(const NDArray<float>* input, const NDArray<float>* diagonal, NDArray<float>* output);
template void matrixSetDiag<float16>(const NDArray<float16>* input, const NDArray<float16>* diagonal, NDArray<float16>* output);
template void matrixSetDiag<double>(const NDArray<double>* input, const NDArray<double>* diagonal, NDArray<double>* output);
template void matrixSetDiag<int>(const NDArray<int>* input, const NDArray<int>* diagonal, NDArray<int>* output);
template void matrixSetDiag<Nd4jLong>(const NDArray<Nd4jLong>* input, const NDArray<Nd4jLong>* diagonal, NDArray<Nd4jLong>* output);


}
}
}