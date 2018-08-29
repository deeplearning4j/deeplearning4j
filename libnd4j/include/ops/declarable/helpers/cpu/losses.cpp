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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 29.08.2018
//


#include<ops/declarable/helpers/losses.h>
#include <helpers/ShapeUtils.h>


namespace nd4j 	  {
namespace ops 	  {
namespace helpers {


//////////////////////////////////////////////////////////////////////////
template <typename T>
void sparseSoftmaxCrossEntropyLossWithLogits(const NDArray<T>& labels, const NDArray<T>& logits, NDArray<T>& output) {


    NDArray<T> maxAlongDim = logits.template reduceAlongDims<simdOps::Max<T>>({-1}, true);
    NDArray<T> logitsExp = (logits - maxAlongDim).template transform<simdOps::Exp<T>>();
    NDArray<T> logSoftMax = ( logitsExp / logitsExp.template reduceAlongDims<simdOps::Sum<T>>({-1}, true) ).template transform<simdOps::Log<T>>();
        
    const Nd4jLong labelsLen = labels.lengthOf();

    std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(logits.rankOf(), {-1});
 
#pragma omp parallel for schedule(guided) 
    for(Nd4jLong i = 0; i < labelsLen; ++i)
        output(i) = -logSoftMax(i, dimsToExclude)(labels(i));
}


template void sparseSoftmaxCrossEntropyLossWithLogits<float>(const NDArray<float>& labels, const NDArray<float>& logits, NDArray<float>& output);
template void sparseSoftmaxCrossEntropyLossWithLogits<float16>(const NDArray<float16>& labels, const NDArray<float16>& logits, NDArray<float16>& output);
template void sparseSoftmaxCrossEntropyLossWithLogits<double>(const NDArray<double>& labels, const NDArray<double>& logits, NDArray<double>& output);

}
}
}