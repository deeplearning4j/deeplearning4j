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
void sparseSoftmaxCrossEntropyLossWithLogits(const NDArray& labels, const NDArray& logits, NDArray& output) {

    auto maxAlongDim = logits.reduceAlongDims(reduce::Max, {-1}, true);
    auto logitsExp = (logits - maxAlongDim).transform(transform::Exp, nullptr);
    auto logSoftMax = ( logitsExp / logitsExp.reduceAlongDims(reduce::Sum, {-1}, true) ).transform(transform::Log);
        
    const Nd4jLong labelsLen = labels.lengthOf();

    std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(logits.rankOf(), {-1});
 
#pragma omp parallel for schedule(guided) 
    for(Nd4jLong i = 0; i < labelsLen; ++i) {

        auto subArr = logSoftMax(i, dimsToExclude);

        // FIXME: double
        output.p(i, -subArr.e<double>(labels.e<Nd4jLong>(i)));
    }
}

    void reduceZeroCountWeights(NDArray* weightsBroad, Nd4jLong sizeAtRestDims, 
        NDArray& numOfNonZeroWeights) {
        for(int i = 0; i < numOfNonZeroWeights.lengthOf(); ++i)
            for(int j = 0; j < sizeAtRestDims; ++j)
                if(weightsBroad->e<float>(i*sizeAtRestDims + j) != 0.f)
                    numOfNonZeroWeights.p<Nd4jLong>(i, 1 + numOfNonZeroWeights.e<Nd4jLong>(i));
    }
}
}
}