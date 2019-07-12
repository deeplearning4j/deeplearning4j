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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.03.2019
//

#include <ops/declarable/helpers/gather.h>
#include <numeric>

namespace nd4j {
namespace ops {
namespace helpers {

////////////////////////////////////////////////////////////////////////
void gather(nd4j::LaunchContext * context, const NDArray* input, const NDArray* indices, NDArray* output, const std::vector<int>& intArgs) {

    int axis = intArgs.size() > 0 ? intArgs[0] : 0;
    const int inputRank = input->rankOf();
    if(axis < 0)
        axis += inputRank;

    const int numOfIntArgs = intArgs.size();

    if (indices != nullptr) {  

        // first case: indices consist of only one scalar
        if(indices->isScalar()) {

            if(input->rankOf() <= 1){
                //For scalar indices, rank 0 or 1 input: can't do tensor along dimension 0 as this is whole array... instead, we want to get a scalar
                auto idx = indices->e<Nd4jLong>(0);
                auto scalarNDArray = input->e(idx);
                output->assign(scalarNDArray);
            } 
            else {
                NDArray inSubArr = (*input)(indices->e<Nd4jLong>(0), {axis});
                output->assign(inSubArr);
            }
        }
        else {

            std::vector<int> dimsOut(indices->rankOf());
            std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... axis+indices->rankOf()-1
            const Nd4jLong numOfSubArrs = indices->lengthOf();
            PRAGMA_OMP_PARALLEL_FOR_IF(numOfSubArrs > Environment::getInstance()->tadThreshold())
            for(int i = 0; i < numOfSubArrs; ++i) {
                NDArray subArrOut = (*output)(i, dimsOut);
                NDArray subArrIn  = (*input)(indices->e<Nd4jLong>(i), {axis});
                subArrOut.assign(subArrIn);
            }
        }
    } 
    else {
                
        // we only allow scalar/vector case here
        if (numOfIntArgs == 2) { // scalar case
            output->assign((*input)(intArgs[1], {axis}));
        }
        else { // vector case
            const Nd4jLong numOfSubArrs = intArgs.size() - 1;
            PRAGMA_OMP_PARALLEL_FOR_IF(numOfSubArrs > Environment::getInstance()->tadThreshold())
            for(int i = 0; i < numOfSubArrs; ++i) {
                NDArray subArrOut = (*output)(i, {axis});
                NDArray subArrIn  = (*input)(intArgs[i+1], {axis});
                subArrOut.assign(subArrIn);
            }
        }
    }    
}


}
}
}