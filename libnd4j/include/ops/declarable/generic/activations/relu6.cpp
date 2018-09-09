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
// @author Yurii Shyrma, created on 16.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_relu6)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(relu6, 1, 1, true, 1, 0) {
    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    input->applyTransform(nd4j::transform::RELU6, output, &T_ARG(0));
    
    return Status::OK();
}


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(relu6_bp, 2, 1, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);
    auto gradI = OUTPUT_VARIABLE(0);
    
    auto derivative = LAMBDA_TT(inp, grad) {
        if((T)0. < inp && inp < (T)6.)
            return grad;                    // derivative = 1
        else 
            return (T)0.;                   // derivative = 0
    };

    input->applyPairwiseLambda(gradO, derivative, gradI);
    
    return Status::OK();
}



}
}

#endif