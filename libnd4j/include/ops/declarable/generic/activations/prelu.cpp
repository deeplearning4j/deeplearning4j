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
// @author Yurii Shyrma, created on 24.07.2018
//


#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_prelu)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(prelu, 2, 1, true, 0, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* alpha  = INPUT_VARIABLE(1);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    auto preluFunc = LAMBDA_TT(i, a) { if (i < static_cast<T>(0)) return i * a; else return i; };

    input->applyPairwiseLambda(alpha, preluFunc, output);
    
    return Status::OK();
}


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(prelu_bp, 3, 2, true, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* alpha = INPUT_VARIABLE(1);
    NDArray<T>* dLdO  = INPUT_VARIABLE(2);
    
    NDArray<T>* dLdI = OUTPUT_VARIABLE(0);
    NDArray<T>* dLdA = OUTPUT_VARIABLE(1);
    
    auto derivI = LAMBDA_TTT(i, a, grO) {if (i < static_cast<T>(0)) return a * grO; else return grO; };
    auto derivA = LAMBDA_TTT(i, a, grO) {if (i < static_cast<T>(0)) return i * grO; else return static_cast<T>(0); };
        
    input->applyTriplewiseLambda(alpha, dLdO, derivI, dLdI);
    input->applyTriplewiseLambda(alpha, dLdO, derivA, dLdA);    
    
    return Status::OK();
}



}
}

#endif