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
#if NOT_EXCLUDED(OP_thresholdedrelu)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(thresholdedrelu, 1, 1, true, 1, 0) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);    
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    const T theta = T_ARG(0);

    REQUIRE_TRUE(theta >= static_cast<T>(0), 0, "THRESHOLDED_RELU OP: input float argument theta must be >= 0, but got %f instead !", theta);

    auto func = LAMBDA_T(i, theta) { if (i > theta) return i; else return static_cast<T>(0); };

    input->applyLambda(func, output);
    
    return Status::OK();
}


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(thresholdedrelu_bp, 2, 1, true, 1, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* dLdO  = INPUT_VARIABLE(1);
    
    NDArray<T>* dLdI = OUTPUT_VARIABLE(0);

    const T theta = T_ARG(0);

    REQUIRE_TRUE(theta >= static_cast<T>(0), 0, "THRESHOLDED_RELU_BP OP: input float argument theta must be >= 0, but got %f instead !", theta);

    auto derivative = LAMBDA_TT(i, grO, theta) {if (i > theta) return grO; else static_cast<T>(0); };

    input->applyPairwiseLambda(dLdO, derivative, dLdI);    
    
    return Status::OK();
}



}
}

#endif