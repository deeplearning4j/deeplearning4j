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
#include <ops/declarable/helpers/legacy_helpers.h>
namespace nd4j {
namespace ops  {


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(thresholdedrelu, 1, 1, true, 0, 0) {
    auto input  = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    auto scalar = block.numT() > 0 ? block.getTArguments()->at(0) : 0.0;

// FIXME: we should have proper extra set here
    input->applyScalar(scalar::RELU, scalar, output);

    return Status::OK();
}


////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(thresholdedrelu_bp, 2, 1, true, 0, 0) {
    auto input = INPUT_VARIABLE(0);
    auto dLdO  = INPUT_VARIABLE(1);
    
    auto dLdI = OUTPUT_VARIABLE(0);

    //const T theta = block.getTArguments()->size() == 0 ? static_cast<T>(1) : T_ARG(0);

    // REQUIRE_TRUE(theta >= static_cast<T>(0), 0, "THRESHOLDED_RELU_BP OP: input float argument theta must be >= 0, but got %f instead !", theta);
/*
    auto derivative = LAMBDA_TT(i, grO, theta) {if (i > theta) return grO; else return static_cast<T>(0); };

    input->applyPairwiseLambda(dLdO, derivative, dLdI);
*/

    // FIXME: we should have proper extra set here
    //input->applyPairwiseTransform(pairwise::RELUDerivativeE, dLdO, dLdI, nullptr);
    helpers::reluDerivative(input, dLdO, dLdI);
    return Status::OK();
}



}
}

#endif