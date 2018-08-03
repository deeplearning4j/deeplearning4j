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
// Created by GS <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_dropout)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/dropout.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(dropout, 1, 1, true, 1, 1) {
    NDArray<T>* input   = INPUT_VARIABLE(0); // lookup param

    NDArray<T>* reduceShape = nullptr; // this param is optional
    NDArray<T>* output  = OUTPUT_VARIABLE(0); // 
    
    int seed = INT_ARG(0);
    
    T probValue = T_ARG(0); 
    if (block.width() > 1)
        reduceShape = INPUT_VARIABLE(1);

    REQUIRE_TRUE(probValue > T(0.f) && probValue <= T(1.f), 0, "dropout: Probability should be with range 0 to 1.");

    if (probValue == T(1.0)) {
        *output = *input;
        return ND4J_STATUS_OK;
    }
    //nd4j::random::RandomBuffer* rng = block.getRNG();
    
    //if (rng == nullptr)
    //    return ND4J_STATUS_BAD_RNG;

    return helpers::dropOutFunctor(input, output, reduceShape, seed, probValue);
}

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(dropout_bp, 2, 1, false, 1, 1) {
    NDArray<T>* input   = INPUT_VARIABLE(0); // lookup param
    NDArray<T>* gradOut   = INPUT_VARIABLE(1); // lookup param

    NDArray<T>* reduceShape = nullptr; // this param is optional
    NDArray<T>* output  = OUTPUT_VARIABLE(0); // 
    
    int seed = INT_ARG(0);
    
    T probValue = T_ARG(0); 
    if (block.width() > 2)
        reduceShape = INPUT_VARIABLE(2);

    REQUIRE_TRUE(probValue > T(0.f) && probValue <= T(1.f), 0, "dropout_bp: Probability should be with range 0 to 1.");
    if (probValue == T(1.0)) {
        output->assign(T(0.f)); // fill up output with 0 
        return ND4J_STATUS_OK;
    }

    //nd4j::random::RandomBuffer* rng = block.getRNG();
    //if (rng == nullptr)
    //    return ND4J_STATUS_BAD_RNG;

    REQUIRE_TRUE(helpers::dropOutFunctorBP(input, gradOut, output, reduceShape, seed, probValue) == ND4J_STATUS_OK, 0, "dropout_bp: Cannot backprop dropout." );

    return ND4J_STATUS_OK;
}

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(alpha_dropout_bp, 2, 1, false, 4, 1) {
    NDArray<T>* input   = INPUT_VARIABLE(0); // lookup param
    NDArray<T>* gradOut   = INPUT_VARIABLE(1); // lookup param

    NDArray<T>* reduceShape = nullptr; // this param is optional
    NDArray<T>* output  = OUTPUT_VARIABLE(0); // 

    if (block.width() > 2)
        reduceShape = INPUT_VARIABLE(2);

    int seed = INT_ARG(0);
    
    T probValue   = T_ARG(0); 
    T alphaValue  = T_ARG(0); 
    T alpha1Value = T_ARG(2); 
    T betaValue   = T_ARG(3); 

    REQUIRE_TRUE(probValue > T(0.f) && probValue <= T(1.f), 0, "dropout_bp: Probability should be with range 0 to 1.");
    if (probValue == T(1.0)) {
        output->assign(T(0.f)); // fill up output with 0 
        return ND4J_STATUS_OK;
    }

    //nd4j::random::RandomBuffer* rng = block.getRNG();
    //if (rng == nullptr)
    //    return ND4J_STATUS_BAD_RNG;

    return helpers::alphaDropOutFunctorBP(input, gradOut, output, reduceShape, seed, probValue, alphaValue, alpha1Value, betaValue);
}

}
}

#endif