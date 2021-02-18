/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dropout)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/dropout.h>

namespace sd {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(dropout, 1, 1, true, 1, 1) {
    auto input = INPUT_VARIABLE(0); // lookup param

    NDArray *reduceShape = nullptr; // this param is optional
    auto output  = OUTPUT_NULLIFIED(0); //
    
    int seed = INT_ARG(0);

    // FIXME: float?
    double probValue = T_ARG(0);
    if (block.width() > 1)
        reduceShape = INPUT_VARIABLE(1);

    REQUIRE_TRUE(probValue > 0.f && probValue <= 1.f, 0, "dropout: Probability should be with range 0 to 1.");

    if (probValue == 1.0) {
        *output = *input;
        return Status::OK();
    }

    return helpers::dropOutFunctor(block, input, output, reduceShape, seed, probValue);
}

        DECLARE_TYPES(dropout) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes({ALL_FLOATS})
                    ->setSameMode(true);
        }

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(dropout_bp, 2, 1, false, 1, 1) {
    NDArray* input   = INPUT_VARIABLE(0); // lookup param
    NDArray* gradOut   = INPUT_VARIABLE(1); // lookup param

    NDArray* reduceShape = nullptr; // this param is optional
    NDArray* output  = OUTPUT_NULLIFIED(0); //
    
    int seed = INT_ARG(0);
    
    double probValue = T_ARG(0); 
    if (block.width() > 2)
        reduceShape = INPUT_VARIABLE(2);

    REQUIRE_TRUE((probValue > 0. && probValue <= 1.), 0, "dropout_bp: Probability should be with range 0 to 1.");
    if (probValue == 1.0) {
        output->assign(0.f); // fill up output with 0
        return ND4J_STATUS_OK;
    }

    REQUIRE_TRUE(helpers::dropOutFunctorBP(block, input, gradOut, output, reduceShape, seed, probValue) == ND4J_STATUS_OK, 0, "dropout_bp: Cannot backprop dropout." );

    return ND4J_STATUS_OK;
}

DECLARE_TYPES(dropout_bp) {
    getOpDescriptor()
            ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS})
            ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CONFIGURABLE_OP_IMPL(alpha_dropout_bp, 2, 1, false, 4, 1) {
    NDArray* input   = INPUT_VARIABLE(0); // lookup param
    NDArray* gradOut   = INPUT_VARIABLE(1); // lookup param

    NDArray* reduceShape = nullptr; // this param is optional
    NDArray* output  = OUTPUT_VARIABLE(0); //

    if (block.width() > 2)
        reduceShape = INPUT_VARIABLE(2);

    int seed = INT_ARG(0);
    
    double probValue   = T_ARG(0);
    double alphaValue  = T_ARG(1);
    double alpha1Value = T_ARG(2);
    double betaValue   = T_ARG(3);

    REQUIRE_TRUE(probValue > 0. && probValue <= 1., 0, "dropout_bp: Probability should be with range 0 to 1.");
    if (probValue == 1.0) {
        output->assign(0.); // fill up output with 0
        return ND4J_STATUS_OK;
    }

    return helpers::alphaDropOutFunctorBP(block, input, gradOut, output, reduceShape, seed, probValue, alphaValue, alpha1Value, betaValue);
}
        DECLARE_TYPES(alpha_dropout_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS})
                    ->setSameMode(true);
        }
}
}

#endif