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
    auto input = INPUT_VARIABLE(0); // lookup param

    NDArray *reduceShape = nullptr; // this param is optional
    auto output  = OUTPUT_VARIABLE(0); //
    
    int seed = INT_ARG(0);
    
    T probValue = T_ARG(0); 
    if (block.width() > 1)
        reduceShape = INPUT_VARIABLE(1);

    REQUIRE_TRUE(probValue > 0.f && probValue <= 1.f, 0, "dropout: Probability should be with range 0 to 1.");

    if (probValue == 1.0f) {
        *output = *input;
        return Status::OK();
    }
    nd4j::random::RandomBuffer* rng = block.getRNG();
    
    if (rng == nullptr)
        return ND4J_STATUS_BAD_RNG;

    return helpers::dropOutFunctor(rng, input, output, reduceShape, seed, probValue);
}

}
}

#endif