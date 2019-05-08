/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author George Shulinok (sgazeos@gmail.com), created on 13.11.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_fake_quant_with_min_max_vars)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/fake_quantization.h>
namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(fake_quant_with_min_max_vars, 3, 1, true, 0, 0) {

            auto x = INPUT_VARIABLE(0);
            auto min = INPUT_VARIABLE(1);
            auto max = INPUT_VARIABLE(2);
            auto output  = OUTPUT_VARIABLE(0);
            bool narrowed = false; //INT_ARG(1);
            int numBits = 8; //INT_ARG(0);
            if (block.getIArguments()->size() == 2) {
                narrowed =INT_ARG(1);
                numBits = INT_ARG(0);
                REQUIRE_TRUE(numBits > 1 && numBits < 17, 0, "fake_quant_with_min_max_vars: Number of bits for quatization should be in between 2 and 16, but %i was given.", numBits);
            }

            helpers::fakeQuantWithMinMaxVars(x, min, max, numBits, narrowed, output);
            return ND4J_STATUS_OK;
        }

        DECLARE_TYPES(fake_quant_with_min_max_vars) {
            getOpDescriptor()
            -> setAllowedOutputTypes({ALL_FLOATS})
            -> setAllowedInputTypes({ALL_INTS, ALL_FLOATS});
        }
    }
}

#endif