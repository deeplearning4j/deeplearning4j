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
// @author George Shulinok <sgazeos@gmail.com>, created on 08.10.2019
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_fake_quant_with_min_max_vars_per_channel)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/fake_quantization.h>
namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(fake_quant_with_min_max_vars_per_channel, 1, 1, true, 0, 0) {

            auto x = INPUT_VARIABLE(0);
            auto min = INPUT_VARIABLE(1);
            auto max = INPUT_VARIABLE(2);

            REQUIRE_TRUE(block.width() == 3 || block.getTArguments()->size() == 2, 0, "fake_quant_with_min_max_vars_per_channel: No minimum/maximum values provided by either input arrays or TArgs");
            auto depth = x->sizeAt(-1);
            REQUIRE_TRUE(min->rankOf() == 1 && max->rankOf() == 1 && min->lengthOf() == max->lengthOf(), 0,
                    "fake_quant_with_min_max_vars_per_channel: Min and Max should be 1D tensors with the same length");
            REQUIRE_TRUE(depth == min->lengthOf(), 0, "fake_quant_with_min_max_vars_per_channel: Min length should be"
                                                      " %lld, but %lld occurs.", depth, min->lengthOf());

            REQUIRE_TRUE(depth == max->lengthOf(), 0, "fake_quant_with_min_max_vars_per_channel: Max length should be"
                                                      "%lld, but %lld occurs.", depth, max->lengthOf());
            auto output  = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->dataType() == output->dataType(), 0, "fake_quant_with_min_max_vars_per_channel: input and output data types must be the same");

            int numBits = 8;
            if (block.getIArguments() && block.getIArguments()->size())
                numBits = INT_ARG(0);
            bool narrowed = false;
            //INT_ARG(1);
            if (block.getIArguments()->size() == 2) {
                numBits = INT_ARG(0);
                narrowed = INT_ARG(1);
                REQUIRE_TRUE(numBits > 1 && numBits < 17, 0, "fake_quant_with_min_max_vars_per_channel: Number of bits"
                                                             " for quatization should be in between 2 and 16, but %i "
                                                             "was given.", numBits);
            }
            helpers::fakeQuantWithMinMaxVarsPerChannel(block.launchContext(), x, min, max, numBits, narrowed, output);
            return ND4J_STATUS_OK;
        }

        DECLARE_TYPES(fake_quant_with_min_max_vars_per_channel) {
            getOpDescriptor()
            -> setAllowedOutputTypes({ALL_FLOATS})
            -> setAllowedInputTypes({ALL_INTS, ALL_FLOATS});
        }

        DECLARE_SYN(fake_quant_with_min_max_args_per_channel, fake_quant_with_min_max_vars_per_channel);
    }
}

#endif