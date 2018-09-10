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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_roll)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/roll.h>

namespace nd4j {
namespace ops {

    CONFIGURABLE_OP_IMPL(roll, 1, 1, true, 0, 1) {
        auto output = OUTPUT_VARIABLE(0);
        auto input = INPUT_VARIABLE(0);
        bool shiftIsLinear = true;
        //std::vector<int> axes(input->rankOf());
        int shift = INT_ARG(0);
        int inputLen = input->lengthOf();
        if (block.isInplace()) output = input;
        if (shift < 0) {
        // convert shift to positive value between 1 and inputLen - 1
            shift -= inputLen * (shift / inputLen - 1);
        }
        else
        // cut shift to value between 1 and inputLen - 1
            shift %= inputLen;

        if (block.numI() > 1)
            shiftIsLinear = false;
        if (shiftIsLinear) {
            helpers::rollFunctorLinear(input, output, shift, block.isInplace());
        }
        else {
            std::vector<int> axes(block.numI() - 1);
            for (unsigned e = 0; e < axes.size(); ++e) {
                int axe = INT_ARG(e + 1);
                REQUIRE_TRUE(axe < input->rankOf() && axe >= -input->rankOf(), 0, "roll: axe value should be between -%i and %i, but %i was given.",
                    input->rankOf(), input->rankOf() - 1, axe);
                axes[e] = (axe < 0? (input->rankOf() + axe) : axe);
            }
            helpers::rollFunctorFull(input, output, shift, axes, block.isInplace());
        }

        return ND4J_STATUS_OK;
    }
}
}

#endif