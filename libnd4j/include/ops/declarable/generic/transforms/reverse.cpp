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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 02.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_reverse)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reverse.h>


namespace nd4j {
namespace ops  {

    CONFIGURABLE_OP_IMPL(reverse, 1, 1, true, 0, -2) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);
        std::vector<int> axis;
        auto isLegacy = false;

        if (block.width() > 1) {
            // dynamic input
            axis = INPUT_VARIABLE(1)->template asVectorT<int>();
            if (block.numI() > 0)
                isLegacy = INT_ARG(0) == 1;

        } else if (block.numI() > 0) {
            axis = *block.getIArguments();
        }

        for (int e = 0; e < axis.size(); e++)
            if (axis[e] < 0)
                axis[e] += input->rankOf();

        // if full range of axis - then clear all axis for default behavior
        if (axis.size() == input->rankOf()) {
            bool dropAll = true;
            for (int e = 0; e < axis.size(); e++)
                if (e != axis[e]) {
                    dropAll = false;
                    break;
                }
           if (dropAll)
                axis.clear();
        }
            
        helpers::reverse(input, output, &axis, isLegacy);
   
        return Status::OK();
    }

    DECLARE_SYN(reverse_v2, reverse);

    CUSTOM_OP_IMPL(reverse_bp, 2, 1, false, 0, -2) {
        auto input = INPUT_VARIABLE(0);
        auto eps = block.width() == 3 ? INPUT_VARIABLE(2) : INPUT_VARIABLE(1);

        auto output = OUTPUT_VARIABLE(0);
        std::vector<int> axis;

        if (block.width() == 3) {
            // dynamic input
            axis = INPUT_VARIABLE(1)->template asVectorT<int>();
        } else if (block.numI() > 0) {
            axis = *block.getIArguments();
        }

        for (int e = 0; e < axis.size(); e++)
            if (axis[e] < 0)
                axis[e] += input->rankOf();

        // we just reverse back original array
        helpers::reverse(eps, output, &axis, false);

        return Status::OK();
    }

    DECLARE_SHAPE_FN(reverse_bp) {
        auto in = inputShape->at(0);
        Nd4jLong *out;
        COPY_SHAPE(in, out);

        return SHAPELIST(out);
    }

}
}

#endif