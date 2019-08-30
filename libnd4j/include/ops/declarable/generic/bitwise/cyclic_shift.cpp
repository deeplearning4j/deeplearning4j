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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_cyclic_shift_bits)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/shift.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(cyclic_shift_bits, 1, 1, true, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(block.numI() > 0 || block.width() > 1, 0, "cyclic_shift_bits: actual shift value is missing");

            uint32_t shift = 0;
            if (block.width() > 1) {
                shift = INPUT_VARIABLE(1)->e<uint32_t>(0);
            } else if (block.numI() > 0) {
                shift = INT_ARG(0);
            };

            helpers::cyclic_shift_bits(block.launchContext(), *input, *output, shift);

            REQUIRE_TRUE(shift > 0 && shift < input->sizeOfT() * 8, 0, "cyclic_shift_bits: can't shift beyond size of data type")

            return Status::OK();
        }

        DECLARE_TYPES(cyclic_shift_bits) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_INTS})
                    ->setSameMode(true);
        }
    }
}

#endif