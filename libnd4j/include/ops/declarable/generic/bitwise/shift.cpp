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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_shift_bits)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/shift.h>

namespace sd {
    namespace ops {
        BROADCASTABLE_OP_IMPL(shift_bits, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            BROADCAST_CHECK_EMPTY(x,y,z);

            x->applyTrueBroadcast(BroadcastIntOpsTuple::custom(scalar::ShiftLeft, pairwise::ShiftLeft, broadcast::ShiftLeft), *y, *z, false);

            return Status::OK();
        }

        DECLARE_TYPES(shift_bits) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_INTS})
                    ->setSameMode(true);
        }
    }
}

#endif