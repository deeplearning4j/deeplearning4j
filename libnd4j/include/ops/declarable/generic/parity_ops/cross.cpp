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
#if NOT_EXCLUDED(OP_cross)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/cross.h>

namespace nd4j {
namespace ops {
    OP_IMPL(cross, 2, 1, false) {
        auto a = INPUT_VARIABLE(0);
        auto b = INPUT_VARIABLE(1);

        REQUIRE_TRUE(a->lengthOf() == b->lengthOf(), 0, "Cross: A and B lengths should match");
        REQUIRE_TRUE(a->rankOf() >= 1 && b->rankOf() >= 1, 0, "Cross: A and B should have rank >= 1");

        // TODO: we might want to lift this restriction
        REQUIRE_TRUE(a->isSameShape(b),0, "Cross: A and B should have equal shape");
        REQUIRE_TRUE(a->sizeAt(-1) == 3, 0, "Cross: outer dimension of A and B should be equal to 3");

        auto o = OUTPUT_VARIABLE(0);

        if (a->lengthOf() == 3) {
            helpers::_cross(a, b, o);
        } else {
            helpers::_crossBatched(a, b, o);
        }

        return ND4J_STATUS_OK;
    }
}
}

#endif