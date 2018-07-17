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
#if NOT_EXCLUDED(OP_get_seed)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(get_seed, -2, 1, false, 0, 0) {
            REQUIRE_TRUE(block.getRNG() != nullptr, 0, "RNG should be defined in Graph");
            auto rng = block.getRNG();
            auto z = OUTPUT_VARIABLE(0);

            z->putScalar(0, (T) rng->getSeed());

            return ND4J_STATUS_OK;
        }

        DECLARE_SHAPE_FN(get_seed) {
            Nd4jLong *newshape;
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);

            newshape[0] = 2;
            newshape[1] = 1;
            newshape[2] = 1;
            newshape[3] = 1;
            newshape[4] = 1;
            newshape[5] = 0;
            newshape[6] = 1;
            newshape[7] = 99;

            return SHAPELIST(newshape);
        }
    }
}

#endif