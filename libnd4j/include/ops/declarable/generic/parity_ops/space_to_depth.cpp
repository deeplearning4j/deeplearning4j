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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_space_to_depth)

#include <ops/declarable/headers/parity_ops.h>
#include <array>
#include <ops/declarable/helpers/s_t_d.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(space_to_depth, 1, 1, false, 0, 2) {
        int block_size = INT_ARG(0);
        bool isNHWC = INT_ARG(1) == 1;

        auto input = INPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 4, 0, "SpaceToDepth: input should be 4D array, but got %f instead", input->rankOf());

        int bS = input->sizeAt(0);
        int iD = isNHWC ? input->sizeAt(3) : input->sizeAt(1);
        int iH = isNHWC ? input->sizeAt(1) : input->sizeAt(2);
        int iW = isNHWC ? input->sizeAt(2) : input->sizeAt(3);

        REQUIRE_TRUE(iH % block_size == 0 && iW % block_size == 0, 0, "SpaceToDepth: input Height & Width should be divisible by block_size");

        auto output = OUTPUT_VARIABLE(0);

        helpers::_spaceTodepth(input, output, block_size, isNHWC);        

        return ND4J_STATUS_OK;
    }
    

    DECLARE_SHAPE_FN(space_to_depth) {
        auto in = inputShape->at(0);
        int block_size = INT_ARG(0);
        bool isNHWC = INT_ARG(1) == 1;

        int bS = shape::sizeAt(in, 0);
        int iD = isNHWC ? shape::sizeAt(in, 3) : shape::sizeAt(in, 1);
        int iH = isNHWC ? shape::sizeAt(in, 1) : shape::sizeAt(in, 2);
        int iW = isNHWC ? shape::sizeAt(in, 2) : shape::sizeAt(in, 3);

        int oD = iD * block_size * block_size;
        int oH = iH / block_size;
        int oW = iW / block_size;

        Nd4jLong *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), Nd4jLong);
        std::array<Nd4jLong, 4> shape;
        if (isNHWC) 
            shape = {{bS, oH, oW, oD }};
        else 
            shape = {{bS, oD, oH, oW }};
        shape::shapeBuffer(4, shape.data(), newShape);

        return SHAPELIST(newShape);
    }
}
}

#endif