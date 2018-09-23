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
#if NOT_EXCLUDED(OP_depth_to_space)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/d_t_s.h>
#include <array>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(depth_to_space, 1, 1, false, 0, 2) {
        int block_size = INT_ARG(0);
        bool isNHWC = INT_ARG(1) == 1;

        auto input = INPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 4, 0, "DepthToSpace: input should be 4D array, but got %f instead", input->rankOf());

        int bS = input->sizeAt(0);
        int iD = isNHWC ? input->sizeAt(3) : input->sizeAt(1);
        int iH = isNHWC ? input->sizeAt(1) : input->sizeAt(2);
        int iW = isNHWC ? input->sizeAt(2) : input->sizeAt(3);

        REQUIRE_TRUE(iD % (block_size * block_size) == 0, 0, "DepthToSpace: input number of channels should be divisible by square(block_size)");

        auto output = OUTPUT_VARIABLE(0);

        helpers::_depthToSpace(input, output, block_size, isNHWC);   

        STORE_RESULT(output);     

        return ND4J_STATUS_OK;
    }
    

    DECLARE_SHAPE_FN(depth_to_space) {
        auto in = inputShape->at(0);
        auto block_size = INT_ARG(0);
        bool isNHWC = INT_ARG(1) == 1;

        int bS = shape::sizeAt(in, 0);
        int iD = isNHWC ? shape::sizeAt(in, 3) : shape::sizeAt(in, 1);
        int iH = isNHWC ? shape::sizeAt(in, 1) : shape::sizeAt(in, 2);
        int iW = isNHWC ? shape::sizeAt(in, 2) : shape::sizeAt(in, 3);

        int oD = iD / (block_size * block_size);
        int oH = iH * block_size;
        int oW = iW * block_size;

        Nd4jLong *newShape;
        ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(4), Nd4jLong);
        std::array<Nd4jLong, 4> shape;
        if (isNHWC) 
            shape = {{bS, oH, oW, oD }};
        else 
            shape = {{bS, oD, oH, oW }};

        shape::shapeBuffer(4, block.dataType(), shape.data(), newShape);

        return SHAPELIST(newShape);
    }
}
}

#endif