/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#if NOT_EXCLUDED(OP_space_to_depth)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_d.h>

#include <array>
#include <limits>

namespace sd {
namespace ops {


    DECLARE_TYPES(space_to_depth) {
        getOpDescriptor()
                ->setAllowedInputTypes({ALL_FLOATS, ALL_INTS, sd::DataType::BOOL})
                ->setSameMode(true);
      getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING);
}

    CUSTOM_OP_IMPL(space_to_depth, 1, 1, false, 0, 2) {
        const LongType block_size = INT_ARG(0);
        REQUIRE_TRUE(block_size > 0 && block_size <= std::numeric_limits<int>::max(), 0,
                     "SpaceToDepth: block_size must be positive and fit the helper's int argument");

        bool isNHWC = INT_ARG(1) == 1;

        auto input = INPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 4, 0, "SpaceToDepth: input should be 4D array, but got %i instead", input->rankOf());

        LongType bS = input->sizeAt(0);
        LongType iD = isNHWC ? input->sizeAt(3) : input->sizeAt(1);
        LongType iH = isNHWC ? input->sizeAt(1) : input->sizeAt(2);
        LongType iW = isNHWC ? input->sizeAt(2) : input->sizeAt(3);

        REQUIRE_TRUE(iH % block_size == 0 && iW % block_size == 0, 0, "SpaceToDepth: input Height & Width should be divisible by block_size");
        REQUIRE_TRUE(iD <= std::numeric_limits<LongType>::max() / (block_size * block_size), 0,
                     "SpaceToDepth: output channel dimension overflows LongType");

        auto output = OUTPUT_VARIABLE(0);

        helpers::_spaceTodepth(block.launchContext(), *input, output, static_cast<int>(block_size), isNHWC);
        return Status::OK;
    }
    

    DECLARE_SHAPE_FN(space_to_depth) {
        auto in = inputShape->at(0);
        const LongType block_size = INT_ARG(0);
        REQUIRE_TRUE(block_size > 0 && block_size <= std::numeric_limits<int>::max(), 0,
                     "SpaceToDepth: block_size must be positive and fit the helper's int argument");
        bool isNHWC = INT_ARG(1) == 1;

        REQUIRE_TRUE(shape::rank(in) == 4, 0, "SpaceToDepth: input must be rank 4");
        LongType bS = shape::sizeAt(in, static_cast<sd::LongType>(0));
        LongType iD = isNHWC ? shape::sizeAt(in, static_cast<sd::LongType>(3)) : shape::sizeAt(in, static_cast<sd::LongType>(1));
        LongType iH = isNHWC ? shape::sizeAt(in, static_cast<sd::LongType>(1)) : shape::sizeAt(in, static_cast<sd::LongType>(2));
        LongType iW = isNHWC ? shape::sizeAt(in, static_cast<sd::LongType>(2)) : shape::sizeAt(in, static_cast<sd::LongType>(3));

        REQUIRE_TRUE(iH % block_size == 0 && iW % block_size == 0, 0,
                     "SpaceToDepth: input Height & Width must be divisible by block_size");
        const LongType blockArea = block_size * block_size;
        REQUIRE_TRUE(iD <= std::numeric_limits<LongType>::max() / blockArea, 0,
                     "SpaceToDepth: output channel dimension overflows LongType");
        const LongType oD = iD * blockArea;
        const LongType oH = iH / block_size;
        const LongType oW = iW / block_size;
        
        std::array<sd::LongType, 4> shape;
        if (isNHWC) 
            shape = {{bS, oH, oW, oD }};
        else 
            shape = {{bS, oD, oH, oW }};

        auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(in), 'c', 4, shape.data(),0);
        return SHAPELIST(newShape);
    }

}  // namespace ops
}  // namespace sd

#endif
