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

namespace sd {
namespace ops {


    DECLARE_TYPES(space_to_depth) {
        getOpDescriptor()
                ->setAllowedInputTypes(sd::DataType::ANY)
                ->setSameMode(true);
    }

    CUSTOM_OP_IMPL(space_to_depth, 1, 1, false, 0, 2) {
        int block_size = INT_ARG(0);
        REQUIRE_TRUE(block_size > 0,0, "SpaceToDepth: input should be > 0");

        bool isNHWC = INT_ARG(1) == 1;

        auto input = INPUT_VARIABLE(0);

        REQUIRE_TRUE(input->rankOf() == 4, 0, "SpaceToDepth: input should be 4D array, but got %f instead", input->rankOf());

        int bS = input->sizeAt(0);
        int iD = isNHWC ? input->sizeAt(3) : input->sizeAt(1);
        int iH = isNHWC ? input->sizeAt(1) : input->sizeAt(2);
        int iW = isNHWC ? input->sizeAt(2) : input->sizeAt(3);

        REQUIRE_TRUE(iH % block_size == 0 && iW % block_size == 0, 0, "SpaceToDepth: input Height & Width should be divisible by block_size");

        auto output = OUTPUT_VARIABLE(0);

        if (shape::strideDescendingCAscendingF(input->shapeInfo()))
            helpers::_spaceTodepth(block.launchContext(), *input, output, block_size, isNHWC);
        else {
          NDArray *inputDup = input->dup(input->ordering());
          helpers::_spaceTodepth(block.launchContext(), *inputDup, output, block_size, isNHWC);
        }
        return Status::OK;
    }
    

    DECLARE_SHAPE_FN(space_to_depth) {
        auto in = inputShape->at(0);
        int block_size = INT_ARG(0);
        REQUIRE_TRUE(block_size > 0,0, "SpaceToDepth: input should be > 0");
        bool isNHWC = INT_ARG(1) == 1;

        int bS = shape::sizeAt(in, static_cast<sd::LongType>(0));
        int iD = isNHWC ? shape::sizeAt(in, static_cast<sd::LongType>(3)) : shape::sizeAt(in, static_cast<sd::LongType>(1));
        int iH = isNHWC ? shape::sizeAt(in, static_cast<sd::LongType>(1)) : shape::sizeAt(in, static_cast<sd::LongType>(2));
        int iW = isNHWC ? shape::sizeAt(in, static_cast<sd::LongType>(2)) : shape::sizeAt(in, static_cast<sd::LongType>(3));

        int oD = iD * block_size * block_size;
        int oH = iH / block_size;
        int oW = iW / block_size;
        
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
