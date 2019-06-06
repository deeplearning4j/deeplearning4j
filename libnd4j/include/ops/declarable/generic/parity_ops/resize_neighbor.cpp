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
//  @author sgazeos@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_resize_bilinear)

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_resize.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(resize_nearest_neighbor, 1, 1, false, 0, -2) {

            auto image = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);
            int width;
            int height;
            bool center = false; // - default value
            if (block.width() > 1) {
                auto newImageSize = INPUT_VARIABLE(1);
                REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_nearest_neighbor: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
                REQUIRE_TRUE(block.numI() <= 1, 0, "resize_nearest_neighbor: Resize params already given by the second param. Int params are expensive.");
                width = newImageSize->e<int>(0);
                height = newImageSize->e<int>(1);
                if (block.numI() == 1) {
                    center = 0 != INT_ARG(0);
                }
            }
            else {
                REQUIRE_TRUE(block.numI() <= 3, 0, "resize_nearest_neighbor: Neither resize width nor height are provided.");
                width = INT_ARG(0);
                height = INT_ARG(1);
                if (block.numI() == 3)
                    center = 0 != INT_ARG(2);
            }

            return helpers::resizeNeighborFunctor(block.launchContext(), image, width, height, center, output);
        }

        DECLARE_SHAPE_FN(resize_nearest_neighbor) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(0);

            Nd4jLong* outputShape;

            int width;
            int height;
            if (block.width() > 1) {
                auto newImageSize = INPUT_VARIABLE(1);
                REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_nearest_neighbor: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
                REQUIRE_TRUE(block.numI() <= 1, 0, "resize_nearest_neighbor: Resize params already given by the second param. Int params are expensive.");
                width = newImageSize->e<int>(0);
                height = newImageSize->e<int>(1);
            }
            else {
                REQUIRE_TRUE(block.numI() <= 3, 0, "resize_nearest_neighbor: Neither resize width nor height are provided.");
                width = INT_ARG(0);
                height = INT_ARG(1);
            }
            
            ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(4), Nd4jLong);
            outputShape[0] = 4;
            outputShape[1] = in[1];
            outputShape[2] = width;
            outputShape[3] = height;
            outputShape[4] = in[4];
            ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));

            shapeList->push_back(CONSTANT(outputShape));
            return shapeList;
        }
        DECLARE_TYPES(resize_nearest_neighbor) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    }
}

#endif