/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
#if NOT_EXCLUDED(OP_resize_area)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_resize.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(resize_area, 1, 1, false, 0, -2) {

            auto image = INPUT_VARIABLE(0);
            int width;
            int height;

            if (block.width() == 2) {
                auto size = INPUT_VARIABLE(1); // integer vector with shape {2} and content (new_height, new_width)
                REQUIRE_TRUE(size->rankOf() == 1, size->lengthOf() == 2, 0, "resize_area: Resize params is a pair of values, not %i.", size->lengthOf());
                size->syncToHost();
                width = size->e<int>(1);
                height = size->e<int>(0);
            }
            else {
                REQUIRE_TRUE(block.numI() == 2, 0, "resize_area: Resize params already given by the second param. Int params are expensive.");
                width = INT_ARG(1);
                height = INT_ARG(0);
            }

            auto output = OUTPUT_VARIABLE(0);
            if (output->isEmpty()) return Status::OK();
            auto inRank = image->rankOf();

            REQUIRE_TRUE(inRank == 3 || inRank == 4, 0, "resize_area: Source tensor should have rank 4, but %i given.", inRank);
            REQUIRE_TRUE(output->rankOf() == inRank, 0, "resize_area: Source tensor and output should have the same rank, but  %i and %i given.", inRank, output->rankOf());
            REQUIRE_TRUE(width > 0 , 0, "resize_area: picture width should be positive 32 bit integer, but %i given", width);
            REQUIRE_TRUE(height > 0 , 0, "resize_area: picture height should be positive 32 bit integer, but %i given", height);
            REQUIRE_TRUE(image->lengthOf() > 0, 0, "resize_area: Only non-zero images allowed to processing.");

            auto alignCorners = false;
            if (block.numB() > 0) {
                alignCorners = B_ARG(0);
            }

            auto source = inRank == 4?image->reshape(image->ordering(), {image->sizeAt(0), image->sizeAt(1), image->sizeAt(2), image->sizeAt(3)}):image->reshape(image->ordering(), {1, image->sizeAt(0), image->sizeAt(1), image->sizeAt(2)});
            auto target = inRank == 4?output->reshape(output->ordering(), {output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3)}):output->reshape(output->ordering(), {1, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2)});

            return helpers::resizeAreaFunctor(block.launchContext(), &source, width, height, alignCorners, &target);
        }

        DECLARE_SHAPE_FN(resize_area) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(0);

            Nd4jLong* outputShape;
            auto inRank = shape::rank(in);
            int width;
            int height;
            if (block.width() == 2) {
                auto newImageSize = INPUT_VARIABLE(1);
                REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0,
                             "resize_area: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
                REQUIRE_TRUE(block.numI() <= 1, 0,
                             "resize_area: Resize params already given by the second param. Int params are expensive.");
                width = newImageSize->e<int>(0);
                height = newImageSize->e<int>(1);
            }
            else {
                REQUIRE_TRUE(block.numI() == 2, 0, "resize_area: Resize params ommited as pair ints nor int tensor.");
                width = INT_ARG(1);
                height = INT_ARG(0);
            }

            REQUIRE_TRUE(inRank == 4 || inRank == 3, 0, "resize_area: Source tensor should have rank 4, but %i given.", inRank);
            
            ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
            outputShape[0] = inRank;
            if (inRank == 4) {
                outputShape[1] = in[1];
                outputShape[2] = width;
                outputShape[3] = height;
                outputShape[4] = in[4];
            }
            else {
                outputShape[1] = width;
                outputShape[2] = height;
                outputShape[3] = in[3];
            }
            ShapeUtils::updateStridesAndType(outputShape, DataType::FLOAT32, shape::order(in));

            shapeList->push_back(CONSTANT(outputShape));
            return shapeList;
        }
        DECLARE_TYPES(resize_area) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
                    ->setAllowedInputTypes(1, DataType::INT32)
                    ->setAllowedOutputTypes({DataType::FLOAT32});
        }

    }
}

#endif