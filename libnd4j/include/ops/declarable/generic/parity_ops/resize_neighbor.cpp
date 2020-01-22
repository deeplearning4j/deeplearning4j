/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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
#if NOT_EXCLUDED(OP_resize_nearest_neighbor)

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_resize.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(resize_nearest_neighbor, 1, 1, false, 0, -2) {

            auto image = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);
            auto inRank = image->rankOf();
            int width;
            int height;
            bool alignCorners = false; // - default value
            if (output->isEmpty()) return Status::OK();
            if (block.width() > 1) {
                auto newImageSize = INPUT_VARIABLE(1);
                REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_nearest_neighbor: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
                REQUIRE_TRUE(block.numI() <= 1, 0, "resize_nearest_neighbor: Resize params already given by the second param. Int params are expensive.");
                height = newImageSize->e<int>(0);
                width = newImageSize->e<int>(1);
            }
            else {
                REQUIRE_TRUE(block.numI() == 2, 0, "resize_nearest_neighbor: Neither resize width nor height are provided.");
                height = INT_ARG(0);
                width = INT_ARG(1);
            }
            if (block.numB() > 0)
                alignCorners = B_ARG(0);
            bool halfPixelCenter = false;

            if (block.numB() > 1)
                halfPixelCenter = B_ARG(1);
            REQUIRE_TRUE(width <= (1 << 24) || height <= (1 << 24), 0, "resize_nearest_neighbor: the image resize should be limited to 2^24 pixels both for height and width, but %d and %d were given.", height, width);
            REQUIRE_TRUE(inRank == 4 || inRank == 3, 0, "resize_nearest_neighbor: Input should be 4D tensor, but rank %i occured");
            REQUIRE_TRUE(inRank == output->rankOf(), 0, "resize_nearest_neighbor: Input and output ranks should be equals, but %i and %i occured.", inRank, output->rankOf());
            REQUIRE_TRUE(image->dataType() == output->dataType(), 0, "resize_nearest_neighbor: Input and output types should be the same, but `%s' occured instead.", DataTypeUtils::asString(output->dataType()).c_str());
            REQUIRE_TRUE(!halfPixelCenter || (halfPixelCenter && !alignCorners), 0, "resize_nearest_neighbor: `half_pixel_centers' should be false or true only when `align_corners' is false");
            REQUIRE_TRUE(((alignCorners && height > 2) || (height > 0)) && ((alignCorners && width > 1) || (width > 0)), 0,  "resize_nearest_neighbor: Wrong input or output size to resize (width = %d, height = %d)", width, height);

            auto source = inRank == 4?*image:image->reshape(image->ordering(), {1, image->sizeAt(0), image->sizeAt(1), image->sizeAt(2)});
            auto target = inRank == 4?*output:output->reshape(output->ordering(), {1, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2)});

            return helpers::resizeNeighborFunctor(block.launchContext(), inRank==4?image:&source, width, height, alignCorners, halfPixelCenter, inRank == 4 ? output : &target);
        }

        DECLARE_SHAPE_FN(resize_nearest_neighbor) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(0);
            auto inRank = shape::rank(in);
            Nd4jLong* outputShape;

            REQUIRE_TRUE(inRank == 4 || inRank == 3, 0, "resize_nearest_neighbor: input image should be 4D "
                                                        "tensor, but input has rank %i",
                         inRank);

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

            ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
            outputShape[0] = inRank;
            if (inRank == 4) {
                outputShape[1] = in[1];
                outputShape[2] = width;
                outputShape[3] = height;
                outputShape[4] = in[4];
            }
            else { // input shape is 3D, so result also should be 3D
                outputShape[1] = width;
                outputShape[2] = height;
                outputShape[3] = in[3];
            }
            ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));

            shapeList->push_back(CONSTANT(outputShape));
            return shapeList;
        }
        DECLARE_TYPES(resize_nearest_neighbor) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_INTS, ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
        }

    }
}

#endif