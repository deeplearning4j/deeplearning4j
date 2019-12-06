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
        CUSTOM_OP_IMPL(resize_bilinear, 1, 1, false, 0, -2) {

            NDArray* image = INPUT_VARIABLE(0);
            NDArray* output = OUTPUT_VARIABLE(0);
            int width;
            int height;
            bool alignCorners = false; // - default value
            auto inRank = image->rankOf();
            if (output->isEmpty()) return Status::OK();

            REQUIRE_TRUE( inRank == 4 || inRank == 3, 0, "resize_bilinear: input image should be 4D "
                                                                          "tensor, but input has rank %i",
                                                                          image->rankOf());
            REQUIRE_TRUE(inRank == output->rankOf(), 0, "resize_bilinear: Input and output ranks should be equals, but %i and %i occured.", inRank, output->rankOf());

            auto source = inRank == 4?image->reshape(image->ordering(), {image->sizeAt(0), image->sizeAt(1), image->sizeAt(2), image->sizeAt(3)}):image->reshape(image->ordering(), {1, image->sizeAt(0), image->sizeAt(1), image->sizeAt(2)});
            auto target = inRank == 4?output->reshape(output->ordering(), {output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3)}):output->reshape(output->ordering(), {1, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2)});

            if (block.width() > 1) {
                auto newImageSize = INPUT_VARIABLE(1);
                REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_bilinear: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
                REQUIRE_TRUE(block.numI() <= 1, 0, "resize_bilinear: Resize params already given by the second param. Int params are expensive.");
                height = newImageSize->e<int>(0);
                width = newImageSize->e<int>(1);
            }
            else {
                REQUIRE_TRUE(block.numI() > 1, 0, "resize_bilinear: Neither resize width nor height are provided.");
                height = INT_ARG(0);
                width = INT_ARG(1);
            }

            if (block.numB() > 0)
                alignCorners = B_ARG(0);
            bool halfPixelCenter = false;

            if (block.numB() > 1)
                halfPixelCenter = B_ARG(1);

            REQUIRE_TRUE(!halfPixelCenter || (halfPixelCenter && !alignCorners), 0, "resize_bilinear: `half_pixel_centers' should be false or true only when `align_corners' is false");

            return helpers::resizeBilinearFunctor(block.launchContext(), inRank==4?image:&source, width, height, alignCorners, halfPixelCenter, inRank == 4 ? output : &target);
        }

        DECLARE_SHAPE_FN(resize_bilinear) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(0);

            Nd4jLong* outputShape;
            auto inRank = shape::rank(in);
            REQUIRE_TRUE(inRank == 4 || inRank == 3, 0, "resize_bilinear: input image should be 4D "
                                                                          "tensor, but input has rank %i",
                                                                          inRank);

            int width;
            int height;
            if (block.width() > 1) {
                auto newImageSize = INPUT_VARIABLE(1);
                REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_bilinear: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
                REQUIRE_TRUE(block.numI() <= 1, 0, "resize_bilinear: Resize params already given by the second param. Int params are expensive.");
                width = newImageSize->e<int>(0);
                height = newImageSize->e<int>(1);
            }
            else {
                REQUIRE_TRUE(block.numI() == 2, 0, "resize_bilinear: Neither resize width nor height are provided.");
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
            if (DataTypeUtils::isR(ArrayOptions::dataType(in))) {
                ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));
            }
            else {
                ShapeUtils::updateStridesAndType(outputShape, DataType::FLOAT32, shape::order(in));
            }

            shapeList->push_back(CONSTANT(outputShape));
            return shapeList;
        }
        DECLARE_TYPES(resize_bilinear) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    }
}

#endif