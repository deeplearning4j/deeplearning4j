/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
#if NOT_EXCLUDED(OP_resize_bicubic)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_resize.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(resize_bicubic, 2, 1, false, 0, 0) {

            auto image = INPUT_VARIABLE(0);
            auto size = INPUT_VARIABLE(1); // integer vector with shape {2} and content (new_height, new_width)
            size->syncToHost();
            auto output = OUTPUT_VARIABLE(0);
            int width;
            int height;
            auto inRank = image->rankOf();
            if (output->isEmpty()) return Status::OK();

            REQUIRE_TRUE(inRank == 3 || inRank == 4, 0, "resize_bicubic: Source tensor should have rank 4, but %i given.", inRank);
            REQUIRE_TRUE(output->rankOf() == inRank, 0, "resize_bicubic: Source tensor and output should have the same rank, but  %i and %i given.", inRank, output->rankOf());
            REQUIRE_TRUE(size->rankOf() == 1, size->lengthOf() == 2, 0, "resize_bicubic: Resize params is a pair of values, not %i.", size->lengthOf());
            REQUIRE_TRUE(block.numI() <= 1, 0, "resize_bicubic: Resize params already given by the second param. Int params are expensive.");
            width = size->e<int>(1);
            height = size->e<int>(0);
            REQUIRE_TRUE(width > 0 , 0, "resize_bicubic: picture width should be positive 32 bit integer, but %i given", width);
            REQUIRE_TRUE(height > 0 , 0, "resize_bicubic: picture height should be positive 32 bit integer, but %i given", height);
            //REQUIRE_TRUE(image->sizeAt(1) > 3 && image->sizeAt(2) > 3, 0, "resize_cubic: To use bicubic algorithm need at least 16 pixels as source.");
            REQUIRE_TRUE(width > 3 && height > 3, 0, "resize_bicubic: To use bicubic algorithm need at least 16 pixels as target.");
            REQUIRE_TRUE(image->lengthOf() > 0, 0, "resize_bicubic: Only non-zero images allowed to processing.");
//            auto method = 1; //kResizeBilinear;
//            if (block.numI() == 1) {
//                method = INT_ARG(0);
//            }
            auto alignCorners = false;
            auto halfPixelAlign = false;
            if (block.numB() > 0) {
                alignCorners = block.getBArguments()->at(0);
                if (block.numB()> 1)
                    halfPixelAlign = block.getBArguments()->at(1);
            }
            REQUIRE_TRUE(!halfPixelAlign || (halfPixelAlign && !alignCorners), 0, "resize_bicubic: `half_pixel_centers' should be false or true only when `align_corners' is false");

            auto source = inRank == 4?image->reshape(image->ordering(), {image->sizeAt(0), image->sizeAt(1), image->sizeAt(2), image->sizeAt(3)}):image->reshape(image->ordering(), {1, image->sizeAt(0), image->sizeAt(1), image->sizeAt(2)});
            auto target = inRank == 4?output->reshape(output->ordering(), {output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3)}):output->reshape(output->ordering(), {1, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2)});

            return helpers::resizeBicubicFunctorA(block.launchContext(), &source, width, height, alignCorners, halfPixelAlign, &target);
        }

        DECLARE_SHAPE_FN(resize_bicubic) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(0);

            Nd4jLong* outputShape;
            auto inRank = shape::rank(in);
            int width;
            int height;
            auto newImageSize = INPUT_VARIABLE(1);
            REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_bilinear: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
            REQUIRE_TRUE(block.numI() <= 1, 0, "resize_bilinear: Resize params already given by the second param. Int params are expensive.");
            width = newImageSize->e<int>(0);
            height = newImageSize->e<int>(1);

            REQUIRE_TRUE(inRank == 4 || inRank == 3, 0, "resize_bicubic: Source tensor should have rank 4, but %i given.", inRank);
            
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
            ShapeUtils::updateStridesAndType(outputShape, in, shape::order(in));

            shapeList->push_back(CONSTANT(outputShape));
            return shapeList;
        }
        DECLARE_TYPES(resize_bicubic) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {DataType::INT32})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    }
}

#endif