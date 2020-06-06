/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_resize_images)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_resize.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(resize_images, 1, 1, false, 0, 0) {

            auto image = INPUT_VARIABLE(0);

            auto output = OUTPUT_VARIABLE(0);
            int width = output->sizeAt(2);
            int height = output->sizeAt(1);
            int method = helpers::ImageResizeMethods::kResizeBilinear;
            if (block.width() > 1) {
                auto size = INPUT_VARIABLE(1);
                REQUIRE_TRUE(size->lengthOf() == 2, 0, "resize_images: Resize params is a pair of values, not %lld.", size->lengthOf());
//                width = size->e<int>(1);
//                height = size->e<int>(0);
                if (block.width() > 2) {
                    auto methodT = INPUT_VARIABLE(2);

                    REQUIRE_TRUE(methodT->isZ() && methodT->isScalar(), 0, "resize_images: Method tensor should be integer scalar, but rank of %i tensor given.", methodT->rankOf());
                    method = methodT->e<int>(0);
                }
                else if (block.numI() == 1) {
                    method = I_ARG(0);
                }
            }
            else {
                REQUIRE_TRUE(block.numI() > 1 && block.numI() < 4, 0, "resize_images: Method and size should be given properly.");
                if(block.numI() == 3) { // full stack of args
//                    height = I_ARG(0);
//                    width = I_ARG(1);
                    method = I_ARG(2);
                }
                else if (block.numI() == 2) {
//                    height = I_ARG(0);
//                    width = I_ARG(1);
                }
            }
            bool preserveAspectRatio = false; // - default value
            bool alignCorners = false;
            if (block.numB()) {
                alignCorners = B_ARG(0);
                if (block.numB() > 1)
                    preserveAspectRatio = B_ARG(1);
            }
            REQUIRE_TRUE(method >= helpers::ImageResizeMethods::kResizeFirst && method <= helpers::ImageResizeMethods::kResizeOldLast, 0, "resize_images: Resize method should be between %i and %i, but %i was given.", (int)helpers::ImageResizeMethods::kResizeFirst, (int)helpers::ImageResizeMethods::kResizeOldLast, (int)method);
            REQUIRE_TRUE(method == helpers::ImageResizeMethods::kResizeNearest || output->dataType() == DataType::FLOAT32, 0, "image_resize: Output data type should be FLOAT32 for this method %i", (int)method );

            auto inRank = image->rankOf();
            REQUIRE_TRUE(inRank >=3 && inRank <=4, 0, "image_resize: Input rank should be 4 or 3, but %i given.", inRank);

            auto source = inRank == 4?image->reshape(image->ordering(), {image->sizeAt(0), image->sizeAt(1), image->sizeAt(2), image->sizeAt(3)}):image->reshape(image->ordering(), {1, image->sizeAt(0), image->sizeAt(1), image->sizeAt(2)});
            auto target = inRank == 4?output->reshape(output->ordering(), {output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3)}, false) : output->reshape(output->ordering(), {1, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2)}, false);

            return helpers::resizeImagesFunctor(block.launchContext(), &source, width, height, (helpers::ImageResizeMethods)method, alignCorners, &target);
        }

        DECLARE_SHAPE_FN(resize_images) {
            auto in = inputShape->at(0);

            Nd4jLong* outputShape;

            int width;
            int height;
            if (block.width() > 1) {
                auto size = INPUT_VARIABLE(1);
                REQUIRE_TRUE(size->lengthOf() == 2, 0, "resize_images: Resize params is a pair of values, not %lld.", size->lengthOf());
                width = size->e<int>(1);
                height = size->e<int>(0);
            }
            else {
                REQUIRE_TRUE(block.numI() > 1 && block.numI() < 4, 0, "resize_images: Method and size should be given properly.");
                if(block.numI() == 3) { // full stack of args
                    height = I_ARG(0);
                    width = I_ARG(1);
                }
                else if (block.numI() == 2) {
                    height = I_ARG(0);
                    width = I_ARG(1);
                }
            }

            double ratio = shape::sizeAt(in, 1) / (0.0 + shape::sizeAt(in, 2));
            if (block.numB() > 1) {
                if (B_ARG(1)) {
                    width = math::nd4j_ceil<double, int>(height / ratio);
                }
            }

            std::vector<Nd4jLong> shape;
            if (shape::rank(in) == 4)
                shape = {in[1], height, width, in[4]};
            else if (shape::rank(in) == 3)
                shape = {height, width, in[3]};

            auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(DataType::FLOAT32, shape::order(in), shape);
            return SHAPELIST(outShape);
        }
        DECLARE_TYPES(resize_images) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS, ALL_INTS})
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes({DataType::FLOAT32});
        }

    }
}

#endif