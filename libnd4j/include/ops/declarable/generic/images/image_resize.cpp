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
//  @author sgazeos@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_image_resize)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_resize.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(image_resize, 2, 1, false, 0, 0) {

            auto image = INPUT_VARIABLE(0);
            auto size = INPUT_VARIABLE(1);

            auto output = OUTPUT_VARIABLE(0);

            int width;
            int height;
            bool antialias = false;
            REQUIRE_TRUE(size->lengthOf() == 2, 0, "image_resize: Resize params is a pair of values, not %lld.", size->lengthOf());
            width = size->e<int>(1);
            height = size->e<int>(0);
            if (block.numB() == 2) {
                antialias = B_ARG(1);
            }

            auto method = helpers::ImageResizeMethods::kResizeBilinear;
            if (block.numI() == 1) {
                method = (helpers::ImageResizeMethods)INT_ARG(0);
            }
            REQUIRE_TRUE(method == helpers::ImageResizeMethods::kResizeNearest || output->dataType() == DataType::FLOAT32, 0, "image_resize: Output data type should be FLOAT32 for this method %i", (int)method );
            REQUIRE_TRUE(method >= helpers::ImageResizeMethods::kResizeFirst && method <= helpers::ImageResizeMethods::kResizeLast, 0, "image_resize: Resize method should be between %i and %i, but %i was given.", (int)helpers::ImageResizeMethods::kResizeFirst, (int)helpers::ImageResizeMethods::kResizeLast, (int)method);
            auto inRank = image->rankOf();
            REQUIRE_TRUE(inRank >=3 && inRank <=4, 0, "image_resize: Input rank should be 4 or 3, but %i given.", image->rankOf());
            auto source = inRank == 4?image->reshape(image->ordering(), {image->sizeAt(0), image->sizeAt(1), image->sizeAt(2), image->sizeAt(3)}):image->reshape(image->ordering(), {1, image->sizeAt(0), image->sizeAt(1), image->sizeAt(2)});
            auto target = inRank == 4?output->reshape(output->ordering(), {output->sizeAt(0), output->sizeAt(1), output->sizeAt(2), output->sizeAt(3)}, false) : output->reshape(output->ordering(), {1, output->sizeAt(0), output->sizeAt(1), output->sizeAt(2)}, false);

            return helpers::resizeFunctor(block.launchContext(), image, width, height, method, antialias, output);
        }

        DECLARE_SHAPE_FN(image_resize) {
            auto in = inputShape->at(0);

            Nd4jLong* outputShape;
            auto method = helpers::ImageResizeMethods::kResizeBilinear;
            if (block.numI() == 1) {
                method = (helpers::ImageResizeMethods)INT_ARG(0);
            }

            int width;
            int height;
            double ratio = shape::sizeAt(in, 1) / (0.0 + shape::sizeAt(in, 2));
            auto newImageSize = INPUT_VARIABLE(1);
            REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_bilinear: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
            REQUIRE_TRUE(block.numI() <= 1, 0, "resize_bilinear: Resize params already given by the second param. Int params are expensive.");
            width = newImageSize->e<int>(1);
            height = newImageSize->e<int>(0);
            if (block.numB() > 0) {
                if (B_ARG(0)) {
                    width = math::nd4j_ceil<double, int>(height / ratio);
                }
            }
            auto dtype = DataType::FLOAT32;
            if (method == helpers::ImageResizeMethods::kResizeNearest)
                dtype = ArrayOptions::dataType(in);
            auto shape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', shape::rank(in) == 4?std::vector<Nd4jLong>{in[1], height, width, in[4]}:std::vector<Nd4jLong>{ height, width, in[4]});

            return SHAPELIST(shape);
        }
        DECLARE_TYPES(image_resize) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes({ALL_FLOATS, ALL_INTS});
        }

    }
}

#endif