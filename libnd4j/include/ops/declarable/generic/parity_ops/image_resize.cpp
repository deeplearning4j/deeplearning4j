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
            bool preserveAspectRatio = false; // - default value
            bool antialias = false;
            REQUIRE_TRUE(size->lengthOf() == 2, 0, "resize_bilinear: Resize params is a pair of values, not %lld.", size->lengthOf());
            width = size->e<int>(0);
            height = size->e<int>(1);
            if (block.getBArguments()->size()) {
                preserveAspectRatio = B_ARG(0);
                if (block.getBArguments()->size() > 1)
                    antialias = B_ARG(1);
            }

            auto method = helpers::ImageResizeMethods::kResizeBilinear;
            if (block.numI() == 1) {
                method = (helpers::ImageResizeMethods)INT_ARG(0);
            }

            return helpers::resizeFunctor(block.launchContext(), image, width, height, method, preserveAspectRatio, antialias, output);
        }

        DECLARE_SHAPE_FN(image_resize) {
            auto shapeList = SHAPELIST(); 
            auto in = inputShape->at(0);

            Nd4jLong* outputShape;

            int width;
            int height;
            auto newImageSize = INPUT_VARIABLE(1);
            REQUIRE_TRUE(newImageSize->lengthOf() == 2, 0, "resize_bilinear: Resize params is a pair of values, not %i.", newImageSize->lengthOf());
            REQUIRE_TRUE(block.numI() <= 1, 0, "resize_bilinear: Resize params already given by the second param. Int params are expensive.");
            width = newImageSize->e<int>(0);
            height = newImageSize->e<int>(1);
            
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
        DECLARE_TYPES(image_resize) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    }
}

#endif