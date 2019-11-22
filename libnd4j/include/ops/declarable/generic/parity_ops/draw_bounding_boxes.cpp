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
//  @author George A. Shulinok <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_draw_bounding_boxes)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/image_draw_bounding_boxes.h>
namespace nd4j {
    namespace ops {
        OP_IMPL(draw_bounding_boxes, 3, 1, true) {

            auto images = INPUT_VARIABLE(0);
            auto boxes = INPUT_VARIABLE(1);

            auto colors = (NDArray*) nullptr;
            if (block.width() > 2) // TF v.1.x ommits color set for boxes, and use color 1.0 for fill up
                colors = INPUT_VARIABLE(2); // but v.2.y require color set

            auto output = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(images->dataType() == output->dataType(), 0, "draw_bounding_boxes: Input and Output types "
                                                                      "should be equals, but %d and %d occured.",
                                                                      (int)images->dataType(), (int)output->dataType());
            REQUIRE_TRUE(images->rankOf() == 4, 0, "draw_bounding_boxes: Images input should be 4D tensor, but %i occured.",
                    images->rankOf());
            REQUIRE_TRUE(boxes->rankOf() == 3, 0, "draw_bounding_boxes: Boxes should be 3D tensor, but %i occured.",
                    boxes->rankOf());
            if (colors) {

                REQUIRE_TRUE(colors->rankOf() == 2, 0, "draw_bounding_boxes: Color set should be 2D matrix, but %i occured.",
                        colors->rankOf());
                REQUIRE_TRUE(colors->sizeAt(1) >= images->sizeAt(3), 0, "draw_bounding_boxes: Color set last dim "
                                                                        "should be not less than images depth, but "
                                                                        "%lld and %lld occured.",
                                                                        colors->sizeAt(1), images->sizeAt(3));
            }
            REQUIRE_TRUE(boxes->sizeAt(0) == images->sizeAt(0), 0, "draw_bounding_boxes: Batches for images and boxes "
                                                                   "should be the same, but %lld and %lld occured.",
                                                                   images->sizeAt(0), boxes->sizeAt(0));
            helpers::drawBoundingBoxesFunctor(block.launchContext(), images, boxes, colors, output);
            return ND4J_STATUS_OK;
        }

        DECLARE_TYPES(draw_bounding_boxes) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {HALF, FLOAT32})// TF allows HALF and FLOAT32 only
                    ->setAllowedInputTypes(1, {FLOAT32}) // as TF
                    ->setAllowedInputTypes(2, {FLOAT32}) // as TF
                    ->setAllowedOutputTypes({HALF, FLOAT32}); // TF allows HALF and FLOAT32 only
        }
    }
}

#endif
