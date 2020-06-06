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
// Created by GS <sgazeos@gmail.com> at 10/17/2019
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_suppression.h>

#if NOT_EXCLUDED(OP_image_non_max_suppression_overlaps)

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(non_max_suppression_overlaps, 2, 1, false, 0, 0) {
            auto boxes = INPUT_VARIABLE(0);
            auto scales = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);
            int maxOutputSize; // = INT_ARG(0);
            if (block.width() > 2)
                maxOutputSize = INPUT_VARIABLE(2)->e<int>(0);
            else if (block.getIArguments()->size() == 1)
                maxOutputSize = INT_ARG(0);
            else
                REQUIRE_TRUE(false, 0, "image.non_max_suppression_overlaps: Max output size argument cannot be retrieved.");
            REQUIRE_TRUE(boxes->rankOf() == 2, 0, "image.non_max_suppression_overlaps: The rank of boxes array should be 2, but %i is given", boxes->rankOf());
            REQUIRE_TRUE(boxes->sizeAt(0) == boxes->sizeAt(1), 0, "image.non_max_suppression_overlaps: The boxes array should be square, but {%lld, %lld} is given", boxes->sizeAt(0), boxes->sizeAt(1));
            REQUIRE_TRUE(scales->rankOf() == 1 && scales->lengthOf() == boxes->sizeAt(0), 0, "image.non_max_suppression_overlaps: The rank of scales array should be 1, but %i is given", boxes->rankOf());

//            if (scales->lengthOf() < maxOutputSize)
//                maxOutputSize = scales->lengthOf();
            double overlapThreshold = 0.5;
            double scoreThreshold = -DataTypeUtils::infOrMax<double>();
            if (block.getTArguments()->size() > 0)
                overlapThreshold = T_ARG(0);
            if (block.getTArguments()->size() > 1)
                scoreThreshold = T_ARG(1);

            // TODO: refactor helpers to multithreaded facility
            helpers::nonMaxSuppressionGeneric(block.launchContext(), boxes, scales, maxOutputSize, overlapThreshold,
                    scoreThreshold, output);
            return Status::OK();
        }

        DECLARE_SHAPE_FN(non_max_suppression_overlaps) {
            auto in = inputShape->at(0);
            int outRank = shape::rank(in);

            int maxOutputSize;
            if (block.width() > 2)
                maxOutputSize = INPUT_VARIABLE(2)->e<int>(0);
            else if (block.getIArguments()->size() == 1)
                maxOutputSize = INT_ARG(0);
            else
                REQUIRE_TRUE(false, 0, "image.non_max_suppression: Max output size argument cannot be retrieved.");

            double overlapThreshold = 0.5;
            double scoreThreshold = 0.;

            Nd4jLong boxSize = helpers::nonMaxSuppressionGeneric(block.launchContext(), INPUT_VARIABLE(0),
                    INPUT_VARIABLE(1), maxOutputSize, overlapThreshold, scoreThreshold, nullptr); //shape::sizeAt(in, 0);
            if (boxSize < maxOutputSize) 
                maxOutputSize = boxSize;

            auto outputShape = ConstantShapeHelper::getInstance().vectorShapeInfo(maxOutputSize, DataType::INT32);

            return SHAPELIST(outputShape);
        }
        DECLARE_TYPES(non_max_suppression_overlaps) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedInputTypes(2, {ALL_INTS})
                    ->setAllowedOutputTypes({ALL_INDICES});
        }

    }
}
#endif
