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
// Created by GS <sgazeos@gmail.com> at 3/30/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/image_suppression.h>

#if NOT_EXCLUDED(OP_image_non_max_suppression)

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(non_max_suppression, 2, 1, false, 0, 0) {
            auto boxes = INPUT_VARIABLE(0);
            auto scales = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);
            int maxOutputSize; // = INT_ARG(0);
            if (block.width() > 2)
                maxOutputSize = INPUT_VARIABLE(2)->e<int>(0);
            else if (block.getIArguments()->size() == 1)
                maxOutputSize = INT_ARG(0);
            else
                REQUIRE_TRUE(false, 0, "image.non_max_suppression: Max output size argument cannot be retrieved.");

            REQUIRE_TRUE(boxes->rankOf() == 2, 0, "image.non_max_suppression: The rank of boxes array should be 2, but %i is given", boxes->rankOf());
            REQUIRE_TRUE(scales->rankOf() == 1 && scales->lengthOf() == boxes->sizeAt(0), 0, "image.non_max_suppression: The rank of boxes array should be 2, but %i is given", boxes->rankOf());

            if (scales->lengthOf() < maxOutputSize)
                maxOutputSize = scales->lengthOf();
            double threshold = 0.5;
            if (block.getTArguments()->size() > 0)
                threshold = T_ARG(0);

            helpers::nonMaxSuppressionV2(block.launchContext(), boxes, scales, maxOutputSize, threshold, output);
            return Status::OK();
        }

        DECLARE_SHAPE_FN(non_max_suppression) {
            auto in = inputShape->at(0);
            int outRank = shape::rank(in);
            Nd4jLong *outputShape = nullptr;

            int maxOutputSize;
            if (block.width() > 2)
                maxOutputSize = INPUT_VARIABLE(2)->e<int>(0);
            else if (block.getIArguments()->size() == 1)
                maxOutputSize = INT_ARG(0);
            else
                REQUIRE_TRUE(false, 0, "image.non_max_suppression: Max output size argument cannot be retrieved.");


            Nd4jLong boxSize = shape::sizeAt(in, 0);
            if (boxSize < maxOutputSize) 
                maxOutputSize = boxSize;

            outputShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(maxOutputSize, DataType::INT32);

            return SHAPELIST(outputShape);
        }
        DECLARE_TYPES(non_max_suppression) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_INDICES});
        }

    }
}
#endif
