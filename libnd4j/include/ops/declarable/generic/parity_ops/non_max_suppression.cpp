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


namespace sd {
    namespace ops {
#if NOT_EXCLUDED(OP_image_non_max_suppression)
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

            double overlayThreshold = 0.5;
            double scoreThreshold = - DataTypeUtils::infOrMax<float>();

            if (block.width() > 3) {
                overlayThreshold = INPUT_VARIABLE(3)->e<double>(0);
            }
            else if (block.getTArguments()->size() > 0) {
                overlayThreshold = T_ARG(0);
            }

            if (block.width() > 4) {
                scoreThreshold = INPUT_VARIABLE(4)->e<double>(0);
            }
            else if (block.getTArguments()->size() > 1) {
                scoreThreshold = T_ARG(1);
            }
            if (boxes->isEmpty() || scales->isEmpty())
                return Status::OK();

            if (output->isEmpty())
                return Status::OK();

            REQUIRE_TRUE(boxes->rankOf() == 2, 0, "image.non_max_suppression: The rank of boxes array should be 2, "
                                                  "but %i is given", boxes->rankOf());
            REQUIRE_TRUE(boxes->sizeAt(1) == 4, 0, "image.non_max_suppression: The last dimension of boxes array "
                                                   "should be 4, but %i is given", boxes->sizeAt(1));
            REQUIRE_TRUE(scales->rankOf() == 1 && scales->lengthOf() == boxes->sizeAt(0), 0,
                    "image.non_max_suppression: The rank of scales array should be 1, but %i is given", boxes->rankOf());
            REQUIRE_TRUE(overlayThreshold >= 0. && overlayThreshold <= 1., 0, "image.non_max_suppressio: The overlay "
                                                                              "threashold should be in [0, 1], but "
                                                                              "%lf is given.", overlayThreshold);
            REQUIRE_TRUE(boxes->dataType() == scales->dataType(), 0,
                    "image.non_max_suppression: Boxes and scores inputs should have the same data type, but %s and %s "
                    "were given.", DataTypeUtils::asString(boxes->dataType()).c_str(),
                    DataTypeUtils::asString(scales->dataType()).c_str());
            helpers::nonMaxSuppression(block.launchContext(), boxes, scales, maxOutputSize, overlayThreshold,
                    scoreThreshold, output);
            return Status::OK();
        }

        DECLARE_SHAPE_FN(non_max_suppression) {
            auto in = inputShape->at(0);
            int outRank = shape::rank(in);
            const Nd4jLong *outputShape = nullptr;

            int maxOutputSize;
            if (block.width() > 2)
                maxOutputSize = INPUT_VARIABLE(2)->e<int>(0);
            else if (block.getIArguments()->size() == 1)
                maxOutputSize = INT_ARG(0);
            else
                REQUIRE_TRUE(false, 0, "image.non_max_suppression: Max output size argument cannot be retrieved.");

            if (maxOutputSize > 0) {
                auto actualIndicesCount = shape::sizeAt(in, 0);
                if (block.getTArguments()->size() > 1 || block.width() > 4) {
                    auto scoreThreshold =
                            block.getTArguments()->size() > 1 ? T_ARG(1) : INPUT_VARIABLE(4)->e<double>(0);
                    auto scales = INPUT_VARIABLE(1);
                    scales->syncToHost();
                    for (auto e = 0; e < scales->lengthOf(); e++) {
                        if (scales->e<float>(e) < (float) scoreThreshold) {
                            actualIndicesCount--;
                        }
                    }
                }
                if (actualIndicesCount < maxOutputSize)
                    maxOutputSize = actualIndicesCount;
            }
            outputShape = ConstantShapeHelper::getInstance().vectorShapeInfo(maxOutputSize, DataType::INT32);

            return SHAPELIST(outputShape);
        }
        DECLARE_TYPES(non_max_suppression) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_INDICES});
        }
#endif
#if NOT_EXCLUDED(OP_image_non_max_suppression_v3)
        DECLARE_TYPES(non_max_suppression_v3) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_INDICES});
        }

        CUSTOM_OP_IMPL(non_max_suppression_v3, 2, 1, false, 0, 0) {
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

            double overlayThreshold = 0.5;
            double scoreThreshold = - DataTypeUtils::infOrMax<float>();

            if (block.width() > 3) {
                overlayThreshold = INPUT_VARIABLE(3)->e<double>(0);
            }
            else if (block.getTArguments()->size() > 0) {
                overlayThreshold = T_ARG(0);
            }

            if (block.width() > 4) {
                scoreThreshold = INPUT_VARIABLE(4)->e<double>(0);
            }
            else if (block.getTArguments()->size() > 1) {
                scoreThreshold = T_ARG(1);
            }
            if (boxes->isEmpty() || scales->isEmpty())
                return Status::OK();
            if (output->isEmpty())
                return Status::OK();

            REQUIRE_TRUE(boxes->rankOf() == 2, 0, "image.non_max_suppression: The rank of boxes array should be 2, but "
                                                  "%i is given", boxes->rankOf());
            REQUIRE_TRUE(boxes->sizeAt(1) == 4, 0, "image.non_max_suppression: The last dimension of boxes array should "
                                                   "be 4, but %i is given", boxes->sizeAt(1));
            REQUIRE_TRUE(scales->rankOf() == 1 && scales->lengthOf() == boxes->sizeAt(0), 0,
                    "image.non_max_suppression: The rank of scales array should be 1, but %i is given", boxes->rankOf());
            REQUIRE_TRUE(overlayThreshold >= 0. && overlayThreshold <= 1., 0,
                    "image.non_max_suppression_v3: The overlay threashold should be in [0, 1], but %lf given.",
                    overlayThreshold);
            REQUIRE_TRUE(boxes->dataType() == scales->dataType(), 0,
                         "image.non_max_suppression_v3: Boxes and scores inputs should have the same data type, but %s and %s "
                         "were given.", DataTypeUtils::asString(boxes->dataType()).c_str(),
                         DataTypeUtils::asString(scales->dataType()).c_str());

            helpers::nonMaxSuppressionV3(block.launchContext(), boxes, scales, maxOutputSize, overlayThreshold,
                    scoreThreshold, output);
            return Status::OK();
        }

        DECLARE_SHAPE_FN(non_max_suppression_v3) {
            auto in = inputShape->at(0);
            int outRank = shape::rank(in);


            int maxOutputSize;
            if (block.width() > 2)
                maxOutputSize = INPUT_VARIABLE(2)->e<int>(0);
            else if (block.getIArguments()->size() == 1)
                maxOutputSize = INT_ARG(0);
            else
            REQUIRE_TRUE(false, 0, "image.non_max_suppression: Max output size argument cannot be retrieved.");
            auto boxes = INPUT_VARIABLE(0);
            auto scales = INPUT_VARIABLE(1);

            double overlayThreshold = 0.5;
            double scoreThreshold = - DataTypeUtils::infOrMax<float>();

            if (block.width() > 3) {
                overlayThreshold = INPUT_VARIABLE(3)->e<double>(0);
            }
            else if (block.getTArguments()->size() > 0) {
                overlayThreshold = T_ARG(0);
            }

            if (block.width() > 4) {
                scoreThreshold = INPUT_VARIABLE(4)->e<double>(0);
            }
            else if (block.getTArguments()->size() > 1) {
                scoreThreshold = T_ARG(1);
            }

            auto len = maxOutputSize;
            if (len > 0)
                len = helpers::nonMaxSuppressionV3(block.launchContext(), boxes, scales, maxOutputSize, overlayThreshold, scoreThreshold, nullptr);

            auto outputShape = ConstantShapeHelper::getInstance().vectorShapeInfo(len, DataType::INT32);

            return SHAPELIST(outputShape);
        }
#endif

    }
}
