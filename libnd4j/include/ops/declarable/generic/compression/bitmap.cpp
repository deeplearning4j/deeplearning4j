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
// @author George A. Shulinok <sgazeos@gmail.com>
//

#include <system/op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/compression.h>

#if NOT_EXCLUDED(OP_decode_bitmap)
namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(decode_bitmap, 2, 1, true, 0, 0) {
            const auto encoded = INPUT_VARIABLE(1);
            auto updates = OUTPUT_VARIABLE(0);

            helpers::decodeBitmap(block.launchContext(), encoded, updates);
            return Status::OK();
        }

        DECLARE_SHAPE_FN(decode_bitmap) {
            auto weights = INPUT_VARIABLE(0);

            return SHAPELIST(weights->shapeInfo());
        }

        DECLARE_TYPES(decode_bitmap) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, DataType::INT32)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}
#endif

#if NOT_EXCLUDED(OP_encode_bitmap)
namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(encode_bitmap, 1, 3, true, 1, 0) {
            auto input = INPUT_VARIABLE(0);
            auto encoded = OUTPUT_NULLIFIED(1);
            auto counter = OUTPUT_NULLIFIED(2);

            float threshold = T_ARG(0);

            encoded->p(0, (int) input->lengthOf());
            encoded->p(1, (int) input->lengthOf());
            encoded->p(2, reinterpret_cast<int *>(&threshold)[0]);
            encoded->p(3, 1); // flag for BITMAP_ENCODING

            auto result = helpers::encodeBitmap(block.launchContext(), input, encoded, threshold);
            counter->p(0, result);
            counter->syncToDevice();

            return Status::OK();
        }

        DECLARE_SHAPE_FN(encode_bitmap) {
            auto input = inputShape->at(0);

            auto outputLength = shape::length(input) / 16 + 5;
            auto encodedShape = ConstantShapeHelper::getInstance()->vectorShapeInfo(outputLength, DataType::INT32);
            auto encodedCounter = ConstantShapeHelper::getInstance()->scalarShapeInfo(DataType::INT32);
            return SHAPELIST(input, encodedShape, encodedCounter);
        }

        DECLARE_TYPES(encode_bitmap) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, DataType::INT32)
                    ->setAllowedInputTypes(2, DataType::INT32);
        }
    }
}
#endif