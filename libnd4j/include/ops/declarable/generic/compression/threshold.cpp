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
// @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/threshold.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(encode_threshold, 1, 2, true, 1, 0) {
            auto x = INPUT_VARIABLE(0);
            auto updated = OUTPUT_VARIABLE(0);
            auto encoded = OUTPUT_NULLIFIED(1);

            float threshold = T_ARG(0);

            REQUIRE_TRUE(x->lengthOf() <= DataTypeUtils::max<int>(), 0, "encode_threshold: gradients array must have length <= MAX_INT");
            REQUIRE_TRUE(encoded->lengthOf() >= 4, 0, "encode_threshold: array for encoded updates can't have less than 4 elements");
//            REQUIRE_TRUE(x->platformBuffer() == updated->platformBuffer(), 0, "encode_threshold: gradients array must be the same at input and output");

            // filling header bytes
            encoded->p(0, encoded->lengthOf() - 4);
            encoded->p(1, (int) x->lengthOf());
            encoded->p(2, reinterpret_cast<int *>(&threshold)[0]);
            encoded->p(3, 0); // flag for FLEXIBLE_ENCODING

            // if there's no updates to process - just skip execution
            if (encoded->lengthOf() == 4)
                return Status::OK();

            helpers::thresholdEncode(*x, *encoded, threshold);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(encode_threshold) {
            auto x = INPUT_VARIABLE(0);
            // we have limit option here
            int boundary = block.numI() > 0 ? I_ARG(0) : DataTypeUtils::max<int>();
            float threshold = T_ARG(0);

            REQUIRE_TRUE(boundary >= 0, 0, "encode_threshold: boundary must be positive");
            REQUIRE_TRUE(x->lengthOf() <= DataTypeUtils::max<int>(), 0, "encode_threshold: gradients array must have length <= MAX_INT");

            // we must calculate number of elements that >= threshold
            auto elements = sd::math::nd4j_min<int>(helpers::thresholdEstimate(*x, threshold), boundary);
            if (elements < 2)
                elements = 0;

            // result array must have 4 additional int elements for header
            return SHAPELIST(x->shapeInfo(), sd::ConstantShapeHelper::getInstance().vectorShapeInfo(elements + 4, DataType::INT32));
        }

        DECLARE_TYPES(encode_threshold) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedOutputTypes(0, {ALL_FLOATS})
                    ->setAllowedOutputTypes(1, DataType::INT32);
        }

        CUSTOM_OP_IMPL(decode_threshold, 2, 1, true, 0, 0) {
            auto weights = INPUT_VARIABLE(0);
            auto encoded = INPUT_VARIABLE(1);
            auto updates = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(encoded->lengthOf() >= 4, 0, "decode_threshold: encoded array can't have length < 4");
            REQUIRE_TRUE(updates->lengthOf() == encoded->e<int>(1), 0, "decode_threshold: updates array must have length equal to [%i]", encoded->e<int>(1));
            REQUIRE_TRUE(encoded->e<int>(3) == 0, 0, "decode_threshold: encoded array doesn't look like threshold-encoded");

            helpers::thresholdDecode(*encoded, *updates);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(decode_threshold) {
            auto weights = inputShape->at(0);
            return SHAPELIST(weights);
        }

        DECLARE_TYPES(decode_threshold) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, DataType::INT32)
                    ->setAllowedOutputTypes(0,{ALL_FLOATS});
        }
    }
}