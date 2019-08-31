/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_bits_hamming_distance)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/helpers.h>
#include <ops/declarable/helpers/hamming.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(bits_hamming_distance, 2, 1, true, 0, 0) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->lengthOf() == y->lengthOf(), 0, "bits_hamming_distance: both arguments must have the same length");
            REQUIRE_TRUE(x->dataType() == y->dataType(), 0, "bits_hamming_distance: both arguments must have the same data type");

            helpers::hamming(block.launchContext(), *x, *y, *output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(bits_hamming_distance) {
            return SHAPELIST(ConstantShapeHelper::getInstance()->scalarShapeInfo(nd4j::DataType::INT64));
        }

        DECLARE_TYPES(bits_hamming_distance) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_INTS})
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes(0, {ALL_INDICES})
                    ->setSameMode(true);
        }
    }
}

#endif