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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/flatten.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(flatten, -1, 1, false, 0, 1) {
            auto output = OUTPUT_VARIABLE(0);
            auto zType = output->dataType();
            auto xType = INPUT_VARIABLE(0)->dataType();

            REQUIRE_TRUE(xType == zType, 0, "Flatten: output array must have same data type as input arrays");
            std::vector<NDArray*> arrays(block.width());
            for (int e = 0; e < block.width(); e++) {
                auto input = INPUT_VARIABLE(e);

                REQUIRE_TRUE(xType == input->dataType(), 0, "Flatten: all input arrays must have the same data type");

                arrays[e] = input;
            }

            char order = (char) INT_ARG(0);
            helpers::flatten(arrays, output, order);

            return Status::OK();
        }

        DECLARE_TYPES(flatten) {
            getOpDescriptor()->setAllowedInputTypes({ALL_INTS, ALL_FLOATS, nd4j::DataType::BOOL});
            getOpDescriptor()->setAllowedOutputTypes(0, {ALL_FLOATS, ALL_INTS, nd4j::DataType::BOOL});
        }

        DECLARE_SHAPE_FN(flatten) {
            Nd4jLong length = 0;
            nd4j::DataType dtype = ArrayOptions::dataType(inputShape->at(0));
            for (int e = 0; e < inputShape->size(); e++) {
                length += shape::length(inputShape->at(e));
                REQUIRE_TRUE(dtype == ArrayOptions::dataType(inputShape->at(e)), 0, "Flatten: all input arrays must have the same datatype");
            }

            return SHAPELIST(ShapeBuilders::createVectorShapeInfo(dtype, length, block.getWorkspace()));
        }
    }
}

#endif