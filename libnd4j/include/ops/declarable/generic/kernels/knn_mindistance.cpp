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
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_knn_mindistance)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/knn.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(knn_mindistance, 3, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto lowest = INPUT_VARIABLE(1);
            auto highest = INPUT_VARIABLE(2);

            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->lengthOf() == lowest->lengthOf() && input->lengthOf() == highest->lengthOf(), 0, "knn_mindistance: all input arrays must have same length");
            REQUIRE_TRUE(input->dataType() == lowest->dataType() && input->dataType() == highest->dataType() && input->dataType() == output->dataType(), 0, "knn_mindistance: all inputs must have the same data type");

            helpers::knn_mindistance(*input, *lowest, *highest, *output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(knn_mindistance) {
            auto input = inputShape->at(0);

            // always return scalar here
            return SHAPELIST(ConstantShapeHelper::getInstance()->scalarShapeInfo(ArrayOptions::dataType(input)));
        }

        DECLARE_TYPES(knn_mindistance) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif