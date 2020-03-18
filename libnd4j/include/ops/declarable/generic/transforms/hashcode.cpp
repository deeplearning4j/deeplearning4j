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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_hashcode)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>
#include <ops/declarable/helpers/hashcode.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(hashcode, 1, 1, false, 0, 0) {
            REQUIRE_TRUE(block.width() == 1, 0, "hashcode: this op can't be applied along dimension");

            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(output->isScalar(), 0, "hashcode: this op requires scalar output");

            helpers::hashCode(block.launchContext(), *input, *output);

            return Status::OK();
        };

        DECLARE_SHAPE_FN(hashcode) {
            return SHAPELIST(ConstantShapeHelper::getInstance()->scalarShapeInfo(sd::DataType::INT64));
        }


        DECLARE_TYPES(hashcode) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS})
                    ->setAllowedInputTypes(1, {ALL_INTS})
                    ->setAllowedOutputTypes({sd::DataType::INT64});
        };
    }
}

#endif

