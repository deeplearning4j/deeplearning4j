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
#if NOT_EXCLUDED(OP_check_numerics)

#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {

        CUSTOM_OP_IMPL(check_numerics, 2, 1, true, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto message = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            auto allFinite = input->reduceNumber(reduce::BoolOps::IsFinite);
            REQUIRE_TRUE(allFinite.e<bool>(0), 0, "CheckNumerics: %s", message->e<std::string>(0).c_str());

            if (!block.isInplace())
                output->assign(input);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(check_numerics) {
            return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(inputShape->at(0))));
        }

        DECLARE_TYPES(check_numerics) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_FLOATS})
                    ->setAllowedInputTypes(1, sd::DataType::UTF8)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif