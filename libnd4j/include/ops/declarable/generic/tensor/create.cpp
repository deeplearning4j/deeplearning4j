/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
#if NOT_EXCLUDED(OP_shapes_of)

#include <ops/declarable/CustomOperations.h>

namespace sd {
    namespace ops {

        CUSTOM_OP_IMPL(create, 1, 1, false, 0, 1) {
            auto init = block.numB() > 0 ? B_ARG(0) : true;

            if (init)
                OUTPUT_VARIABLE(0)->nullify();

            return Status::OK();
        }

        DECLARE_SHAPE_FN(create) {
            auto shapeInput = INPUT_VARIABLE(0);
            auto order = (char) INT_ARG(0);
            auto dtype = DataTypeUtils::fromInt(INT_ARG(1));

            REQUIRE_TRUE(order == 'c' || order == 'f', 0, "create: order must be either c or f");

            auto shape = shapeInput->getBufferAsVector<Nd4jLong>();

            return SHAPELIST(sd::ConstantShapeHelper::getInstance().createShapeInfo(dtype, order, shape));
        }

        DECLARE_TYPES(create) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_INTS})
                    ->setAllowedOutputTypes(sd::DataType::ANY);
        }
    }
}

#endif