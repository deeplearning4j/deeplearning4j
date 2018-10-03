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

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_fill)

#include <ops/declarable/headers/parity_ops.h>

namespace nd4j {
    namespace ops {
        
        CUSTOM_OP_IMPL(fill, 1, 1, false, -2, 0) {
            auto shapeArray = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            auto w = block.width();
            auto i = block.numI();
            auto t = block.numT();

            REQUIRE_TRUE( w > 1 || t > 0 || i > 0, 0, "Fill: either additional variable should exist, or scalar value should be present");
            if (w > 1) {
                output->assign(INPUT_VARIABLE(1));
            } else {
                if (t > 0) {
                    output->assign(T_ARG(0));
                } else if (i > 0) {
                    output->assign(INT_ARG(0));
                }
            }

            STORE_RESULT(output);

            return Status::OK();
        };

        
        DECLARE_SHAPE_FN(fill) {

            auto shapeArray = INPUT_VARIABLE(0);
            const int len = shapeArray->lengthOf();
            Nd4jLong *newShape = nullptr;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(len), Nd4jLong);            

            newShape[0] = len;
            for (int e = 0; e < shapeArray->lengthOf(); e++)
                newShape[e+1] = shapeArray->e<Nd4jLong>(e);
            
            ShapeUtils::updateStridesAndType(newShape, shapeArray->dataType(), 'c');

            return SHAPELIST(newShape);
        };
    }
}

#endif