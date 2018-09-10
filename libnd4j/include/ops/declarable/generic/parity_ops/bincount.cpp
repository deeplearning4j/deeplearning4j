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
#if NOT_EXCLUDED(OP_bincount)

//#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/weights.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(bincount, 1, 1, false, 0, 0) {
            auto values = INPUT_VARIABLE(0);
            
            NDArray *weights = nullptr;
            if (block.width() > 1) {
                weights = INPUT_VARIABLE(1);
                REQUIRE_TRUE(values->isSameShape(weights), 0, "bincount: the input and weights shapes should be equals");
            }
            int maxLength = -1;
            int minLength = 0;
            int maxIndex = values->argMax();
            maxLength = int((*values)(maxIndex))  + 1;

            if (block.numI() > 0) {
                minLength = nd4j::math::nd4j_max(INT_ARG(0), 0);
                if (block.numI() == 2) 
                    maxLength = nd4j::math::nd4j_min(maxLength, INT_ARG(1));
            }

            auto result = OUTPUT_VARIABLE(0);
            result->assign(0.0f);
             
            helpers::adjustWeights(values, weights, result, minLength, maxLength);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(bincount) {
            auto shapeList = SHAPELIST(); 
            auto in = INPUT_VARIABLE(0);

            int maxIndex = in->argMax();
            int maxLength = int((*in)(maxIndex))  + 1;

            if (block.numI() > 0)
                maxLength = nd4j::math::nd4j_max(maxLength, INT_ARG(0));

            if (block.numI() > 1) 
                maxLength = nd4j::math::nd4j_min(maxLength, INT_ARG(1));

            Nd4jLong* newshape;
            
            ALLOCATE(newshape, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);

            shape::shapeVector(maxLength,  newshape);

            shapeList->push_back(newshape); 
            return shapeList;
        }

    }
}

#endif