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
// Created by george@skymind.io on 6/1/2018.
//
#include <ops/declarable/helpers/reduce_dot.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_dot_bp)

    DECLARE_SHAPE_FN(reduce_dot_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;// = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims);
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

    CUSTOM_OP_IMPL(reduce_dot_bp, 3, 1, false, 0, 0) {

            auto inputX = INPUT_VARIABLE(0);
            auto inputY = INPUT_VARIABLE(1);
            auto epsilon = INPUT_VARIABLE(2);
            auto output = OUTPUT_VARIABLE(0);
            //
            // L(x,y) = SUM(x_i * y_i)
            // dL/dx_i = y_i
            //    
            //REQUIRE_TRUE(output->isSameShape(epsilon), 0, "reduce_sum_bp: The second param shape should be the same as result shape.");
            if (epsilon->isScalar()) {
                output->assign(epsilon);
                output->template applyPairwiseTransform<simdOps::Multiply<T>>(inputY, output, nullptr);
            }
            else {
                auto axes = *block.getIArguments();
                helpers::reduceDotBP(inputX, inputY, epsilon, output, axes);
            }

            return ND4J_STATUS_OK;
    }
#endif

}
}
