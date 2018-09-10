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
    
        Nd4jLong* outShapeInfo1;// = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims);
        Nd4jLong* outShapeInfo2;// = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims);
        COPY_SHAPE(inputShape->at(0), outShapeInfo1);
        COPY_SHAPE(inputShape->at(1), outShapeInfo2);

        return SHAPELIST(outShapeInfo1, outShapeInfo2);
    }

    CUSTOM_OP_IMPL(reduce_dot_bp, 3, 2, false, 0, 0) {
            auto inputX = INPUT_VARIABLE(0);
            auto inputY = INPUT_VARIABLE(1);
            auto epsilon = INPUT_VARIABLE(2);
            auto output1 = OUTPUT_VARIABLE(0);
            auto output2 = OUTPUT_VARIABLE(1);
            //
            // L(x,y) = SUM(x_i * y_i)
            // dL/dx_i = y_i
            //    
            //REQUIRE_TRUE(output->isSameShape(epsilon), 0, "reduce_sum_bp: The second param shape should be the same as result shape.");
            if (epsilon->isScalar()) {
                output1->assign(epsilon);
                output1->applyPairwiseTransform(pairwise::Multiply, inputY, output1, nullptr);
                output2->assign(epsilon);
                output2->applyPairwiseTransform(pairwise::Multiply, inputX, output2, nullptr);
            }
            else {
                auto axes = *block.getIArguments();
                helpers::reduceDotBP(inputX, inputY, epsilon, output1, output2, axes);
            }

            return Status::OK();
    }
#endif

}
}
