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
// Created by george@skymind.io on 6/6/2018.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reduce_minmax.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_min)

    CUSTOM_OP_IMPL(reduce_min, 1, 1, false, 0, 0) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);
        std::vector<int> axes = *block.getIArguments();

        for(const auto& item : axes)
            REQUIRE_TRUE(item > -input->shapeInfo()[0] || item <input->shapeInfo()[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
        input->reduceAlongDimension(reduce::Min, output, axes, keepDims);

        return Status::OK();
    }

    DECLARE_SHAPE_FN(reduce_min) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        std::vector<int> dimensions = *block.getIArguments();
        Nd4jLong* outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShape->at(0)));
        return SHAPELIST(outShapeInfo);
    }
#endif 
#if NOT_EXCLUDED(OP_reduce_min_bp)

    DECLARE_SHAPE_FN(reduce_min_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;// = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

    CUSTOM_OP_IMPL(reduce_min_bp, 2, 1, false, 0, 0) {
      //       dL/dIn  = dL/dOut                   if in_i == out (== min(in))
      //               = 0                         otherwise
            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);
			
			output->assign(0.0);

            const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;

            // FIXME:: double
            double keepDimsT = (keepDims? 1.f : 0.f);

            nd4j::ops::reduce_min op;
            std::vector<Nd4jLong> axes;

            if (block.numI() > 0) {
                for (int e = 0; e < block.numI(); e++)
                    axes.emplace_back(INT_ARG(e));// = *block.getIArguments();
            }
            // FIXME: double
            std::vector<double> tVec(1);
            tVec[0] = (keepDims ? 1.0 : 0.0);
            std::vector<NDArray*> inputVec({input});
            std::unique_ptr<ResultSet> tmpResult(op.execute(inputVec, tVec, axes, false));
            if (tmpResult->status() != Status::OK())
                return tmpResult->status();
       
            auto tempMin = tmpResult->at(0); // out
            REQUIRE_TRUE(tempMin->isSameShape(epsilon), 0, "reduce_min_bp: The second param shape should be an equal with reduce_min output.");
            helpers::minMaxReduceFunctor(input, epsilon, tempMin, output);

            return Status::OK();
    }
#endif

}
}
