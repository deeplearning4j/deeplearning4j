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
#include <ops/declarable/helpers/axis.h>
namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_min)

    CUSTOM_OP_IMPL(reduce_min, 1, 1, false, 0, 0) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);
            std::vector<int> axes;
            if (block.width() > 1) {
                auto axesVector = INPUT_VARIABLE(1);
                helpers::adjustAxis(input, axesVector, axes);
            }
            else
                axes = *block.getIArguments();

        for(const auto& item : axes)
            REQUIRE_TRUE(item > -input->shapeInfo()[0] || item <input->shapeInfo()[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        bool keepDims = false;//: false;
        if (block.getBArguments()->size() > 0)
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size() > 0) // for compatibility
            keepDims = (bool)T_ARG(0);
        input->reduceAlongDimension(reduce::Min, output, axes, keepDims);

        return Status::OK();
    }

        DECLARE_TYPES(reduce_min) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setSameMode(true);
        }

    DECLARE_SHAPE_FN(reduce_min) {

        bool keepDims = false;//: false;
        if (block.getBArguments()->size() > 0)
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size() > 0)
            keepDims = (bool)T_ARG(0);

        std::vector<int> axes;
        if (block.width() > 1) {
            auto axesVector = INPUT_VARIABLE(1);
            helpers::adjustAxis(INPUT_VARIABLE(0), axesVector, axes);
        }
        else if (block.getIArguments()->size())
            axes = *block.getIArguments();
        Nd4jLong* outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), axes, inputShape->at(0), keepDims, false, block.getWorkspace());
        ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShape->at(0)));
        return SHAPELIST(outShapeInfo);
    }
#endif 
#if NOT_EXCLUDED(OP_reduce_min_bp)

    DECLARE_SHAPE_FN(reduce_min_bp) {


        Nd4jLong* outShapeInfo;// = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

        DECLARE_TYPES(reduce_min_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    CUSTOM_OP_IMPL(reduce_min_bp, 2, 1, false, 0, 0) {
      //       dL/dIn  = dL/dOut                   if in_i == out (== min(in))
      //               = 0                         otherwise
            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);
			
			output->assign(0.0);

            bool keepDims = false;//: false;
            if (block.getBArguments()->size() > 0)
                keepDims = B_ARG(0);
            else if (block.getTArguments()->size() > 0)
                keepDims = (bool)T_ARG(0);

            nd4j::ops::reduce_min op;
            std::vector<Nd4jLong> axes;
            std::vector<int> axesInt;
            if (block.width() > 2) {
                auto axesVector = INPUT_VARIABLE(2);
                helpers::adjustAxis(input, axesVector, axesInt);
            }
            else
                axesInt = *block.getIArguments();

            for (size_t e = 0; e < axesInt.size(); e++)
                axes.emplace_back(axesInt[e]);// = *block.getIArguments();

            std::vector<NDArray*> inputVec({input});
            std::unique_ptr<ResultSet> tmpResult(op.execute(inputVec, {}, axes, {keepDims}, false));
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
