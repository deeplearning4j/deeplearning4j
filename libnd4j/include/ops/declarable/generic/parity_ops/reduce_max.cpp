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

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reduce_minmax.h>
#include <ops/declarable/helpers/axis.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_max)

    CUSTOM_OP_IMPL(reduce_max, 1, 1, false, 0, 0) {
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
        else if (block.getTArguments()->size() > 0)
            keepDims = (bool)T_ARG(0);

        input->reduceAlongDimension(reduce::Max, output, axes, keepDims);
        return Status::OK();
    }

    DECLARE_SHAPE_FN(reduce_max) {

        bool keepDims = false;//: false;
        if (block.getBArguments()->size() > 0)
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size() > 0)
            keepDims = (bool)T_ARG(0);

        auto axes = *block.getIArguments();
        if (block.width() > 1) {
            auto axesVector = INPUT_VARIABLE(1);
            helpers::adjustAxis(INPUT_VARIABLE(0), axesVector, axes);
        }

        Nd4jLong* outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), axes, inputShape->at(0), keepDims, false, block.getWorkspace());
        ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShape->at(0)));

        return SHAPELIST(outShapeInfo);
    }

        DECLARE_TYPES(reduce_max) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setSameMode(true);
        }

#endif 
#if NOT_EXCLUDED(OP_reduce_max_bp)

    DECLARE_SHAPE_FN(reduce_max_bp) {    

        Nd4jLong* outShapeInfo;// = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

        DECLARE_TYPES(reduce_max_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    CUSTOM_OP_IMPL(reduce_max_bp, 2, 1, false, 0, 0) {

            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);
			
			output->assign(0.0);

            // at first step we build fwd activation
            nd4j::ops::reduce_max op;
            std::vector<int> axes;
            std::vector<Nd4jLong> axesLong;

            if (block.width() > 2) {
                auto axesVector = INPUT_VARIABLE(2);
                helpers::adjustAxis(input, axesVector, axes);
            }
            else
                axes = *block.getIArguments();

            for (size_t e = 0; e < axes.size(); e++)
                axesLong.emplace_back(axes[e]);// = *block.getIArguments();

            bool keepDims = false;//: false;

            if (block.getBArguments()->size() > 0)
                keepDims = B_ARG(0);

            else if (block.getTArguments()->size() > 0)
                keepDims = (bool)T_ARG(0);

            // FIXME:: double!!!
            //std::vector<double> tVec(1);
            //tVec[0] = (keepDims? 1.0 : 0.0);
            std::vector<NDArray*> inputVec({input});
            std::vector<bool> emptyBool({keepDims});
            std::unique_ptr<ResultSet> tmpResult(op.execute(inputVec, {}, axesLong, emptyBool, false));
            if (tmpResult->status() != ND4J_STATUS_OK)
                return tmpResult->status();

            auto tempMax = tmpResult->at(0);
            REQUIRE_TRUE(tempMax->isSameShape(epsilon), 0, "reduce_max_bp: The second param shape should be an equal with reduce_max output.");
            helpers::minMaxReduceFunctor(input, epsilon, tempMax, output);
            return Status::OK();
    }
#endif

}
}
