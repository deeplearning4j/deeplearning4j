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
// Created by george@skymind.io on 6/4/2018.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/reduce_minmax.h>
#include <ops/declarable/helpers/axis.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_norm_max)

    CUSTOM_OP_IMPL(reduce_norm_max, 1, 1, false, 0, 0) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);
            auto axes = *block.getIArguments();

        if (block.width() > 1) {
            auto axesVector = INPUT_VARIABLE(1);
            helpers::adjustAxis(input, axesVector, axes);
        }
//            else if (block.getIArguments()->size())
        bool keepDims = false;
        if (block.getBArguments()->size())
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size())
            keepDims = (bool)T_ARG(0);

        for(const auto& item : axes)
            REQUIRE_TRUE(item > -input->shapeInfo()[0] && item < input->shapeInfo()[0], 0, "REDUCE_NORM_MAX OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        input->reduceAlongDimension(reduce::NormMax, output, axes, keepDims);

        return Status::OK();
    }

    DECLARE_SHAPE_FN(reduce_norm_max) {

        auto axes = *block.getIArguments();
        if (block.width() > 1) {
            auto axesVector = INPUT_VARIABLE(1);
            helpers::adjustAxis(INPUT_VARIABLE(0), axesVector, axes);
        }
//            else if (block.getIArguments()->size())
        bool keepDims = false;
        if (block.getBArguments()->size())
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size())
            keepDims = (bool)T_ARG(0);

        Nd4jLong* outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), axes, inputShape->at(0), keepDims, false, block.getWorkspace());
        ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShape->at(0)));

        return SHAPELIST(outShapeInfo);
    }

        DECLARE_TYPES(reduce_norm_max) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
#endif 
#if NOT_EXCLUDED(OP_reduce_norm_max_bp)

    DECLARE_SHAPE_FN(reduce_norm_max_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;// = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

        DECLARE_TYPES(reduce_norm_max_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    CUSTOM_OP_IMPL(reduce_norm_max_bp, 2, 1, false, 0, 0) {

            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

			output->assign(0.0);

            auto axes = *block.getIArguments();
            if (block.width() > 2) {
                auto axesVector = INPUT_VARIABLE(2);
                helpers::adjustAxis(input, axesVector, axes);
            }
//            else if (block.getIArguments()->size())
            bool keepDims = false;
            if (block.getBArguments()->size())
                keepDims = B_ARG(0);
            else if (block.getTArguments()->size())
                keepDims = (bool)T_ARG(0);
            std::vector<Nd4jLong> axesLong;
            for (size_t i = 0; i < axes.size(); i++)
                axesLong.emplace_back(axes[i]);
            //std::vector<NDArray*> inputVec({input});
            nd4j::ops::reduce_norm_max op;
            std::unique_ptr<ResultSet> tmpResult(op.execute({input}, {}, axesLong, {keepDims}, false));
            if (tmpResult->status() != Status::OK())
                return tmpResult->status();

            auto normMax = tmpResult->at(0);

            helpers::minMaxReduceFunctor(input, epsilon, normMax, output, true);

            return Status::OK();
    }
#endif

}
}
