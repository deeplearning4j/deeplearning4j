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
#include <ops/declarable/helpers/axis.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_sum)

    CUSTOM_OP_IMPL(reduce_sum, 1, 1, false, 0, 0) {
        auto input = INPUT_VARIABLE(0);
        auto output = OUTPUT_VARIABLE(0);
        //std::vector<int> axes = *block.getIArguments();
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
            REQUIRE_TRUE(item >= -input->shapeInfo()[0] && item < input->shapeInfo()[0], 0, "REDUCE_SUM OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        //const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
        input->reduceAlongDimension(reduce::Sum, output, axes, keepDims);

        return Status::OK();
    }

    DECLARE_SHAPE_FN(reduce_sum) {

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
        //ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShape->at(0)));
        return SHAPELIST(outShapeInfo);
    }

        DECLARE_TYPES(reduce_sum) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setSameMode(true);
        }
#endif 
#if NOT_EXCLUDED(OP_reduce_sum_bp)

    DECLARE_SHAPE_FN(reduce_sum_bp) {    

        //std::vector<int> dimensions = *block.getIArguments();
        Nd4jLong* outShapeInfo;// = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
        COPY_SHAPE(inputShape->at(0), outShapeInfo);
        return SHAPELIST(outShapeInfo);
    }

        DECLARE_TYPES(reduce_sum_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

    CUSTOM_OP_IMPL(reduce_sum_bp, 2, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto epsilon = INPUT_VARIABLE(1);
            auto output = OUTPUT_VARIABLE(0);

            if (epsilon->isScalar()) {
                output->assign(epsilon);
            }
            else {
                auto axes = *block.getIArguments();
                if (block.width() > 2) {
                    auto axesVector = INPUT_VARIABLE(2);
                    helpers::adjustAxis(input, axesVector, axes);
                }

                std::vector<int> dimensions; //(input->rankOf() - axes.size());
                for (Nd4jLong e = 0; e < input->rankOf(); e++) {
                    if (std::find(axes.begin(), axes.end(), e) == axes.end()) {
                        dimensions.emplace_back(e);
                    }
                }
                std::unique_ptr<ResultSet> outList(output->allTensorsAlongDimension(dimensions));
                for (Nd4jLong e = 0; e < outList->size(); ++e) {
                    outList->at(e)->assign(epsilon);
                }
            }

            return Status::OK();
    }
#endif

}
}
