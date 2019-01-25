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

#include <ops/declarable/helpers/reduce_product.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_prod)

    CUSTOM_OP_IMPL(reduce_prod, 1, 1, false, 0, 0) {
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
            REQUIRE_TRUE(item > -input->shapeInfo()[0] || item <input->shapeInfo()[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        input->reduceAlongDimension(reduce::Prod, output, axes, keepDims);

        return Status::OK();
    }

    DECLARE_TYPES(reduce_prod) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(reduce_prod) {

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

        auto outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), axes, inputShape->at(0), keepDims, false, block.getWorkspace());
        //ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShape->at(0)));
        return SHAPELIST(outShapeInfo);
    }
#endif 
#if NOT_EXCLUDED(OP_reduce_prod_bp)

    DECLARE_SHAPE_FN(reduce_prod_bp) {    

        const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
        Nd4jLong* outShapeInfo;
        COPY_SHAPE(inputShape->at(0), outShapeInfo);

        return SHAPELIST(outShapeInfo);
    }

    DECLARE_TYPES(reduce_prod_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
    }

    CUSTOM_OP_IMPL(reduce_prod_bp, 2, 1, false, 0, 0) {
        auto input = INPUT_VARIABLE(0);
        auto epsilon = INPUT_VARIABLE(1);
        auto output = OUTPUT_VARIABLE(0);

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
        for (size_t e = 0; e < axes.size(); e++)
            axesLong.emplace_back(axes[e]);// = *block.getIArguments();

        nd4j::ops::reduce_prod op;
        std::unique_ptr<ResultSet> tmpResult(op.execute({input}, {}, axesLong, {keepDims}, false));
        if (tmpResult->status() != Status::OK())
            return tmpResult->status();
        auto tempProd = tmpResult->at(0);
        REQUIRE_TRUE(tempProd->isSameShape(epsilon), 0, "reduce_prod_bp: The the second param and reduce_sum output should have the equal shapes.");
    
        // tempProd has equal shape with epsilon
        if (epsilon->isScalar()) {
            helpers::reduceProductBPScalar(input, epsilon, tempProd, output);
        }
        else { // result 

            helpers::reduceProductBP(input, epsilon, tempProd, output, axes);
        }

        return Status::OK();
    }
#endif

}
}
