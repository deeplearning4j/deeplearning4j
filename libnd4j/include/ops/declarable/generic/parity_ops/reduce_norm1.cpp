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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_norm1)

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_norm1, 1, 1, false, 0, 0) {

    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    std::vector<int> dimensions;
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }
    else if (block.getIArguments()->size())
        dimensions = *block.getIArguments();

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_NORM1 OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->shapeInfo()[0] && item < input->shapeInfo()[0], 0, "REDUCE_NORM1 OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

    bool keepDims = false;
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    input->reduceAlongDimension(reduce::Norm1, *output, dimensions, keepDims);

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_norm1) {

    bool keepDims = false;
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    std::vector<int> dimensions;
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(INPUT_VARIABLE(0)->rankOf(), axesVector, dimensions);
    }
    else if (block.getIArguments()->size())
        dimensions = *block.getIArguments();

    REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_NORM1 OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_NORM1 OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);

    return SHAPELIST(ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace()));
}

DECLARE_TYPES(reduce_norm1) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}
#endif
#if NOT_EXCLUDED(OP_reduce_norm1_bp)

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_norm1_bp, 2, 1, false, 0, 0) {
    // L = Sum abs(x_i) for all i = 1 to N
    // dL/dx_i = 1 if x_i >= 0 and -1 when x_i < 0
    // out_i = epsilon_i if x_i > 0 and -epsilon_i when x_i < 0
    // when gradO is non a scalar, using dimensions to split output onto gradO like parts
    // and use LAMBDA with that formula for it.

    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);
    auto gradI = OUTPUT_VARIABLE(0);

    input->applyTransform(nd4j::transform::Sign, *gradI);

    if (gradO->lengthOf() == 1) {
        *gradI *= *gradO;
    }
    else {

        bool keepDims = false;
        auto dimensions = *block.getIArguments();

        if (block.width() > 2) {
            auto axesVector = INPUT_VARIABLE(2);
            helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
        }

        if (block.getBArguments()->size())
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size())
            keepDims = (bool)T_ARG(0);

        REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_NORM1_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

        for(const auto& item : dimensions)
            REQUIRE_TRUE(item >= -input->rankOf() && item < input->rankOf(), 0, "REDUCE_NORM1_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        // *** calculations *** //

        if(!keepDims) {
            auto gradOShapeKeepDims = ShapeUtils::evalReduceShapeInfo(gradO->ordering(), dimensions, *input, true, false, block.getWorkspace());
            *gradI *= gradO->reshape(gradO->ordering(), ShapeUtils::pullShapeFromShapeInfo(gradOShapeKeepDims));  // for example could be something like [a,b] -> [1,a,1,b]
        } else
            *gradI *= *gradO;
    }

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_norm1_bp) {

    auto dimensions = *block.getIArguments();
    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(INPUT_VARIABLE(0)->rankOf(), axesVector, dimensions);
    }

    REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_NORM1_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_NORM1_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !", inputShape->at(0)[0], inputShape->at(0)[0], item);

    Nd4jLong* outShapeInfo;
    COPY_SHAPE(inputShape->at(0), outShapeInfo);

    return SHAPELIST(CONSTANT(outShapeInfo));
}

DECLARE_TYPES(reduce_norm1_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}


#endif

}
}
