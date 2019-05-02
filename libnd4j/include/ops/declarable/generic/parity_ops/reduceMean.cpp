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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.06.2018
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

namespace nd4j    {
namespace ops     {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_mean, 1, 1, false, 0, 0) {
    auto input   = INPUT_VARIABLE(0);
    auto output  = OUTPUT_VARIABLE(0);

    auto dimensions = *block.getIArguments();
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }

    bool keepDims = false;
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_MEAN OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->rankOf() && item < input->rankOf(), 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);
    
    input->reduceAlongDimension(reduce::Mean, output, dimensions, keepDims);

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_mean) {

    auto dimensions = *block.getIArguments();
    auto in = inputShape->at(0);
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(shape::rank(in), axesVector, dimensions);
    }

    bool keepDims = false;
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    REQUIRE_TRUE(dimensions.size() <= in[0], 0, "REDUCE_MEAN OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());
    
    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_MEAN OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);

    auto outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(in), dimensions, in, keepDims, false, block.getWorkspace());

    return SHAPELIST(outShapeInfo);
}

DECLARE_TYPES(reduce_mean) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_mean_bp, 2, 1, false, 0, 0) {
    auto input  = INPUT_VARIABLE(0);
    auto gradO  = INPUT_VARIABLE(1);

    auto gradI  = OUTPUT_VARIABLE(0);

    auto dimensions = *block.getIArguments();
    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }

    bool keepDims = false;
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_MEAN_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->rankOf() && item < input->rankOf(), 0, "REDUCE_MEAN_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);
    
    if(gradO->lengthOf() == 1) {
        gradI->assign(gradO->e(0) / input->lengthOf());
    }
    else {
        
        (*gradI).assign((gradO->lengthOf() + 0.) / input->lengthOf());

        if(!keepDims) {
            auto gradOShapeKeepDims = ShapeUtils::evalReduceShapeInfo(gradO->ordering(), dimensions, *input, true, false, block.getWorkspace());
            gradO = gradO->reshape(gradO->ordering(), ShapeUtils::pullShapeFromShapeInfo(gradOShapeKeepDims));  // for example could be something like [a,b] -> [1,a,1,b]
        }

        *gradI *= *gradO;

        if(!keepDims)
            delete gradO;
    }

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_mean_bp) {
    auto in = inputShape->at(0);
    auto dimensions = *block.getIArguments();
    auto rank = shape::rank(in);

    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(rank, axesVector, dimensions);
    }
    REQUIRE_TRUE(dimensions.size() <= rank, 0, "REDUCE_MEAN_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());
    
    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -rank || item < rank, 0, "REDUCE_MEAN_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , rank, rank, item);
    
    Nd4jLong* gradIshapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIshapeInfo);

    return SHAPELIST(gradIshapeInfo);
}


DECLARE_TYPES(reduce_mean_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}



}
}
