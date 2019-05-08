/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_sqnorm)

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_sqnorm, 1, 1, false, 0, 0) {
    
    auto input = INPUT_VARIABLE(0);
    auto gradI = OUTPUT_VARIABLE(0);
    
    bool keepDims = false;

    auto dimensions = *block.getIArguments();

    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(input, axesVector, dimensions);
    }    
        
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_SQNORM OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->rankOf() && item < input->rankOf(), 0, "REDUCE_SQNORM OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

    input->reduceAlongDimension(reduce::SquaredNorm, gradI, dimensions, keepDims);

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_sqnorm) {

    auto dimensions = *block.getIArguments();
    bool keepDims = false;
    
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(INPUT_VARIABLE(0), axesVector, dimensions);
    }
        
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_SQNORM OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());
    
    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_SQNORM OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);

    Nd4jLong* outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());    

    return SHAPELIST(outShapeInfo);
}

DECLARE_TYPES(reduce_sqnorm) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY) 
        ->setAllowedOutputTypes({ALL_FLOATS});
}

#endif 

#if NOT_EXCLUDED(OP_reduce_sqnorm_bp)
 
//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_sqnorm_bp, 2, 1, false, 0, 0) {

    auto input  = INPUT_VARIABLE(0);
    auto gradO  = INPUT_VARIABLE(1);
    auto gradI  = OUTPUT_VARIABLE(0);

    if (gradO->lengthOf() == 1) {
        gradI->assign( 2 * (*input) * gradO->e(0));        
    }
    else {
        
        bool keepDims = false;
        auto dimensions = *block.getIArguments();
        
        if (block.width() > 2) {
            auto axesVector = INPUT_VARIABLE(2);
            helpers::adjustAxis(input, axesVector, dimensions);
        }
                
        if (block.getBArguments()->size())
            keepDims = B_ARG(0);
        else if (block.getTArguments()->size())
            keepDims = (bool)T_ARG(0);

        REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_SQNORM_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

        for(const auto& item : dimensions)
            REQUIRE_TRUE(item >= -input->rankOf() && item < input->rankOf(), 0, "REDUCE_SQNORM_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

        // *** calculations *** //

        if(!keepDims) {

            Nd4jLong* gradOShapeKeepDims = ShapeUtils::evalReduceShapeInfo(gradO->ordering(), dimensions, *input, true, false, block.getWorkspace());
            gradO = gradO->reshape(gradO->ordering(), ShapeUtils::pullShapeFromShapeInfo(gradOShapeKeepDims));  // for example could be something like [a,b] -> [1,a,1,b]
            RELEASE(gradOShapeKeepDims, block.getWorkspace());
        }

        gradI->assign(2. * (*input) * *gradO);

        if(!keepDims)
            delete gradO;
    }
    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_sqnorm_bp) {

    if(shape::length(inputShape->at(1)) > 1) {
        
        auto dimensions = *block.getIArguments();
        if (block.width() > 2) {
            auto axesVector = INPUT_VARIABLE(2);
            helpers::adjustAxis(INPUT_VARIABLE(0), axesVector, dimensions);
        }

        REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_SQNORM_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());
    
        for(const auto& item : dimensions)
            REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_SQNORM_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);
    }
    
    Nd4jLong* gradIshapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIshapeInfo);

    return SHAPELIST(gradIshapeInfo);
}

DECLARE_TYPES(reduce_sqnorm_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}
   
#endif

}
}
