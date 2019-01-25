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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 04.06.2018
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

namespace nd4j    {
namespace ops     {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_variance, 1, 1, false, 0, 0) {
    auto input   = INPUT_VARIABLE(0);
    auto output  = OUTPUT_VARIABLE(0);

    bool keepDims      = false;//block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    bool biasCorrected = false;//block.getTArguments()->size() > 1 ? (bool)T_ARG(1) : false;

    auto dimensions = *block.getIArguments();
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(input, axesVector, dimensions);
    }
//            else if (block.getIArguments()->size())
    if (block.getBArguments()->size()) {
        keepDims = B_ARG(0);
        if (block.getBArguments()->size() > 1)
            biasCorrected = B_ARG(1);
    }
    else if (block.getTArguments()->size()) {
        keepDims = (bool)T_ARG(0);
        if (block.getTArguments()->size() > 1)
            biasCorrected = (bool)T_ARG(1);
    }

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_VARIANCE OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item > -input->rankOf() || item < input->rankOf(), 0, "REDUCE_VARIANCE OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);
        
    input->varianceAlongDimension(variance::SummaryStatsVariance, output, biasCorrected, dimensions);

    return Status::OK();
}

        DECLARE_TYPES(reduce_variance) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }


DECLARE_SHAPE_FN(reduce_variance) {
    bool keepDims      = false;//block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;

    auto dimensions = *block.getIArguments();
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(INPUT_VARIABLE(0), axesVector, dimensions);
    }
//            else if (block.getIArguments()->size())
    if (block.getBArguments()->size()) {
        keepDims = B_ARG(0);
    }
    else if (block.getTArguments()->size()) {
        keepDims = (bool)T_ARG(0);
    }

    REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_VARIANCE OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());
    
    for(const auto& item : dimensions)
        REQUIRE_TRUE(item > -inputShape->at(0)[0] || item < inputShape->at(0)[0], 0, "REDUCE_VARIANCE OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);

    Nd4jLong* outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(inputShape->at(0)), dimensions, inputShape->at(0), keepDims, false, block.getWorkspace());
    ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShape->at(0)));

    return SHAPELIST(outShapeInfo);
}


        DECLARE_TYPES(reduce_variance_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_variance_bp, 2, 1, false, 0, 0) {
    auto input  = INPUT_VARIABLE(0);
    auto gradO  = INPUT_VARIABLE(1);

    auto gradI  = OUTPUT_VARIABLE(0);

    bool keepDims      = false;//block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    bool biasCorrected = false;//block.getTArguments()->size() > 1 ? (bool)T_ARG(1) : false;

    auto dimensions = *block.getIArguments();
    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(input, axesVector, dimensions);
    }
//            else if (block.getIArguments()->size())
    if (block.getBArguments()->size()) {
        keepDims = B_ARG(0);
        if (block.getBArguments()->size() > 1)
            biasCorrected = B_ARG(1);
    }
    else if (block.getTArguments()->size()) {
        keepDims = (bool)T_ARG(0);
        if (block.getTArguments()->size() > 1)
            biasCorrected = (bool)T_ARG(1);
    }

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_VARIANCE OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item > -input->rankOf() || item < input->rankOf(), 0, "REDUCE_VARIANCE OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);        

    const Nd4jLong N = input->lengthOf() / gradO->lengthOf();
    const Nd4jLong NminusOne = biasCorrected ? N - 1 : N;
    const double factor1 = 2.0 / NminusOne;
    const double factor2 = 2.0 / (N * NminusOne);
    
    auto mean = input->reduceAlongDims(reduce::Mean, dimensions, true);
    
    gradI->assign( (*input - mean) * (2.0f / NminusOne));                                    // automatic broadcasting happens here

    Nd4jLong* gradOShapeKeepDims = ShapeUtils::evalReduceShapeInfo(input->ordering(), dimensions, *input, true, false, block.getWorkspace());
    const bool isGradOShapeBroadcast = shape::equalsSoft(gradOShapeKeepDims, gradO->getShapeInfo());
        
    if(!isGradOShapeBroadcast)
        gradO = gradO->reshape(gradO->ordering(), ShapeUtils::pullShapeFromShapeInfo(gradOShapeKeepDims));  // for example could be something like [a,b] -> [1,a,1,b]
    
    *gradI *= *gradO;                                                  // automatic broadcasting happens here
            
    if(!isGradOShapeBroadcast)
        delete gradO;

    return Status::OK();
}



DECLARE_SHAPE_FN(reduce_variance_bp) {

    auto dimensions = *block.getIArguments();
    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(INPUT_VARIABLE(0), axesVector, dimensions);
    }
//            else if (block.getIArguments()->size())

    REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_VARIANCE OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());
    
    for(const auto& item : dimensions)
        REQUIRE_TRUE(item > -inputShape->at(0)[0] || item < inputShape->at(0)[0], 0, "REDUCE_VARIANCE OP: the input dimension to reduce along must be in range (-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);
    
    Nd4jLong* gradIshapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIshapeInfo);
        
    return SHAPELIST(gradIshapeInfo);
}


}
}
