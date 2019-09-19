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
CUSTOM_OP_IMPL(reduce_stdev, 1, 1, false, 0, 0) {
    auto input   = INPUT_VARIABLE(0);
    auto output  = OUTPUT_VARIABLE(0);

    bool keepDims      = false;//block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    bool biasCorrected = false;//block.getTArguments()->size() > 1 ? (bool)T_ARG(1) : false;

    auto dimensions = ArrayUtils::toIntVector(*block.getIArguments());
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }

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

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_STDEV OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->rankOf() && item < input->rankOf(), 0, "REDUCE_STDEV OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

    input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, output, biasCorrected, dimensions);

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_stdev) {
        auto in = inputShape->at(0);
        auto rank = shape::rank(in);
        bool keepDims      = false;//block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    auto dimensions = ArrayUtils::toIntVector(*block.getIArguments());

    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(rank, axesVector, dimensions);
    }

    if (block.getBArguments()->size()) {
        keepDims = B_ARG(0);
    }
    else if (block.getTArguments()->size()) {
        keepDims = (bool)T_ARG(0);
    }

    REQUIRE_TRUE(dimensions.size() <= rank, 0, "REDUCE_STDEV OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_STDEV OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);

    Nd4jLong* outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(in), dimensions, in, keepDims, false, block.getWorkspace());

    return SHAPELIST(outShapeInfo);
}

DECLARE_TYPES(reduce_stdev) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_stdev_bp, 2, 1, false, 0, 0) {
    auto input  = INPUT_VARIABLE(0);
    auto gradO  = INPUT_VARIABLE(1);

    auto gradI  = OUTPUT_VARIABLE(0);

    bool keepDims      = false;//block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    bool biasCorrected = false;//block.getTArguments()->size() > 1 ? (bool)T_ARG(1) : false;

    auto dimensions = ArrayUtils::toIntVector(*block.getIArguments());
    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }

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

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_STDEV_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->rankOf() && item < input->rankOf(), 0, "REDUCE_STDEV_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

    const Nd4jLong N = input->lengthOf() / gradO->lengthOf();
    const Nd4jLong NminusOne = biasCorrected ? N - 1 : N;

    auto mean = input->reduceAlongDims(reduce::Mean, dimensions, true);

    NDArray variance(mean.getShapeInfo(), true, block.launchContext());                    // create empty array with shape matching shape of mean array
    input->varianceAlongDimension(variance::SummaryStatsStandardDeviation, &variance, biasCorrected, dimensions);

    gradI->assign( (*input - mean) / (variance * NminusOne));                              // automatic broadcasting happens here

    if(!keepDims) {
        auto gradOShapeKeepDims = ShapeUtils::evalReduceShapeInfo(gradO->ordering(), dimensions, *input, true, false, block.getWorkspace());
        *gradI *= gradO->reshape(gradO->ordering(), ShapeUtils::pullShapeFromShapeInfo(gradOShapeKeepDims));  // for example could be something like [a,b] -> [1,a,1,b]
    }
    else
        *gradI *= *gradO;           // automatic broadcasting happens here

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_stdev_bp) {
    auto in = inputShape->at(0);
    auto rank = shape::rank(in);
    auto dimensions = ArrayUtils::toIntVector(*block.getIArguments());
    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(rank, axesVector, dimensions);
    }

    REQUIRE_TRUE(dimensions.size() <= rank, 0, "REDUCE_STDEV_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_STDEV_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);

    Nd4jLong* gradIshapeInfo(nullptr);
    COPY_SHAPE(in, gradIshapeInfo);

    return SHAPELIST(CONSTANT(gradIshapeInfo));
}

DECLARE_TYPES(reduce_stdev_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(nd4j::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}


}
}
