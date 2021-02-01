/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/transforms.h>
#include <ops/declarable/helpers/axis.h>

namespace sd {
namespace ops {
#if NOT_EXCLUDED(OP_reduce_norm_max)

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_norm_max, 1, 1, false, 0, 0) {

    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    std::vector<int> dimensions;
    if (block.width() > 1) {
        auto axesVector = INPUT_VARIABLE(1);
        helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }
    else if (block.getIArguments()->size())
        dimensions = *block.getIArguments();

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_NORM_MAX OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->shapeInfo()[0] && item < input->shapeInfo()[0], 0, "REDUCE_NORM_MAX OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

    bool keepDims = false;
    if (block.getBArguments()->size())
        keepDims = B_ARG(0);
    else if (block.getTArguments()->size())
        keepDims = (bool)T_ARG(0);

    input->reduceAlongDimension(reduce::NormMax, *output, dimensions, keepDims);

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_norm_max) {

    auto in = inputShape->at(0);
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

    REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_NORM_MAX OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_NORM_MAX OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , inputShape->at(0)[0], inputShape->at(0)[0], item);

    return SHAPELIST(ShapeUtils::evalReduceShapeInfo(shape::order(in), dimensions, in, keepDims, false, block.getWorkspace()));
}

DECLARE_TYPES(reduce_norm_max) {
    getOpDescriptor()
        ->setAllowedInputTypes(sd::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}
#endif

#if NOT_EXCLUDED(OP_reduce_norm_max_bp)

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(reduce_norm_max_bp, 2, 1, false, 0, 0) {

    auto input = INPUT_VARIABLE(0);
    auto gradO = INPUT_VARIABLE(1);
    auto gradI = OUTPUT_VARIABLE(0);

    std::vector<int> dimensions = *block.getIArguments();

    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(input->rankOf(), axesVector, dimensions);
    }

    REQUIRE_TRUE(dimensions.size() <= input->rankOf(), 0, "REDUCE_NORM_MAX_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -input->shapeInfo()[0] && item < input->shapeInfo()[0], 0, "REDUCE_NORM_MAX_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !" , input->rankOf(), input->rankOf(), item);

    // *** calculations *** //

    *gradI = 0;

    if(gradO->lengthOf() == 1) {

        auto indOfAbsMaxElem = input->indexReduceNumber(sd::indexreduce::IndexAbsoluteMax);
        const Nd4jLong ind = indOfAbsMaxElem.t<Nd4jLong>(0);
        const int sign = input->e<float>(ind) >= 0 ? 1 : -1;
        gradI->p(ind, sign * gradO->e(0));
    }
    else {

        auto indicesArr = input->applyIndexReduce(sd::indexreduce::IndexAbsoluteMax, dimensions);
        helpers::scatterSimple(block.launchContext(), 6, *gradI, *gradO, indicesArr, ShapeUtils::evalDimsToExclude(gradI->rankOf(), dimensions));      // 6 corresponds to copy operation
        *gradI *= input->transform(sd::transform::Sign);
    }

    return Status::OK();
}

DECLARE_SHAPE_FN(reduce_norm_max_bp) {

    auto dimensions = *block.getIArguments();
    if (block.width() > 2) {
        auto axesVector = INPUT_VARIABLE(2);
        helpers::adjustAxis(INPUT_VARIABLE(0)->rankOf(), axesVector, dimensions);
    }

    REQUIRE_TRUE(dimensions.size() <= inputShape->at(0)[0], 0, "REDUCE_NORM_MAX_BP OP: the number of dimensions to reduce along must be <= input array rank, but got %i instead" , dimensions.size());

    for(const auto& item : dimensions)
        REQUIRE_TRUE(item >= -inputShape->at(0)[0] && item < inputShape->at(0)[0], 0, "REDUCE_NORM_MAX_BP OP: the input dimension to reduce along must be in range [-%i, %i), but got %i instead !", inputShape->at(0)[0], inputShape->at(0)[0], item);

    Nd4jLong* outShapeInfo;
    COPY_SHAPE(inputShape->at(0), outShapeInfo);

    return SHAPELIST(CONSTANT(outShapeInfo));
}

DECLARE_TYPES(reduce_norm_max_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(sd::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}




#endif

}
}
