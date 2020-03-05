/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_space_to_batch_nd)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace sd {
namespace ops  {


CUSTOM_OP_IMPL(space_to_batch_nd, 3, 1, false, 0, 0) {

    // 4D example, numOfSpatialDims = 2 - two spatial dimensions
    // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockShape[0]*blockShape[1], (iH + padBottom + padTop)/blockSize[0], (iW + padLeft + padRight)/blockSize[1], iC]

    auto input      = INPUT_VARIABLE(0);
    auto blockShape = INPUT_VARIABLE(1);
    auto padding    = INPUT_VARIABLE(2);

    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(blockShape->rankOf() == 1, 0, "SpaceToBatchND: rank of blockShape array must be equal to one, but got %i instead !", blockShape->rankOf());

    const uint numOfSpatialDims = blockShape->sizeAt(0);

    REQUIRE_TRUE(input->rankOf() == output->rankOf(), 0, "SpaceToBatchND: rank of input and output array must be the same, but got %i and %i correspondingly !", input->rankOf(), output->rankOf());

    if(padding->sizeAt(0) != numOfSpatialDims || padding->sizeAt(1) != 2) {
        const std::string expectedpaddingShape = "[" + std::to_string(numOfSpatialDims) + ", 2]";   // [numOfSpatialDims, 2]
        REQUIRE_TRUE(false, 0, "SpaceToBatchND: operation expects padding shape to be %s, but got %s instead", expectedpaddingShape.c_str(), ShapeUtils::shapeAsString(padding).c_str());
    }

    // FIXME - should we use this time-consuming validation ?
    for (uint i = 0; i < numOfSpatialDims; ++i) {
        const uint padLeft       = padding->e<uint>(i,0);
        const uint padRight      = padding->e<uint>(i,1);
        const Nd4jLong blockSize = blockShape->e<Nd4jLong>(i);
        REQUIRE_TRUE((input->sizeAt(i + 1) + padLeft + padRight) % blockSize == 0, 0, "SpaceToBatchND: after padding, spatial dimensions of input array must be divisible by blockSize !");
    }

    helpers::spaceToBatchND(block.launchContext(), *input, *blockShape, *padding, *output);

    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(space_to_batch_nd) {

    getOpDescriptor()->setAllowedInputTypes(0, sd::DataType::ANY)
                     ->setAllowedInputTypes(1, {ALL_INTS})
                     ->setAllowedInputTypes(2, {ALL_INTS})
                     ->setSameMode(true);
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(space_to_batch_nd) {

    auto inputShapeInfo   = inputShape->at(0);
    auto blockShapeInfo   = inputShape->at(1);
    auto paddingShapeInfo = inputShape->at(2);

    REQUIRE_TRUE(blockShapeInfo[0] == 1, 0, "SpaceToBatchND: rank of blockShape array must be equal to one, but got %i instead !", blockShapeInfo[0]);

    const uint numOfSpatialDims = blockShapeInfo[1];

    if(paddingShapeInfo[1] != numOfSpatialDims || paddingShapeInfo[2] != 2) {
        const std::string expectedpaddingShape = "[" + std::to_string(numOfSpatialDims) + ", 2]";   // [numOfSpatialDims, 2]
        REQUIRE_TRUE(false, 0, "SpaceToBatchND: operation expects padding shape to be %s, but got %s instead", expectedpaddingShape.c_str(), ShapeUtils::shapeAsString(paddingShapeInfo).c_str());
    }

    std::vector<Nd4jLong> outShape(inputShapeInfo + 1, inputShapeInfo + 1 + inputShapeInfo[0]);

    outShape[0] *= INPUT_VARIABLE(1)->reduceNumber(sd::reduce::Prod).e<Nd4jLong>(0);

    for (uint i = 0; i < numOfSpatialDims; ++i)
        outShape[i + 1] = (outShape[i + 1] + INPUT_VARIABLE(2)->e<uint>(i,0) + INPUT_VARIABLE(2)->e<uint>(i,1)) / INPUT_VARIABLE(1)->e<Nd4jLong>(i);

    return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outShape));
}

}
}

#endif
