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
#if NOT_EXCLUDED(OP_batch_to_space_nd)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace sd {
namespace ops  {


CUSTOM_OP_IMPL(batch_to_space_nd, 3, 1, false, 0, 0) {

    // 4D example, numOfSpatialDims = 2 - two spatial dimensions
    // [bS*blockShape[0]*blockShape[1], iH, iW, iC] is rearranged/permuted to [bS, iH*blockShape[0] - cropTop  - cropBottom, iW*blockShape[1] - cropLeft - cropRight, iC]

    auto input      = INPUT_VARIABLE(0);
    auto blockShape = INPUT_VARIABLE(1);
    auto crop       = INPUT_VARIABLE(2);

    auto output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(blockShape->rankOf() == 1, 0, "BatchToSpaceND: rank of blockShape array must be equal to one, but got %i instead !", blockShape->rankOf());

    const uint numOfSpatialDims = blockShape->sizeAt(0);

    const auto product = blockShape->reduceNumber(sd::reduce::Prod).e<Nd4jLong>(0);
    REQUIRE_TRUE(input->sizeAt(0) % product == 0, 0, "BatchToSpaceND: first dimension of input array must be divisible by product of blockShape array elements (= %lld), but got first dimension equal to %i", product, input->sizeAt(0));

    if(crop->sizeAt(0) != numOfSpatialDims || crop->sizeAt(1) != 2) {
        const std::string expectedCropShape = "[" + std::to_string(numOfSpatialDims) + ", 2]";   // [numOfSpatialDims, 2]
        REQUIRE_TRUE(false, 0, "BatchToSpaceND: operation expects padding shape to be %s, but got %s instead", expectedCropShape.c_str(), ShapeUtils::shapeAsString(crop).c_str());
    }

    // FIXME - should we use this time-consuming validation ?
    for (uint i = 0; i < numOfSpatialDims; ++i) {
        const auto cropLeft      = crop->e<uint>(i,0);
        const auto cropRight     = crop->e<uint>(i,1);
        const auto outSpatialDim = input->sizeAt(i + 1) * blockShape->e<Nd4jLong>(i) - cropLeft - cropRight;
        REQUIRE_TRUE(outSpatialDim >= 0, 0, "BatchToSpaceND: crop left/right values are too big and cause negative output spatial dimension/dimensions !");
    }

    helpers::batchToSpaceND(block.launchContext(), *input, *blockShape, *crop, *output);

    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(batch_to_space_nd) {

    getOpDescriptor()->setAllowedInputTypes(0, sd::DataType::ANY)
                     ->setAllowedInputTypes(1, {ALL_INTS})
                     ->setAllowedInputTypes(2, {ALL_INTS})
                     ->setSameMode(true);
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(batch_to_space_nd) {

    auto inputShapeInfo = inputShape->at(0);
    auto blockShapeInfo = inputShape->at(1);
    auto cropShapeInfo  = inputShape->at(2);

    REQUIRE_TRUE(blockShapeInfo[0] == 1, 0, "BatchToSpaceND: rank of blockShape array must be equal to one, but got %i instead !", blockShapeInfo[0]);

    const auto product = INPUT_VARIABLE(1)->reduceNumber(sd::reduce::Prod).e<Nd4jLong>(0);
    REQUIRE_TRUE(inputShapeInfo[1] % product == 0, 0, "BatchToSpaceND: first dimension of input array must be divisible by product of blockShape array elements (= %lld), but got first dimension equal to %i", product, inputShapeInfo[1]);

    const auto numOfSpatialDims = blockShapeInfo[1];

    if(cropShapeInfo[1] != numOfSpatialDims || cropShapeInfo[2] != 2) {
        const std::string expectedCropShape = "[" + std::to_string(numOfSpatialDims) + ", 2]";   // [numOfSpatialDims, 2]
        REQUIRE_TRUE(false, 0, "BatchToSpaceND: operation expects padding shape to be %s, but got %s instead", expectedCropShape.c_str(), ShapeUtils::shapeAsString(cropShapeInfo).c_str());
    }


    std::vector<Nd4jLong> outShape(inputShapeInfo + 1, inputShapeInfo + 1 + inputShapeInfo[0]);

    outShape[0] /= product;

    for (uint i = 0; i < numOfSpatialDims; ++i)
        outShape[i + 1] = outShape[i + 1] * INPUT_VARIABLE(1)->e<Nd4jLong>(i) - INPUT_VARIABLE(2)->e<uint>(i,0) - INPUT_VARIABLE(2)->e<uint>(i,1);

    return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outShape));
}


}
}

#endif