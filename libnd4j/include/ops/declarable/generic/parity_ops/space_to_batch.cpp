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
// @author raver119@gmail.com, created on 19.01.18.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_space_to_batch)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace nd4j {
namespace ops {


CUSTOM_OP_IMPL(space_to_batch, 2, 1, false, 0, 1) {

    // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockSize*blockSize, (iH + padBottom + padTop)/blockSize, (iW + padLeft + padRight)/blockSize, iC]

    auto input   = INPUT_VARIABLE(0);
    auto padding = INPUT_VARIABLE(1);

    auto output = OUTPUT_VARIABLE(0);

    const uint blockSize = INT_ARG(0);
    REQUIRE_TRUE(blockSize >= 2, 0, "SpaceToBatch: integer parameter block_size must be >= 2, but got %i instead", blockSize);

    REQUIRE_TRUE(input->rankOf() == 4,  0, "SpaceToBatch: rank of input array must be equal 4, but got %i instead",  input->rankOf());
    REQUIRE_TRUE(output->rankOf() == 4, 0, "SpaceToBatch: rank of output array must be equal 4, but got %i instead", output->rankOf());

    const std::string expectedpaddingShape = "[2, 2]";
    const std::string actualpaddingShape = ShapeUtils::shapeAsString(padding);
    REQUIRE_TRUE(actualpaddingShape == expectedpaddingShape, 0, "SpaceToBatch: operation expects padding shape to be {2, 2}, but got %s instead", actualpaddingShape.c_str());

    const uint padBottom = padding->e<uint>(0,0);
    const uint padTop    = padding->e<uint>(0,1);
    const uint padLeft   = padding->e<uint>(1,0);
    const uint padRight  = padding->e<uint>(1,1);

    REQUIRE_TRUE((input->sizeAt(1) + padBottom + padTop) % blockSize == 0 && (input->sizeAt(2) + padLeft + padRight) % blockSize == 0, 0, "SpaceToBatch: after padding, second and third dimensions of input array must be divisible by blockSize !");

    helpers::spaceToBatch(block.launchContext(), *input, *output, padBottom, padTop, padLeft, padRight, blockSize);

    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(space_to_batch) {

    getOpDescriptor()->setAllowedInputTypes(0, nd4j::DataType::ANY)
                     ->setAllowedInputTypes(1, {ALL_INTS})
                     ->setSameMode(true);
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(space_to_batch) {

    auto inputShapeInfo   = inputShape->at(0);
    auto paddingShapeInfo = inputShape->at(1);

    const uint blockSize = INT_ARG(0);
    REQUIRE_TRUE(blockSize >= 2, 0, "SpaceToBatch: integer parameter block_size must be >= 2, but got %i instead", blockSize);

    const int rank = inputShapeInfo[0];
    REQUIRE_TRUE(rank == 4, 0, "SpaceToBatch: rank of input array must be equal 4, but got %i instead", rank);

    const std::string expectedpaddingShape = "[2, 2]";
    const std::string actualpaddingShape = ShapeUtils::shapeAsString(paddingShapeInfo);
    REQUIRE_TRUE(actualpaddingShape == expectedpaddingShape, 0, "SpaceToBatch: operation expects padding shape to be {2, 2}, but got %s instead", actualpaddingShape.c_str());

    const uint padBottom = INPUT_VARIABLE(1)->e<Nd4jLong>(0,0);
    const uint padTop    = INPUT_VARIABLE(1)->e<Nd4jLong>(0,1);
    const uint padLeft   = INPUT_VARIABLE(1)->e<Nd4jLong>(1,0);
    const uint padRight  = INPUT_VARIABLE(1)->e<Nd4jLong>(1,1);

    REQUIRE_TRUE((inputShapeInfo[2] + padBottom + padTop) % blockSize == 0 && (inputShapeInfo[3] + padLeft + padRight) % blockSize == 0, 0, "SpaceToBatch: after padding, second and third dimensions of input array must be divisible by blockSize !");

    return SHAPELIST(ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', {inputShapeInfo[1] * blockSize * blockSize, (inputShapeInfo[2] + padBottom + padTop) / blockSize, (inputShapeInfo[3] + padLeft + padRight) / blockSize, inputShapeInfo[4]}));
}

}
}

#endif
