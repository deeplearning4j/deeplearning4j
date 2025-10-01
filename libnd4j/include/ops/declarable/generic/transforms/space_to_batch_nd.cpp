
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_space_to_batch_nd)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(space_to_batch_nd, 3, 1, false, 0, 0) {
  // 4D example, numOfSpatialDims = 2 - two spatial dimensions
  // [bS, iH, iW, iC] is rearranged/permuted to [bS*blockShape[0]*blockShape[1], (iH + padBottom + padTop)/blockSize[0],
  // (iW + padLeft + padRight)/blockSize[1], iC]

  auto input = INPUT_VARIABLE(0);
  auto blockShape = INPUT_VARIABLE(1);
  auto padding = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(blockShape->rankOf() == 1, 0,
               "SpaceToBatchND: rank of blockShape array must be equal to one, but got %i instead !",
               blockShape->rankOf());

  const LongType numOfSpatialDims = blockShape->sizeAt(0);

  REQUIRE_TRUE(input->rankOf() == output->rankOf(), 0,
               "SpaceToBatchND: rank of input and output array must be the same, but got %i and %i correspondingly !",
               input->rankOf(), output->rankOf());

  if (padding->sizeAt(0) != numOfSpatialDims || padding->sizeAt(1) != 2) {
    const std::string expectedpaddingShape = "[" + std::to_string(numOfSpatialDims) + ", 2]";  // [numOfSpatialDims, 2]
    REQUIRE_TRUE(false, 0, "SpaceToBatchND: operation expects padding shape to be %s, but got %s instead",
                 expectedpaddingShape.c_str(), ShapeUtils::shapeAsString(padding).c_str());
  }

  // FIXME - should we use this time-consuming validation ?
  for (LongType i = 0; i < numOfSpatialDims; ++i) {
    const LongType padLeft = padding->e<LongType>(i, 0);
    const LongType padRight = padding->e<LongType>(i, 1);
    const LongType blockSize = blockShape->e<LongType>(i);
    REQUIRE_TRUE((input->sizeAt(i + 1) + padLeft + padRight) % blockSize == 0, 0,
                 "SpaceToBatchND: after padding, spatial dimensions of input array must be divisible by blockSize !");
  }

  if (shape::strideDescendingCAscendingF(input->shapeInfo()))
    helpers::spaceToBatchND(block.launchContext(), *input, *blockShape, *padding, *output);
  else {
    NDArray *inputDup = input->dup(input->ordering());
    helpers::spaceToBatchND(block.launchContext(), *inputDup, *blockShape, *padding, *output);
  }
  return Status::OK;
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(space_to_batch_nd) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedInputTypes(1, {ALL_INTS})
      ->setAllowedInputTypes(2, {ALL_INTS})
      ->setSameMode(true);
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(space_to_batch_nd) {
  auto inputShapeInfo = inputShape->at(0);
  auto blockShapeInfo = inputShape->at(1);
  auto paddingShapeInfo = inputShape->at(2);

  REQUIRE_TRUE(blockShapeInfo[0] == 1, 0,
               "SpaceToBatchND: rank of blockShape array must be equal to one, but got %i instead !",
               blockShapeInfo[0]);

  const LongType numOfSpatialDims = blockShapeInfo[1];

  if (paddingShapeInfo[1] != numOfSpatialDims || paddingShapeInfo[2] != 2) {
    const std::string expectedpaddingShape = "[" + std::to_string(numOfSpatialDims) + ", 2]";  // [numOfSpatialDims, 2]
    REQUIRE_TRUE(false, 0, "SpaceToBatchND: operation expects padding shape to be %s, but got %s instead",
                 expectedpaddingShape.c_str(), ShapeUtils::shapeAsString(paddingShapeInfo).c_str());
  }

  std::vector<LongType> outShape(inputShapeInfo + 1, inputShapeInfo + 1 + inputShapeInfo[0]);

  outShape[0] *= INPUT_VARIABLE(1)->reduceNumber(reduce::Prod).e<LongType>(0);

  for (LongType i = 0; i < numOfSpatialDims; ++i)
    outShape[i + 1] =
        (outShape[i + 1] + INPUT_VARIABLE(2)->e<LongType>(i, 0) + INPUT_VARIABLE(2)->e<LongType>(i, 1)) /
        INPUT_VARIABLE(1)->e<LongType>(i);

  return SHAPELIST(
      ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShapeInfo), 'c', outShape));
}

}  // namespace ops
}  // namespace sd

#endif
