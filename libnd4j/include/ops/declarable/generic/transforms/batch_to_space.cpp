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
#if NOT_EXCLUDED(OP_batch_to_space)

#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/helpers/s_t_b.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(batch_to_space, 2, 1, false, 0, 1) {
  // [bS*blockSize*blockSize, H/blockSize, W/blockSize, iC] is rearranged/permuted to [bS, oH, oW, iC]
  // oH = H - cropTop  - cropBottom
  // oW = W - cropLeft - cropRight

  auto input = INPUT_VARIABLE(0);
  auto crop = INPUT_VARIABLE(1);

  auto output = OUTPUT_VARIABLE(0);

  const LongType blockSize = INT_ARG(0);
  REQUIRE_TRUE(blockSize >= 2, 0, "BatchToSpace: integer parameter block_size must be >= 2, but got %i instead",
               blockSize);

  const int rank = input->rankOf();
  const int dim0 = input->sizeAt(0);
  REQUIRE_TRUE(rank == 4, 0, "BatchToSpace: rank of input array must be equal 4, but got %i instead", rank);
  REQUIRE_TRUE(dim0 % (blockSize * blockSize) == 0, 0,
               "BatchToSpace: first dimension of input array must be divisible by blockSize * blockSize (that is by "
               "%i), but got first dimension equal to %i",
               blockSize * blockSize, dim0);

  if (crop->sizeAt(0) != 2 || crop->sizeAt(1) != 2)
  REQUIRE_TRUE(false, 0, "BatchToSpace: operation expects crop shape to be {2, 2}, but got %s instead",
               ShapeUtils::shapeAsString(crop).c_str());

  const LongType cropBottom = crop->e<LongType>(0, 0);
  const LongType cropTop = crop->e<LongType>(0, 1);
  const LongType cropLeft = crop->e<LongType>(1, 0);
  const LongType cropRight = crop->e<LongType>(1, 1);

  const int oH = input->sizeAt(1) * blockSize - cropBottom - cropTop;  // top and bottom
  const int oW = input->sizeAt(2) * blockSize - cropLeft - cropRight;  // left and right
  REQUIRE_TRUE(oH >= 0, 0,
               "BatchToSpace: crop top/bottom values are too big and cause negative output height dimension !");
  REQUIRE_TRUE(oW >= 0, 0,
               "BatchToSpace: crop left/right values are too big and cause negative output width dimension !");

  if (shape::strideDescendingCAscendingF(input->shapeInfo()))
    helpers::batchToSpace(block.launchContext(), *input, *output, cropBottom, cropTop, cropLeft, cropRight, blockSize);
  else {
    auto dupped = input->dup(input->ordering());
    helpers::batchToSpace(block.launchContext(), *dupped, *output, cropBottom, cropTop, cropLeft, cropRight, blockSize);
    delete dupped;
  }
  return Status::OK;
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(batch_to_space) {
  getOpDescriptor()->setAllowedInputTypes(0, ANY)->setAllowedInputTypes(1, {ALL_INTS})->setSameMode(true);
}

////////////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(batch_to_space) {
  auto inputShapeInfo = inputShape->at(0);
  auto cropShapeInfo = inputShape->at(1);

  const LongType blockSize = INT_ARG(0);
  REQUIRE_TRUE(blockSize >= 2, 0, "BatchToSpace: integer parameter block_size must be >= 2, but got %i instead",
               blockSize);

  const int rank = inputShapeInfo[0];
  const int dim0 = inputShapeInfo[1];
  REQUIRE_TRUE(rank == 4, 0, "BatchToSpace: rank of input array must be equal 4, but got %i instead", rank);
  REQUIRE_TRUE(dim0 % (blockSize * blockSize) == 0, 0,
               "BatchToSpace: first dimension of input array must be divisible by blockSize * blockSize (that is by "
               "%i), but got first dimension equal to %i",
               blockSize * blockSize, dim0);

  if (cropShapeInfo[1] != 2 || cropShapeInfo[2] != 2)
  REQUIRE_TRUE(false, 0, "BatchToSpace: operation expects crop shape to be {2, 2}, but got %s instead",
               ShapeUtils::shapeAsString(cropShapeInfo).c_str());

  const LongType cropBottom = INPUT_VARIABLE(1)->e<LongType>(0, 0);
  const LongType cropTop = INPUT_VARIABLE(1)->e<LongType>(0, 1);
  const LongType cropLeft = INPUT_VARIABLE(1)->e<LongType>(1, 0);
  const LongType cropRight = INPUT_VARIABLE(1)->e<LongType>(1, 1);

  const int oH = inputShapeInfo[2] * blockSize - cropTop - cropBottom;  // top and bottom
  const int oW = inputShapeInfo[3] * blockSize - cropLeft - cropRight;  // left and right
  REQUIRE_TRUE(oH >= 0, 0,
               "BatchToSpace: crop top/bottom values are too big and cause negative output height dimension !");
  REQUIRE_TRUE(oW >= 0, 0,
               "BatchToSpace: crop left/right values are too big and cause negative output width dimension !");

  // we always give out C order here
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(inputShapeInfo), 'c', {dim0 / (blockSize * blockSize), oH, oW, inputShapeInfo[4]}));
}

}  // namespace ops
}  // namespace sd

#endif
