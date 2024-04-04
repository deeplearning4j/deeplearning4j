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
// Created by raver119 on 17.10.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_col2im)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/col2im.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(col2im, 1, 1, false, 0, 9) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_NULLIFIED(0);

  REQUIRE_TRUE(x->rankOf() == 6, 0, "col2im input should be 6D, but got %i instead", x->rankOf());
  REQUIRE_TRUE(z->rankOf() == 4, 0, "col2im output should be 4D, but got %i instead", z->rankOf());

  LongType strideY = INT_ARG(0);
  LongType strideX = INT_ARG(1);
  LongType padHeight = INT_ARG(2);
  LongType padWidth = INT_ARG(3);
  LongType imgHeight = INT_ARG(4);
  LongType imgWidth = INT_ARG(5);
  LongType dY = INT_ARG(6);  // Dilation in height/y dimension
  LongType dX = INT_ARG(7);  // Dilation in width/x dimension

  LaunchContext* ctx = block.launchContext();
  helpers::col2im(*ctx, x, z, strideY, strideX, padHeight, padWidth, imgHeight, imgWidth, dY, dX);

  return Status::OK;
}
DECLARE_SHAPE_FN(col2im) {
  auto inShape = inputShape->at(0);

  LongType bS = shape::shapeOf(inShape)[0];
  LongType iD = shape::shapeOf(inShape)[1];

  LongType sY = INT_ARG(0);
  LongType sX = INT_ARG(1);
  LongType pY = INT_ARG(2);
  LongType pX = INT_ARG(3);
  LongType inY = INT_ARG(4);
  LongType inX = INT_ARG(5);
  LongType dY = INT_ARG(6);  // Dilation, height/y dimension
  LongType dX = INT_ARG(7);  // Dilation, width/x dimension
  bool isSameMode = INT_ARG(8) > 0;

  LongType* zShape;
  ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(4), sd::LongType);

  zShape[0] = 4;
  zShape[1] = bS;
  zShape[2] = iD;
  zShape[3] = inY;
  zShape[4] = inX;

  zShape[shape::shapeInfoLength(zShape) - 2] = 1;
  zShape[shape::shapeInfoLength(zShape) - 1] = 99;

  ShapeUtils::updateStridesAndType(zShape, inShape, 'c');

  return SHAPELIST(CONSTANT(zShape));
}

DECLARE_TYPES(col2im) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedOutputTypes(0, INHERIT)
      ->setSameMode(true);
}
}  // namespace ops
}  // namespace sd

#endif
