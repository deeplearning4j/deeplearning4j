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
#if NOT_EXCLUDED(OP_im2col)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/col2im.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(im2col, 1, 1, false, 0, 9) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_NULLIFIED(0);

  REQUIRE_TRUE(x->rankOf() == 4, 0, "im2col input should be 4D, but got %i instead", x->rankOf());
  REQUIRE_TRUE(z->rankOf() == 6, 0, "im2col output should be 6D, but got %i instead", z->rankOf());

  LongType kernelHeight = INT_ARG(0);
  LongType kernelWidth = INT_ARG(1);
  LongType strideY = INT_ARG(2);
  LongType strideX = INT_ARG(3);
  LongType padHeight = INT_ARG(4);
  LongType padWidth = INT_ARG(5);
  LongType dY = INT_ARG(6);  // Dilation, height/y dimension
  LongType dX = INT_ARG(7);  // Dilation, width/x dimension
  bool isSameMode = INT_ARG(8) > 0;
  double zeroPadVal = 0.0;
  if (block.getTArguments()->size() > 0) zeroPadVal = T_ARG(0);

  // FIXME: zeropad value is void
  LaunchContext* ctx = block.launchContext();
  helpers::im2col(*ctx, *x, *z, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth, dY, dX,
                           NDArrayFactory::create(zeroPadVal, block.launchContext()));

  return Status::OK;
}

DECLARE_SHAPE_FN(im2col) {
  auto inShape = inputShape->at(0);

  LongType bS = shape::shapeOf(inShape)[0];
  LongType iD = shape::shapeOf(inShape)[1];
  LongType inY = shape::shapeOf(inShape)[2];
  LongType inX = shape::shapeOf(inShape)[3];

  LongType kY = INT_ARG(0);
  LongType kX = INT_ARG(1);
  LongType sY = INT_ARG(2);
  LongType sX = INT_ARG(3);
  LongType pY = INT_ARG(4);
  LongType pX = INT_ARG(5);
  LongType dY = INT_ARG(6);  // Dilation, height/y dimension
  LongType dX = INT_ARG(7);  // Dilation, width/x dimension
  int paddingMode = INT_ARG(8);
  bool isSameMode = INT_ARG(8) == 1;
  // output is always 6d for im2col
  LongType* zShape;
  ALLOCATE(zShape, block.getWorkspace(), shape::shapeInfoLength(6), sd::LongType);

  LongType oY = 0;
  LongType oX = 0;

  ConvolutionUtils::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, paddingMode);

  if (isSameMode) ConvolutionUtils::calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sY, sX, dY, dX);

  zShape[0] = 6;
  zShape[1] = bS;
  zShape[2] = iD;
  zShape[3] = kY;
  zShape[4] = kX;
  zShape[5] = oY;
  zShape[6] = oX;

  zShape[shape::shapeInfoLength(zShape) - 2] = 1;
  zShape[shape::shapeInfoLength(zShape) - 1] = 99;

  ShapeUtils::updateStridesAndType(zShape, inShape, 'c');

  return SHAPELIST(CONSTANT(zShape));
}

CUSTOM_OP_IMPL(im2col_bp, 2, 1, false, 0, 9) {
  auto input = INPUT_VARIABLE(0);
  auto gradAtOutput = INPUT_VARIABLE(1);
  auto z = OUTPUT_NULLIFIED(0);

  REQUIRE_TRUE(input->rankOf() == 4, 0, "im2col_bp input should be 4D, but got %i instead", input->rankOf());
  REQUIRE_TRUE(gradAtOutput->rankOf() == 6, 0,
               "im2col_bp gradient at output (input idx 1) should be 6D, but got %i instead", gradAtOutput->rankOf());
  REQUIRE_TRUE(z->rankOf() == 4, 0, "im2col_bp output (grad at input) should be 4D, but got %i instead", z->rankOf());

  LongType kernelHeight = INT_ARG(0);
  LongType kernelWidth = INT_ARG(1);
  LongType strideY = INT_ARG(2);
  LongType strideX = INT_ARG(3);
  LongType pH = INT_ARG(4);
  LongType pW = INT_ARG(5);
  LongType dY = INT_ARG(6);  // Dilation, height/y dimension
  LongType dX = INT_ARG(7);  // Dilation, width/x dimension
  int paddingMode = INT_ARG(8);
  double zeroPadVal = 0.0;
  if (block.getTArguments()->size() > 0) zeroPadVal = T_ARG(0);

  // Assuming NCHW format here
  int imgH = input->sizeAt(2);
  int imgW = input->sizeAt(3);

  LaunchContext* ctx = block.launchContext();
  // FIXME:: all helpers should accept NDArray
  helpers::col2im(*ctx, gradAtOutput, z, strideY, strideX, pH, pW, imgH, imgW, dY, dX);

  return Status::OK;
}

DECLARE_TYPES(im2col) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedOutputTypes(0, INHERIT)
      ->setSameMode(true);
}

DECLARE_TYPES(im2col_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, ANY)
      ->setAllowedOutputTypes(0, INHERIT)
      ->setSameMode(true);
}

DECLARE_SHAPE_FN(im2col_bp) {
  LongType* inShape;
  COPY_SHAPE(inputShape->at(0), inShape);

  return SHAPELIST(CONSTANT(inShape));
}
}  // namespace ops
}  // namespace sd

#endif
