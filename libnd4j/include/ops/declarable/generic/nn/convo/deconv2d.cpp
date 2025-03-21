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
// @author raver119@gmail.com
// @author Yurii Shyrma
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_deconv2d)

#include <helpers/MmulHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/col2im.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(deconv2d, 2, 1, false, 0, 9) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_NULLIFIED(0);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "CUSTOM DECONV2D OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "CUSTOM DECONV2D OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) height
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) width
  LongType sH = INT_ARG(2);                                                          // strides height
  LongType sW = INT_ARG(3);                                                          // strides width
  sd::LongType pH = INT_ARG(4);                                                          // paddings height
  sd::LongType  pW = INT_ARG(5);                                                          // paddings width
  LongType dH = INT_ARG(6);                                                          // dilations height
  LongType dW = INT_ARG(7);                                                          // dilations width
  int isSameMode = INT_ARG(8);                                                  // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;             // INT_ARG(9): 0-NCHW,  1-NHWC
  int wFormat = block.getIArguments()->size() > 10
                ? INT_ARG(10)
                : 0;  // 0 - [kH, kW, oC, iC], 1 - [iC, oC, kH, kW], 2 - [iC, kH, kW, oC]

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWoC, indWiC, indWkH, indOoH);

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, oC, iC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
  REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
               "CUSTOM DECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i "
               "instead !",
               oC, bias->rankOf(), bias->lengthOf());

  std::vector<LongType> outputPermute = {0,3,1,2};
  if (!isNCHW) output = new NDArray(output->permute(outputPermute, false, false));  // [bS, oH, oW, oC] -> [bS, oC, oH, oW]

  std::vector<LongType> colPermut;
  if (1 == wFormat)
    colPermut = {1, 2, 3, 0, 4, 5};
  else
    colPermut = {2, 3, 1, 0, 4, 5};

  if (isSameMode)  // Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not
    // deconv) forward pass
    ConvolutionUtils::calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);

  std::vector<sd::LongType> colShape = {bS, oC, kH, kW, iH, iW};
  NDArray columns(input->ordering(), colShape, input->dataType(), block.launchContext());

  //----- calculation of output -----//
  // NHWC: [kH, kW, oC, iC] x [bS, iH, iW, iC] = [kH, kW, oC, bS, iH, iW]
  // NHWC: [iC, oC, kH, kW] x [bS, iH, iW, iC] = [oC, kH, kW, bS, iH, iW]
  // NHWC: [iC, kH, kW, oC] x [bS, iH, iW, iC] = [kH, kW, oC, bS, iH, iW]
  std::vector<LongType> firstDims = {indWiC};
  std::vector<LongType> secondDims = {indIOioC};
  sd::MmulHelper::tensorDot(weights, input, &columns, firstDims, secondDims, colPermut);
  LaunchContext* ctx = block.launchContext();
  helpers::col2im(*ctx, &columns, output, sH, sW, pH, pW, oH, oW, dH,
                  dW);  // [bS, oC, kH, kW, iH, iW] is de-convoluted to [bS, oC, oH, oW]

  //----- add biases if required -----//
  if (bias)
    helpers::addBias(block, *output, *bias, *output, true);

  if (!isNCHW) delete output;

  return sd::Status::OK;
}
DECLARE_TYPES(deconv2d) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(deconv2d) {
  auto inputShapeInfo = inputShape->at(0);    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto weightsShapeInfo = inputShape->at(1);  // [kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]
  auto biasShapeInfo = block.width() > 2 ? inputShape->at(2) : nullptr;  // [oC]

  const int rank = 4;
  REQUIRE_TRUE(shape::rank(inputShapeInfo) == rank, 0,
               "CUSTOM DECONV2D OP: rank of input array must be equal to %i, but got %i instead !", rank,
               shape::rank(inputShapeInfo));
  REQUIRE_TRUE(shape::rank(weightsShapeInfo) == rank, 0,
               "CUSTOM DECONV2D OP: rank of weights array must be equal to %i, but got %i instead !", rank,
               shape::rank(weightsShapeInfo));

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(0)));  // filter(kernel) height
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(1)));  // filter(kernel) width
  LongType sH = INT_ARG(2);                                                                          // strides height
  LongType sW = INT_ARG(3);                                                                          // strides width
  LongType pH = INT_ARG(4);                                                                          // paddings height
  LongType pW = INT_ARG(5);                                                                          // paddings width
  LongType dH = INT_ARG(6);                                                                          // dilations height
  LongType dW = INT_ARG(7);                                                                          // dilations width
  int isSameMode = INT_ARG(8);                                                                  // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 1-NHWC, 0-NCHW
  int wFormat = block.getIArguments()->size() > 10
                ? INT_ARG(10)
                : 0;  // 0 - [kH, kW, oC, iC], 1 - [iC, oC, kH, kW], 2 - [iC, kH, kW, oC]

  LongType indIOioC, indIiH, indWoC(0 == wFormat ? 2 : (1 == wFormat ? 1 : 3));
  if (!isNCHW) {
    indIOioC = 3;
    indIiH = 1;
  } else {
    indIOioC = 1;
    indIiH = 2;
  }

  const LongType bS = inputShapeInfo[1];             // batch size
  const LongType iH = inputShapeInfo[indIiH + 1];    // input height
  const LongType iW = inputShapeInfo[indIiH + 2];    // input width
  const LongType iC = inputShapeInfo[indIOioC + 1];  // input channels
  const LongType oC = weightsShapeInfo[indWoC + 1];  // output channels

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, oC, iC);
  REQUIRE_TRUE(shape::shapeEquals(4, expectedWeightsShape.data(), shape::rank(weightsShapeInfo),
                                  shape::shapeOf(weightsShapeInfo)),
               0, "CUSTOM DECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
  if (biasShapeInfo)
  REQUIRE_TRUE(shape::rank(biasShapeInfo) <= 2 && oC == shape::length(biasShapeInfo), 0,
               "CUSTOM DECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i "
               "instead !",
               oC, biasShapeInfo[0], shape::length(biasShapeInfo));

  LongType oH, oW;  // output height, width
  ConvolutionUtils::calcOutSizeDeconv2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  sd::LongType outputShape[4];

  outputShape[0] = bS;
  if (isNCHW) {
    outputShape[1] = oC;
    outputShape[2] = oH;
    outputShape[3] = oW;
  } else {
    outputShape[1] = oH;
    outputShape[2] = oW;
    outputShape[3] = oC;
  }

  auto desc = new  ShapeDescriptor(ArrayOptions::dataType(weightsShapeInfo), shape::order(inputShapeInfo), outputShape, 4);
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
}

DECLARE_TYPES(deconv2d_bp) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(deconv2d_bp, 3, 2, false, 0, 9) {
  auto input = INPUT_VARIABLE(0);                               // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);                             // [kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
               ? INPUT_VARIABLE(3)
               : INPUT_VARIABLE(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW), gradI
  auto gradW = OUTPUT_VARIABLE(1);  // [kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  REQUIRE_TRUE(input->rankOf() == 4, 0,
               "CUSTOM DECONV2D_BP OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 4, 0,
               "CUSTOM DECONV2D_BP OP: rank of weights array must be equal to 4 , but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(
      gradO->rankOf() == 4, 0,
      "CUSTOM DECONV2D_BP OP: rank of output gradients (next epsilon) array must be equal to 4, but got %i instead !",
      gradO->rankOf());

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) height
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) width
  LongType sH = INT_ARG(2);                                                          // strides height
  LongType sW = INT_ARG(3);                                                          // strides width
  sd::LongType pH = INT_ARG(4);                                                          // paddings height
  sd::LongType pW = INT_ARG(5);                                                          // paddings width
  LongType dH = INT_ARG(6);                                                          // dilations height
  LongType dW = INT_ARG(7);                                                          // dilations width
  int isSameMode = INT_ARG(8);                                                  // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;             // INT_ARG(9): 1-NHWC, 0-NCHW
  int wFormat = block.getIArguments()->size() > 10
                ? INT_ARG(10)
                : 0;  // 0 - [kH, kW, oC, iC], 1 - [iC, oC, kH, kW], 2 - [iC, kH, kW, oC]

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWoC, indWiC, indWkH, indOoH);

  LongType trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizeDeconv2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, oC, iC);
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "CUSTOM DECONV2D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got "
               "%s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
  REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
               "CUSTOM DECONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
               "%i instead !",
               oC, bias->rankOf(), bias->lengthOf());

  if (isSameMode) {  // SAME
    // Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward
    // pass
    ConvolutionUtils::calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);
  }

  // ----- calculation of gradI -> pass it through conv2d_ff ----- //
  sd::ops::conv2d conv2d;

  const sd::Status status =
      conv2d.execute({gradO, weights}, {gradI}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, !isNCHW, wFormat}, {});
  if (status != sd::Status::OK) return status;

  // -----prepare permutation arrays and axes for dot product ----- //
  std::vector<LongType> inputAxes;

  if (!isNCHW) {
    std::vector<LongType> permuteDims = {0,3,1,2};
    gradO = new NDArray(gradO->permute(permuteDims, false, false));  // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
    inputAxes = {0, 1, 2};                              // bS, iH, iW
  } else
    inputAxes = {0, 2, 3};  // bS, iH, iW

  std::vector<LongType> gradWAxes;  // empty for wFormat = 1
  if (0 == wFormat)
    gradWAxes = {3, 2, 0, 1};
  else if (2 == wFormat)
    gradWAxes = {0, 3, 1, 2};
  std::vector<sd::LongType> colShape = {bS, oC, kH, kW, iH, iW};

  // ----- calculation of gradW ----- //
  NDArray columns(input->ordering(), colShape, input->dataType(), block.launchContext());

  LaunchContext* ctx = block.launchContext();
  NDArray zero = NDArrayFactory::create(0.f, input->getContext());
  helpers::im2col(
      *ctx, *gradO, columns, kH, kW, sH, sW, pH, pW, dH, dW,
     zero );  // [bS, oC, oH, oW] is convoluted to [bS, oC, kH, kW, iH, iW]
  std::vector<LongType> mulDims = {0,4,5};
  MmulHelper::tensorDot(input, &columns, gradW, inputAxes, mulDims,
                        gradWAxes);  // [bS, iC, iH, iW]/[bS, iH, iW, iC] x [bS, oC, kH, kW, iH, iW] = [iC, oC, kH, kW]

  // ----- calculation of gradB ----- //
  if (gradB) {
    std::vector<LongType> bShape = {gradB->lengthOf()};
    if (gradB->rankOf() == 2) gradB = new NDArray(gradB->reshape(gradB->ordering(), bShape, false));
    std::vector<sd::LongType> axesForReduction = {0, 2, 3};  // bS, oH, oW
    gradO->reduceAlongDimension(reduce::Sum, gradB, &axesForReduction);  // sum over bS, oH, oW
    if (gradB != OUTPUT_VARIABLE(2)) delete gradB;
  }

  if (!isNCHW) delete gradO;

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(deconv2d_bp) {
  auto inputShapeInfo = inputShape->at(0);    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW)
  auto weightsShapeInfo = inputShape->at(1);  // [kH, kW, oC, iC], [iC, oC, kH, kW], [iC, kH, kW, oC]
  sd::LongType const* biasShapeInfo = block.width() > 3 ? inputShape->at(2) : nullptr;  // [oC]
  auto gradOShapeInfo = block.width() > 3
                        ? inputShape->at(3)
                        : inputShape->at(2);  // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

  const int rank = 4;
  REQUIRE_TRUE(shape::rank(inputShapeInfo) == rank, 0,
               "CUSTOM DECONV2D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank,
               shape::rank(inputShapeInfo));
  REQUIRE_TRUE(shape::rank(weightsShapeInfo) == rank, 0,
               "CUSTOM DECONV2D_BP OP: rank of weights array must be equal to %i , but got %i instead !", rank,
               shape::rank(weightsShapeInfo));
  REQUIRE_TRUE(
      shape::rank(gradOShapeInfo) == rank, 0,
      "CUSTOM DECONV2D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !",
      rank, shape::rank(gradOShapeInfo));

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(0)));  // filter(kernel) height
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(1)));  // filter(kernel) width
  LongType sH = INT_ARG(2);                                                                          // strides height
  LongType sW = INT_ARG(3);                                                                          // strides width
  LongType pH = INT_ARG(4);                                                                          // paddings height
  LongType pW = INT_ARG(5);                                                                          // paddings width
  LongType dH = INT_ARG(6);                                                                          // dilations height
  LongType dW = INT_ARG(7);                                                                          // dilations width
  int isSameMode = INT_ARG(8);                                                                  // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;  // INT_ARG(9): 1-NHWC, 0-NCHW
  int wFormat = block.getIArguments()->size() > 10
                ? INT_ARG(10)
                : 0;  // 0 - [kH, kW, oC, iC], 1 - [iC, oC, kH, kW], 2 - [iC, kH, kW, oC]

  LongType indIOioC, indIiH, indOoH, indWoC(0 == wFormat ? 2 : (1 == wFormat ? 1 : 3));
  if (!isNCHW) {
    indIOioC = 3;
    indIiH = 1;
    indOoH = 1;
  } else {
    indIOioC = 1;
    indIiH = 2;
    indOoH = 2;
  }

  const LongType bS = inputShapeInfo[1];             // batch size
  const LongType iH = inputShapeInfo[indIiH + 1];    // input height
  const LongType iW = inputShapeInfo[indIiH + 2];    // input width
  const LongType iC = inputShapeInfo[indIOioC + 1];  // input channels
  const LongType oC = weightsShapeInfo[indWoC + 1];  // output channels

  LongType trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizeDeconv2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, oC, iC);
  REQUIRE_TRUE(
      shape::shapeEquals(4, expectedGradOShape.data(), shape::rank(gradOShapeInfo), shape::shapeOf(gradOShapeInfo)), 0,
      "CUSTOM DECONV2D_BP OP: wrong shape of output gradients next epsilon) array, expected is %s, but got %s instead "
      "!",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
  REQUIRE_TRUE(shape::shapeEquals(4, expectedWeightsShape.data(), shape::rank(weightsShapeInfo),
                                  shape::shapeOf(weightsShapeInfo)),
               0, "CUSTOM DECONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
  if (biasShapeInfo)
  REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0,
               "CUSTOM DECONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
               "%i instead !",
               oC, biasShapeInfo[0], shape::length(biasShapeInfo));

  auto gradIShapeInfo =
      ShapeBuilders::copyShapeInfoAndType(inputShapeInfo, gradOShapeInfo, false, block.getWorkspace());
  auto gradWShapeInfo =
      ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, gradOShapeInfo, false, block.getWorkspace());

  auto shapes = SHAPELIST(CONSTANT(gradIShapeInfo), CONSTANT(gradWShapeInfo));

  if (biasShapeInfo != nullptr) {
    auto gradBShapeInfo =
        ShapeBuilders::copyShapeInfoAndType(biasShapeInfo, gradOShapeInfo, false, block.getWorkspace());
    shapes->push_back(CONSTANT(gradBShapeInfo));
  }

  return shapes;
}

}  // namespace ops
}  // namespace sd

#endif
