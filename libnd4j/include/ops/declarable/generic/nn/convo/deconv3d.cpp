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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.09.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_deconv3d)

#include <helpers/MmulHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(deconv3d, 2, 1, false, 0, 13) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;  // [oC]

  auto output = OUTPUT_VARIABLE(0);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "CUSTOM DECONV3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CUSTOM DECONV3D OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) depth
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) height
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(weights->sizeAt(2));  // filter(kernel) width
  LongType sD = INT_ARG(3);                                                          // strides depth
  LongType sH = INT_ARG(4);                                                          // strides height
  LongType sW = INT_ARG(5);                                                          // strides width
  LongType pD = INT_ARG(6);                                                          // paddings depth
  LongType pH = INT_ARG(7);                                                          // paddings height
  LongType pW = INT_ARG(8);                                                          // paddings width
  LongType dD = INT_ARG(9);                                                          // dilations depth
  LongType dH = INT_ARG(10);                                                         // dilations height
  LongType dW = INT_ARG(11);                                                         // dilations width
  int isSameMode = INT_ARG(12);                                                 // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]


  sd_printf("Deconv3d: getting sizes\n",0);
  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM DECONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i "
                 "instead !",
                 oC, bias->rankOf(), bias->lengthOf());


  sd_printf("isNCDHW %d\n",isNCDHW);
  if (!isNCDHW) output = new NDArray(output->permute({0, 4, 1, 2, 3}));  // [bS, oD, oH, oW, oC] -> [bS, oC, oD, oH, oW]

  std::vector<LongType> colPermut;
  if (1 == wFormat)
    colPermut = {1, 2, 3, 4, 0, 5, 6, 7};
  else
    colPermut = {2, 3, 4, 1, 0, 5, 6, 7};

  if (isSameMode)  // Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not
                   // deconv) forward pass
    ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
  sd_printf("Deconv3d: calculated padding\n",0);

  NDArray columns(input->ordering(), {bS, oC, kD, kH, kW, iD, iH, iW}, input->dataType(), block.launchContext());

  //----- calculation of output -----//
  // [kD, kH, kW, oC, iC] x [bS, iD, iH, iW, iC] = [kD, kH, kW, oC, bS, iD, iH, iW]
  // [iC, oC, kD, kH, kW] x [bS, iD, iH, iW, iC] = [oC, kD, kH, kW, bS, iD, iH, iW]
  // [iC, kD, kH, kW, oC] x [bS, iD, iH, iW, iC] = [kD, kH, kW, oC, bS, iD, iH, iW]
  sd::MmulHelper::tensorDot(weights, input, &columns, {indWiC}, {indIOioC},
                            colPermut);  // [bS, oC, kD, kH, kW, iD, iH, iW] -> [kD, kH, kW, oC, bS, iD, iH, iW]
  sd_printf("Deconv3d: after tensorDot\n",0);

  ConvolutionUtils::col2vol(block, columns, *output, sD, sH, sW, pD, pH, pW, dD, dH,
                            dW);  // [bS, oC, kD, kH, kW, iD, iH, iW] is de-convoluted to [bS, oC, oD, oH, oW]
  sd_printf("Deconv3d: after col2vol\n",0);

  //----- add biases if required -----//
  if (bias)
    helpers::addBias(block, *output, *bias, *output, true);

  sd_printf("addBias\n",0);
  //if (!isNCDHW) delete output;

  return sd::Status::OK;
}

DECLARE_TYPES(deconv3d) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(deconv3d) {
  auto inputShapeInfo = inputShape->at(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NDCHW)
  auto weightsShapeInfo = inputShape->at(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto biasShapeInfo = block.width() > 2 ? inputShape->at(2) : nullptr;  // [oC]

  const int rank = 5;
  REQUIRE_TRUE(shape::rank(inputShapeInfo) == rank, 0,
               "CUSTOM DECONV3D OP: rank of input array must be equal to %i, but got %i instead !", rank,
               shape::rank(inputShapeInfo));
  REQUIRE_TRUE(shape::rank(weightsShapeInfo) == rank, 0,
               "CUSTOM DECONV3D OP: rank of weights array must be equal to %i, but got %i instead !", rank,
               shape::rank(weightsShapeInfo));

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(0)));  // filter(kernel) depth
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(1)));  // filter(kernel) height
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(2)));  // filter(kernel) width
  LongType sD = INT_ARG(3);                                                                          // strides depth
  LongType sH = INT_ARG(4);                                                                          // strides height
  LongType sW = INT_ARG(5);                                                                          // strides width
  LongType pD = INT_ARG(6);                                                                          // paddings depth
  LongType pH = INT_ARG(7);                                                                          // paddings height
  LongType pW = INT_ARG(8);                                                                          // paddings width
  LongType dD = INT_ARG(9);                                                                          // dilations depth
  LongType dH = INT_ARG(10);                                                                         // dilations height
  LongType dW = INT_ARG(11);                                                                         // dilations width
  int isSameMode = INT_ARG(12);                                                                 // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;  // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]

  LongType indIOioC, indIiD, indWoC(0 == wFormat ? 3 : (1 == wFormat ? 1 : 4));
  if (!isNCDHW) {
    indIOioC = 4;
    indIiD = 1;
  } else {
    indIOioC = 1;
    indIiD = 2;
  }

  const LongType bS = inputShapeInfo[1];             // batch size
  const LongType iD = inputShapeInfo[indIiD + 1];    // input depth
  const LongType iH = inputShapeInfo[indIiD + 2];    // input height
  const LongType iW = inputShapeInfo[indIiD + 3];    // input width
  const LongType iC = inputShapeInfo[indIOioC + 1];  // input channels
  const LongType oC = weightsShapeInfo[indWoC + 1];  // output channels

  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
  REQUIRE_TRUE(shape::shapeEquals(5, expectedWeightsShape.data(), shape::rank(weightsShapeInfo),
                                  shape::shapeOf(weightsShapeInfo)),
               0, "CUSTOM DECONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
  if (biasShapeInfo)
    REQUIRE_TRUE(shape::rank(biasShapeInfo) <= 2 && oC == shape::length(biasShapeInfo), 0,
                 "CUSTOM DECONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i "
                 "instead !",
                 oC, shape::rank(biasShapeInfo), shape::length(biasShapeInfo));

  LongType oD, oH, oW;  // output depth, height, width
  ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW,
                                        isSameMode);



 std::initializer_list<sd::LongType> outputShape;

  if (isNCDHW) {
    outputShape = {bS,oC,oD,oH,oW};
  } else {
    outputShape = {bS,oD,oH,oW,oC};
  }

  ShapeDescriptor *shapeDescriptor = new ShapeDescriptor(ArrayOptions::dataType(inputShapeInfo), shape::order(inputShapeInfo),
                                                         outputShape);
  auto outputShapeInfo = ConstantShapeHelper::getInstance().createShapeInfo(shapeDescriptor);
  delete shapeDescriptor;
  return SHAPELIST(outputShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(deconv3d_bp, 3, 2, false, 0, 13) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weights = INPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;  // [oC]
  auto gradO = block.width() > 3
                   ? INPUT_VARIABLE(3)
                   : INPUT_VARIABLE(2);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), gradI
  auto gradW = OUTPUT_VARIABLE(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;  // [oC]

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "CUSTOM DECONV3D_BP OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 5, 0,
               "CUSTOM DECONV3D_BP OP: rank of weights array must be equal to 5 , but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(
      gradO->rankOf() == 5, 0,
      "CUSTOM DECONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !",
      gradO->rankOf());

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) depth
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) height
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(weights->sizeAt(2));  // filter(kernel) width
  LongType sD = INT_ARG(3);                                                          // strides depth
  LongType sH = INT_ARG(4);                                                          // strides height
  LongType sW = INT_ARG(5);                                                          // strides width
  LongType pD = INT_ARG(6);                                                          // paddings depth
  LongType pH = INT_ARG(7);                                                          // paddings height
  LongType pW = INT_ARG(8);                                                          // paddings width
  LongType dD = INT_ARG(9);                                                          // dilations depth
  LongType dH = INT_ARG(10);                                                         // dilations height
  LongType dW = INT_ARG(11);                                                         // dilations width
  int isSameMode = INT_ARG(12);                                                 // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;          // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                             indIOioC, indIOioD, indWoC, indWiC, indWkD);

  LongType trueoD, trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizeDeconv3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH,
                                        iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx(
      {bS, oC, trueoD, trueoH, trueoW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "CUSTOM DECONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got "
               "%s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
  if (bias)
    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                 "CUSTOM DECONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
                 "%i instead !",
                 oC, bias->rankOf(), bias->lengthOf());

  if (isSameMode)  // Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not
                   // deconv) forward pass
    ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  // ----- calculation of gradI -> pass it through conv3d_ff ----- //
  sd::ops::conv3dnew conv3d;
  const sd::Status status =
      conv3d.execute({gradO, weights}, {gradI}, {},
                     {kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isSameMode, !isNCDHW, wFormat}, {});
  if (status != sd::Status::OK) return status;

  // -----prepare permutation arrays and axes for dot product ----- //
  std::vector<LongType> inputAxesForDot;

  if (!isNCDHW) {
    gradO = new NDArray(gradO->permute({0, 4, 1, 2, 3}));  // [bS, oD, oH, oW, oC] -> [bS, oC, oD, oH, oW]
    inputAxesForDot = {0, 1, 2, 3};                        // bS, iD, iH, iW
  } else
    inputAxesForDot = {0, 2, 3, 4};  // bS, iD, iH, iW

  std::vector<LongType> gradWAxes;  // empty for wFormat = 1
  if (0 == wFormat)
    gradWAxes = {4, 3, 0, 1, 2};
  else if (2 == wFormat)
    gradWAxes = {0, 4, 1, 2, 3};

  // ----- calculation of gradW ----- //
  auto columns = NDArrayFactory::create(input->ordering(), {bS, oC, kD, kH, kW, iD, iH, iW}, input->dataType(),
                                        block.launchContext());
  ConvolutionUtils::vol2col(block, *gradO, columns, sD, sH, sW, pD, pH, pW, dD, dH,
                            dW);  // [bS, oC, oD, oH, oW] is deconvoluted to [bS, oC, kD, kH, kW, iD, iH, iW]
  MmulHelper::tensorDot(input, &columns, gradW, inputAxesForDot, {0, 5, 6, 7},
                        gradWAxes);  // [bS, iC, iD, iH, iW]/[bS, iD, iH, iW, iC] x [bS, oC, kD, kH, kW, iD, iH, iW] =
                                     // [iC, oC, kD, kH, kW]

  // ----- calculation of gradB ----- //
  if (gradB) {
    if (gradB->rankOf() == 2) gradB = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}, false));
    gradO->reduceAlongDimension(reduce::Sum, *gradB, {0, 2, 3, 4});  // sum over bS, oD, oH, oW
    if (gradB != OUTPUT_VARIABLE(2)) delete gradB;
  }

  if (!isNCDHW) delete gradO;

  return sd::Status::OK;
}

DECLARE_TYPES(deconv3d_bp) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::ANY)
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(deconv3d_bp) {
  auto inputShapeInfo = inputShape->at(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto weightsShapeInfo = inputShape->at(1);  // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
  auto biasShapeInfo = block.width() > 3 ? inputShape->at(2) : nullptr;  // [oC]
  auto gradOShapeInfo =
      block.width() > 3
          ? inputShape->at(3)
          : inputShape->at(2);  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

  const int rank = 5;
  REQUIRE_TRUE(shape::rank(inputShapeInfo) == rank, 0,
               "CUSTOM DECONV3D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank,
               shape::rank(inputShapeInfo));
  REQUIRE_TRUE(shape::rank(weightsShapeInfo) == rank, 0,
               "CUSTOM DECONV3D_BP OP: rank of weights array must be equal to %i , but got %i instead !", rank,
               shape::rank(weightsShapeInfo));
  REQUIRE_TRUE(
      shape::rank(gradOShapeInfo) == rank, 0,
      "CUSTOM DECONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !",
      rank, shape::rank(gradOShapeInfo));

  LongType kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(0)));  // filter(kernel) depth
  LongType kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(1)));  // filter(kernel) height
  LongType kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<LongType>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(2)));  // filter(kernel) width
  LongType sD = INT_ARG(3);                                                                          // strides depth
  LongType sH = INT_ARG(4);                                                                          // strides height
  LongType sW = INT_ARG(5);                                                                          // strides width
  LongType pD = INT_ARG(6);                                                                          // paddings depth
  LongType pH = INT_ARG(7);                                                                          // paddings height
  LongType pW = INT_ARG(8);                                                                          // paddings width
  LongType dD = INT_ARG(9);                                                                          // dilations depth
  LongType dH = INT_ARG(10);                                                                         // dilations height
  LongType dW = INT_ARG(11);                                                                         // dilations width
  int isSameMode = INT_ARG(12);                                                                 // 0-SAME,  1-VALID
  int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;  // INT_ARG(13): 1-NDHWC, 0-NCDHW
  int wFormat = block.getIArguments()->size() > 14
                    ? INT_ARG(14)
                    : 0;  // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]

  LongType indIOioC, indIiD, indWoC(0 == wFormat ? 3 : (1 == wFormat ? 1 : 4));
  if (!isNCDHW) {
    indIOioC = 4;
    indIiD = 1;
  } else {
    indIOioC = 1;
    indIiD = 2;
  }

  const LongType bS = inputShapeInfo[1];             // batch size
  const LongType iD = inputShapeInfo[indIiD + 1];    // input depth
  const LongType iH = inputShapeInfo[indIiD + 2];    // input height
  const LongType iW = inputShapeInfo[indIiD + 3];    // input width
  const LongType iC = inputShapeInfo[indIOioC + 1];  // input channels
  const LongType oC = weightsShapeInfo[indWoC + 1];  // output channels

  LongType trueoD, trueoH, trueoW;  // true output depth, height, width
  ConvolutionUtils::calcOutSizeDeconv3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH,
                                        iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx(
      {bS, oC, trueoD, trueoH, trueoW, 0, indIOioC, indIiD, indIiD + 1, indIiD + 2});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
  REQUIRE_TRUE(
      shape::shapeEquals(5, expectedGradOShape.data(), shape::rank(gradOShapeInfo), shape::shapeOf(gradOShapeInfo)), 0,
      "CUSTOM DECONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead "
      "!",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
  REQUIRE_TRUE(shape::shapeEquals(5, expectedWeightsShape.data(), shape::rank(weightsShapeInfo),
                                  shape::shapeOf(weightsShapeInfo)),
               0, "CUSTOM DECONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
  if (biasShapeInfo)
    REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0,
                 "CUSTOM DECONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, "
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
