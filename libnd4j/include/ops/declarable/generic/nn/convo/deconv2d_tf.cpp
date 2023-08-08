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

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(deconv2d_tf, 3, 1, false, 0, 9) {
  auto gradO = INPUT_VARIABLE(2);       // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  auto weights = INPUT_VARIABLE(1);     // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradIShape = INPUT_VARIABLE(0);  // [4] - shape of input of conv2d (that is shape of gradI)

  auto gradI = OUTPUT_NULLIFIED(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

  LongType kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<LongType>(weights->sizeAt(0));  // filter(kernel) height
  LongType kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<LongType>(weights->sizeAt(1));  // filter(kernel) width
  LongType sH = INT_ARG(2);                                                          // strides height
  LongType sW = INT_ARG(3);                                                          // strides width
  LongType pH = INT_ARG(4);                                                          // paddings height
  LongType pW = INT_ARG(5);                                                          // paddings width
  LongType dH = INT_ARG(6);                                                          // dilations height
  LongType dW = INT_ARG(7);                                                          // dilations width
  int isSameMode = INT_ARG(8);                                                  // 0-VALID, 1-SAME
  int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;             // INT_ARG(9): 1-NHWC, 0-NCHW
  int wFormat = block.getIArguments()->size() > 10
                    ? INT_ARG(10)
                    : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  const LongType rank = gradO->rankOf();

  REQUIRE_TRUE(weights->rankOf() == rank, 0,
               "CUSTOM DECONV2D_TF OP: rank of weights array must be equal to 4, but got %i instead !",
               weights->rankOf());
  REQUIRE_TRUE(gradIShape->rankOf() == 1, 0,
               "CUSTOM DECONV2D_TF OP: rank of array with output shape must be equal to 1, but got %i instead !",
               gradIShape->rankOf());
  REQUIRE_TRUE(gradIShape->lengthOf() == rank, 0,
               "CUSTOM DECONV2D_TF OP: length of array with output shape must be equal to 4, but got %i instead !",
               gradIShape->lengthOf());

  // create empty conv2d input array
  NDArray *input = new NDArray(gradO->ordering(), gradIShape->asVectorT<sd::LongType>(), gradO->dataType(), block.launchContext());

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                             indIiH, indWiC, indWoC, indWkH, indOoH);

  LongType trueoH, trueoW;  // true output height, width
  ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, oC, trueoH, trueoW, 0, indIOioC, indOoH, indOoH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "CUSTOM DECONV2D_TF OP: wrong shape of input array, basing on array with output shape expected is %s, "
               "but got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0,
               "CUSTOM DECONV2D_TF OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());

  ConvolutionUtils::conv2dBP(block, input, weights, nullptr, gradO, gradI, nullptr, nullptr, kH, kW, sH, sW, pH, pW,
                             dH, dW, isSameMode, isNCHW, wFormat);


  delete input;
  return sd::Status::OK;
}

DECLARE_TYPES(deconv2d_tf) {
  getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(deconv2d_tf) {
  auto gradOShapeInfo = inputShape->at(2);       // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  auto weightsShapeInfo = inputShape->at(1);     // [kH, kW, iC, oC], [oC, iC, kH, kW], [oC, kH, kW, iC]
  auto gradIShapeShapeInfo = inputShape->at(0);  // [4]

  const int rank = 4;

  REQUIRE_TRUE(shape::rank(weightsShapeInfo) == rank, 0,
               "CUSTOM DECONV2D_TF OP: rank of weights array must be equal to %i, but got %i instead !", rank,
               shape::rank(weightsShapeInfo));
  REQUIRE_TRUE(shape::rank(gradOShapeInfo) == rank, 0,
               "CUSTOM DECONV2D_TF OP: rank of input array must be equal to %i, but got %i instead !", rank,
               shape::rank(gradOShapeInfo));
  REQUIRE_TRUE(shape::rank(gradIShapeShapeInfo) == 1, 0,
               "CUSTOM DECONV2D_TF OP: rank of array with output shape must be equal to %i, but got %i instead !", 1,
               shape::rank(gradIShapeShapeInfo));

  const LongType kH =
      INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(0)));  // filter(kernel) height
  const LongType kW =
      INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(shape::sizeAt(weightsShapeInfo, static_cast<sd::LongType>(1)));  // filter(kernel) width
  const LongType sH = INT_ARG(2);                                                               // strides height
  const LongType sW = INT_ARG(3);                                                               // strides width
  const LongType pH = INT_ARG(4);                                                               // paddings height
  const LongType pW = INT_ARG(5);                                                               // paddings width
  const LongType dH = INT_ARG(6);                                                               // dilations height
  const LongType dW = INT_ARG(7);                                                               // dilations width
  const int isSameMode = INT_ARG(8);                                                       // 0-VALID, 1-SAME
  const int isNCHW = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;                  // INT_ARG(9): 1-NHWC, 0-NCHW
  const int wFormat = block.getIArguments()->size() > 10
                          ? INT_ARG(10)
                          : 0;  // 0 - [kH, kW, iC, oC], 1 - [oC, iC, kH, kW], 2 - [oC, kH, kW, iC]

  LongType indIOioC, indIiH, indWoC(0 == wFormat ? 3 : 0), indOoH;
  if (!isNCHW) {
    indIOioC = 3;
    indIiH = 1;
    indOoH = 1;
  } else {
    indIOioC = 1;
    indIiH = 2;
    indOoH = 2;
  }

  std::vector<sd::LongType> gradIShape = INPUT_VARIABLE(0)->template asVectorT<sd::LongType>();

  const LongType bS = gradIShape[0];                 // batch size
  const LongType iH = gradIShape[indIiH];            // input height
  const LongType iW = gradIShape[indIiH + 1];        // input width
  const LongType iC = gradIShape[indIOioC];          // input channels
  const LongType oC = weightsShapeInfo[indWoC + 1];  // output channels
  const LongType oH = gradOShapeInfo[indOoH + 1];    // input height
  const LongType oW = gradOShapeInfo[indOoH + 2];    // input width

  LongType trueiH, trueiW;  // output height, width
  ConvolutionUtils::calcOutSizeDeconv2D(trueiH, trueiW, kH, kW, sH, sW, pH, pW, dH, dW, oH, oW, isSameMode);

  std::vector<sd::LongType> expectedGradIShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, trueiH, trueiW, 0, indIOioC, indIiH, indIiH + 1});
  std::vector<sd::LongType> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kH, kW, iC, oC);
  REQUIRE_TRUE(expectedGradIShape == gradIShape, 0,
               "CUSTOM DECONV2D_TF OP: wrong shape of array with output shape, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedGradIShape).c_str(), ShapeUtils::shapeAsString(gradIShape).c_str());
  REQUIRE_TRUE(shape::shapeEquals(4, expectedWeightsShape.data(), shape::rank(weightsShapeInfo),
                                  shape::shapeOf(weightsShapeInfo)),
               0, "CUSTOM DECONV2D_TF OP: wrong shape of weights array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedWeightsShape).c_str(),
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str());

  sd::LongType shape[4];
  shape[0] = bS;

  if (isNCHW) {
    shape[1] = iC;
    shape[2] = iH;
    shape[3] = iW;
  } else {
    shape[1] = iH;
    shape[2] = iW;
    shape[3] = iC;
  }

  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(weightsShapeInfo),
                                                                      shape::order(gradOShapeInfo), 4, shape));
}

}  // namespace ops
}  // namespace sd

#endif
