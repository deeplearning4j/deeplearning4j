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
// @author raver119@gmail.com, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com), changed on 14.05.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_avgpool2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(avgpool2d, 1, 1, false, 0, 10) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_NULLIFIED(0);

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same
  // mode;

  const LongType kH = INT_ARG(0);
  const LongType kW = INT_ARG(1);
  const LongType sH = INT_ARG(2);
  const LongType sW = INT_ARG(3);
  LongType pH = INT_ARG(4);
  LongType pW = INT_ARG(5);
  const LongType dH = INT_ARG(6);
  const LongType dW = INT_ARG(7);
  const auto isSameMode = static_cast<bool>(INT_ARG(8));
  const auto extraParam0 = INT_ARG(9);
  const int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 0-NCHW, 1-NHWC

  REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D op: input should have rank of 4, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

  LongType oH = 0;
  LongType oW = 0;

  const LongType iH = static_cast<LongType>(isNCHW ? input->sizeAt(2) : input->sizeAt(1));
  const LongType iW = static_cast<LongType>(isNCHW ? input->sizeAt(3) : input->sizeAt(2));

  if (!isNCHW) {
    input = new NDArray(input->permute({0, 3, 1, 2}, false));    // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    output = new NDArray(output->permute({0, 3, 1, 2}, false));  // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
  }

  ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  if (isSameMode) ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 -
  // poolingMode; 9 - divisor;
  ConvolutionUtils::pooling2d(block, *input, *output, kH, kW, sH, sW, pH, pW, dH, dW, AVG_POOL,
                              extraParam0);

  if (!isNCHW) {
     delete input;
      delete output;
  }

  return Status::OK;
}

DECLARE_SYN(AvgPool2D, avgpool2d);
DECLARE_SYN(AvgPool, avgpool2d);
DECLARE_SYN(avgpool, avgpool2d);

DECLARE_TYPES(avgpool2d) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(avgpool2d) {
  auto inShape = inputShape->at(0);
  auto shapeOf = shape::shapeOf(inShape);

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same
  // mode;
  const LongType kH = INT_ARG(0);
  const LongType kW = INT_ARG(1);
  const LongType sH = INT_ARG(2);
  const LongType sW = INT_ARG(3);
  const LongType pH = INT_ARG(4);
  const LongType pW = INT_ARG(5);
  const LongType dH = INT_ARG(6);
  const LongType dW = INT_ARG(7);
  const int isSameMode = INT_ARG(8);

  const int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 0-NCHW, 1-NHWC

  REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

  const LongType bS = shapeOf[0];
  const LongType iD = isNCHW ? shapeOf[1] : shapeOf[3];
  const LongType iH = isNCHW ? shapeOf[2] : shapeOf[1];
  const LongType iW = isNCHW ? shapeOf[3] : shapeOf[2];

  const char order = shape::order(inShape);  // output order must be equal to input order

  // calculate output Height/Width
  LongType oH, oW;
  ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

  // allocate memory for new shape
  LongType *newShape = new LongType[4];
  if (isNCHW) {
    newShape[0] = bS;
    newShape[1] = iD;
    newShape[2] = oH;
    newShape[3] = oW;
  } else {
    newShape[0] = bS;
    newShape[1] = oH;
    newShape[2] = oW;
    newShape[3] = iD;
  }
  auto desc = new ShapeDescriptor(ArrayOptions::dataType(inShape), shape::order(inShape), newShape, 4);
  auto ret =  SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
  if (Environment::getInstance().isDeleteShapeInfo()) delete desc;
  delete[] newShape;
  return ret;
}

DECLARE_TYPES(avgpool2d_bp) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool2d_bp, 2, 1, false, 0, 10) {
  auto input = INPUT_VARIABLE(0);    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto gradO = INPUT_VARIABLE(1);    // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  auto gradI = OUTPUT_NULLIFIED(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

  LongType kH = INT_ARG(0);          // filter(kernel) height
  LongType kW = INT_ARG(1);          // filter(kernel) width
  LongType sH = INT_ARG(2);          // strides height
  LongType sW = INT_ARG(3);          // strides width
  LongType pH = INT_ARG(4);          // paddings height
  LongType pW = INT_ARG(5);          // paddings width
  LongType dH = INT_ARG(6);          // dilations height
  LongType dW = INT_ARG(7);          // dilations width
  int isSameMode = INT_ARG(8);  // 0-VALID, 1-SAME
  int extraParam0 = INT_ARG(9);
  int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 0-NCHW, 1-NHWC

  REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D_BP op: input should have rank of 4, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D_BP op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  std::vector<LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oH, oW, 0, indIOioC, indIiH, indIiH + 1});
  std::vector<LongType> expectedGradIShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, iH, iW, 0, indIOioC, indIiH, indIiH + 1});
  REQUIRE_TRUE(
      gradO->isSameShape(expectedGradOShape), 0,
      "AVGPOOL2D_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(
      gradI->isSameShape(expectedGradIShape), 0,
      "AVGPOOL2D_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradIShape).c_str(), ShapeUtils::shapeAsString(gradI).c_str());

  if (!isNCHW) {
    input = new NDArray(input->permute({0, 3, 1, 2}, false));  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    gradI = new NDArray(gradI->permute({0, 3, 1, 2}, false));  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
    gradO = new NDArray(gradO->permute({0, 3, 1, 2}, false));  // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
  }

  if (isSameMode)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 -
  // poolingMode; 9 - divisor;
  ConvolutionUtils::pooling2dBP(block, *input, *gradO, *gradI, kH, kW, sH, sW, pH, pW, dH, dW, 1, extraParam0);

  if (!isNCHW) {
    delete input;
    delete gradI;
    delete gradO;
  }

  return Status::OK;
}

DECLARE_SHAPE_FN(avgpool2d_bp) {
  REQUIRE_TRUE(inputShape->at(0)[0] == 4, 0, "AVGPOOL2D_BP op: input array must be 4D, but got %i instead!",
               inputShape->at(0)[0]);
  REQUIRE_TRUE(inputShape->at(1)[0] == 4, 0,
               "AVGPOOL2D_BP op: output's gradient array (next epsilon) must be 4D, but got %i instead!",
               inputShape->at(1)[0]);

  auto desc = new  ShapeDescriptor(inputShape->at(0), ArrayOptions::dataType(inputShape->at(1)));
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(desc));
}

}  // namespace ops
}  // namespace sd

#endif
