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

#include <ops/declarable/helpers/convolutions.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

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
  const auto paddingMode = static_cast<bool>(INT_ARG(8));
  const auto extraParam0 = INT_ARG(9);
  const int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 0-NCHW, 1-NHWC

  REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D CUDNN op: input should have rank of 4, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D CUDNN op: dilation must not be zero, but got instead {%i, %i}", dH,
               dW);

  LongType oH = 0;
  LongType oW = 0;

  const LongType iH = static_cast<LongType>(isNCHW ? input->sizeAt(2) : input->sizeAt(1));
  const LongType iW = static_cast<LongType>(isNCHW ? input->sizeAt(3) : input->sizeAt(2));

  ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

  if (paddingMode) ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  const cudnnPoolingMode_t mode =
      (extraParam0 == 0) ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

  pooling2dCUDNN(block.launchContext(), input, output, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW, mode);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool2d, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN AVGPOOL2d OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT),
               makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT)) &&
      req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT),
                   {INT32, HALF, FLOAT32, DOUBLE});
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);   // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto gradO = INPUT_VARIABLE(1);   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

  const LongType kH = INT_ARG(0);           // filter(kernel) height
  const LongType kW = INT_ARG(1);           // filter(kernel) width
  const LongType sH = INT_ARG(2);           // strides height
  const LongType sW = INT_ARG(3);           // strides width
  LongType pH = INT_ARG(4);                 // paddings height
  LongType pW = INT_ARG(5);                 // paddings width
  const LongType dH = INT_ARG(6);           // dilations height
  const LongType dW = INT_ARG(7);           // dilations width
  const auto paddingMode = INT_ARG(8);  // 0-VALID, 1-SAME
  const auto extraParam0 = INT_ARG(9);
  const auto isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 0-NCHW, 1-NHWC

  REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D_BP CUDNN op: input should have rank of 4, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D_BP CUDNN op: dilation must not be zero, but got instead {%i, %i}", dH,
               dW);

  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  std::vector<LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oH, oW, 0, indIOioC, indIiH, indIiH + 1});
  std::vector<LongType> expectedGradIShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, iH, iW, 0, indIOioC, indIiH, indIiH + 1});
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "AVGPOOL2D_BP CUDNN op: wrong shape of output's gradients array (next epsilon), expected is %s, but got "
               "%s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(
      gradI->isSameShape(expectedGradIShape), 0,
      "AVGPOOL2D_BP CUDNN op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradIShape).c_str(), ShapeUtils::shapeAsString(gradI).c_str());

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  const cudnnPoolingMode_t mode =
      (extraParam0 == 0) ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

  pooling2dBpCUDNN(block.launchContext(), input, gradO, gradI, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW, mode);

  return Status::OK;
}

PLATFORM_CHECK(avgpool2d_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);   // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto gradO = INPUT_VARIABLE(1);   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

  Requirements req("CUDNN AVGPOOL2d_BP OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
               makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT),
                   makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT)) &&
      req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT),
                   {INT32, HALF, FLOAT32, DOUBLE}) &&
      req.expect(
          makeShapeInfoVariable(input, SHAPE_MSG_INPUT0), makeShapeInfoVariable(gradI, SHAPE_MSG_OUTPUT),
          [](const decltype(input)& l, const decltype(gradI)& r) {
            return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
          },
          EXPECTED_EQ_MSG);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
