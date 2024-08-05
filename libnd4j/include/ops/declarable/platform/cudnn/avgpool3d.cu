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
PLATFORM_IMPL(avgpool3dnew, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto output = OUTPUT_VARIABLE(0);  // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)

  LongType kD = INT_ARG(0);            // filter(kernel) depth
  LongType kH = INT_ARG(1);            // filter(kernel) height
  LongType kW = INT_ARG(2);            // filter(kernel) width
  LongType sD = INT_ARG(3);            // strides depth
  LongType sH = INT_ARG(4);            // strides height
  LongType sW = INT_ARG(5);            // strides width
  LongType pD = INT_ARG(6);            // paddings depth
  LongType pH = INT_ARG(7);            // paddings height
  LongType pW = INT_ARG(8);            // paddings width
  LongType dD = INT_ARG(9);            // dilations depth
  LongType dH = INT_ARG(10);           // dilations height
  LongType dW = INT_ARG(11);           // dilations width
  int paddingMode = INT_ARG(12);  // 1-SAME,  0-VALID
  int extraParam0 = INT_ARG(13);
  int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 0-NCDHW, 1-NDHWC

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "AVGPOOL3DNEW CUDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
               "AVGPOOL3DNEW CUDNN OP: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  std::vector<LongType> expectedOutputShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oD, oH, oW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  REQUIRE_TRUE(output->isSameShape(expectedOutputShape), 0,
               "AVGPOOL3DNEW CUDNN OP: wrong shape of output array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expectedOutputShape).c_str(), ShapeUtils::shapeAsString(output).c_str());

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  const cudnnPoolingMode_t mode =
      (extraParam0 == 0) ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

  pooling3dCUDNN(block.launchContext(), input, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isNCDHW, mode);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool3dnew, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  Requirements req("CUDNN AVGPOOL3d OP");
  req.expectEq(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT),
               makeInfoVariable(output->dataType(), TYPE_MSG_OUTPUT)) &&
      req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT),
                   {INT32, HALF, FLOAT32, DOUBLE});
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool3dnew_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);   // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto gradO = INPUT_VARIABLE(1);   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon

  const LongType kD = INT_ARG(0);            // filter(kernel) depth
  const LongType kH = INT_ARG(1);            // filter(kernel) height
  const LongType kW = INT_ARG(2);            // filter(kernel) width
  const LongType sD = INT_ARG(3);            // strides depth
  const LongType sH = INT_ARG(4);            // strides height
  const LongType sW = INT_ARG(5);            // strides width
  LongType pD = INT_ARG(6);                  // paddings depth
  LongType pH = INT_ARG(7);                  // paddings height
  LongType pW = INT_ARG(8);                  // paddings width
  const LongType dD = INT_ARG(9);            // dilations depth
  const LongType dH = INT_ARG(10);           // dilations height
  const LongType dW = INT_ARG(11);           // dilations width
  const int isSameMode = INT_ARG(12);   // 1-SAME,  0-VALID
  const int extraParam0 = INT_ARG(13);  // define what divisor to use while averaging
  const int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 0-NCDHW, 1-NDHWC

  REQUIRE_TRUE(input->rankOf() == 5, 0, "AVGPOOL3DNEW_BP CUDNN OP: input should have rank of 5, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
               "AVGPOOL3DNEW_BP CUDNN OP: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  std::vector<LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oD, oH, oW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  std::vector<LongType> expectedGradIShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, iD, iH, iW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "AVGPOOL3DNEW_BP CUDNN: wrong shape of output's gradients array (next epsilon), expected is %s, but got "
               "%s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
  REQUIRE_TRUE(
      gradI->isSameShape(expectedGradIShape), 0,
      "AVGPOOL3DNEW_BP CUDNN: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !",
      ShapeUtils::shapeAsString(expectedGradIShape).c_str(), ShapeUtils::shapeAsString(gradI).c_str());

  if (isSameMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  const cudnnPoolingMode_t mode =
      (extraParam0 == 0) ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

  pooling3dBpCUDNN(block.launchContext(), input, gradO, gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isNCDHW,
                   mode);

  return Status::OK;
}

PLATFORM_CHECK(avgpool3dnew_bp, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);   // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto gradO = INPUT_VARIABLE(1);   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon

  Requirements req("CUDNN AVGPOOL3d_BP OP");
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
