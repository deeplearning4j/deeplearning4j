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
// @author saudet
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/helpers/convolutions.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

using namespace dnnl;
using namespace samediff;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same
  // mode;

  const sd::LongType kH = INT_ARG(0);
  const sd::LongType kW = INT_ARG(1);
  const sd::LongType sH = INT_ARG(2);
  const sd::LongType sW = INT_ARG(3);
  sd::LongType pH = INT_ARG(4);
  sd::LongType pW = INT_ARG(5);
  const sd::LongType dH = INT_ARG(6);
  const sd::LongType dW = INT_ARG(7);
  const auto paddingMode = INT_ARG(8);
  const auto extraParam0 = INT_ARG(9);
  const int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 0-NCHW, 1-NHWC

  REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D MKLDNN op: input should have rank of 4, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D MKLDNN op: dilation must not be zero, but got instead {%i, %i}", dH,
               dW);

  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  if (paddingMode) ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  auto mode = (extraParam0 == 0) ? algorithm::pooling_avg_exclude_padding : algorithm::pooling_avg_include_padding;

  onednnUtils::poolingONEDNN(input, output, 0, kH, kW, 0, sH, sW, 0, pH, pW, isNCHW, mode);

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool2d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  Requirements req("ONEDNN AVGPOOL2d OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(sd::ONEDNNStream::isSupported({input, output}), ONEDNN_STREAM_NOT_SUPPORTED);
  if (req) onednnUtils::checkPoolingONEDNN(req, block, input, output);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool2d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);   // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
  auto gradO = INPUT_VARIABLE(1);   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

  sd::LongType kH = INT_ARG(0);           // filter(kernel) height
  sd::LongType kW = INT_ARG(1);           // filter(kernel) width
  sd::LongType sH = INT_ARG(2);           // strides height
  sd::LongType sW = INT_ARG(3);           // strides width
  sd::LongType pH = INT_ARG(4);           // paddings height
  sd::LongType pW = INT_ARG(5);           // paddings width
  sd::LongType dH = INT_ARG(6);           // dilations height
  sd::LongType dW = INT_ARG(7);           // dilations width
  int paddingMode = INT_ARG(8);  // 0-VALID, 1-SAME
  int extraParam0 = INT_ARG(9);
  int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 0-NCHW, 1-NHWC

  REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D_BP MKLDNN op: input should have rank of 4, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D_BP MKLDNN op: dilation must not be zero, but got instead {%i, %i}", dH,
               dW);

  sd::LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  sd::LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oH, oW, 0, indIOioC, indIiH, indIiH + 1});
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "AVGPOOL2D_BP MKLDNN op: wrong shape of output's gradients array (next epsilon), expected is %s, but "
               "got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

  auto mode = (extraParam0 == 0) ? algorithm::pooling_avg_exclude_padding : algorithm::pooling_avg_include_padding;

  onednnUtils::poolingBpONEDNN(input, gradO, gradI, 0, kH, kW, 0, sH, sW, 0, pH, pW, isNCHW, mode);

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool2d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  Requirements req("ONEDNN AVGPOOL2d_BP OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(sd::ONEDNNStream::isSupported({input, output}), ONEDNN_STREAM_NOT_SUPPORTED);
  if (req) onednnUtils::checkPoolingONEDNN(req, block,input, gradO);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
