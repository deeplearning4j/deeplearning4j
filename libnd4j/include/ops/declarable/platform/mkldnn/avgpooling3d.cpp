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

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool3dnew, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto output = OUTPUT_VARIABLE(0);  // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)

  sd::LongType kD = INT_ARG(0);            // filter(kernel) depth
  sd::LongType kH = INT_ARG(1);            // filter(kernel) height
  sd::LongType kW = INT_ARG(2);            // filter(kernel) width
  sd::LongType sD = INT_ARG(3);            // strides depth
  sd::LongType sH = INT_ARG(4);            // strides height
  sd::LongType sW = INT_ARG(5);            // strides width
  sd::LongType pD = INT_ARG(6);            // paddings depth
  sd::LongType pH = INT_ARG(7);            // paddings height
  sd::LongType pW = INT_ARG(8);            // paddings width
  sd::LongType dD = INT_ARG(9);            // dilations depth
  sd::LongType dH = INT_ARG(10);           // dilations height
  sd::LongType dW = INT_ARG(11);           // dilations width
  int paddingMode = INT_ARG(12);  // 1-SAME,  0-VALID
  int extraParam0 = INT_ARG(13);
  int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 0-NCDHW, 1-NDHWC

  REQUIRE_TRUE(input->rankOf() == 5, 0,
               "AVGPOOL3DNEW MKLDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
  REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
               "AVGPOOL3DNEW MKLDNN OP: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  auto mode = (extraParam0 == 0) ? algorithm::pooling_avg_exclude_padding : algorithm::pooling_avg_include_padding;

  onednnUtils::poolingONEDNN(input, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, isNCDHW, mode);

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool3dnew, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  Requirements req("ONEDNN AVGPOOL3d OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(sd::ONEDNNStream::isSupported({input, output}), ONEDNN_STREAM_NOT_SUPPORTED);
  if (req) onednnUtils::checkPoolingONEDNN(req, block, input, output);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool3dnew_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);   // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
  auto gradO = INPUT_VARIABLE(1);   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
  auto gradI = OUTPUT_VARIABLE(0);  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon

  const sd::LongType kD = INT_ARG(0);            // filter(kernel) depth
  const sd::LongType kH = INT_ARG(1);            // filter(kernel) height
  const sd::LongType kW = INT_ARG(2);            // filter(kernel) width
  const sd::LongType sD = INT_ARG(3);            // strides depth
  const sd::LongType sH = INT_ARG(4);            // strides height
  const sd::LongType sW = INT_ARG(5);            // strides width
  sd::LongType pD = INT_ARG(6);                  // paddings depth
  sd::LongType pH = INT_ARG(7);                  // paddings height
  sd::LongType pW = INT_ARG(8);                  // paddings width
  const sd::LongType dD = INT_ARG(9);            // dilations depth
  const sd::LongType dH = INT_ARG(10);           // dilations height
  const sd::LongType dW = INT_ARG(11);           // dilations width
  const int paddingMode = INT_ARG(12);  // 1-SAME,  0-VALID
  const int extraParam0 = INT_ARG(13);  // define what divisor to use while averaging
  const int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 0-NCDHW, 1-NDHWC

  REQUIRE_TRUE(input->rankOf() == 5, 0, "AVGPOOL3DNEW_BP MKLDNN op: input should have rank of 5, but got %i instead",
               input->rankOf());
  REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
               "AVGPOOL3DNEW_BP MKLDNN op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

  sd::LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  sd::LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  std::vector<sd::LongType> expectedGradOShape =
      ShapeUtils::composeShapeUsingDimsAndIdx({bS, iC, oD, oH, oW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
  REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,
               "AVGPOOL3DNEW_BP MKLDNN op: wrong shape of output's gradients array (next epsilon), expected is %s, but "
               "got %s instead !",
               ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());

  if (paddingMode)  // SAME
    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

  auto mode = (extraParam0 == 0) ? algorithm::pooling_avg_exclude_padding : algorithm::pooling_avg_include_padding;

  onednnUtils::poolingBpONEDNN(input, gradO, gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, isNCDHW, mode);

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool3dnew_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);
  Requirements req("ONEDNN AVGPOOL3d_BP OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(sd::ONEDNNStream::isSupported({input, output}), ONEDNN_STREAM_NOT_SUPPORTED);
  if (req) onednnUtils::checkPoolingONEDNN(req, block, input, gradO);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
