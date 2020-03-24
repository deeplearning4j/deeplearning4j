/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

using namespace dnnl;

namespace sd      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(maxpool3dnew, ENGINE_CPU) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)

    int kD = INT_ARG(0);                                                        // filter(kernel) depth
    int kH = INT_ARG(1);                                                        // filter(kernel) height
    int kW = INT_ARG(2);                                                        // filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int paddingMode = INT_ARG(12);                                               // 1-SAME,  0-VALID
    // int extraParam0 = INT_ARG(13);                                           // unnecessary for max case, required only for avg and pnorm cases
    int isNCDHW  = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 1-NDHWC, 0-NCDHW

    REQUIRE_TRUE(input->rankOf() == 5, 0, "MAXPOOL3DNEW MKLDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0, "MAXPOOL3DNEW MKLDNN op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    if(paddingMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    mkldnnUtils::poolingMKLDNN(input, output, kD,kH,kW, sD,sH,sW, pD,pH,pW, isNCDHW, algorithm::pooling_max);

    return Status::OK();

}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(maxpool3dnew, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    return block.isUseMKLDNN() && sd::MKLDNNStream::isSupported({input, output});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(maxpool3dnew_bp, ENGINE_CPU) {

    auto input = INPUT_VARIABLE(0);                          // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto gradO = INPUT_VARIABLE(1);                          // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
    auto gradI = OUTPUT_VARIABLE(0);                         // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon

    const int kD = INT_ARG(0);                                                  // filter(kernel) depth
    const int kH = INT_ARG(1);                                                  // filter(kernel) height
    const int kW = INT_ARG(2);                                                  // filter(kernel) width
    const int sD = INT_ARG(3);                                                  // strides depth
    const int sH = INT_ARG(4);                                                  // strides height
    const int sW = INT_ARG(5);                                                  // strides width
          int pD = INT_ARG(6);                                                  // paddings depth
          int pH = INT_ARG(7);                                                  // paddings height
          int pW = INT_ARG(8);                                                  // paddings width
    const int dD = INT_ARG(9);                                                  // dilations depth
    const int dH = INT_ARG(10);                                                 // dilations height
    const int dW = INT_ARG(11);                                                 // dilations width
    const int paddngMode = INT_ARG(12);                                         // 1-SAME,  0-VALID
    // int extraParam0 = INT_ARG(13);                                           // unnecessary for max case, required only for avg and pnorm cases
    int isNCDHW  = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 1-NDHWC, 0-NCDHW

    REQUIRE_TRUE(input->rankOf() == 5, 0, "MAXPOOL3DNEW_BP MKLDNN op: input should have rank of 5, but got %i instead", input->rankOf());
    REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0, "MAXPOOL3DNEW_BP MKLDNN op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;               // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::vector<Nd4jLong> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx({bS,iC,oD,oH,oW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2});
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0, "MAXPOOL3DNEW_BP MKLDNN op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());

    if(paddngMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    mkldnnUtils::poolingBpMKLDNN(input, gradO, gradI, kD,kH,kW, sD,sH,sW, pD,pH,pW, isNCDHW, algorithm::pooling_max);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(maxpool3dnew_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    return block.isUseMKLDNN() && sd::MKLDNNStream::isSupported({input, output});
}

}
}
}