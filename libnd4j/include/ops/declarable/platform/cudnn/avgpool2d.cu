/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include "cudnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace nd4j      {
namespace ops       {
namespace platforms {


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool2d, ENGINE_CUDA) {

    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;
    const auto kH = INT_ARG(0);
    const auto kW = INT_ARG(1);
    const auto sH = INT_ARG(2);
    const auto sW = INT_ARG(3);
          auto pH = INT_ARG(4);
          auto pW = INT_ARG(5);
    const auto dH = INT_ARG(6);
    const auto dW = INT_ARG(7);
    const auto paddingMode = static_cast<bool>(INT_ARG(8));
    const auto extraParam0 = INT_ARG(9);
    const int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;       // INT_ARG(10): 0-NCHW, 1-NHWC

    REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D CUDNN op: input should have rank of 4, but got %i instead", input->rankOf());
    REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D CUDNN op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

    int oH = 0;
    int oW = 0;

    const int iH = static_cast<int>(isNCHW ? input->sizeAt(2) : input->sizeAt(1));
    const int iW = static_cast<int>(isNCHW ? input->sizeAt(3) : input->sizeAt(2));

    ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

    if (paddingMode)
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    const cudnnPoolingMode_t mode = (extraParam0 == 0) ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    pooling2dCUDNN(block.launchContext(), input, output, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW, mode);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(avgpool2d, ENGINE_CUDA) {

    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    const auto goodType  = input->dataType() == DataType::DOUBLE || input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::HALF || input->dataType() == DataType::INT32;

    return goodType && input->dataType() == output->dataType();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(avgpool2d_bp, ENGINE_CUDA) {

    auto input = INPUT_VARIABLE(0);                          // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto gradO = INPUT_VARIABLE(1);                          // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    auto gradI = OUTPUT_VARIABLE(0);                         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

    const auto  kH = INT_ARG(0);                                                        // filter(kernel) height
    const auto  kW = INT_ARG(1);                                                        // filter(kernel) width
    const auto  sH = INT_ARG(2);                                                        // strides height
    const auto  sW = INT_ARG(3);                                                        // strides width
          auto  pH = INT_ARG(4);                                                        // paddings height
          auto  pW = INT_ARG(5);                                                        // paddings width
    const auto  dH = INT_ARG(6);                                                        // dilations height
    const auto  dW = INT_ARG(7);                                                        // dilations width
    const auto  paddingMode = INT_ARG(8);                                               // 0-VALID, 1-SAME
    const auto  extraParam0 = INT_ARG(9);
    const auto  isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;         // INT_ARG(10): 0-NCHW, 1-NHWC

    REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D_BP CUDNN op: input should have rank of 4, but got %i instead", input->rankOf());
    REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D_BP CUDNN op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::vector<Nd4jLong>  expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx({bS,iC,oH,oW,  0,indIOioC,indIiH,indIiH+1});
    std::vector<Nd4jLong>  expectedGradIShape = ShapeUtils::composeShapeUsingDimsAndIdx({bS,iC,iH,iW,  0,indIOioC,indIiH,indIiH+1});
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0, "AVGPOOL2D_BP CUDNN op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(gradI->isSameShape(expectedGradIShape), 0, "AVGPOOL2D_BP CUDNN op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradIShape).c_str(), ShapeUtils::shapeAsString(gradI).c_str());

    if(paddingMode)                       // SAME
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    const cudnnPoolingMode_t mode = (extraParam0 == 0) ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;

    pooling2dBpCUDNN(block.launchContext(), input, gradO, gradI, kH, kW, sH, sW, pH, pW, dH, dW, isNCHW, mode);

    return Status::OK();
}

PLATFORM_CHECK(avgpool2d_bp, ENGINE_CUDA) {

    auto input = INPUT_VARIABLE(0);                          // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto gradO = INPUT_VARIABLE(1);                          // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    auto gradI = OUTPUT_VARIABLE(0);                         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

    const auto goodType = input->dataType() == DataType::DOUBLE || input->dataType() == DataType::FLOAT32 || input->dataType() == DataType::HALF || input->dataType() == DataType::INT32;

    return goodType && (input->dataType() == gradO->dataType())
                    && (input->dataType() == gradI->dataType())
                    && shape::haveSameShapeAndStrides(input->getShapeInfo(), gradI->getShapeInfo());
}


}
}
}
