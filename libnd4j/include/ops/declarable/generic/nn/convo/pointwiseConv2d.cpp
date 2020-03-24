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
// @author Yurii Shyrma, created on 20.03.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>

namespace sd {
namespace ops  {



CUSTOM_OP_IMPL(pointwise_conv2d, 2, 1, false, 0, 0) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [1, 1, iC, oC], [oC, iC, 1, 1], [oC, 1, 1, iC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, iH, iW, oC] (NHWC) or [bS, oC, iH, iW] (NCHW)

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM POINTWISECONV2D OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM POINTWISECONV2D OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2, 0, "CUSTOM POINTWISECONV2D OP: rank of biases array must be equal <= 2, but got %i instead !", bias->rankOf());

    int kH = 1;                                                             // filter(kernel) height
    int kW = 1;                                                             // filter(kernel) width
    int sH = 1;                                                             // strides height
    int sW = 1;                                                             // strides width
    int pH = 0;                                                             // paddings height
    int pW = 0;                                                             // paddings width
    int dH = 1;                                                             // dilations height
    int dW = 1;                                                             // dilations width
    int isNCHW  = block.getIArguments()->size() > 0 ? !INT_ARG(0) : 1;      // INT_ARG(0): 0-NCHW, 1-NHWC
    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;       // 0 - [1, 1, iC, oC], 1 - [oC, iC, 1, 1], 2 - [oC, 1, 1, iC]

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, wFormat, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, 1, 1, iC, oC);
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM POINTWISECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM POINTWISECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    ConvolutionUtils::conv2d(block, input, weights, bias, output, kH,kW, sH,sW, pH,pW, dH,dW, 1/*isSameMode*/, isNCHW, wFormat);

    return Status::OK();
}

    DECLARE_TYPES(pointwise_conv2d) {
        getOpDescriptor()
                ->setAllowedInputTypes(sd::DataType::ANY)
                ->setAllowedOutputTypes({ALL_FLOATS});
    }


DECLARE_SHAPE_FN(pointwise_conv2d) {

    Nd4jLong* inputShapeInfo  = inputShape->at(0);                                   // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    Nd4jLong* weightsShapeInfo  = inputShape->at(1);                                 // [1, 1, iC, oC], [oC, iC, 1, 1], [oC, 1, 1, iC]
    Nd4jLong* biasShapeInfo = block.width() > 2 ? inputShape->at(2) : nullptr;       // [oC]

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM POINTWISECONV2D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM POINTWISECONV2D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);

    int isNCHW = block.getIArguments()->size() > 0 ? !INT_ARG(0) : 1;       // INT_ARG(0): 0-NCHW, 1-NHWC
    int wFormat = block.getIArguments()->size() > 1 ? INT_ARG(1) : 0;       // 0 - [1, 1, iC, oC], 1 - [oC, iC, 1, 1], 2 - [oC, 1, 1, iC]

    int indIOioC, indWoC(0 == wFormat ? 3 : 0);
    if(!isNCHW)
        indIOioC = 3;
    else
        indIOioC = 1;

    const int bS = inputShapeInfo[1];                            // batch size
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels
    const int oC = weightsShapeInfo[indWoC+1];                   // output channels

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, 1, 1, iC, oC);
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(weightsShapeInfo, expectedWeightsShape), 0, "POINTWISECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "POINTWISECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    auto outputShapeInfo = ShapeBuilders::copyShapeInfoAndType(inputShapeInfo, weightsShapeInfo, true, block.getWorkspace());

    // do not forget to put oC instead of iC in outputShapeInfo
    outputShapeInfo[indIOioC + 1] = oC;

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(CONSTANT(outputShapeInfo));
}


}
}
