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
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(pointwise_conv2d, 2, 1, false, 0, 0) {
    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [1,  1,  iC, oC] (NHWC) or [oC, iC,  1,  1] (NCHW)
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
    int isNCHW = block.getIArguments()->size() > 0 ? !INT_ARG(0) : 1;       // 1-NCHW, 0-NHWC
    
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({oC,iC,1,1,  indWoC,indWiC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM POINTWISECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias) 
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM POINTWISECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());                
        
    ConvolutionUtils::conv2d(input, weights, bias, output, kH,kW, sH,sW, pH,pW, dH,dW, 1/*isSameMode*/, isNCHW);

    return Status::OK();
}


DECLARE_SHAPE_FN(pointwise_conv2d) {
    auto inputShapeInfo  = inputShape->at(0);                                   // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weightsShapeInfo  = inputShape->at(1);                                 // [1,  1,  iC, oC] (NHWC) or [oC, iC,  1,  1] (NCHW)
    auto biasShapeInfo = block.width() > 2 ? inputShape->at(2) : nullptr;       // [oC]

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM POINTWISECONV2D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM POINTWISECONV2D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);

    int isNCHW = block.getIArguments()->size() > 0 ? !INT_ARG(0) : 1;       // 1-NCHW, 0-NHWC

    int indIOioC, indWkH, indWoC, indWiC;
    if(!isNCHW) {
        indIOioC = 3; indWkH = 0; indWoC = 3; indWiC = 2;
    }
    else {        
        indIOioC = 1; indWkH = 2; indWoC = 0; indWiC = 1;              
    }    

    const int bS = inputShapeInfo[1];                            // batch size
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels        
    const int oC = weightsShapeInfo[indWoC+1];                   // output channels

    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,1,1,  indWiC,indWoC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "POINTWISECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo) 
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "POINTWISECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));    

    Nd4jLong* outputShapeInfo = nullptr;
    COPY_SHAPE(inputShapeInfo, outputShapeInfo);  

    // do not forget to put oC instead of iC in outputShapeInfo
    outputShapeInfo[indIOioC + 1] = oC;                                   

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);        
}

}
}