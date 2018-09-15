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
// created by Yurii Shyrma on 08.03.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_depthwise_conv2d)

#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <declarable/generic/helpers/convolutions.h>


namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(depthwise_conv2d, 2, 1, false, 0, 9) {
    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, mC] (NHWC) or [mC, iC, kH, kW] (NCHW)
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC] = iC*mC
    
    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DEPTHWISECONV2D OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DEPTHWISECONV2D OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
                                     
    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 0-NCHW,  1-NHWC

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier

    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({mC,iC,kH,kW,  indWmC,indWiC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM DEPTHWISECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    REQUIRE_TRUE(output->sizeAt(indIOioC) == iC*mC, 0, "CUSTOM DEPTHWISECONV2D OP: the output_channels must be equal to input_channels * channels_multiplier = %i !", iC*mC);
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DEPTHWISECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    ConvolutionUtils::depthwiseConv2d({input, weights, bias}, output, {kH,kW,sH,sW,pH,pW,dH,dW,isSameMode,isNCHW});
    
    return Status::OK();
}



DECLARE_SHAPE_FN(depthwise_conv2d) {
    auto inputShapeInfo   = inputShape->at(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weightsShapeInfo = inputShape->at(1);                                    // [kH, kW, iC, mC] (NHWC) or [mC, iC, kH, kW] (NCHW)
    auto biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;      // [oC] = iC*mC

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DEPTHWISECONV2D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DEPTHWISECONV2D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NDHWC, 1-NCDHW

    int indIOioC, indIiH, indWkH, indWmC, indWiC;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indWmC = 3; indWiC = 2;
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indWmC = 0; indWiC = 1;              
    }    

    const int bS = inputShapeInfo[1];                            // batch size
    const int iH = inputShapeInfo[indIiH+1];                     // input height
    const int iW = inputShapeInfo[indIiH+2];                     // input width
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels        
    const int mC = weightsShapeInfo[indWmC+1];                   // channels multiplier(oC = iC*mC)
    const int oC = iC*mC;                                        // output channels

    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,mC,kH,kW,  indWiC,indWmC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "DEPTHWISECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo) 
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "DEPTHWISECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));    

    int oH, oW;                                         // output height, width
    ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
    
    Nd4jLong* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), Nd4jLong);

    outputShapeInfo[0] = rank;
    outputShapeInfo[1] = bS;

    if (isNCHW) {
        outputShapeInfo[2] = oC;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
    } else {
        outputShapeInfo[2] = oH;
        outputShapeInfo[3] = oW;
        outputShapeInfo[4] = oC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}



////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(depthwise_conv2d_bp, 3, 2, false, 0, 9) {
    auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC] = [iC*mC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 4, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of output gradients (next epsilon) array must be equal to 4, but got %i instead !", gradO->rankOf());
                                     
    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NHWC, 1-NCHW    

    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier(oC = iC*mC), output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);
    mC = weights->sizeAt(indWmC);                           // channels multiplier    

    int trueoH, trueoW;          // correct output height, width
    ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({mC,iC,kH,kW,  indWmC,indWiC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0,  "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());        

    ConvolutionUtils::depthwiseConv2dBP({input, weights, bias, gradO}, {gradI, gradW, gradB}, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, isNCHW});

    return Status::OK();
}



DECLARE_SHAPE_FN(depthwise_conv2d_bp) {
    auto inputShapeInfo   = inputShape->at(0);
    auto weightsShapeInfo = inputShape->at(1);
    auto biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;
    auto gradOShapeInfo   = block.width() > 3 ? inputShape->at(3) : inputShape->at(2);

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);
    REQUIRE_TRUE(gradOShapeInfo[0]   == rank, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !", rank, gradOShapeInfo[0]);

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NHWC, 1-NCHW    

    int indIOioC, indIiH, indWkH, indWmC, indWiC, indOoH;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indWmC = 3; indWiC = 2; 
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indWmC = 0; indWiC = 1;              
    }    

    const int bS = inputShapeInfo[1];                            // batch size
    const int iH = inputShapeInfo[indIiH+1];                     // input height
    const int iW = inputShapeInfo[indIiH+2];                     // input width
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels        
    const int mC = weightsShapeInfo[indWmC+1];                   // channels multiplier(oC = iC*mC)
    const int oC = iC*mC;                                        // output channels

    int trueoH, trueoW;          // correct output height, width
    ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indIiH,indIiH+1}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({mC,iC,kH,kW,  indWmC,indWiC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradOShapeInfo), 0,  "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if(biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));        

    Nd4jLong* gradIshapeInfo(nullptr), *gradWshapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, gradIshapeInfo);
    COPY_SHAPE(weightsShapeInfo, gradWshapeInfo);

    if(biasShapeInfo) {
        Nd4jLong* gradBshapeInfo(nullptr);
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWshapeInfo, gradBshapeInfo);
    }     

    return SHAPELIST(gradIshapeInfo, gradWshapeInfo);        
}




}
}
#endif