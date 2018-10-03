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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.03.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_avgpool3dnew)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew, 1, 1, false, 0, 14) {
    
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
    int isSameMode  = INT_ARG(12);                                              // 1-SAME,  0-VALID
    int extraParam0 = INT_ARG(13);
    int isNCDHW  = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(input->rankOf() == 5, 0, "AVGPOOL3DNEW OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());    
    REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0, "AVGPOOL3DNEW op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::string expectedOutputShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,iC,oD,oH,oW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    REQUIRE_TRUE(expectedOutputShape == ShapeUtils::shapeAsString(output), 0, "AVGPOOL3D op: wrong shape of output array, expected is %s, but got %s instead !", expectedOutputShape.c_str(), ShapeUtils::shapeAsString(output).c_str());

    if(!isNCDHW) {
        input  = input->permute({0, 4, 1, 2, 3});                                                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
        output = output->permute({0, 4, 1, 2, 3});                                                      // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
    }    

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
    
    //T extraParams[] = {};    
    ConvolutionUtils::pooling3d(*input, *output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, 1, extraParam0);
   
    if(!isNCDHW) {              
        delete input;
        delete output;
    }
        
    return Status::OK();
}

DECLARE_SHAPE_FN(avgpool3dnew) {

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
    int isSameMode = INT_ARG(12);                                               // 1-SAME,  0-VALID
    int isNCDHW  = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0, "AVGPOOL3DNEW op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);
    
    auto inputShapeInfo = inputShape->at(0);

    int idxID, idxIC;    
    if(isNCDHW) { idxID = 2; idxIC = 1;}
    else        { idxID = 1; idxIC = 4;}

    int bS = inputShapeInfo[1];                          // batch size
    int iC = inputShapeInfo[idxIC+1];                    // input channels            
    int iD = inputShapeInfo[idxID+1];                    // input depth
    int iH = inputShapeInfo[idxID+2];                    // input height
    int iW = inputShapeInfo[idxID+3];                    // input width

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);
    
    Nd4jLong* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), Nd4jLong);

    outputShapeInfo[0] = 5;
    outputShapeInfo[1] = bS;

    if (isNCDHW) {    
        outputShapeInfo[2] = iC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = iC;
    }
    // TF DOC: A Tensor. Has the same type as input.
    ShapeUtils::updateStridesAndType(outputShapeInfo, inputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew_bp, 2, 1, false, 0, 14) {
    
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
    const int isSameMode = INT_ARG(12);                                         // 1-SAME,  0-VALID
    const int extraParam0 = INT_ARG(13);                                        // define what divisor to use while averaging 
    const int isNCDHW  = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1; // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(input->rankOf() == 5, 0, "AVGPOOL3DNEW_BP op: input should have rank of 5, but got %i instead", input->rankOf());    
    REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0, "AVGPOOL3DNEW_BP op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::string expectedGradOShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,iC,oD,oH,oW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    std::string expectedGradIShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,iC,iD,iH,iW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0, "AVGPOOL3D_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedGradIShape == ShapeUtils::shapeAsString(gradI), 0, "AVGPOOL3D_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !", expectedGradIShape.c_str(), ShapeUtils::shapeAsString(gradI).c_str());

    if(!isNCDHW) {
        gradI = gradI->permute({0, 4, 1, 2, 3});                                   // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
        gradO = gradO->permute({0, 4, 1, 2, 3});                                   // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]                        
    }

    if(isSameMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
    
    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - poolingMode; 9 - divisor;    
    ConvolutionUtils::pooling3dBP(*input, *gradO, *gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, 1, extraParam0);

    if(!isNCDHW) {
        delete gradI;
        delete gradO;
    }    

    return Status::OK();
}


DECLARE_SHAPE_FN(avgpool3dnew_bp) {

    Nd4jLong* gradIshapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIshapeInfo);
    ArrayOptions::setDataType(gradIshapeInfo, ArrayOptions::dataType(inputShape->at(1)));
        
    return SHAPELIST(gradIshapeInfo);        
}



}
}

#endif