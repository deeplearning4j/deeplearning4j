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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 18.09.2018
//

#include <ops/declarable/generic/helpers/convolutions.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/col2im.h>
#include <NDArrayFactory.h>
#include <MmulHelper.h>

namespace nd4j {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::calcPadding2D(int& pH, int& pW, int oH, int oW, int iH, int iW, int kH, int kW, int sH, int sW, int dH, int dW) {
    
    int eKH, eKW;
    if (dH == 1 && dW == 1) {
        eKH = kH;
        eKW = kW;
    } else {
        eKH = kH + (kH - 1) * (dH - 1);
        eKW = kW + (kW - 1) * (dW - 1);
    }

    pH = ((oH - 1) * sH + eKH - iH) / 2; //Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
    pW = ((oW - 1) * sW + eKW - iW) / 2;
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::calcPadding3D(int& pD, int& pH, int& pW, const int oD, const int oH, const int oW, const int iD, const int iH, const int iW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int dD, const int dH, const int dW) {

    int eKD, eKH, eKW;            
    if (dD == 1 && dH == 1 && dW == 1) {
        eKD = kD;
        eKH = kH;
        eKW = kW;
    } else {
        eKD = kD + (kD - 1) * (dD - 1);
        eKH = kH + (kH - 1) * (dH - 1);
        eKW = kW + (kW - 1) * (dW - 1);
    }

    pD = ((oD - 1) * sD + eKD - iD) / 2;       // Note that padBottom is 1 bigger than this if bracketed term is not divisible by 2
    pH = ((oH - 1) * sH + eKH - iH) / 2; 
    pW = ((oW - 1) * sW + eKW - iW) / 2;

}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::getSizesAndIndexesConv2d(const bool isNCHW, const Nd4jLong* inShapeInfo, const Nd4jLong* outShapeInfo, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indOoH = 1; indWoC = 3; indWiC = 2;
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indOoH = 2; indWoC = 0; indWiC = 1;              
    }    

    bS = inShapeInfo[1];                          // batch size
    iC = inShapeInfo[indIOioC+1];                   // input channels        
    iH = inShapeInfo[indIiH+1];                     // input height
    iW = inShapeInfo[indIiH+2];                   // input width
    oC = outShapeInfo[indIOioC+1];                  // output channels
    oH = outShapeInfo[indOoH+1];                    // output height
    oW = outShapeInfo[indOoH+2];                  // output width    
}

//////////////////////////////////////////////////////////////////////////
void ConvolutionUtils::getSizesAndIndexesConv2d(const bool isNCHW, const NDArray& input, const NDArray& output, int& bS, int& iC, int& iH, int& iW, int& oC, int& oH, int& oW, int& indIOioC, int& indIiH, int& indWiC, int& indWoC, int& indWkH, int& indOoH) {

    getSizesAndIndexesConv2d(isNCHW, input.getShapeInfo(), output.getShapeInfo(), bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void conv2d_(const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    // bias    [oC]    
    // output  [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
                                         
    // kH  filter(kernel) height
    // kW  filter(kernel) width
    // sH  strides height
    // sW  strides width
    // pH  paddings height
    // pW  paddings width
    // dH  dilations height
    // dW  dilations width
    // isSameMode 0-VALID, 1-SAME
    // isNCHW     1-NCHW,  0-NHWC
    
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
    
    std::vector<int> weightsAxesForDot = {indWiC, indWkH, indWkH+1};                                                        // iC, kH, kW
    
    std::vector<int> permutForOutput;
    if(!isNCHW)
        input = input->permute({0, 3, 1, 2});                                       // [bS, iH, iW, iC] -> [bS, iC, iH, iW] if NHWC
    else
        permutForOutput = {0, indOoH, indOoH+1, indIOioC};                          // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
     
    if(isSameMode)                       // SAME        
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());        

    //----- calculation of output -----//        
    LaunchContext ctx;
    helpers::im2col(ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, NDArrayFactory::_scalar(0.f, input->getWorkspace()));  // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    MmulHelper::tensorDot(&columns, weights, output, {1,2,3}, weightsAxesForDot, permutForOutput);      // [bS, iC, kH, kW, oH, oW] x [kH, kW, iC, oC]/[oC, iC, kH, kW] = [bS, oH, oW, oC]

    //----- add biases if required -----//
    if(bias)
        output->applyBroadcast(broadcast::Add, {indIOioC}, bias);

    if(!isNCHW)
        delete input;                
}

void ConvolutionUtils::conv2d(const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    
    BUILD_TRIPLE_SELECTOR(input->dataType(), weights->dataType(), output->dataType(), conv2d_, (input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
}

//////////////////////////////////////////////////////////////////////////
template <typename X, typename Y, typename Z>
void conv2dBP_(const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {

    // input   [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    // weights [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    // bias    [oC]
    // gradO   [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    
    // gradI    [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    // gradW    [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    // gradB    [oC]
                                     
    // kH         filter(kernel) height
    // kW         filter(kernel) width
    // sH         strides height
    // sW         strides width
    // pH         paddings height
    // pW         paddings width
    // dH         dilations height
    // dW         dilations width
    // isSameMode 0-VALID, 1-SAME
    // isNCHW     0-NHWC, 1-NCHW    

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::vector<int> gradOaxesForDot, permutForGradW, permutForColumns;    

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradI = gradI->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradOaxesForDot  = {0, 1, 2};                                           // bS, oH, oW        
        permutForGradW   = {2, 0, 1, 3};                                        // [kH, kW, iC, oC] -> [iC, kH, kW, oC]        
        permutForColumns = {2, 3, 1, 0, 4, 5};                                  // [bS, iC, kH, kW, oH, oW] -> [kH, kW, iC, bS, oH, oW]
    }
    else {
        gradOaxesForDot  = {0, 2, 3};                                           // bS, oH, oW
        permutForGradW   = {1, 2, 3, 0};                                        // [oC, iC, kH, kW] -> [iC, kH, kW, oC]
        permutForColumns = {1, 2, 3, 0, 4, 5};                                  // [bS, iC, kH, kW, oH, oW] -> [iC, kH, kW, bS, oH, oW]
    }
    
    if(isSameMode)                       // SAME        
        ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);
   
    NDArray columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, input->getWorkspace());
    
    // ----- calculation of gradW ----- // 
    if(gradW) {
        LaunchContext ctx;
        helpers::im2col(ctx, *input, columns, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, NDArrayFactory::_scalar(0.f, input->getWorkspace()));   // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]        
        nd4j::MmulHelper::tensorDot(&columns, gradO, gradW, {0,4,5}, gradOaxesForDot, permutForGradW);       // [bS, iC, kH, kW, oH, oW] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [iC, kH, kW, oC]
    }

    // ----- calculation of gradB ----- // 
    if(gradB) {        
        NDArray* gradBR = gradB;
        if(gradB->rankOf() == 2) 
            gradBR = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->reduceAlongDimension(reduce::Sum, gradBR, gradOaxesForDot);                          // sum over bS, oH, oW
        if(gradBR != gradB) 
            delete gradBR;
    }

    //----- calculation of gradI -----//
    nd4j::MmulHelper::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, permutForColumns);  // [kH, kW, iC, oC]/[oC, iC, kH, kW]] x [bS, oH, oW, oC]/[bS, oC, oH, oW] = [kH, kW, iC, bS, oH, oW]/[iC, kH, kW, bS, oH, oW]
    LaunchContext ctx;
    helpers::col2im(ctx, columns, *gradI, sH, sW, pH, pW, iH, iW, dH, dW);                          // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]    
  
    if(!isNCHW) {
        delete input;
        delete gradI;
    }
}
void ConvolutionUtils::conv2dBP(const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
    
    BUILD_TRIPLE_SELECTOR(input->dataType(), weights->dataType(), gradO->dataType(), conv2dBP_, (input, weights, bias, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
}






BUILD_TRIPLE_TEMPLATE(template void conv2d_,   (const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
BUILD_TRIPLE_TEMPLATE(template void conv2dBP_, (const NDArray* input, const NDArray* weights, const NDArray* bias, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}
}
