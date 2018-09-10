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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 05.09.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_deconv3d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <MmulHelper.h>

namespace nd4j {
namespace ops  {
  
CUSTOM_OP_IMPL(deconv3d, 2, 1, false, 0, 13) {
            
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM DECONV3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM DECONV3D OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

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
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 0-NDHWC, 1-NCDHW    

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWoC, indWiC, indWkD);

    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({iC,oC,kD,kH,kW,  indWiC,indWoC,indWkD,indWkD+1,indWkD+2}));            
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weights), 0, "CUSTOM DECONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weights).c_str());        
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    std::vector<int> permutForColumns;

    if(!isNCDHW) {
        output  = output->permute({0, 4, 1, 2, 3});                             // [bS, oD, oH, oW, oC] -> [bS, oC, oD, oH, oW] 
        permutForColumns = {2, 3, 4, 1, 0, 5, 6, 7};                            // [bS, oC, kD, kH, kW, iD, iH, iW] -> [kD, kH, kW, oC, bS, iD, iH, iW]
    }
    else
        permutForColumns = {1, 2, 3, 4, 0, 5, 6, 7};                            // [bS, oC, kD, kH, kW, iD, iH, iW] -> [oC, kD, kH, kW, bS, iD, iH, iW]

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, oC, kD, kH, kW, iD, iH, iW}, block.getWorkspace());

    //----- calculation of output -----//
    // NDHWC: [kD, kH, kW, oC, iC] x [bS, iD, iH, iW, iC] = [kD, kH, kW, oC, bS, iD, iH, iW]
    // NCDHW: [iC, oC, kD, kH, kW] x [bS, iC, iD, iH, iW] = [oC, kD, kH, kW, bS, iD, iH, iW]
    nd4j::MmulHelper<T>::tensorDot(weights, input, &columns, {indWiC}, {indIOioC}, permutForColumns);
    ConvolutionUtils<T>::col2vol(columns, *output, sD, sH, sW, pD, pH, pW, dD, dH, dW);                            // [bS, oC, kD, kH, kW, iD, iH, iW] is de-convoluted to [bS, oC, oD, oH, oW]
           
    //----- add biases if required -----//
    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({1}, bias);

    if(!isNCDHW)
        delete output;
    
    return Status::OK();

}

DECLARE_SHAPE_FN(deconv3d) {

    auto inputShapeInfo   = inputShape->at(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NDCHW)
    auto weightsShapeInfo = inputShape->at(1);                                    // [kD, kH, kW, oC, iC] (NDHWC) or [iC, oC, kD, kH, kW] (NDCHW)
    auto biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;      // [oC]

    const int rank = 5;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DECONV3D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DECONV3D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);

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
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 0-NDHWC, 1-NCDHW    

    int indIOioC, indIiD, indWkD, indWoC, indWiC;
    if(!isNCDHW) {
        indIOioC = 4; indIiD = 1; indWkD = 0; indWiC = 4; indWoC = 3;
    }
    else {        
        indIOioC = 1; indIiD = 2; indWkD = 2; indWiC = 0; indWoC = 1;              
    }    

    const int bS = inputShapeInfo[1];                           // batch size
    const int iD = inputShapeInfo[indIiD+1];                    // input depth
    const int iH = inputShapeInfo[indIiD+2];                    // input height
    const int iW = inputShapeInfo[indIiD+3];                    // input width
    const int iC = inputShapeInfo[indIOioC+1];                  // input channels        
    const int oC = weightsShapeInfo[indWoC+1];                  // output channels

    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({iC,oC,kD,kH,kW,  indWiC,indWoC,indWkD,indWkD+1,indWkD+2}));                
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weightsShapeInfo), 0, "CUSTOM DECONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM DECONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    int oD, oH, oW;                                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);
    
    Nd4jLong* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), Nd4jLong);

    outputShapeInfo[0] = rank;
    outputShapeInfo[1] = bS;

    if (isNCDHW) {
        outputShapeInfo[2] = oC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = oC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(deconv3d_bp, 3, 2, false, 0, 13) {

    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, oC, iC] (NDHWC) or [iC, oC, kD, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), gradI
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, oC, iC] (NDHWC) or [iC, oC, kD, kH, kW] (NCDHW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM DECONV3D_BP OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM DECONV3D_BP OP: rank of weights array must be equal to 5 , but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf()   == 5, 0, "CUSTOM DECONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !", gradO->rankOf());


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
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 0-NDHWC, 1-NCDHW    

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWoC, indWiC, indWkD);

    int trueoD, trueoH, trueoW;          // true output height, width
    ConvolutionUtils<T>::calcOutSizeDeconv3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({iC,oC,kD,kH,kW,  indWiC,indWoC,indWkD,indWkD+1,indWkD+2}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradO), 0,  "CUSTOM DECONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weights), 0, "CUSTOM DECONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

     // ----- calculation of gradI -> pass it through conv3d_ff ----- //
    nd4j::ops::conv3dnew<T> conv3d;
    const Nd4jStatus status = conv3d.execute({gradO, weights}, {gradI}, {}, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW,  dD,dH,dW,  isSameMode,  !isNCDHW});
    if (status != ND4J_STATUS_OK)
        return status;

    // -----prepare permutation arrays and axes for dot product ----- //
    std::vector<int> permutForGradW, inputAxesForDot;
    if(!isNCDHW) {
        gradO = gradO->permute({0, 4, 1, 2, 3});                                // [bS, oD, oH, oW, oC] -> [bS, oC, oD, oH, oW]
        inputAxesForDot = {0, 1, 2, 3};                                         // bS, iD, iH, iW
        permutForGradW = {4, 3, 0, 1, 2};                                       // [kD, kH, kW, oC, iC] -> [iC, oC, kD, kH, kW]
    }
    else
        inputAxesForDot = {0, 2, 3, 4};                                         // bS, iD, iH, iW

    // ----- calculation of gradW ----- //
    NDArray<T> columns(input->ordering(), {bS, oC, kD, kH, kW, iD, iH, iW}, block.getWorkspace());  
    ConvolutionUtils<T>::vol2col(*gradO, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                  // [bS, oC, oD, oH, oW] is deconvoluted to [bS, oC, kD, kH, kW, iD, iH, iW]
    MmulHelper<T>::tensorDot(input, &columns, gradW, inputAxesForDot, {0, 5, 6, 7}, permutForGradW);    // [bS, iC, iD, iH, iW]/[bS, iD, iH, iW, iC] x [bS, oC, kD, kH, kW, iD, iH, iW] = [iC, oC, kD, kH, kW]

    // ----- calculation of gradB ----- //
    if(gradB) {
        if(gradB->rankOf() == 2)
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, {0, 2, 3, 4});                                // sum over bS, oD, oH, oW
        if(gradB != OUTPUT_VARIABLE(2))
            delete gradB;
    }

    if(!isNCDHW)
        delete gradO;

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(deconv3d_bp) {

    auto inputShapeInfo   = inputShape->at(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weightsShapeInfo = inputShape->at(1);                                                // [kD, kH, kW, oC, iC] (NDHWC) or [iC, oC, kD, kH, kW] (NCDHW)
    Nd4jLong* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;             // [oC]
    Nd4jLong* gradOShapeInfo   = block.width() > 3 ? inputShape->at(3) : inputShape->at(2);   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    const int rank = 5;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DECONV3D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DECONV3D_BP OP: rank of weights array must be equal to %i , but got %i instead !", rank, weightsShapeInfo[0]);
    REQUIRE_TRUE(gradOShapeInfo[0]   == rank, 0, "CUSTOM DECONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !", rank, gradOShapeInfo[0]);

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
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 0-NDHWC, 1-NCDHW    

    int indIOioC, indIiD, indWkD, indWoC, indWiC;
    if(!isNCDHW) {
        indIOioC = 4; indIiD = 1; indWkD = 0; indWiC = 4; indWoC = 3;
    }
    else {        
        indIOioC = 1; indIiD = 2; indWkD = 2; indWiC = 0; indWoC = 1;              
    }    

    const int bS = inputShapeInfo[1];                           // batch size
    const int iD = inputShapeInfo[indIiD+1];                    // input depth
    const int iH = inputShapeInfo[indIiD+2];                    // input height
    const int iW = inputShapeInfo[indIiD+3];                    // input width
    const int iC = inputShapeInfo[indIOioC+1];                  // input channels        
    const int oC = weightsShapeInfo[indWoC+1];                  // output channels

    int trueoD, trueoH, trueoW;          // true output depth, height, width
    ConvolutionUtils<T>::calcOutSizeDeconv3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIiD,indIiD+1,indIiD+2}));
    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({iC,oC,kD,kH,kW,  indWiC,indWoC,indWkD,indWkD+1,indWkD+2}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradOShapeInfo), 0,  "CUSTOM DECONV3D_BP OP: wrong shape of output gradients next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradOShapeInfo).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weightsShapeInfo), 0, "CUSTOM DECONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weightsShapeInfo).c_str());
    if(biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM DECONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    Nd4jLong *gradIShapeInfo(nullptr), *gradWShapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, gradIShapeInfo);
    COPY_SHAPE(weightsShapeInfo, gradWShapeInfo);

    auto shapes = SHAPELIST(gradIShapeInfo, gradWShapeInfo);

    if (biasShapeInfo != nullptr) {
        Nd4jLong *gradBShapeInfo(nullptr);
        COPY_SHAPE(biasShapeInfo, gradBShapeInfo);
        shapes->push_back(gradBShapeInfo);
    }

    return shapes;
}



}
}

#endif