/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author Yurii Shyrma, created on 05.02.2018
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_conv3dnew)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/addBias.h>
#include <helpers/MmulHelper.h>

namespace sd {
namespace ops  {

CUSTOM_OP_IMPL(conv3dnew, 2, 1, false, 0, 13) {
    auto input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM CONV3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(weights->sizeAt(2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int paddingMode = INT_ARG(12);                                              // 0-SAME,  1-VALID
    int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;        // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    REQUIRE_TRUE(paddingMode < 2, 0, "CUSTOM CONV3D OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");
    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM CONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW, paddingMode);

    nd4j_debug("MKL-DNN is not used for conv3dnew!\n", 0);

    std::vector<int> permutForOutput;

    if (isNCDHW)
        permutForOutput = {0,2,3,4,1};                                        // [bS, oC, oD, oH, oW] -> [bS, oD, oH, oW, oC]
    else
        input = new NDArray(input->permute({0,4,1,2,3}));

    std::vector<int> wAxes;
    if(0 == wFormat)
        wAxes = {3,0,1,2};
    else if(1 == wFormat)
        wAxes = {1,2,3,4};
    else
        wAxes = {4,1,2,3};

    NDArray columns(input->ordering(), {bS, iC, kD, kH, kW, oD, oH, oW}, input->dataType(), block.launchContext());
    ConvolutionUtils::vol2col(block, *input, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                 // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]
    // [bS, iC, kD, kH, kW, oD, oH, oW] x [kD, kH, kW, iC, oC] = [bS, oD, oH, oW, oC]
    // [bS, iC, kD, kH, kW, oD, oH, oW] x [oC, iC, kD, kH, kW] = [bS, oD, oH, oW, oC]
    // [bS, iC, kD, kH, kW, oD, oH, oW] x [oC, kD, kH, kW, iC] = [bS, oD, oH, oW, oC]
    MmulHelper::tensorDot(&columns, weights, output, {1,2,3,4}, wAxes, permutForOutput);

    if(bias)
        // output->applyBroadcast(broadcast::Add, {indIOioC}, bias);
        helpers::addBias(block, *output, *bias, *output, isNCDHW);

     if(!isNCDHW)
        delete input;

    return Status::OK();
}

   DECLARE_TYPES(conv3dnew) {
        getOpDescriptor()
                ->setAllowedInputTypes(0, sd::DataType::ANY)
                ->setAllowedInputTypes(1, {ALL_FLOATS})
                ->setAllowedInputTypes(2, {ALL_FLOATS})
                ->setAllowedOutputTypes({ALL_FLOATS});
    }

DECLARE_SHAPE_FN(conv3dnew) {

    auto inputShapeInfo   = inputShape->at(0);                                  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weightsShapeInfo = inputShape->at(1);                                  // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;    // [oC]

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int paddingMode = INT_ARG(12);                                              // 1-SAME,  0-VALID;
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;         // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    const int rank = 5;
    REQUIRE_TRUE(paddingMode < 2, 0, "CUSTOM CONV3D OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM CONV3D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM CONV3D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo);

    int indIOioC, indIiD, indWoC(0 == wFormat ? 4 : 0);
    if(!isNCDHW) {
        indIOioC = 4; indIiD = 1;
    }
    else {
        indIOioC = 1; indIiD = 2;
    }

    int bS = inputShapeInfo[1];                           // batch size
    int iD = inputShapeInfo[indIiD+1];                    // input depth
    int iH = inputShapeInfo[indIiD+2];                    // input height
    int iW = inputShapeInfo[indIiD+3];                    // input width
    int iC = inputShapeInfo[indIOioC+1];                  // input channels
    int oC = weightsShapeInfo[indWoC+1];                  // output channels

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(weightsShapeInfo, expectedWeightsShape), 0, "CUSTOM CONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM CONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, paddingMode);

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

    ShapeUtils::updateStridesAndType(outputShapeInfo, weightsShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(CONSTANT(outputShapeInfo));
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(conv3dnew_bp, 3, 2, false, 0, 13) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM CONV3D_BP OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D_BP OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 5, 0, "CUSTOM CONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !", gradO->rankOf());

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(weights->sizeAt(2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int paddingMode = INT_ARG(12);                                              // 1-SAME,  0-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;         // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, paddingMode);

    REQUIRE_TRUE(paddingMode < 2, 0, "CUSTOM CONV3D_BP OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");
    std::vector<Nd4jLong> expectedGradOShape   = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2});
    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,  "CUSTOM CONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW, paddingMode);

    nd4j_debug("MKL-DNN is not used for conv3dnew_bp!\n", 0);

    std::vector<int> gradOaxesForDot;

    if(!isNCDHW) {
        gradOaxesForDot  = {0,1,2,3};                                           // bS, oD, oH, oW
        input = new NDArray(input->permute({0,4,1,2,3}));                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
        gradI = new NDArray(gradI->permute({0,4,1,2,3}));                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
    }
    else {
        gradOaxesForDot  = {0,2,3,4};                                           // bS, oD, oH, oW
    }

    std::vector<int> wPermut, colPermut;

    if(0 == wFormat) {
        wPermut   = {3,0,1,2,4};
        colPermut = {2,3,4,1,0,5,6,7};
    }
    else if(1 == wFormat) {
        wPermut   = {1,2,3,4,0};
        colPermut = {1,2,3,4,0,5,6,7};
    }
    else {
        wPermut   = {4,1,2,3,0};
        colPermut = {2,3,4,1,0,5,6,7};
    }

    // ----- calculation of gradW and gradB ----- //
    NDArray columns(input->ordering(), {bS, iC, kD, kH, kW, oD, oH, oW}, input->dataType(), block.launchContext());
    ConvolutionUtils::vol2col(block, *input, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                   // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]
    MmulHelper::tensorDot(&columns, gradO, gradW, {0,5,6,7}, gradOaxesForDot, wPermut);     // [bS, iC, kD, kH, kW, oD, oH, oW] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [iC, kD, kH, kW, oC]

    //----- calculation of gradO -----//
    if(gradB) {
        if(gradB->rankOf() == 2)
            gradB = new NDArray(gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()}, false));
        gradO->reduceAlongDimension(reduce::Sum, *gradB, gradOaxesForDot);                          // sum over bS oD oH oW
        if(gradB != OUTPUT_VARIABLE(2))
            delete gradB;
    }

    //----- calculation of gradI -----//
    // [kD, kH, kW, iC, oC] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [kD, kH, kW, iC, bS, oD, oH, oW]
    // [oC, iC, kD, kH, kW] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [kD, kH, kW, iC, bS, oD, oH, oW]
    // [oC, kD, kH, kW, iC] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [kD, kH, kW, iC, bS, oD, oH, oW]
    MmulHelper::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, colPermut);
    ConvolutionUtils::col2vol(block, columns, *gradI, sD, sH, sW, pD, pH, pW, dD, dH, dW);                   // columns [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to  [bS, iC, iD, iH, iW]

    if(!isNCDHW) {
        delete input;
        delete gradI;
    }

    return Status::OK();
}

   DECLARE_TYPES(conv3dnew_bp) {
        getOpDescriptor()
                ->setAllowedInputTypes(0, sd::DataType::ANY)
                ->setAllowedInputTypes(1, {ALL_FLOATS})
                ->setAllowedInputTypes(2, {ALL_FLOATS})
                ->setAllowedInputTypes(3, {ALL_FLOATS})
                ->setAllowedOutputTypes({ALL_FLOATS});
    }


DECLARE_SHAPE_FN(conv3dnew_bp) {

    auto inputShapeInfo   = inputShape->at(0);                                              // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weightsShapeInfo = inputShape->at(1);                                              // [kD, kH, kW, iC, oC], [oC, iC, kD, kH, kW], [oC, kD, kH, kW, iC]
    Nd4jLong const* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;                // [oC]
    Nd4jLong const* gradOShapeInfo   = block.width() > 3 ? inputShape->at(3) : inputShape->at(2);      // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(shape::sizeAt(weightsShapeInfo, 2));// filter(kernel) width
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
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;         // 0-[kD, kH, kW, iC, oC], 1-[oC, iC, kD, kH, kW], 2-[oC, kD, kH, kW, iC]

    const int rank = 5;
    REQUIRE_TRUE(paddingMode < 2, 0, "CUSTOM CONV3D OP: causal padding mode (paddingMode = 2) is not allowed for this operation !");
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM CONV3D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM CONV3D_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo);
    REQUIRE_TRUE(gradOShapeInfo[0]   == rank, 0, "CUSTOM CONV3D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !", rank, gradOShapeInfo);

    int indIOioC, indIiD, indWoC(0 == wFormat ? 4 : 0);
    if(!isNCDHW) {
        indIOioC = 4; indIiD = 1;
    }
    else {
        indIOioC = 1; indIiD = 2;
    }

    int bS = inputShapeInfo[1];                           // batch size
    int iD = inputShapeInfo[indIiD+1];                    // input depth
    int iH = inputShapeInfo[indIiD+2];                    // input height
    int iW = inputShapeInfo[indIiD+3];                    // input width
    int iC = inputShapeInfo[indIOioC+1];                  // input channels
    int oC = weightsShapeInfo[indWoC+1];                  // output channels

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, paddingMode);

    std::vector<Nd4jLong> expectedGradOShape   = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIiD,indIiD+1,indIiD+2});
    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, iC, oC);
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(gradOShapeInfo, expectedGradOShape),   0, "CUSTOM CONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(weightsShapeInfo, expectedWeightsShape), 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if(biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM CONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    auto gradIshapeInfo = ShapeBuilders::copyShapeInfoAndType(inputShapeInfo,   gradOShapeInfo, false, block.getWorkspace());
    auto gradWshapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, gradOShapeInfo, false, block.getWorkspace());

    if(biasShapeInfo) {
        auto gradBshapeInfo = ShapeBuilders::copyShapeInfoAndType(biasShapeInfo, gradOShapeInfo, false, block.getWorkspace());
        return SHAPELIST(CONSTANT(gradIshapeInfo), CONSTANT(gradWshapeInfo), CONSTANT(gradBshapeInfo));
    }

    return SHAPELIST(CONSTANT(gradIshapeInfo), CONSTANT(gradWshapeInfo));
}
}
}

#endif
