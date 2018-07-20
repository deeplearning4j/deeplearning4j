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
//  @author raver119@gmail.com
//  @author Yurii Shyrma


#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_conv1d)

#include <ops/declarable/CustomOperations.h>
#include <declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(conv1d, 2, 1, false, 0, 4) {

    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kW, iC, oC] (NWC) or [oC, iC, kW] (NCW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oW, oC] (NWC) or [bS, oC, oW] (NCW)

    int kW = INT_ARG(0);                                                        // filter(kernel) width
    int sW = INT_ARG(1);                                                        // strides width
    int pW = INT_ARG(2);                                                        // paddings width
    int isSameMode = INT_ARG(3);                                                // 0-VALID, 1-SAME
    int isNCW      = block.getIArguments()->size() > 4 ? !INT_ARG(4) : 1;       // 0-NCH,  1-NHC

    const int rank = 3;
    REQUIRE_TRUE(input->rankOf()   == rank, 0, "CUSTOM CONV1D OP: rank of input array must be equal to %i, but got %i instead !", rank, input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == rank, 0, "CUSTOM CONV1D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weights->rankOf());

    int indIOioC, indIiW, indWkW, indWoC, indWiC;
    if(!isNCW) {
        indIOioC = 2; indIiW = 1; indWkW = 0; indWoC = 2;
    }
    else {
        indIOioC = 1; indIiW = 2; indWkW = 2; indWoC = 0;
    }

    int bS = input->sizeAt(0);                        // batch size
    int iW = input->sizeAt(indIiW);                   // input width
    int iC = input->sizeAt(indIOioC);                 // input channels
    int oC = weights->sizeAt(indWoC);                 // output channels

    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({iC,oC,kW,  1,indWoC,indWkW}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weights), 0, "CUSTOM CONV1D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV1D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    std::vector<Nd4jLong> reshapeForInput, reshapeForOutput, reshapeForWeights;
    if(!isNCW) {
        reshapeForInput   = {input->sizeAt(0), 1, input->sizeAt(1), input->sizeAt(2)};                  // [bS, iW, iC] -> [bS, 1, iW, iC]
        reshapeForOutput  = {output->sizeAt(0), 1, output->sizeAt(1), output->sizeAt(2)};               // [bS, oW, oC] -> [bS, 1, oW, oC]
        reshapeForWeights = {1, weights->sizeAt(0), weights->sizeAt(1), weights->sizeAt(2)};            // [kW, iC, oC] -> [1, kW, iC, oC]
    }
    else {
        reshapeForInput   = {input->sizeAt(0),  input->sizeAt(1),  1, input->sizeAt(2)};                // [bS, iC, iW] -> [bS, iC, 1, iW]
        reshapeForOutput  = {output->sizeAt(0), output->sizeAt(1), 1, output->sizeAt(2)};               // [bS, oC, oW] -> [bS, oC, 1, oW]
        reshapeForWeights = {weights->sizeAt(0),weights->sizeAt(1),1, weights->sizeAt(2)};              // [oC, iC, kW] -> [oC, iC, 1, kW]
    }

    NDArray<T>* inputReshaped   = input  ->reshape(input->ordering(),   reshapeForInput);
    NDArray<T>* outputReshaped  = output ->reshape(output->ordering(),  reshapeForOutput);
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(), reshapeForWeights);

    ConvolutionUtils<T>::conv2d({inputReshaped, weightsReshaped, bias}, outputReshaped, {1,kW,  1,sW,  0,pW,  1,1,  isSameMode,  isNCW});

    delete inputReshaped;
    delete outputReshaped;
    delete weightsReshaped;

    return Status::OK();
}


DECLARE_SHAPE_FN(conv1d) {

    auto inputShapeInfo   = inputShape->at(0);
    auto weightsShapeInfo = inputShape->at(1);
    Nd4jLong* biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;

    int kW = INT_ARG(0);                                                        // filter(kernel) width
    int sW = INT_ARG(1);                                                        // strides width
    int pW = INT_ARG(2);                                                        // paddings width
    int isSameMode = INT_ARG(3);                                                // 0-VALID, 1-SAME
    int isNCW  = block.getIArguments()->size() > 4 ? !INT_ARG(4) : 1;           // 0-NWC, 1-NCW

    int indIOioC, indIiW, indWkW, indWoC, indWiC;
    if(!isNCW) {
        indIOioC = 2; indIiW = 1; indWkW = 0; indWoC = 2;
    }
    else {
        indIOioC = 1; indIiW = 2; indWkW = 2; indWoC = 0;
    }

    const int rank = 3;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM CONV1D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM CONV1D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo);

    int bS = inputShapeInfo[1];                         // batch size
    int iW = inputShapeInfo[indIiW+1];                   // input width
    int iC = inputShapeInfo[indIOioC+1];                   // input channels
    int oC = weightsShapeInfo[indWoC+1];                 // output channels

    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({iC,oC,kW,  1,indWoC,indWkW}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weightsShapeInfo), 0, "CUSTOM CONV1D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM CONV1D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    int oH, oW;                                         // output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(oH,oW,  1,kW,  1,sW,  0,pW,  1,1,  1,iW, isSameMode);

    Nd4jLong* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

    outputShapeInfo[0] = 3;
    outputShapeInfo[1] = bS;

    if (isNCW) {
        outputShapeInfo[2] = oC;
        outputShapeInfo[3] = oW;
    } else {
        outputShapeInfo[2] = oW;
        outputShapeInfo[3] = oC;
    }

    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}



//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(conv1d_bp, 3, 2, false, 0, 4) {

    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kW, iC, oC] (NWC) or [oC, iC, kW] (NCW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oW, oC] (NWC) or [bS, oC, oW] (NCW), epsilon_next

    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW), epsilon
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kW, iC, oC] (NWC) or [oC, iC, kW] (NCW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    int kW = INT_ARG(0);                                                        // filter(kernel) width
    int sW = INT_ARG(1);                                                        // strides width
    int pW = INT_ARG(2);                                                        // paddings width
    int isSameMode = INT_ARG(3);                                                // 0-VALID, 1-SAME
    int isNCW  = block.getIArguments()->size() > 4 ? !INT_ARG(4) : 1;           // 0-NWC, 1-NCW

    const int rank = 3;
    REQUIRE_TRUE(input->rankOf()   == rank, 0, "CUSTOM CONV1D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == rank, 0, "CUSTOM CONV1D_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank, weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf()   == rank, 0, "CUSTOM CONV1D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !", rank, gradO->rankOf());
    int indIOioC, indIiW, indWkW, indWoC;
    if(!isNCW) {
        indIOioC = 2; indIiW = 1; indWkW = 0; indWoC = 2;
    }
    else {
        indIOioC = 1; indIiW = 2; indWkW = 2; indWoC = 0;
    }

    const int bS = input->sizeAt(0);                          // batch size
    const int iW = input->sizeAt(indIiW);                     // input width
    const int iC = input->sizeAt(indIOioC);                   // input channels
    const int oC = weights->sizeAt(indWoC);                    // output channels

    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(trueoH,trueoW, 1,kW, 1,sW, 0,pW, 1,1, 1,iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,oC,trueoW,  0,indIOioC,indIiW}));
    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({oC,iC,kW,  indWoC,1,indWkW}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradO), 0,  "CUSTOM CONV1D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weights), 0, "CUSTOM CONV1D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV1D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    std::vector<Nd4jLong> reshapeForInput, reshapeForGradO, reshapeForWeights;
    if(!isNCW) {
        reshapeForInput   = {input->sizeAt(0), 1, input->sizeAt(1), input->sizeAt(2)};                  // [bS, iW, iC] -> [bS, 1, iW, iC]
        reshapeForGradO   = {gradO->sizeAt(0), 1, gradO->sizeAt(1), gradO->sizeAt(2)};                  // [bS, oW, oC] -> [bS, 1, oW, oC]
        reshapeForWeights = {1, weights->sizeAt(0), weights->sizeAt(1), weights->sizeAt(2)};            // [kW, iC, oC] -> [1, kW, iC, oC]
    }
    else {
        reshapeForInput   = {input->sizeAt(0), input->sizeAt(1), 1, input->sizeAt(2)};                  // [bS, iC, iW] -> [bS, iC, 1, iW]
        reshapeForGradO   = {gradO->sizeAt(0), gradO->sizeAt(1), 1, gradO->sizeAt(2)};                  // [bS, oC, oW] -> [bS, oC, 1, oW]
        reshapeForWeights = {weights->sizeAt(0),weights->sizeAt(1), 1, weights->sizeAt(2)};             // [oC, iC, kW] -> [oC, iC, 1, kW]
    }

    NDArray<T>* inputReshaped   = input  ->reshape(input->ordering(),  reshapeForInput);
    NDArray<T>* gradIReshaped   = gradI  ->reshape(gradI->ordering(),  reshapeForInput);
    NDArray<T>* gradOReshaped   = gradO  ->reshape(gradO->ordering(),  reshapeForGradO);
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(),reshapeForWeights);
    NDArray<T>* gradWReshaped   = gradW  ->reshape(gradW->ordering(),  reshapeForWeights);

    ConvolutionUtils<T>::conv2dBP({inputReshaped, weightsReshaped, bias, gradOReshaped}, {gradIReshaped, gradWReshaped, gradB}, {1,kW,  1,sW,  0,pW,  1,1,  isSameMode,  isNCW});

    delete inputReshaped;
    delete gradIReshaped;
    delete gradOReshaped;
    delete weightsReshaped;
    delete gradWReshaped;

    return Status::OK();
}


DECLARE_SHAPE_FN(conv1d_bp) {

    auto inputShapeInfo   = inputShape->at(0);                                               // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW)
    auto weightsShapeInfo = inputShape->at(1);                                               // [kW, iC, oC] (NWC) or [oC, iC, kW] (NCW)
    Nd4jLong* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;                 // [oC]
    Nd4jLong* gradOShapeInfo   = block.width() > 3 ? inputShape->at(3) : inputShape->at(2);       // [bS, oW, oC] (NWC) or [bS, oC, oW] (NCW), epsilon_next

    const int rank = 3;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM CONV1D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM CONV1D_BP OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);
    REQUIRE_TRUE(gradOShapeInfo[0]   == rank, 0, "CUSTOM CONV1D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !", rank, gradOShapeInfo[0]);

    int kW = INT_ARG(0);                                                        // filter(kernel) width
    int sW = INT_ARG(1);                                                        // strides width
    int pW = INT_ARG(2);                                                        // paddings width
    int isSameMode = INT_ARG(3);                                                // 0-VALID, 1-SAME
    int isNCW  = block.getIArguments()->size() > 4 ? !INT_ARG(4) : 1;          // 0-NWC, 1-NCW

    int indIOioC, indIiW, indWkW, indWoC;
    if(!isNCW) {
        indIOioC = 2; indIiW = 1; indWkW = 0; indWoC = 2;
    }
    else {
        indIOioC = 1; indIiW = 2; indWkW = 2; indWoC = 0;
    }

    const int bS = inputShapeInfo[1];                            // batch size
    const int iW = inputShapeInfo[indIiW+1];                     // input width
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels
    const int oC = weightsShapeInfo[indWoC+1];                   // output channels

    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(trueoH,trueoW, 1,kW, 1,sW, 0,pW, 1,1, 1,iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,oC,trueoW,  0,indIOioC,indIiW}));
    std::string expectedWeightsShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({oC,iC,kW,  indWoC,1,indWkW}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradOShapeInfo), 0,  "CUSTOM CONV1D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradOShapeInfo).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils<T>::shapeAsString(weightsShapeInfo), 0, "CUSTOM CONV1D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils<T>::shapeAsString(weightsShapeInfo).c_str());
    if(biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM CONV1D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));


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