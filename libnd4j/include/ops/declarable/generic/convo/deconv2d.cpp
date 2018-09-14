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
// @author Yurii Shyrma
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_deconv2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <MmulHelper.h>

namespace nd4j {
namespace ops  {
  
CUSTOM_OP_IMPL(deconv2d, 2, 1, false, 0, 9) {
    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, oC, iC] (NHWC) or [iC, oC, kH, kW] (NCHW)
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    auto *output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DECONV2D OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DECONV2D OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 1-NCHW,  0-NHWC

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,kH,kW,  indWiC,indWoC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM DECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    std::vector<int> permutForColumns;

    if(!isNCHW) {
        output  = output->permute({0, 3, 1, 2});                                // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
        permutForColumns = {2, 3, 1, 0, 4, 5};                                  // [bS, oC, kH, kW, iH, iW] -> [kH, kW, oC, bS, iH, iW]
    }
    else
        permutForColumns = {1, 2, 3, 0, 4, 5};                                  // [bS, oC, kH, kW, iH, iW] -> [oC, kH, kW, bS, iH, iW]

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray columns(input->ordering(), {bS, oC, kH, kW, iH, iW}, block.getWorkspace());
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) oH, (T) oW, (T) dH, (T) dW});

    //----- calculation of output -----//
    // NHWC: [kH, kW, oC, iC] x [bS, iH, iW, iC] = [kH, kW, oC, bS, iH, iW]
    // NCHW: [iC, oC, kH, kW] x [bS, iC, iH, iW] = [oC, kH, kW, bS, iH, iW]
    nd4j::MmulHelper<T>::tensorDot(weights, input, &columns, {indWiC}, {indIOioC}, permutForColumns);
    columns.applyTransform(transform::Col2Im, output, extrasCol2Im.data());                            // [bS, oC, kH, kW, iH, iW] is de-convoluted to [bS, oC, oH, oW]
           
    //----- add biases if required -----//
    if(bias)
        output->applyBroadcast(BroadcastOpsTuple::Add(), {1}, bias);

    if(!isNCHW)
        delete output;
    
    return Status::OK();

}

DECLARE_SHAPE_FN(deconv2d) {

    auto inputShapeInfo   = inputShape->at(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weightsShapeInfo = inputShape->at(1);                                    // [kH, kW, oC, iC] (NHWC) or [iC, oC, kH, kW] (NCHW)
    auto biasShapeInfo    = block.width() > 2 ? inputShape->at(2) : nullptr;      // [oC]

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DECONV2D OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DECONV2D OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);

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

    int indIOioC, indIiH, indWkH, indWoC, indWiC;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indWiC = 3; indWoC = 2;
    }
    else {
        indIOioC = 1; indIiH = 2; indWkH = 2; indWiC = 0; indWoC = 1;
    }

    const int bS = inputShapeInfo[1];                            // batch size
    const int iH = inputShapeInfo[indIiH+1];                     // input height
    const int iW = inputShapeInfo[indIiH+2];                     // input width
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels
    const int oC = weightsShapeInfo[indWoC+1];                   // output channels

    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,kH,kW,  indWiC,indWoC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "CUSTOM DECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if (biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM DECONV2D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

    int oH, oW;                                         // output height, width
    ConvolutionUtils<T>::calcOutSizeDeconv2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
    
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
CUSTOM_OP_IMPL(deconv2d_bp, 3, 2, false, 0, 9) {
    auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, oC, iC] (NDHWC) or [iC, oC, kH, kW] (NCDHW)
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), gradI
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, oC, iC] (NDHWC) or [iC, oC, kH, kW] (NCDHW)
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DECONV2D_BP OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DECONV2D_BP OP: rank of weights array must be equal to 4 , but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf()   == 4, 0, "CUSTOM DECONV2D_BP OP: rank of output gradients (next epsilon) array must be equal to 4, but got %i instead !", gradO->rankOf());


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

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils<T>::calcOutSizeDeconv2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,kH,kW,  indWiC,indWoC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0,  "CUSTOM DECONV2D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM DECONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

     // ----- calculation of gradI -> pass it through conv2d_ff ----- //
    nd4j::ops::conv2d<T> conv2d;
    const Nd4jStatus status = conv2d.execute({gradO, weights}, {gradI}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW,  isSameMode,  !isNCHW});
    if (status != ND4J_STATUS_OK)
        return status;

    // -----prepare permutation arrays and axes for dot product ----- //
    std::vector<int> permutForGradW, inputAxesForDot;
    if(!isNCHW) {
        gradO = gradO->permute({0, 3, 1, 2});                                   // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
        inputAxesForDot = {0, 1, 2};                                            // bS, iH, iW
        permutForGradW = {3, 2, 0, 1};                                          // [kH, kW, oC, iC] -> [iC, oC, kH, kW]
    }
    else
        inputAxesForDot = {0, 2, 3};                                            // bS, iH, iW

    // ----- calculation of gradW ----- //
    NDArray columns(input->ordering(), {bS, oC, kH, kW, iH, iW}, block.getWorkspace());
    std::vector<double> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW, (T)0.f, (T)0.f});
    gradO->applyTransform(transform::Im2col, &columns, extrasIm2Col.data());                          // [bS, oC, oH, oW] is convoluted to [bS, oC, kH, kW, iH, iW]
    MmulHelper<T>::tensorDot(input, &columns, gradW, inputAxesForDot, {0, 4, 5}, permutForGradW);           // [bS, iC, iH, iW]/[bS, iH, iW, iC] x [bS, oC, kH, kW, iH, iW] = [iC, oC, kH, kW]

    // ----- calculation of gradB ----- //
    if(gradB) {
        if(gradB->rankOf() == 2)
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->reduceAlongDimension(reduce::Sum, gradB, {0, 2, 3});                                // sum over bS, oH, oW
        if(gradB != OUTPUT_VARIABLE(2))
            delete gradB;
    }

    if(!isNCHW)
        delete gradO;

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(deconv2d_bp) {

    auto inputShapeInfo   = inputShape->at(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    auto weightsShapeInfo = inputShape->at(1);                                                // [kH, kW, oC, iC] (NDHWC) or [iC, oC, kH, kW] (NCDHW)
    Nd4jLong* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;                  // [oC]
    Nd4jLong* gradOShapeInfo   = block.width() > 3 ? inputShape->at(3) : inputShape->at(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

    const int rank = 4;
    REQUIRE_TRUE(inputShapeInfo[0]   == rank, 0, "CUSTOM DECONV2D_BP OP: rank of input array must be equal to %i, but got %i instead !", rank, inputShapeInfo[0]);
    REQUIRE_TRUE(weightsShapeInfo[0] == rank, 0, "CUSTOM DECONV2D_BP OP: rank of weights array must be equal to %i , but got %i instead !", rank, weightsShapeInfo[0]);
    REQUIRE_TRUE(gradOShapeInfo[0]   == rank, 0, "CUSTOM DECONV2D_BP OP: rank of output gradients (next epsilon) array must be equal to %i, but got %i instead !", rank, gradOShapeInfo[0]);

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

    int indIOioC, indIiH, indWkH, indWoC, indWiC, indOoH;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indWiC = 3; indWoC = 2; indOoH = 1;
    }
    else {
        indIOioC = 1; indIiH = 2; indWkH = 2; indWiC = 0; indWoC = 1; indOoH = 2;
    }

    const int bS = inputShapeInfo[1];                            // batch size
    const int iH = inputShapeInfo[indIiH+1];                     // input height
    const int iW = inputShapeInfo[indIiH+2];                     // input width
    const int iC = inputShapeInfo[indIOioC+1];                   // input channels
    const int oC = weightsShapeInfo[indWoC+1];                   // output channels

    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils<T>::calcOutSizeDeconv2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,kH,kW,  indWiC,indWoC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradOShapeInfo), 0,  "CUSTOM DECONV2D_BP OP: wrong shape of output gradients next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradOShapeInfo).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "CUSTOM DECONV2D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    if(biasShapeInfo)
        REQUIRE_TRUE(biasShapeInfo[0] <= 2 && oC == shape::length(biasShapeInfo), 0, "CUSTOM DECONV2D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, biasShapeInfo[0], shape::length(biasShapeInfo));

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