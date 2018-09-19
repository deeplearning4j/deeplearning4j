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

namespace nd4j {
namespace ops  {


////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(deconv2d_tf, 3, 1, false, 0, 9) {
    auto gradO      = INPUT_VARIABLE(2);                                                // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    auto weights    = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    auto gradIShape = INPUT_VARIABLE(0);                                                // [4] - shape of input of conv2d (that is shape of gradI)
            
    auto gradI = OUTPUT_VARIABLE(0);                                                  // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

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

    const int rank = gradO->rankOf();

    REQUIRE_TRUE(weights->rankOf() == rank, 0, "CUSTOM DECONV2D_TF OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradIShape->rankOf() == 1, 0, "CUSTOM DECONV2D_TF OP: rank of array with output shape must be equal to 1, but got %i instead !", gradIShape->rankOf());
    REQUIRE_TRUE(gradIShape->lengthOf() == rank, 0, "CUSTOM DECONV2D_TF OP: length of array with output shape must be equal to 4, but got %i instead !", gradIShape->lengthOf());    

    // create empty conv2d input array    
    auto gradIShapeAsVector = gradIShape->asVectorT<Nd4jLong>();
    NDArray input(gradO->ordering(), gradIShapeAsVector, block.getWorkspace());
    
                                     
    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes       
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, &input, gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);
    
    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    std::string expectedGradOShape   = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,kH,kW,  indWiC,indWoC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0,  "CUSTOM DECONV2D_TF OP: wrong shape of input array, basing on array with output shape expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM DECONV2D_TF OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());

    ConvolutionUtils::conv2dBP(&input, weights, nullptr, gradO,   gradI, nullptr, nullptr,  kH,kW,sH,sW,pH,pW,dH,dW,isSameMode,isNCHW);
    
    return Status::OK();
}

DECLARE_SHAPE_FN(deconv2d_tf) {
    auto gradOShapeInfo   = inputShape->at(2);                                                // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    auto weightsShapeInfo = inputShape->at(1);                                                // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    auto gradIShapeShapeInfo = inputShape->at(0);                                               // [4]

    const int rank = 4;
    
    REQUIRE_TRUE(weightsShapeInfo[0]  == rank, 0, "CUSTOM DECONV2D_TF OP: rank of weights array must be equal to %i, but got %i instead !", rank, weightsShapeInfo[0]);
    REQUIRE_TRUE(gradOShapeInfo[0]    == rank, 0, "CUSTOM DECONV2D_TF OP: rank of input array must be equal to %i, but got %i instead !", rank, gradOShapeInfo[0]);
    REQUIRE_TRUE(gradIShapeShapeInfo[0] == 1,    0, "CUSTOM DECONV2D_TF OP: rank of array with output shape must be equal to %i, but got %i instead !", 1, gradIShapeShapeInfo[0]);    
    

    const int kH = INT_ARG(0);                                                        // filter(kernel) height
    const int kW = INT_ARG(1);                                                        // filter(kernel) width
    const int sH = INT_ARG(2);                                                        // strides height
    const int sW = INT_ARG(3);                                                        // strides width
    const int pH = INT_ARG(4);                                                        // paddings height
    const int pW = INT_ARG(5);                                                        // paddings width
    const int dH = INT_ARG(6);                                                        // dilations height
    const int dW = INT_ARG(7);                                                        // dilations width
    const int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    const int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // 0-NHWC, 1-NCHW    

    int indIOioC, indIiH, indWkH, indWoC, indWiC, indOoH;
    if(!isNCHW) {
        indIOioC = 3; indIiH = 1; indWkH = 0; indWoC = 3; indWiC = 2, indOoH = 1;
    }
    else {        
        indIOioC = 1; indIiH = 2; indWkH = 2; indWoC = 0; indWiC = 1, indOoH = 2;              
    }    

    std::vector<Nd4jLong> gradIShape = INPUT_VARIABLE(0)->template asVectorT<Nd4jLong>();

    const int bS = gradIShape[0];                          // batch size
    const int iH = gradIShape[indIiH];                     // input height
    const int iW = gradIShape[indIiH+1];                   // input width
    const int iC = gradIShape[indIOioC];                   // input channels        
    const int oC = weightsShapeInfo[indWoC+1];             // output channels
    const int oH = gradOShapeInfo[indOoH+1];               // input height
    const int oW = gradOShapeInfo[indOoH+2];               // input width

    int trueiH, trueiW;                                         // output height, width
    ConvolutionUtils::calcOutSizeDeconv2D(trueiH, trueiW, kH, kW, sH, sW, pH, pW, dH, dW, oH, oW, isSameMode);
    
    std::string expectedGradIShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({bS,iC,trueiH,trueiW,  0,indIOioC,indIiH,indIiH+1}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,kH,kW,  indWiC,indWoC,indWkH,indWkH+1}));
    REQUIRE_TRUE(expectedGradIShape == ShapeUtils::shapeAsString(gradIShape), 0,  "CUSTOM DECONV2D_TF OP: wrong shape of array with output shape, expected is %s, but got %s instead !", expectedGradIShape.c_str(), ShapeUtils::shapeAsString(gradIShape).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weightsShapeInfo), 0, "CUSTOM DECONV2D_TF OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weightsShapeInfo).c_str());
    
    Nd4jLong* gradIshapeInfo(nullptr);        
    ALLOCATE(gradIshapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);

    gradIshapeInfo[0] = rank;
    gradIshapeInfo[1] = bS;

    if (isNCHW) {
        gradIshapeInfo[2] = iC;
        gradIshapeInfo[3] = iH;
        gradIshapeInfo[4] = iW;
    } else {
        gradIshapeInfo[2] = iH;
        gradIshapeInfo[3] = iW;
        gradIshapeInfo[4] = iC;
    }
    
    shape::updateStrides(gradIshapeInfo, shape::order(gradOShapeInfo)); 

    return SHAPELIST(gradIshapeInfo);        

}

/*


    CUSTOM_OP_IMPL(deconv2d_tf, 2, 1, false, 0, 9) {
        NDArray<T> *input   = INPUT_VARIABLE(2);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
        NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kH, kW, oC, iC] (NHWC) or [iC, oC, kH, kW] (NCHW)

        NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

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
     int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // 0-NCHW,  1-NHWC

     int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
     int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
     ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

     std::string expectedWeightsShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx({iC,oC,kH,kW,  indWiC,indWoC,indWkH,indWkH+1}));
     REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM DECONV2D OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());

     std::vector<int> permutForColumns;

     if(!isNCHW) {
         output  = output->permute({0, 3, 1, 2});                                // [bS, oH, oW, oC] -> [bS, oC, oH, oW]
         permutForColumns = {2, 3, 1, 0, 4, 5};                                  // [bS, oC, kH, kW, iH, iW] -> [kH, kW, oC, bS, iH, iW]
     }
     else
         permutForColumns = {1, 2, 3, 0, 4, 5};                                  // [bS, oC, kH, kW, iH, iW] -> [oC, kH, kW, bS, iH, iW]

     if(isSameMode)                       // SAME
         ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

        NDArray<T> columns(input->ordering(), {bS, oC, kH, kW, iH, iW}, block.getWorkspace());
        std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) oH, (T) oW, (T) dH, (T) dW});

     //----- calculation of output -----//
     // NHWC: [kH, kW, oC, iC] x [bS, iH, iW, iC] = [kH, kW, oC, bS, iH, iW]
     // NCHW: [iC, oC, kH, kW] x [bS, iC, iH, iW] = [oC, kH, kW, bS, iH, iW]
     nd4j::MmulHelper<T>::tensorDot(weights, input, &columns, {indWiC}, {indIOioC}, permutForColumns);
     columns.template applyTransform<simdOps::Col2Im<T>>(output, extrasCol2Im.data());                            // [bS, oC, kH, kW, iH, iW] is de-convoluted to [bS, oC, oH, oW]

     if(!isNCHW)
         delete output;
    
     return Status::OK();

    }

 DECLARE_SHAPE_FN(deconv2d_tf) {
    
     auto tfShape = INPUT_VARIABLE(0);
     auto inputShapeInfo   = inputShape->at(2);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
     auto weightsShapeInfo = inputShape->at(1);                                    // [kH, kW, oC, iC] (NHWC) or [iC, oC, kH, kW] (NCHW)

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

     auto shapeTF = tfShape->template asVectorT<Nd4jLong>();
     auto shapeND = shape::shapeOf(outputShapeInfo);

     if (!shape::shapeEquals(shapeTF.size(), shapeTF.data(), 4, shapeND)) {
         auto e = ShapeUtils::shapeAsString(shapeTF.size(), shapeTF.data());
         auto a = ShapeUtils::shapeAsString(4, shapeND);
         REQUIRE_TRUE(false, 0, "deconv2d_tf: shape doesn't match TF value: Expected %s vs actual %s", e.c_str(), a.c_str());
     }

     return SHAPELIST(outputShapeInfo);
 }
*/

}
}

#endif