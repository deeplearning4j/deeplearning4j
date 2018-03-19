//
// created by Yurii Shyrma on 05.02.2018
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {



CUSTOM_OP_IMPL(conv3dnew, 2, 1, false, 0, 13) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)
    
    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM CONV3D OP: rank of input array must be equal to 5 !");
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D OP: rank of weights array must be equal to 5 !");
                                     
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
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    REQUIRE_TRUE(weights->sizeAt(indWiC) == iC && weights->sizeAt(indWoC) == oC && weights->sizeAt(indWkD) == kD && weights->sizeAt(indWkD+1) == kH && weights->sizeAt(indWkD+2) == kW, 0, "CUSTOM CONV3D OP: wrong shape of weights array !");
    if (bias) {
        REQUIRE_TRUE(bias->rankOf() == 1,    0, "CUSTOM CONV3D OP: rank of biases array must be equal to 1 !");
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM CONV3D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
    }            

    std::vector<int> weightsAxesForDot, permutForGradW;

    if(!isNCDHW) {
        weightsAxesForDot = {3,0,1,2};                                          // iC, kD, kH, kW
        input = input->permute({0,4,1,2,3});                                    // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
    }
    else {
        weightsAxesForDot = {1,2,3,4};                                          // iC, kD, kH, kW
        permutForGradW    = {0,2,3,4,1};                                        // [bS, oC, oD, oH, oW] -> [bS, oD, oH, oW, oC]
    }
    
    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kD, kH, kW, oD, oH, oW}, block.getWorkspace());            
    ConvolutionUtils<T>::vol2col2(*input, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                 // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        

    // [bS, iC, kD, kH, kW, oD, oH, oW] x [kD, kH, kW, iC, oC]/[oC, iC, kD, kH, kW] = [bS, oD, oH, oW, oC]
    nd4j::NDArrayFactory<T>::tensorDot(&columns, weights, output, {1,2,3,4}, weightsAxesForDot, permutForGradW);

    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({indIOioC}, bias);

    if(!isNCDHW)
        delete input;                
    
    return Status::OK();
}


DECLARE_SHAPE_FN(conv3dnew) {

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
    int isSameMode = INT_ARG(12);                                               // 1-SAME,  0-VALID;
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 1-NDHWC, 0-NCDHW
    
    int indID  = isNCDHW == 0 ? 1 : 2;
    int indIC  = isNCDHW == 0 ? 4 : 1;
    int indOC = isNCDHW == 0 ? 4 : 0;

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);

    int bS = inputShapeInfo[1];                         // batch size
    int iD = inputShapeInfo[indID+1];                    // input depth
    int iH = inputShapeInfo[indID+2];                    // input height
    int iW = inputShapeInfo[indID+3];                    // input width
    int iC = inputShapeInfo[indIC+1];                    // input channels        
    int oC = weightsShapeInfo[indOC+1];                 // output channels

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    if (isNCDHW) {
        outputShapeInfo[0] = 5;
        outputShapeInfo[1] = bS;
        outputShapeInfo[2] = oC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[0] = 5;
        outputShapeInfo[1] = bS;
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = oC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}


////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(conv3dnew_bp, 3, 2, false, 0, 13) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, iC, oC] (NDHWC) or [oC, iC, kD, kH, kW] (NCDHW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]
    
    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM CONV3D_BP OP: rank of input array must be equal to 5 !");
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D_BP OP: rank of weights array must be equal to 5 !");
    REQUIRE_TRUE(gradO->rankOf() == 5, 0, "CUSTOM CONV3D_BP OP: rank of gradO array must be equal to 5 !");

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
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 1-NDHWC, 0-NCDHW    

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils<T>::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    REQUIRE_TRUE(gradO->sizeAt(0)==bS && gradO->sizeAt(indIOioD)==trueoD && gradO->sizeAt(indIOioD+1)==trueoH && gradO->sizeAt(indIOioD+2)==trueoW && gradO->sizeAt(indIOioC)==oC, 0, "CUSTOM CONV3D_BP OP: wrong shape of gradient_output (next epsilon) array !");
    REQUIRE_TRUE(weights->sizeAt(indWiC)==iC && weights->sizeAt(indWoC)==oC &&  weights->sizeAt(indWkD)==kD   && weights->sizeAt(indWkD+1)==kH   && weights->sizeAt(indWkD+2)==kW, 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array !");
    if(bias)
        REQUIRE_TRUE(bias->rankOf()==1 && bias->lengthOf()==oC, 0, "CUSTOM CONV3D_BP OP: wrong shape of biases array !");

    std::vector<int> gradOaxesForDot, permutForGradW, permutForColumns;    

    if(!isNCDHW) {
        input = input->permute({0,4,1,2,3});                                    // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
        gradI = gradI->permute({0,4,1,2,3});                                    // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
        gradOaxesForDot  = {0,1,2,3};                                           // bS, oD, oH, oW        
        permutForGradW   = {3,0,1,2,4};                                         // [kD, kH, kW, iC, oC] -> [iC, kD, kH, kW, oC]        
        permutForColumns = {2,3,4,1,0,5,6,7};                                   // [bS, iC, kD, kH, kW, oD, oH, oW] -> [kD, kH, kW, iC, bS, oD, oH, oW]
    }
    else {
        gradOaxesForDot  = {0,2,3,4};                                           // bS, oD, oH, oW
        permutForGradW   = {1,2,3,4,0};                                         // [oC, iC, kD, kH, kW] -> [iC, kD, kH, kW, oC]
        permutForColumns = {1,2,3,4,0,5,6,7};                                   // [bS, iC, kD, kH, kW, oD, oH, oW] -> [iC, kD, kH, kW, bS, oD, oH, oW]
    }

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);    
    
    // ----- calculation of gradW and gradB ----- //                
    NDArray<T>  columns(input->ordering(), {bS, iC, kD, kH, kW, oD, oH, oW}, block.getWorkspace());      
    ConvolutionUtils<T>::vol2col2(*input, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                         // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        
    nd4j::NDArrayFactory<T>::tensorDot(&columns, gradO, gradW, {0,5,6,7}, gradOaxesForDot, permutForGradW);     // [bS, iC, kD, kH, kW, oD, oH, oW] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [iC, kD, kH, kW, oC]

    if(gradB) {        
        if(gradB->rankOf() == 2) 
            gradB = gradB->reshape(gradB->ordering(), {(int)gradB->lengthOf()});
        gradO->template reduceAlongDimension<simdOps::Sum<T>>(gradB, gradOaxesForDot);                          // sum over bS oD oH oW
        if(gradB != OUTPUT_VARIABLE(2)) 
            delete gradB;
    }

    //----- calculation of gradI -----//            
    nd4j::NDArrayFactory<T>::tensorDot(weights, gradO, &columns, {indWoC}, {indIOioC}, permutForColumns);   // [kD, kH, kW, iC, oC]/[oC, iC, kD, kH, kW]] x [bS, oD, oH, oW, oC]/[bS, oC, oD, oH, oW] = [kD, kH, kW, iC, bS, oD, oH, oW]/[iC, kD, kH, kW, bS, oD, oH, oW]
    ConvolutionUtils<T>::col2vol2(columns, *gradI, sD, sH, sW, pD, pH, pW, dD, dH, dW);                     // columns [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to  [bS, iC, iD, iH, iW]
   
    if(!isNCDHW) {        
        delete input;        
        delete gradI;
    }
    
    return Status::OK();
}



DECLARE_SHAPE_FN(conv3dnew_bp) {

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);
    int* biasShapeInfo    = block.width() > 3 ? inputShape->at(2) : nullptr;  

    int* gradIshapeInfo(nullptr), *gradWshapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, gradIshapeInfo);
    COPY_SHAPE(weightsShapeInfo, gradWshapeInfo);

    if(biasShapeInfo) {
        int* gradBshapeInfo(nullptr);
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWshapeInfo, gradBshapeInfo);
    }     

    return SHAPELIST(gradIshapeInfo, gradWshapeInfo);        
}




}
}
