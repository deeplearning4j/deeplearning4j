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

    if(!isNCDHW) {
        input   = input->permute({0, 4, 1, 2, 3});                                 // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
        weights = weights->permute({3, 0, 1, 2, 4});                               // [kD, kH, kW, iC, oC] -> [iC, kD, kH, kW, oC]                 
    }
    else {        
        output  = output->permute({0, 2, 3, 4, 1});                                // [bS, oC, oD, oH, oW] -> [bS, oD, oH, oW, oC]
        weights = weights->permute({1, 2, 3, 4, 0});                               // [oC, iC, kD, kH, kW] -> [iC, kD, kH, kW, oC]
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iD = input->sizeAt(2);           // input depth
    int iH = input->sizeAt(3);           // input height
    int iW = input->sizeAt(4);           // input width
    int oC = weights->sizeAt(4);         // output channels        
    int oD = output->sizeAt(1);          // output depth
    int oH = output->sizeAt(2);          // output height
    int oW = output->sizeAt(3);          // output width    
    
    REQUIRE_TRUE(weights->sizeAt(0) == iC && weights->sizeAt(1) == kD && weights->sizeAt(2) == kH && weights->sizeAt(3) == kW, 0, "CUSTOM CONV3D OP: wrong shape of weights array !");
    if (bias) {
        REQUIRE_TRUE(bias->rankOf() == 1,    0, "CUSTOM CONV3D OP: rank of biases array must be equal to 1 !");
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM CONV3D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
    }            
    
    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kD, kH, kW, oD, oH, oW}, block.getWorkspace());            
    ConvolutionUtils<T>::vol2col2(*input, columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);      // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        
    
    columns.permutei({0, 5, 6, 7, 1, 2, 3, 4});                                                          // [bS, iC, kD, kH, kW, oD, oH, oW] -> [bS, oD, oH, oW, iC, kD, kH, kW]
    columns.reshapei({bS*oD*oH*oW, iC*kD*kH*kW});
    NDArray<T>* outputReshaped  = output->reshape(output->ordering(), {bS*oD*oH*oW, oC});
    NDArray<T>* weightsReshaped  = weights->reshape(weights->ordering(), {iC*kD*kH*kW, oC});
    NDArrayFactory<T>::mmulHelper(&columns, weightsReshaped, outputReshaped, 1.0, 0.0);                  // [bS*oD*oH*oW, iC*kD*kW*kH] x [iC*kD*kH*kW, oC] = [bS*oD*oH*oW, oC]    

    if(bias)
        outputReshaped->template applyBroadcast<simdOps::Add<T>>({1}, bias);


    if(!isNCDHW)
        delete input;                
    else {
        output->assign(outputReshaped);
        delete output;        
    }    

    delete outputReshaped;
    delete weightsReshaped;
    delete weights;
    
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

    if(!isNCDHW) {
        input   = input->permute({0, 4, 1, 2, 3});                                 // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
        gradI   = gradI->permute({0, 4, 1, 2, 3});                                 // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]        
        weights = weights->permute({3, 0, 1, 2, 4});                               // [kD, kH, kW, iC, oC] -> [iC, kD, kH, kW, oC]         
        gradW   = gradW->permute({3, 0, 1, 2, 4});                                 // [kD, kH, kW, iC, oC] -> [iC, kD, kH, kW, oC]                 
    }
    else {
        gradO   = gradO->permute({0, 2, 3, 4, 1});                                 // [bS, oC, oD, oH, oW] -> [bS, oD, oH, oW, oC]
        weights = weights->permute({1, 2, 3, 4, 0});                               // [oC, iC, kD, kH, kW] -> [iC, kD, kH, kW, oC]
        gradW   = gradW->permute({1, 2, 3, 4, 0});                                 // [oC, iC, kD, kH, kW] -> [iC, kD, kH, kW, oC]
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iD = input->sizeAt(2);           // input depth
    int iH = input->sizeAt(3);           // input height
    int iW = input->sizeAt(4);           // input width
    int oC = weights->sizeAt(4);         // output channels
    int oD = gradO->sizeAt(1);           // output depth
    int oH = gradO->sizeAt(2);           // output height
    int oW = gradO->sizeAt(3);           // output width    

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils<T>::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    REQUIRE_TRUE(gradO->sizeAt(0)==bS   && gradO->sizeAt(1)==trueoD && gradO->sizeAt(2)==trueoH && gradO->sizeAt(3)==trueoW && gradO->sizeAt(4)==oC, 0, "CUSTOM CONV3D_BP OP: wrong shape of gradient_output (next epsilon) array !");    
    REQUIRE_TRUE(weights->sizeAt(0)==iC && weights->sizeAt(1)==kD   && weights->sizeAt(2)==kH   && weights->sizeAt(3)==kW, 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array !");
    if(bias)
        REQUIRE_TRUE(bias->rankOf()==1 && bias->lengthOf()==oC, 0, "CUSTOM CONV3D_BP OP: wrong shape of biases array !");

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);    
    
    NDArray<T>  columns(input->ordering(), {iC, kD, kH, kW, bS, oD, oH, oW}, block.getWorkspace());      
    NDArray<T>* columnsPermuted = columns.permute({4, 0, 1, 2, 3, 5, 6, 7});                            // [iC, kD, kH, kW, bS, oD, oH, oW] -> [bS, iC, kD, kH, kW, oD, oH, oW]
    NDArray<T>* columnsReshaped = columns.reshape(columns.ordering(), {iC*kD*kH*kW, bS*oD*oH*oW});
    NDArray<T>* gradWreshaped   = gradW->reshape(gradW->ordering(),{iC*kD*kH*kW, oC});    
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(), {iC*kD*kH*kW, oC});    
    NDArray<T>* gradOreshaped   = gradO->reshape(gradO->ordering(),{bS*oD*oH*oW, oC});    
    NDArray<T>* gradOreshapedT  = gradOreshaped->transpose();                                           // [bS*oD*oH*oW, oC] -> [oC, bS*oD*oH*oW]

    // ----- calculation of gradW and gradB ----- //                
    ConvolutionUtils<T>::vol2col2(*input, *columnsPermuted, sD, sH, sW, pD, pH, pW, dD, dH, dW);        // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        
    NDArrayFactory<T>::mmulHelper(columnsReshaped, gradOreshaped, gradWreshaped, 1.0, 0.0);             // [iC*kD*kW*kH, bS*oD*oH*oW] x [bS*oD*oH*oW, oC] = [iC*kD*kH*kW, oC]

    if(gradB) {
        NDArray<T>* sum = gradOreshaped->sum({0});                  // sum over bS*oD*oH*oW
        gradB->assign(sum);
        delete sum;
    }

    //----- calculation of gradI -----//            
    NDArrayFactory<T>::mmulHelper(weightsReshaped, gradOreshapedT, columnsReshaped, 1.0, 0.0);             // [iC*kD*kH*kW, oC] x [oC, bS*oD*oH*oW] = [iC*kD*kW*kH, bS*oD*oH*oW]
    ConvolutionUtils<T>::col2vol2(*columnsPermuted, *gradI, sD, sH, sW, pD, pH, pW, dD, dH, dW);           // columns [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to  [bS, iC, iD, iH, iW]

    //----- assign array having separate shape (caused by permute+reshape ops) to output gradW -----///
    gradW->assign(gradWreshaped);

    //----- clean dynamically allocated memory -----//
    delete gradOreshapedT;
    delete columnsPermuted;
    delete columnsReshaped;
    delete gradWreshaped;
    delete weightsReshaped;
    delete gradOreshaped;    
    delete weights;
    delete gradW;

   
    if(!isNCDHW) {        
        delete input;        
        delete gradI;
    }
    else {
        delete gradO;              
            
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
