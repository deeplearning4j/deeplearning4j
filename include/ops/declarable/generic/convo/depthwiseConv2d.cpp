//
// created by Yurii Shyrma on 08.03.2018
//


#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <declarable/generic/helpers/convolutions.h>


namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(depthwise_conv2d, 2, 1, false, 0, 9) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, mC] (NHWC) or [mC, iC, kH, kW] (NCHW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC] = iC*mC
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, iC*mC] (NHWC) or [bS, iC*mC, oH, oW] (NCHW)
    
    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DEPTHWISECONV2D OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DEPTHWISECONV2D OP: rank of weights array must be equal to 4 !");
                                     
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

    if(!isNCHW) {
        input   = input->permute({0, 3, 1, 2});                                 // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        weights = weights->permute({2, 0, 1, 3});                               // [kH, kW, iC, mC] -> [iC, kH, kW, mC]                 
    }
    else {        
        output  = output->permute({0, 2, 3, 1});                                // [bS, iC*mC, oH, oW] -> [bS, oH, oW, iC*mC]
        weights = weights->permute({1, 2, 3, 0});                               // [mC, iC, kH, kW] -> [iC, kH, kW, mC]
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iH = input->sizeAt(2);           // input height
    int iW = input->sizeAt(3);           // input width
    int mC = weights->sizeAt(3);         // channels multiplier, oC = iC*mC    
    int oH = output->sizeAt(1);          // output height
    int oW = output->sizeAt(2);          // output width    
    int oC = output->sizeAt(3);          // output channels
    
    REQUIRE_TRUE(weights->sizeAt(0) == iC && weights->sizeAt(1) == kH && weights->sizeAt(2) == kW, 0, "CUSTOM DEPTHWISECONV2D OP: wrong shape of weights array !");    
    REQUIRE_TRUE(output->sizeAt(3) == iC*mC, 0, "CUSTOM DEPTHWISECONV2D OP: the output_channels must be equal to input_channels * channels_multiplier !");    
    if (bias) {
        REQUIRE_TRUE(bias->rankOf() <= 2 ,   0, "CUSTOM DEPTHWISECONV2D OP: rank of biases array must be equal to 1 or 2!");
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM DEPTHWISECONV2D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
    }            
    
    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, block.getWorkspace());                
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                      // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]

    columns.permutei({1, 0, 4, 5, 2, 3});                                                                   // [bS, iC, kH, kW, oH, oW] -> [iC, bS, oH, oW, kH, kW]
    columns.reshapei({iC, bS*oH*oW, kH*kW});
    NDArray<T>* outputPermuted = output->reshape(output->ordering(), {bS*oH*oW, iC, mC});
    outputPermuted->permutei({1, 0, 2});                                                                    // [bS*oH*oW, iC, mC] -> [iC, bS*oH*oW, mC]
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(), {iC, kH*kW, mC});
    NDArrayFactory<T>::mmulHelper(&columns, weightsReshaped, outputPermuted, 1.0, 0.0);                     // [iC, bS*oH*oW, kW*kH] x [iC, kH*kW, mC] = [iC, bS*oH*oW, mC]    

    outputPermuted->permutei({1, 0, 2});                                                                    // [iC, bS*oH*oW, mC] -> [bS*oH*oW, iC, mC]

    if(bias)
        outputPermuted->template applyBroadcast<simdOps::Add<T>>({1,2}, bias);

    if(!isNCHW)
        delete input;                
    else {        
        output->assign(outputPermuted);
        delete output;        
    }    

    delete outputPermuted;
    delete weightsReshaped;
    delete weights;
    
    return Status::OK();
}



DECLARE_SHAPE_FN(depthwise_conv2d) {

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

    int indIH = isNCHW == 1 ? 2 : 1;
    int indIC = isNCHW == 1 ? 1 : 3;
    int indOC = isNCHW == 1 ? 0 : 3;

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);

    int bS = inputShapeInfo[1];                         // batch size
    int iH = inputShapeInfo[indIH+1];                   // input height
    int iW = inputShapeInfo[indIH+2];                   // input width
    int iC = inputShapeInfo[indIC+1];                   // input channels        
    int mC = weightsShapeInfo[indOC+1];                 // channel multiplier

    int oC = iC * mC;                                   // output channels

    int oH, oW;                                         // output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    outputShapeInfo[0] = 4;
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
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC] = [iC*mC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, mC] (NDHWC) or [mC, iC, kH, kW] (NCDHW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]
    
    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of weights array must be equal to 4 !");
    REQUIRE_TRUE(gradO->rankOf() == 4, 0, "CUSTOM DEPTHWISECONV2D_BP OP: rank of gradO array must be equal to 4 !");
                                     
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

    if(!isNCHW) {
        input   = input->permute({0, 3, 1, 2});                                 // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradI   = gradI->permute({0, 3, 1, 2});                                 // [bS, iH, iW, iC] -> [bS, iC, iH, iW]        
        weights = weights->permute({2, 0, 1, 3});                               // [kH, kW, iC, oC] -> [iC, kH, kW, mC]         
        gradW   = gradW->permute({2, 0, 1, 3});                                 // [kH, kW, iC, oC] -> [iC, kH, kW, mC]                 
    }
    else {
        gradO   = gradO->permute({0, 2, 3, 1});                                 // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
        weights = weights->permute({1, 2, 3, 0});                               // [oC, iC, kH, kW] -> [iC, kH, kW, mC]
        gradW   = gradW->permute({1, 2, 3, 0});                                 // [oC, iC, kH, kW] -> [iC, kH, kW, mC]
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iH = input->sizeAt(2);           // input height
    int iW = input->sizeAt(3);           // input width
    int mC = weights->sizeAt(3);         // channels multiplier    
    int oH = gradO->sizeAt(1);           // output height
    int oW = gradO->sizeAt(2);           // output width    
    int oC = gradO->sizeAt(3);           // output channels

    int trueoH, trueoW;          // correct output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    REQUIRE_TRUE(gradO->sizeAt(0)==bS   && gradO->sizeAt(1)==trueoH && gradO->sizeAt(2)==trueoW && gradO->sizeAt(3)==mC*iC, 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of gradient_output (next epsilon) array !");    
    REQUIRE_TRUE(weights->sizeAt(0)==iC && weights->sizeAt(1)==kH   && weights->sizeAt(2)==kW, 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of weights array !");
    if(bias)
        REQUIRE_TRUE(bias->rankOf()<=2 && bias->lengthOf()==oC, 0, "CUSTOM DEPTHWISECONV2D_BP OP: wrong shape of biases array !");

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T>  columns(input->ordering(), {iC, kH, kW, bS, oH, oW}, block.getWorkspace());        
    NDArray<T>* columnsPermuted = columns.permute({3, 0, 1, 2, 4, 5});                                 // [iC, kH, kW, bS, oH, oW] -> [bS, iC, kH, kW, oH, oW]
    NDArray<T>* columnsReshaped = columns.reshape(columns.ordering(), {iC, kH*kW, bS*oH*oW});
    NDArray<T>* gradWreshaped   = gradW->reshape(gradW->ordering(),{iC, kH*kW, mC});    
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(), {iC, kH*kW, mC});    
    NDArray<T>* gradOPermuted   = gradO->reshape(gradO->ordering(),{bS*oH*oW, iC, mC});    
    gradOPermuted->permutei({1, 0, 2});                                                                 // [bS*oH*oW, iC, mC] -> [iC, bS*oH*oW, mC]

    // ----- calculation of gradW and gradB ----- //            
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(columnsPermuted, extrasIm2Col.data());          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    NDArrayFactory<T>::mmulHelper(columnsReshaped, gradOPermuted, gradWreshaped, 1.0, 0.0);            // [iC, kW*kH, bS*oH*oW] x [iC, bS*oH*oW, mC] = [iC, kH*kW, mC]

    if(gradB) {
        NDArray<T>* sum = gradOPermuted->sum({1});                  // sum over bS*oH*oW
        gradB->assign(sum);
        delete sum;
    }

    //----- calculation of gradI -----//            
    gradOPermuted->permutei({0, 2, 1});                                                                     // [iC, bS*oH*oW, mC] -> [iC, mC, bS*oH*oW]
    NDArrayFactory<T>::mmulHelper(weightsReshaped, gradOPermuted, columnsReshaped, 1.0, 0.0);               // [iC, kH*kW, mC] x [iC, mC, bS*oH*oW] = [iC, kW*kH, bS*oH*oW]
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    columnsPermuted->template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());               // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    //----- assign array that has separately allocated new shape (caused by permute+reshape ops sequence) to output gradW -----///
    gradW->assign(gradWreshaped);

    //----- clean memory being allocated dynamically  -----//    
    delete columnsPermuted;
    delete columnsReshaped;
    delete gradWreshaped;
    delete weightsReshaped;
    delete gradOPermuted;    
    delete weights;
    delete gradW;

   
    if(!isNCHW) {        
        delete input;        
        delete gradI;
    }
    else {
        delete gradO;              
            
    }
    
    return Status::OK();
}



DECLARE_SHAPE_FN(depthwise_conv2d_bp) {

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