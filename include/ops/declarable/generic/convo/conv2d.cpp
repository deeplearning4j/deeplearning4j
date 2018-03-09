//
// created by Yurii Shyrma on 06.03.2018
//

#ifndef LIBND4J_CONVO_OPS_H
#define LIBND4J_CONVO_OPS_H

#include <op_boilerplate.h>
#include <memory>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/OpRegistrator.h>
#include <declarable/generic/helpers/convolutions.h>



namespace nd4j {
    namespace ops {


CUSTOM_OP_IMPL(conv2d, 2, 1, false, 0, 9) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, oC] (NHWC) or [oC, iC, kH, kW] (NCHW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)
    
    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM CONV2D OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM CONV2D OP: rank of weights array must be equal to 4 !");
                                     
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
        weights = weights->permute({2, 0, 1, 3});                               // [kH, kW, iC, oC] -> [iC, kH, kW, oC]                 
    }
    else {        
        output  = output->permute({0, 2, 3, 1});                                // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
        weights = weights->permute({1, 2, 3, 0});                               // [oC, iC, kH, kW] -> [iC, kH, kW, oC]
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iH = input->sizeAt(2);           // input height
    int iW = input->sizeAt(3);           // input width
    int oC = weights->sizeAt(3);         // output channels        
    int oH = output->sizeAt(1);          // output height
    int oW = output->sizeAt(2);          // output width    
    
    REQUIRE_TRUE(weights->sizeAt(0) == iC && weights->sizeAt(1) == kH && weights->sizeAt(2) == kW, 0, "CUSTOM CONV2D OP: wrong shape of weights array !");    
    if (bias) {
        REQUIRE_TRUE(bias->rankOf() <= 2 ,   0, "CUSTOM CONV2D OP: rank of biases array must be equal to 1 or 2!");
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM CONV2D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
    }            
    
    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T> columns(input->ordering(), {bS, iC, kH, kW, oH, oW}, block.getWorkspace());        
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(&columns, extrasIm2Col.data());                        // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]
    
    columns.permutei({0, 4, 5, 1, 2, 3});                                                                     // [bS, iC, kH, kW, oH, oW] -> [bS, oH, oW, iC, kH, kW]
    columns.reshapei({bS*oH*oW, iC*kH*kW});
    NDArray<T>* outputReshaped  = output->reshape(output->ordering(), {bS*oH*oW, oC});
    NDArray<T>* weightsReshaped  = weights->reshape(weights->ordering(), {iC*kH*kW, oC});
    NDArrayFactory<T>::mmulHelper(&columns, weightsReshaped, outputReshaped, 1.0, 0.0);                        // [bS*oH*oW, iC*kW*kH] x [iC*kH*kW, oC] = [bS*oH*oW, oC]    

    if(bias)
        outputReshaped->template applyBroadcast<simdOps::Add<T>>({1}, bias);


    if(!isNCHW)
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



DECLARE_SHAPE_FN(conv2d) {

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
    int oC = weightsShapeInfo[indOC+1];                 // output channels

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
CUSTOM_OP_IMPL(conv2d_bp, 3, 2, false, 0, 9) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC] (NDHWC) or [oC, iC, kH, kW] (NCDHW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NDHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NDHWC) or [bS, iC, iH, iW] (NCDHW), epsilon
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, oC] (NDHWC) or [oC, iC, kH, kW] (NCDHW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]
    
    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM CONV2D_BP OP: rank of input array must be equal to 4 !");
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM CONV2D_BP OP: rank of weights array must be equal to 4 !");
    REQUIRE_TRUE(gradO->rankOf() == 4, 0, "CUSTOM CONV2D_BP OP: rank of gradO array must be equal to 4 !");
                                     
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
        weights = weights->permute({2, 0, 1, 3});                               // [kH, kW, iC, oC] -> [iC, kH, kW, oC]         
        gradW   = gradW->permute({2, 0, 1, 3});                                 // [kH, kW, iC, oC] -> [iC, kH, kW, oC]                 
    }
    else {
        gradO   = gradO->permute({0, 2, 3, 1});                                 // [bS, oC, oH, oW] -> [bS, oH, oW, oC]
        weights = weights->permute({1, 2, 3, 0});                               // [oC, iC, kH, kW] -> [iC, kH, kW, oC]
        gradW   = gradW->permute({1, 2, 3, 0});                                 // [oC, iC, kH, kW] -> [iC, kH, kW, oC]
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iH = input->sizeAt(2);           // input height
    int iW = input->sizeAt(3);           // input width
    int oC = weights->sizeAt(3);         // output channels    
    int oH = gradO->sizeAt(1);           // output height
    int oW = gradO->sizeAt(2);           // output width    

    int trueoH, trueoW;          // correct output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    REQUIRE_TRUE(gradO->sizeAt(0)==bS   && gradO->sizeAt(1)==trueoH && gradO->sizeAt(2)==trueoW && gradO->sizeAt(3)==oC, 0, "CUSTOM CONV2D_BP OP: wrong shape of gradient_output (next epsilon) array !");    
    REQUIRE_TRUE(weights->sizeAt(0)==iC && weights->sizeAt(1)==kH   && weights->sizeAt(2)==kW, 0, "CUSTOM CONV2D_BP OP: wrong shape of weights array !");
    if(bias)
        REQUIRE_TRUE(bias->rankOf()<=2 && bias->lengthOf()==oC, 0, "CUSTOM CONV2D_BP OP: wrong shape of biases array !");

    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::_calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    NDArray<T>  columns(input->ordering(), {iC, kH, kW, bS, oH, oW}, block.getWorkspace());        
    NDArray<T>* columnsPermuted = columns.permute({3, 0, 1, 2, 4, 5});                                 // [iC, kH, kW, bS, oH, oW] -> [bS, iC, kH, kW, oH, oW]
    NDArray<T>* columnsReshaped = columns.reshape(columns.ordering(), {iC*kH*kW, bS*oH*oW});
    NDArray<T>* gradWreshaped   = gradW->reshape(gradW->ordering(),{iC*kH*kW, oC});    
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(), {iC*kH*kW, oC});    
    NDArray<T>* gradOreshaped   = gradO->reshape(gradO->ordering(),{bS*oH*oW, oC});    
    NDArray<T>* gradOreshapedT  = gradOreshaped->transpose();                                           // [bS*oH*oW, oC] -> [oC, bS*oH*oW]

    // ----- calculation of gradW and gradB ----- //            
    std::vector<T> extrasIm2Col({(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T) dW});
    input->template applyTransform<simdOps::Im2col<T>>(columnsPermuted, extrasIm2Col.data());          // [bS, iC, iH, iW] is convoluted to [bS, iC, kH, kW, oH, oW]    
    NDArrayFactory<T>::mmulHelper(columnsReshaped, gradOreshaped, gradWreshaped, 1.0, 0.0);            // [iC*kW*kH, bS*oH*oW] x [bS*oH*oW, oC] = [iC*kH*kW, oC]

    if(gradB) {
        NDArray<T>* sum = gradOreshaped->sum({0});                  // sum over bS*oH*oW
        gradB->assign(sum);
        delete sum;
    }

    //----- calculation of gradI -----//            
    NDArrayFactory<T>::mmulHelper(weightsReshaped, gradOreshapedT, columnsReshaped, 1.0, 0.0);             // [iC*kH*kW, oC] x [oC, bS*oH*oW] = [iC*kW*kH, bS*oH*oW]
    std::vector<T> extrasCol2Im({(T) sH, (T) sW, (T) pH, (T) pW, (T) iH, (T) iW, (T) dH, (T) dW});
    columnsPermuted->template applyTransform<simdOps::Col2Im<T>>(gradI, extrasCol2Im.data());               // [bS, iC, kH, kW, oH, oW] is de-convoluted to [bS, iC, iH, iW]

    //----- assign array having separate new shape (caused by permute+reshape ops) to output gradW -----///
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

   
    if(!isNCHW) {        
        delete input;        
        delete gradI;
    }
    else {
        delete gradO;              
            
    }
    
    return Status::OK();
}



DECLARE_SHAPE_FN(conv2d_bp) {

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

#endif //LIBND4J_CONVO_OPS_H
