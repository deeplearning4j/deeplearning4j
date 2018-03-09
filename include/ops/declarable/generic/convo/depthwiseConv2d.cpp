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
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
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
    
    REQUIRE_TRUE(weights->sizeAt(0) == iC && weights->sizeAt(1) == kH && weights->sizeAt(2) == kW, 0, "CUSTOM CONV2D OP: wrong shape of weights array !");    
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

    if(bias)
        output->template applyBroadcast<simdOps::Add<T>>({3}, bias);

    if(!isNCHW)
        delete input;                
    else {
        outputPermuted->permutei({1, 0, 2});                                                                // [iC, bS*oH*oW, mC] -> [bS*oH*oW, iC, mC]
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




}
}