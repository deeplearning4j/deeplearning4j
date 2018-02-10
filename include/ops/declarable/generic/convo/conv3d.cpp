//
// created by Yurii Shyrma on 05.02.2018
//


#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(conv3dNew, 2, 1, false, 0, 13) {
    
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
    int paddingMode = INT_ARG(12);                                              // 0-SAME,  1-VALID
    int dataFormat  = block.getIArguments()->size() > 13 ? INT_ARG(13) : 0;     // 0-NDHWC, 1-NCDHW    

    // vol2col (im2col for 3d case) works only with NCDHW format    
    if(!dataFormat) {
        input   = input->permute({0, 4, 1, 2, 3});                              // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]        
        weights = weights->permute({4, 3, 0, 1, 2});                            // [kD, kH, kW, iC, oC] -> [oC, iC, kD, kH, kW] 
        output  = output->permute({0, 4, 1, 2, 3});                             // [bS, oD, oH, oW, oC] -> [bS, oC, oD, oH, oW]
    }

    int bS = input->sizeAt(0);           // batch size
    int iC = input->sizeAt(1);           // input channels        
    int iD = input->sizeAt(2);           // input depth
    int iH = input->sizeAt(3);           // input height
    int iW = input->sizeAt(4);           // input width
    int oC = weights->sizeAt(0);         // output channels    
    int oD = output->sizeAt(2);          // output depth
    int oH = output->sizeAt(3);          // output height
    int oW = output->sizeAt(4);          // output width    
    
    REQUIRE_TRUE(weights->sizeAt(1) == iC, 0, "CUSTOM CONV3D OP: wrong shape of weights array, input_inChannels != weights_inChannels");
    REQUIRE_TRUE(weights->sizeAt(2) == kD, 0, "CUSTOM CONV3D OP: weights array has wrong shape, take a careful look at int arguments !");
    REQUIRE_TRUE(weights->sizeAt(3) == kH, 0, "CUSTOM CONV3D OP: weights array has wrong shape, take a careful look at int arguments !");
    REQUIRE_TRUE(weights->sizeAt(4) == kW, 0, "CUSTOM CONV3D OP: weights array has wrong shape, take a careful look at int arguments !");
    if (bias) {
        REQUIRE_TRUE(bias->rankOf() == 1,    0, "CUSTOM CONV3D OP: rank of biases array must be equal to 1 !");
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM CONV3D OP:: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
    }            
    
    if(!paddingMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);    

    NDArray<T>* reshapedWeights = weights->reshape(weights->ordering(), {oC, iC*kD*kH*kW});
    NDArray<T>* reshapedOutput  = output->reshape(output->ordering(), {bS, oC, oD*oH*oW});    
    NDArray<T> columns(input->ordering(), {iC*kD*kW*kH, oD*oH*oW});
    
    ResultSet<T>* inSubArrsList  = NDArrayFactory<T>::allExamples(input);
    ResultSet<T>* outSubArrsList = NDArrayFactory<T>::allExamples(reshapedOutput);

    for(int i = 0; i < bS; ++i) {
        
        ConvolutionUtils<T>::vol2col(inSubArrsList->at(i)->getBuffer(), columns.getBuffer(), iC, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW);                
        NDArrayFactory<T>::mmulHelper(reshapedWeights, &columns, outSubArrsList->at(i), 1.0, 0.0);      // [oC, iC*kD*kH*kW] x [iC*kD*kW*kH, oD*oH*oW] = [oC, oD*oH*oW]
                            
        if(bias)
            outSubArrsList->at(i)->template applyBroadcast<simdOps::Add<T>>({0}, bias);
    }

    delete inSubArrsList;
    delete outSubArrsList;
    delete reshapedWeights;
    delete reshapedOutput;
   
    if(!dataFormat) {
        delete input;
        delete weights;
        delete output;        
    }
    
    return Status::OK();
}


DECLARE_SHAPE_FN(conv3dNew) {

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
    int paddingMode = INT_ARG(12);                                              // 0-SAME,  1-VALID;
    int dataFormat  = block.getIArguments()->size() > 13 ? INT_ARG(13) : 0;     // 0-NDHWC, 1-NCDHW
    
    int indID  = dataFormat == 0 ? 1 : 2;
    int indIC  = dataFormat == 0 ? 4 : 1;
    int indOC = dataFormat == 0 ? 4 : 0;

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);

    int bS = inputShapeInfo[1];                         // batch size
    int iD = inputShapeInfo[indID+1];                    // input depth
    int iH = inputShapeInfo[indID+2];                    // input height
    int iW = inputShapeInfo[indID+3];                    // input width
    int iC = inputShapeInfo[indIC+1];                    // input channels        
    int oC = weightsShapeInfo[indOC+1];                 // output channels

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, paddingMode);

    if(!paddingMode)                        // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);
    outputShapeInfo[0]      = 5;
    outputShapeInfo[1]      = bS;
    outputShapeInfo[indID+1] = oD;
    outputShapeInfo[indID+2] = oH;
    outputShapeInfo[indID+3] = oW;
    outputShapeInfo[indIC+1] = oC;
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));
    
    return new ShapeList(outputShapeInfo);
}


}
}


