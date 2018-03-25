//
// Created by raver119 on 29/10/17.
// changed by Yurii Shyrma 20.03.2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>
#include <memory>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(sconv2d, 2, 1, false, 0, 9) {

    NDArray<T> *input        = INPUT_VARIABLE(0);                                           // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    NDArray<T> *weightsDepth = INPUT_VARIABLE(1);                                           // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW]  (NCHW)
    NDArray<T> *weightsPoint = nullptr;                                                     // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *bias         = nullptr;                                                     // [oC], oC = iC*mC if weightsPoint=nullptr
    NDArray<T> *output       = OUTPUT_VARIABLE(0);                                          // [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW]  (NCHW)
    
    if(block.width() == 3) {
        if((INPUT_VARIABLE(2))->rankOf() == 4)
            weightsPoint = INPUT_VARIABLE(2);
        else
            bias = INPUT_VARIABLE(2);
    }
    else if(block.width() == 4) {
        weightsPoint = INPUT_VARIABLE(2);
        bias = INPUT_VARIABLE(3);
    }

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM SCONV2D OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weightsDepth->rankOf() == 4, 0, "CUSTOM SCONV2D OP: rank of weightsDepth array must be equal to 4, but got %i instead !", weightsDepth->rankOf());
    if(weightsPoint)
        REQUIRE_TRUE(weightsPoint->rankOf() == 4, 0, "CUSTOM SCONV2D OP: rank of weightsPoint array must be equal to 4, but got %i instead !", weightsPoint->rankOf());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() == 1 || bias->rankOf() == 2, 0, "CUSTOM SCONV2D OP: rank of biases array must be equal to 1 or 2, but got %i instead !", bias->rankOf());;           

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
    
    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier


    std::string expectedWeightsDShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({mC,iC,kH,kW,  indWmC,indWiC,indWkH,indWkH+1}));            
    REQUIRE_TRUE(expectedWeightsDShape == ShapeUtils<T>::shapeAsString(*weightsDepth), 0, "CUSTOM SCONV2D OP: wrong shape of weightsDepth array, expected is %s, but got %s instead !", expectedWeightsDShape.c_str(), ShapeUtils<T>::shapeAsString(*weightsDepth).c_str());    
    if(weightsPoint) {        
        std::string expectedWeightsPShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({oC,iC*mC,1,1,  indWmC,indWiC,indWkH,indWkH+1}));            
        REQUIRE_TRUE(expectedWeightsPShape == ShapeUtils<T>::shapeAsString(*weightsPoint), 0, "CUSTOM SCONV2D OP: wrong shape of weightsPoint array, expected is %s, but got %s instead !", expectedWeightsPShape.c_str(), ShapeUtils<T>::shapeAsString(*weightsPoint).c_str());            
    }
    if (bias)        
        REQUIRE_TRUE(oC == bias->lengthOf(), 0, "CUSTOM SCONV2D OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
  
    if (iC == 1) {
        nd4j_debug("CUSTOM SCONV2D OP: for input_channels=1 this op is equivalent to standard conv2d\n","");
        nd4j::ops::conv2d<T> c2d;
        return c2d.execute(&block);
    }
    

    NDArray<T>* outputDepth = output;
    if(weightsPoint)                        // if pointwise convolution is expected
        outputDepth = new NDArray<T>(output->ordering(), !isNCHW ? std::vector<int>({bS, oH, oW, iC*mC}) : std::vector<int>({bS, iC*mC, oH, oW}));    

    // ----- perform depthwise convolution (if weightsPoint is absent then oC = iC*mC) ----- //
    nd4j::ops::depthwise_conv2d<T> op;
    Nd4jStatus status = op.execute({input, weightsDepth, weightsPoint ? nullptr : bias}, {outputDepth}, {}, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, !isNCHW});                                   
    if (status != ND4J_STATUS_OK) 
        return status;
    
    // ----- perform pointwise convolution (oH = iH, oW = iW) ----- //
    if (weightsPoint) {

        nd4j::ops::conv2d<T> op;        
        status = op.execute({outputDepth, weightsPoint, bias}, {output}, {}, {1,1, 1,1, 0,0, 1,1, isSameMode, !isNCHW});  // in this case oH=iH, oW=iW
        delete outputDepth;
    }
    
    return status;
}


DECLARE_SHAPE_FN(sconv2d) {
    
    int* inputShapeInfo        = inputShape->at(0);
    int* weightsDepthShapeInfo = inputShape->at(1);
    int* weightsPointShapeInfo = nullptr;

    if(block.width() == 3 && inputShape->at(2)[0] == 4)        
        weightsPointShapeInfo = inputShape->at(2);
    else if(block.width() == 4) 
        weightsPointShapeInfo = inputShape->at(2);        
    
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
    
    int bS = inputShapeInfo[1];                         // batch size
    int iH = inputShapeInfo[indIH+1];                   // input height
    int iW = inputShapeInfo[indIH+2];                   // input width
    int iC = inputShapeInfo[indIC+1];                   // input channels        
    int mC = weightsDepthShapeInfo[indOC+1];            // channel multiplier

    int oC = weightsPointShapeInfo ? weightsPointShapeInfo[indOC+1] : iC*mC;      // output channels (oC or iC*mC)

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


////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sconv2d_bp, 3, 2, false, 0, 9) {

    NDArray<T> *input        = INPUT_VARIABLE(0);                                           // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW]  (NCHW)
    NDArray<T> *gradO        = INPUT_VARIABLE(1);                                           // [bS, oH, oW, oC]  (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    NDArray<T> *weightsDepth = INPUT_VARIABLE(2);                                           // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW]  (NCHW)
    NDArray<T> *weightsPoint = nullptr;                                                     // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *bias         = nullptr;                                                     // [oC], oC = iC*mC if weightsPoint=nullptr 
    
    NDArray<T> *gradI  = OUTPUT_VARIABLE(0);                                                // [bS, iH, iW, iC]  (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
    NDArray<T> *gradWD = OUTPUT_VARIABLE(1);                                                // [kH, kW, iC, mC]  (NHWC) or [mC, iC, kH, kW] (NCHW)
    NDArray<T> *gradWP = nullptr;                                                           // [1, 1, iC*mC, oC] (NHWC) or [oC, iC*mC, 1, 1] (NCHW)
    NDArray<T> *gradB  = nullptr;                                                           // [oC]

    if(block.width() == 4) {
        if((INPUT_VARIABLE(3))->rankOf() == 4) {
            weightsPoint = INPUT_VARIABLE(3);
            gradWP       = OUTPUT_VARIABLE(2);
        }
        else {
            bias  = INPUT_VARIABLE(3);
            gradB = OUTPUT_VARIABLE(2);
        }
    }
    else if(block.width() == 5) {
        weightsPoint = INPUT_VARIABLE(3);
        bias         = INPUT_VARIABLE(4);
        gradWP       = OUTPUT_VARIABLE(2);
        gradB        = OUTPUT_VARIABLE(3);
    }
        

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM SCONV2D_BP OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(gradO->rankOf()   == 4, 0, "CUSTOM SCONV2D_BP OP: rank of gradI (epsilon_next) array must be equal to 4, but got %i instead !", gradO->rankOf());
    REQUIRE_TRUE(weightsDepth->rankOf() == 4, 0, "CUSTOM SCONV2D_BP OP: rank of weightsDepth array must be equal to 4 !, but got %i instead !", weightsDepth->rankOf());
    if(weightsPoint) {
        REQUIRE_TRUE(weightsPoint->rankOf() == 4, 0, "CUSTOM SCONV2D_BP OP: rank of weightsPoint array must be equal to 4, but got %i instead !", weightsPoint->rankOf());
        REQUIRE_TRUE(gradWP->rankOf() == 4, 0, "CUSTOM SCONV2D_BP OP: rank of weightsPoint gradients array must be equal to 4, but got %i instead !", gradWP->rankOf());
    }
    if(bias) {
        REQUIRE_TRUE(bias->rankOf() == 1  || bias->rankOf()  == 2, 0, "CUSTOM SCONV2D_BP OP: rank of biases array must be equal to 1 or 2, but got %i instead !", bias->rankOf());;           
        REQUIRE_TRUE(gradB->rankOf() == 1 || gradB->rankOf() == 2, 0, "CUSTOM SCONV2D_BP OP: rank of gradients biases array must be equal to 1 or 2, but got %i instead !", gradB->rankOf());;           
    }

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
    
    int bS, iC, iH, iW, mC, oC, oH, oW;                     // batch size, input channels, input height/width, channels multiplier, output channels, output height/width
    int indIOioC, indIiH, indWmC, indWiC, indWkH, indOoH;   // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWmC, indWkH, indOoH);    
    mC = weightsDepth->sizeAt(indWmC);                      // channels multiplier

    

    std::string expectedWeightsDShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({mC,iC,kH,kW,  indWmC,indWiC,indWkH,indWkH+1}));            
    REQUIRE_TRUE(expectedWeightsDShape == ShapeUtils<T>::shapeAsString(*weightsDepth), 0, "CUSTOM SCONV2D_BP OP: wrong shape of weightsDepth array, expected is %s, but got %s instead !", expectedWeightsDShape.c_str(), ShapeUtils<T>::shapeAsString(*weightsDepth).c_str());    
    REQUIRE_TRUE(expectedWeightsDShape == ShapeUtils<T>::shapeAsString(*gradWD),       0, "CUSTOM SCONV2D_BP OP: wrong shape of gradWD array, expected is %s, but got %s instead !", expectedWeightsDShape.c_str(), ShapeUtils<T>::shapeAsString(*gradWD).c_str());
    if(weightsPoint) {
        std::string expectedWeightsPShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({oC,iC*mC,1,1,  indWmC,indWiC,indWkH,indWkH+1}));            
        REQUIRE_TRUE(expectedWeightsPShape == ShapeUtils<T>::shapeAsString(*weightsPoint), 0, "CUSTOM SCONV2D_BP OP: wrong shape of weightsPoint array, expected is %s, but got %s instead !", expectedWeightsPShape.c_str(), ShapeUtils<T>::shapeAsString(*weightsPoint).c_str());
        REQUIRE_TRUE(expectedWeightsPShape == ShapeUtils<T>::shapeAsString(*gradWP),       0, "CUSTOM SCONV2D_BP OP: wrong shape of gradWP array, expected is %s, but got %s instead !", expectedWeightsPShape.c_str(), ShapeUtils<T>::shapeAsString(*gradWP).c_str());
    }
    if (bias) {
        REQUIRE_TRUE(oC == bias->lengthOf(),  0, "CUSTOM SCONV2D_BP OP: length of bias array must be equal to outChannels, but got %i instead", bias->lengthOf());        
        REQUIRE_TRUE(oC == gradB->lengthOf(), 0, "CUSTOM SCONV2D_BP OP: length of gradients bias array must be equal to outChannels, but got %i instead", gradB->lengthOf());
    }
  
    // if (iC == 1) {
    //     nd4j_debug("CUSTOM SCONV2D_BP OP: for input_channels=1 this op is equivalent to standard conv2d_bp \n","");
    //     nd4j::ops::conv2d_bp<T> op;
    //     return op.execute(&block);
    // }
        
    // ----- if weightsPoint is present, perform pointwise backprop first and calculate gradWP at this step ----- //
    if (weightsPoint){           

        nd4j::ops::sconv2d<T> opFF;
        ResultSet<T>* resultFF = opFF.execute({input, weightsDepth}, {}, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, !isNCHW});
        NDArray<T>* inputPoint = resultFF->at(0);          // [bS, oH, oW, mC]  (NHWC) or [bS, mC, oH, oW] (NCHW)

        std::vector<int> gradIDepthShape = ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC*mC,oH,oW,  0,indIOioC,indIiH,indIiH+1});
        NDArray<T>* gradIDepth = new NDArray<T>(inputPoint->ordering(), gradIDepthShape, block.getWorkspace());                 // [bS, oH, oW, iC*mC]  (NHWC) or [bS, iC*mC, oH, oW] (NCHW)

        nd4j::ops::conv2d_bp<T> opBP;
        opBP.execute({inputPoint, weightsPoint, bias, gradO}, {gradIDepth, gradWP, gradB}, {}, {1,1, 1,1, 0,0, 1,1, isSameMode, !isNCHW});      // in this case oH=iH and oW=iW
    
        gradO = gradIDepth;

        bias = gradB = nullptr;                     // if pointwise backprop was done then don't calculate gradB at depthwise_conv2d_bp step

        delete resultFF;
    }    

    // ----- apply depthwise_conv2d_bp ----- //
    nd4j::ops::depthwise_conv2d_bp<T> op;
    Nd4jStatus status = op.execute({input, weightsDepth, bias, gradO}, {gradI, gradWD, gradB}, {}, {kH,kW, sH,sW, pH,pW, dH,dW, isSameMode, !isNCHW});                                   

    if(weightsPoint)
        delete gradO;
    
    return status;

}


DECLARE_SHAPE_FN(sconv2d_bp) {

    int* inputShapeInfo        = inputShape->at(0);
    int* weightsDepthShapeInfo = inputShape->at(2);
    int* weightsPointShapeInfo = nullptr;
    int* biasShapeInfo         = nullptr;

    if(block.width() == 4) {    
        if(inputShape->at(3)[0] == 4)
            weightsPointShapeInfo = inputShape->at(3);
        else 
            biasShapeInfo  = inputShape->at(3);
    }
    else if(block.width() == 5) {
        weightsPointShapeInfo = inputShape->at(3);
        biasShapeInfo         = inputShape->at(4);
    }

    int* gradIshapeInfo(nullptr), *gradWDshapeInfo(nullptr);
    COPY_SHAPE(inputShapeInfo, gradIshapeInfo);
    COPY_SHAPE(weightsDepthShapeInfo, gradWDshapeInfo);

    int* gradWPshapeInfo(nullptr), *gradBshapeInfo(nullptr);
    
    if(weightsPointShapeInfo && biasShapeInfo) {        
        COPY_SHAPE(weightsPointShapeInfo, gradWPshapeInfo);
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWDshapeInfo, gradWPshapeInfo, gradBshapeInfo);
    }

    if(weightsPointShapeInfo && !biasShapeInfo) {        
        COPY_SHAPE(weightsPointShapeInfo, gradWPshapeInfo);        
        return SHAPELIST(gradIshapeInfo, gradWDshapeInfo, gradWPshapeInfo);
    }

    if(!weightsPointShapeInfo && biasShapeInfo) {        
        COPY_SHAPE(biasShapeInfo, gradBshapeInfo);
        return SHAPELIST(gradIshapeInfo, gradWDshapeInfo, gradBshapeInfo);
    }

    return SHAPELIST(gradIshapeInfo, gradWDshapeInfo);
}



}
}