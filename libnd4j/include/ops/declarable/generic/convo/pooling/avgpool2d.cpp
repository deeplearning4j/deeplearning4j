//
// @author raver119@gmail.com, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com), changed on 14.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_avgpool2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(avgpool2d, 1, 1, false, 0, 10) {

    NDArray<T>* input = INPUT_VARIABLE(0);

    REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", input->rankOf());

    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;
    auto argI = *(block.getIArguments());
    auto output = OUTPUT_VARIABLE(0);

    const auto kH = INT_ARG(0);
    const auto kW = INT_ARG(1);
    const auto sH = INT_ARG(2);
    const auto sW = INT_ARG(3);
          int pH = INT_ARG(4);
          int pW = INT_ARG(5);
    const auto dH = INT_ARG(6);
    const auto dW = INT_ARG(7);
    const auto isSameMode = static_cast<bool>(INT_ARG(8));
    const auto extraParam0 = INT_ARG(9);

    REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

    int oH = 0;
    int oW = 0;

    int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;       // 0-NHWC, 1-NCHW    

    const int iH = static_cast<int>(isNCHW ? input->sizeAt(2) : input->sizeAt(1));
    const int iW = static_cast<int>(isNCHW ? input->sizeAt(3) : input->sizeAt(2));

    if (!isNCHW) {
        input  = input->permute({0, 3, 1, 2});                // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
        output = output->permute({0, 3, 1, 2});               // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
    }

    ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    if (isSameMode)
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);            
            
    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - poolingMode; 9 - divisor;    
    T extraParams[] = {static_cast<T>(kH), static_cast<T>(kW), static_cast<T>(sH), static_cast<T>(sW), static_cast<T>(pH), static_cast<T>(pW), static_cast<T>(dH), static_cast<T>(dW), static_cast<T>(1.f), static_cast<T>(extraParam0)};
    ConvolutionUtils<T>::pooling2d(*input, *output, extraParams);

    if (!isNCHW) {
        delete input;
        delete output;
    }

    return Status::OK();
}

DECLARE_SYN(AvgPool2D, avgpool2d);
DECLARE_SYN(AvgPool, avgpool2d);
DECLARE_SYN(avgpool, avgpool2d);

DECLARE_SHAPE_FN(avgpool2d) {
    
    auto inShape = inputShape->at(0);
    auto shapeOf = shape::shapeOf(inShape);

    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;
    std::vector<int> argI = *(block.getIArguments());
    const int kH = INT_ARG(0);
    const int kW = INT_ARG(1);
    const int sH = INT_ARG(2);
    const int sW = INT_ARG(3);
    const int pH = INT_ARG(4);
    const int pW = INT_ARG(5);
    const int dH = INT_ARG(6);
    const int dW = INT_ARG(7);
    const int isSameMode = INT_ARG(8);

    const int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;       // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

    const int bS = shapeOf[0];
    const int iD = isNCHW ? shapeOf[1] : shapeOf[3];
    const int iH = isNCHW ? shapeOf[2] : shapeOf[1];
    const int iW = isNCHW ? shapeOf[3] : shapeOf[2];

    const char order = shape::order(inShape); // output order must be equal to input order

    // calculate output Height/Width
    int oH, oW;
    ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

    // allocate memory for new shape
    Nd4jLong* newShapeInfo = nullptr;
    ALLOCATE(newShapeInfo, block.getWorkspace(), 12, Nd4jLong);
    if (isNCHW) {
        newShapeInfo[0] = 4;        // rank
        newShapeInfo[1] = bS;
        newShapeInfo[2] = iD;
        newShapeInfo[3] = oH;
        newShapeInfo[4] = oW;
    } else {
        newShapeInfo[0] = 4;        // rank
        newShapeInfo[1] = bS;
        newShapeInfo[2] = oH;
        newShapeInfo[3] = oW;
        newShapeInfo[4] = iD;
    }
    shape::updateStrides(newShapeInfo, order);

    return SHAPELIST(newShapeInfo);
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool2d_bp, 2, 1, false, 0, 10) {

    NDArray<T>* input = INPUT_VARIABLE(0);                          // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    NDArray<T>* gradO = INPUT_VARIABLE(1);                          // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next
    NDArray<T>* gradI = OUTPUT_VARIABLE(0);                         // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon

    int kH = INT_ARG(0);                                                        // filter(kernel) height
    int kW = INT_ARG(1);                                                        // filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int extraParam0 = INT_ARG(9);
    int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;         // 0-NHWC, 1-NCHW    

    REQUIRE_TRUE(input->rankOf() == 4, 0, "AVGPOOL2D_BP op: input should have rank of 4, but got %i instead", input->rankOf());
    REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D_BP op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::string expectedGradOShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,oH,oW,  0,indIOioC,indIiH,indIiH+1}));
    std::string expectedGradIShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,iH,iW,  0,indIOioC,indIiH,indIiH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradO), 0, "AVGPOOL2D_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradO).c_str());    
    REQUIRE_TRUE(expectedGradIShape == ShapeUtils<T>::shapeAsString(gradI), 0, "AVGPOOL2D_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !", expectedGradIShape.c_str(), ShapeUtils<T>::shapeAsString(gradI).c_str());

    if(!isNCHW) {
        gradI = gradI->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradO = gradO->permute({0, 3, 1, 2});                                   // [bS, oH, oW, iC] -> [bS, iC, oH, oW]                        
    }
    
    if(isSameMode)                       // SAME        
        ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    // NDArray<T> columnsWrongShape(input->ordering(), {bS, iC, oH, oW, kH, kW}, input->getWorkspace());    
    // NDArray<T>* columns = columnsWrongShape.permute({0, 1, 4, 5, 2, 3});                                // [bS, iC, oH, oW, kH, kW] -> [bS, iC, kH, kW, oH, oW]
    // NDArray<T>* gradOVector = gradO->reshape('c', {(int) gradO->lengthOf(), 1}); 
    // NDArray<T>* columns2d = columnsWrongShape.reshape('c', {bS*iC*oH*oW, kH*kW});
    
    // columns2d->addiColumnVector(gradOVector);

    // columns->template applyTransform<simdOps::Col2Im<T>>(gradI, std::vector<T>({(T)sH, (T)sW, (T)pH, (T)pW, (T)iH, (T)iW, (T)dH, (T)dW}).data());

    // *gradI /= kH*kW; 
        
    NDArray<T> temp;    // does not mean anything, just to fit pooling2dBP signature
    // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - poolingMode; 9 - divisor;
    std::vector<T> argT = {(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T)dW, 1., (T)extraParam0};    
    ConvolutionUtils<T>::pooling2dBP(temp, *gradO, *gradI, argT.data());

    if(!isNCHW) {
        delete gradI;
        delete gradO;
    }
    // delete columns;
    // delete columns2d;
    // delete gradOVector;
    
    return Status::OK();

}

DECLARE_SHAPE_FN(avgpool2d_bp) {
                
    REQUIRE_TRUE(inputShape->at(0)[0] == 4, 0, "AVGPOOL2D_BP op: input array must be 4D, but got %i instead!", inputShape->at(0)[0]);
    REQUIRE_TRUE(inputShape->at(1)[0] == 4, 0, "AVGPOOL2D_BP op: output's gradient array (next epsilon) must be 4D, but got %i instead!", inputShape->at(1)[0]);
    
    Nd4jLong* gradIShapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIShapeInfo);
    
    return SHAPELIST(gradIShapeInfo);
}


}
}

#endif