//
// @author raver119@gmail.com, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com), changed on 14.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_pnormpool2d)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(pnormpool2d, 1, 1, false, 0, 10) {

            REQUIRE_OK(this->validateInputLengthMatch(block));
            REQUIRE_OK(this->validateInputDimensionsMatch(block));
            NDArray<T>* input = INPUT_VARIABLE(0);
            NDArray<T>* output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "PNORMPOOL2D op: input should have rank of 4, but got %i instead", input->rankOf());

            int kY = INT_ARG(0);
            int kX = INT_ARG(1);
            int sY = INT_ARG(2);
            int sX = INT_ARG(3);
            int pY = INT_ARG(4);
            int pX = INT_ARG(5);
            int dY = INT_ARG(6);
            int dX = INT_ARG(7);
            bool isSameMode = INT_ARG(8);
            int extraParam0 = INT_ARG(9);

            REQUIRE_TRUE(dY != 0 && dX != 0, 0, "PNORMPOOL2D op: dilation must not be zero, but got instead {%i, %i}", dY, dX);

            int oY = 0;
            int oX = 0;

            int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;       // 0-NDHWC, 1-NCDHW    

            if (!isNCHW) {
                input  = input->permute({0, 3, 1, 2});                  // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
                output = output->permute({0, 3, 1, 2});                 // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
            }

            const int inY = input->sizeAt(2);
            const int inX = input->sizeAt(3);

            ConvolutionUtils<T>::calcOutSizePool2D(oY, oX, kY, kX, sY, sX, pY, pX, dY, dX, inY, inX, isSameMode);

            if (isSameMode)
                ConvolutionUtils<T>::calcPadding2D(pY, pX, oY, oX, inY, inX, kY, kX, sX, pY, dY, dX);

            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - poolingMode; 9 - divisor;
            std::vector<T> argT = {(T) kY, (T) kX, (T) sY, (T) sX, (T) pY, (T) pX, (T) dY, (T)dX, 2., (T)extraParam0};
            ConvolutionUtils<T>::pooling2d(*input, *output, argT.data());

            if (!isNCHW) {
                delete input;
                delete output;
            }

            return Status::OK();
        }
        DECLARE_SYN(PnormPool2D, pnormpool2d);
        DECLARE_SYN(PnormPool, pnormpool2d);
        DECLARE_SYN(pnormpool, pnormpool2d);


        DECLARE_SHAPE_FN(pnormpool2d) {
            auto inShape = inputShape->at(0);
            auto shapeOf = shape::shapeOf(inShape);

            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;
            std::vector<int> argI = *(block.getIArguments());
            int kH = INT_ARG(0);
            int kW = INT_ARG(1);
            int sH = INT_ARG(2);
            int sW = INT_ARG(3);
            int pH = INT_ARG(4);
            int pW = INT_ARG(5);
            int dH = INT_ARG(6);
            int dW = INT_ARG(7);
            int isSameMode = INT_ARG(8);
            int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;       // 0-NDHWC, 1-NCDHW    

            REQUIRE_TRUE(dH != 0 && dW != 0, 0, "PNORMPOOL2D op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

            int bS = shapeOf[0];
            int iC = isNCHW ? shapeOf[1] : shapeOf[3];
            int iH = isNCHW ? shapeOf[2] : shapeOf[1];
            int iW = isNCHW ? shapeOf[3] : shapeOf[2];
            char order = shape::order(inShape); // output order must be equal to input order

            // calculate output Height/Width
            int oH, oW;
            ConvolutionUtils<T>::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);
            // allocate memory for new shape
            Nd4jLong* newShapeInfo = nullptr;
            ALLOCATE(newShapeInfo, block.getWorkspace(), 12, Nd4jLong);

            newShapeInfo[0] = 4;        // rank
            newShapeInfo[1] = bS;
            if (isNCHW) {
                newShapeInfo[2] = iC;
                newShapeInfo[3] = oH;
                newShapeInfo[4] = oW;
            } else {
                newShapeInfo[2] = oH;
                newShapeInfo[3] = oW;
                newShapeInfo[4] = iC;
            }
            shape::updateStrides(newShapeInfo, order);

            return SHAPELIST(newShapeInfo);
        }

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(pnormpool2d_bp, 2, 1, false, 1, 10) {

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
    int pnorm = INT_ARG(9);
    int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;           // 0-NHWC, 1-NCHW    

    T eps = T_ARG(0);

    REQUIRE_TRUE(input->rankOf() == 4, 0, "PNORMPOOL2D_BP op: input should have rank of 4, but got %i instead", input->rankOf());
    REQUIRE_TRUE(dH != 0 && dW != 0, 0, "PNORMPOOL2D_BP op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

    std::string expectedGradOShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,oH,oW,  0,indIOioC,indIiH,indIiH+1}));
    std::string expectedGradIShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,iH,iW,  0,indIOioC,indIiH,indIiH+1}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradO), 0, "PNORMPOOL2D_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradO).c_str());    
    REQUIRE_TRUE(expectedGradIShape == ShapeUtils<T>::shapeAsString(gradI), 0, "PNORMPOOL2D_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !", expectedGradIShape.c_str(), ShapeUtils<T>::shapeAsString(gradI).c_str());

    if(!isNCHW) {
        input = input->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradI = gradI->permute({0, 3, 1, 2});                                   // [bS, iH, iW, iC] -> [bS, iC, iH, iW]                        
        gradO = gradO->permute({0, 3, 1, 2});                                   // [bS, oH, oW, iC] -> [bS, iC, oH, oW]                        
    }
    
    // if(isSameMode)                       // SAME        
    //     ConvolutionUtils<T>::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

    // NDArray<T> columnsWrongShape(input->ordering(), {bS, iC, oH, oW, kH, kW}, input->getWorkspace());    
    // NDArray<T>* columns = columnsWrongShape.permute({0, 1, 4, 5, 2, 3});                                // [bS, iC, oH, oW, kH, kW] -> [bS, iC, kH, kW, oH, oW]
    // NDArray<T>* gradOVector = gradO->reshape('c', {(int) gradO->lengthOf(), 1}); 
    // NDArray<T>* columns2d = columnsWrongShape.reshape('c', {bS*iC*oH*oW, kH*kW});
    // NDArray<T> pNorm(columns2d->getShapeInfo(), block.getWorkspace());    

    // input->template applyTransform<simdOps::Im2col<T>>(columns, std::vector<T>({(T)kH, (T)kW, (T)sH, (T)sW, (T)pH, (T)pW, (T)dH, (T)dW, (T)0.f, (T)0.f}).data());
    
    // columns2d->template applyTransform<simdOps::Abs<T>>(&pNorm);
    // pNorm.template applyTransform<simdOps::Pow<T>>(&pNorm, std::vector<T>({(T)pnorm}).data());

    // NDArray<T>* denomVec = pNorm.sum({1});    
    // denomVec->template applyTransform<simdOps::Pow<T>>(std::vector<T>({(T)1. - (T)1. / pnorm}).data());    
    // denomVec->template applyScalar<simdOps::Max<T>>(eps); // in case of 0    
    // denomVec->template applyPairwiseTransform<simdOps::ReverseDivide<T>>(gradOVector, denomVec, nullptr);

    // if(pnorm != 2) {
    //     T extraParams[] = {(T)1. - (T)2. / pnorm};
    //     pNorm.template applyTransform<simdOps::Pow<T>>(std::vector<T>({(T)1. - (T)2. / pnorm}).data());
    //     *columns2d *= pNorm;
    // }    
    
    // columns2d->muliColumnVector(denomVec);
    
    // columns->template applyTransform<simdOps::Col2Im<T>>(gradI, std::vector<T>({(T)sH, (T)sW, (T)pH, (T)pW, (T)iH, (T)iW, (T)dH, (T)dW}).data());
    
    std::vector<T> argT = {(T) kH, (T) kW, (T) sH, (T) sW, (T) pH, (T) pW, (T) dH, (T)dW, 2., (T)pnorm};
    ConvolutionUtils<T>::pooling2dBP(*input, *gradO, *gradI, argT.data());

    if(!isNCHW) {
        delete input;
        delete gradI;
        delete gradO;
    }
    // delete columns;
    // delete columns2d;
    // delete gradOVector;
    // delete denomVec;
    
    return Status::OK();
}

DECLARE_SHAPE_FN(pnormpool2d_bp) {
                
    REQUIRE_TRUE(inputShape->at(0)[0] == 4, 0, "PNORMPOOL2D_BP op: input array must be 4D, but got %i instead!", inputShape->at(0)[0]);
    REQUIRE_TRUE(inputShape->at(1)[0] == 4, 0, "PNORMPOOL2D_BP op: output's gradient array (next epsilon) must be 4D, but got %i instead!", inputShape->at(1)[0]);
    
    Nd4jLong* gradIShapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIShapeInfo);
    
    return SHAPELIST(gradIShapeInfo);
}

}
}

#endif