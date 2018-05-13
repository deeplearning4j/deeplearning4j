//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 19.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_maxpool3dnew)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(maxpool3dnew, 1, 1, false, 0, 13) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)
                                     
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
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(input->rankOf() == 5, 0, "MAXPOOL3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());    

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::string expectedOutputShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,oD,oH,oW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    REQUIRE_TRUE(expectedOutputShape == ShapeUtils<T>::shapeAsString(output), 0, "MAXPOOL3D op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !", expectedOutputShape.c_str(), ShapeUtils<T>::shapeAsString(output).c_str());
    REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "MAXPOOL3D OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", iD,iH,iW, kD,kH,kW);    
    REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "MAXPOOL3D OP: pad depth/height/width must not be greater than half of kernel depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", pD,pH,pW, kD,kH,kW);    
    
    if(!isNCDHW) {
        input = input ->permute({0, 4, 1, 2, 3});                                                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
        output = new NDArray<T>(output->ordering(), {bS, iC, oD, oH, oW}, block.getWorkspace());
        input->streamline('c');        
    }    

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);    
    
    // input [bS, iC, iD, iH, iW], output [bS, iC, oD, oH, oW]
    ConvolutionUtils<T>::maxPool3d(*input, *output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW);
   
    if(!isNCDHW) {              
        output->permutei({0, 2, 3, 4, 1});                                 // [bS, iC, oD, oH, oW] -> [bS, oD, oH, oW, iC]
        OUTPUT_VARIABLE(0)->assign(output);
        
        delete input;
        delete output;
    }
        
    return Status::OK();
}
// //////////////////////////////////////////////////////////////////////////
// CUSTOM_OP_IMPL(maxpool3dnew, 1, 1, false, 0, 10) {
    
//     NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
//     NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)
                                     
//     int kD = INT_ARG(0);                                                        // filter(kernel) depth
//     int kH = INT_ARG(1);                                                        // filter(kernel) height
//     int kW = INT_ARG(2);                                                        // filter(kernel) width
//     int sD = INT_ARG(3);                                                        // strides depth
//     int sH = INT_ARG(4);                                                        // strides height
//     int sW = INT_ARG(5);                                                        // strides width
//     int pD = INT_ARG(6);                                                        // paddings depth
//     int pH = INT_ARG(7);                                                        // paddings height
//     int pW = INT_ARG(8);                                                        // paddings width
//     int isSameMode = INT_ARG(9);                                                // 1-SAME,  0-VALID
//     int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;        // 0-NDHWC, 1-NCDHW    

//     REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM MAXPOOL3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());    

//     int idxID, idxIC;
//     if(isNCHW) { idxID = 2; idxIC = 1;}
//     else       { idxID = 1; idxIC = 4;}

//     int bS = input->sizeAt(0);                  // batch size
//     int iC = input->sizeAt(idxIC);              // input channels        
//     int iD = input->sizeAt(idxID);              // input depth
//     int iH = input->sizeAt(idxID+1);            // input height
//     int iW = input->sizeAt(idxID+2);            // input width    
//     int oD = output->sizeAt(idxID);             // output depth
//     int oH = output->sizeAt(idxID+1);           // output height
//     int oW = output->sizeAt(idxID+2);           // output width                

//     REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "CUSTOM MAXPOOL3D OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", iD,iH,iW, kD,kH,kW);    
//     REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "CUSTOM MAXPOOL3D OP: pad depth/height/width must not be greater than half of kernel depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", pD,pH,pW, kD,kH,kW);    
    
//     if(!isNCHW) {
//         input = input ->permute({0, 4, 1, 2, 3});                                                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
//         output = new NDArray<T>(output->ordering(), {bS, iC, oD, oH, oW}, block.getWorkspace());                                // [bS, iC, oD, oH, oW]

//         input->streamline('c');        
//     }    
    
//     if(isSameMode)                       // SAME
//         ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, 1, 1, 1);    
    
//     // input [bS, iC, iD, iH, iW], output [bS, iC, oD, oH, oW]
//     ConvolutionUtils<T>::maxPool3d(*input, *output, kD, kH, kW, sD, sH, sW, pD, pH, pW, 1, 1, 1);
   
//     if(!isNCHW) {              

//         output->permutei({0, 2, 3, 4, 1});                                 // [bS, iC, oD, oH, oW] -> [bS, oD, oH, oW, iC]
//         OUTPUT_VARIABLE(0)->assign(output);
        
//         delete input;
//         delete output;
//     }
        
//     return Status::OK();
// }


DECLARE_SHAPE_FN(maxpool3dnew) {

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
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 0-NDHWC, 1-NCDHW    
    
    int* inputShapeInfo = inputShape->at(0);

    int idxID, idxIC;    
    if(isNCDHW) { idxID = 2; idxIC = 1;}
    else       { idxID = 1; idxIC = 4;}

    int bS = inputShapeInfo[1];                          // batch size
    int iC = inputShapeInfo[idxIC+1];                    // input channels            
    int iD = inputShapeInfo[idxID+1];                    // input depth
    int iH = inputShapeInfo[idxID+2];                    // input height
    int iW = inputShapeInfo[idxID+3];                    // input width

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    outputShapeInfo[0] = 5;
    outputShapeInfo[1] = bS;

    if (isNCDHW) {    
        outputShapeInfo[2] = iC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = iC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}
//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(maxpool3dnew_bp, 2, 1, false, 0, 13) {

    NDArray<T>* input = INPUT_VARIABLE(0);                          // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T>* gradO = INPUT_VARIABLE(1);                          // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
    NDArray<T>* gradI = OUTPUT_VARIABLE(0);                         // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon

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
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // 0-NDHWC, 1-NCDHW    

    REQUIRE_TRUE(input->rankOf() == 5, 0, "MAXPOOL3D_BP op: input should have rank of 5, but got %i instead", input->rankOf());    

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
    ConvolutionUtils<T>::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::string expectedGradOShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,oD,oH,oW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    std::string expectedGradIShape = ShapeUtils<T>::shapeAsString(ShapeUtils<T>::composeShapeUsingDimsAndIdx({bS,iC,iD,iH,iW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2}));
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils<T>::shapeAsString(gradO), 0, "MAXPOOL3D_BP op: wrong shape of output's gradients array (next epsilon), expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils<T>::shapeAsString(gradO).c_str());    
    REQUIRE_TRUE(expectedGradIShape == ShapeUtils<T>::shapeAsString(gradI), 0, "MAXPOOL3D_BP op: wrong shape of input's gradients array (epsilon), expected is %s, but got %s instead !", expectedGradIShape.c_str(), ShapeUtils<T>::shapeAsString(gradI).c_str());

    if(!isNCDHW) {
        input = input->permute({0, 4, 1, 2, 3});                                   // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
        gradI = gradI->permute({0, 4, 1, 2, 3});                                   // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]                        
        gradO = gradO->permute({0, 4, 1, 2, 3});                                   // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]                        
    }

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);    
    
    NDArray<T> columnsWrongShape(input->ordering(), {bS, iC, oD, oH, oW, kD, kH, kW}, input->getWorkspace());    
    NDArray<T>* columns = columnsWrongShape.permute({0, 1, 5, 6, 7, 2, 3, 4});                      // [bS, iC, oD, oH, oW, kD, kH, kW] -> [bS, iC, kD, kH, kW, oD, oH, oW]

    ConvolutionUtils<T>::vol2col(*input, *columns, sD, sH, sW, pD, pH, pW, dD, dH, dW);                 // [bS, iC, iD, iH, iW] is convoluted to [bS, iC, kD, kH, kW, oD, oH, oW]        

    NDArray<T>* columns2d = columnsWrongShape.reshape('c', {bS*iC*oD*oH*oW, kD*kH*kW});
    NDArray<T>* gradOVector = gradO->reshape('c', {(int) gradO->lengthOf(), 1}); 
    T extraParams[] = {(T)1., (T)1.};
    columns2d->template applyTransform<simdOps::IsMax<T>>(extraParams);
    columns2d->muliColumnVector(gradOVector);

    ConvolutionUtils<T>::col2vol(*columns, *gradI, sD, sH, sW, pD, pH, pW, dD, dH, dW);                     // columns [bS, iC, kD, kH, kW, oD, oH, oW] is de-convoluted to  [bS, iC, iD, iH, iW]

    if(!isNCDHW) {
        delete input;
        delete gradI;
        delete gradO;
    }
    delete columns;
    delete columns2d;
    delete gradOVector;
    
    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
// CUSTOM_OP_IMPL(maxpool3dnew_bp, 2, 1, false, 0, 10) {
    
//     NDArray<T> *input = INPUT_VARIABLE(0);                                                  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
//     NDArray<T> *gradO = INPUT_VARIABLE(1);                                                  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
    
//     NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    
//     REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM MAXPOOL3D_BP OP: rank of input array must be equal to 5 !", input->rankOf());    
                                     
//     int kD = INT_ARG(0);                                                        // filter(kernel) depth
//     int kH = INT_ARG(1);                                                        // filter(kernel) height
//     int kW = INT_ARG(2);                                                        // filter(kernel) width
//     int sD = INT_ARG(3);                                                        // strides depth
//     int sH = INT_ARG(4);                                                        // strides height
//     int sW = INT_ARG(5);                                                        // strides width
//     int pD = INT_ARG(6);                                                        // paddings depth
//     int pH = INT_ARG(7);                                                        // paddings height
//     int pW = INT_ARG(8);                                                        // paddings width
//     int isSameMode = INT_ARG(9);                                                // 1-SAME,  0-VALID
//     int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;        // 0-NDHWC, 1-NCDHW    

//     int idxID, idxIC;
//     if(isNCHW) { idxID = 2; idxIC = 1;}
//     else       { idxID = 1; idxIC = 4;}

//     int bS = input->sizeAt(0);                  // batch size
//     int iC = input->sizeAt(idxIC);              // input channels        
//     int iD = input->sizeAt(idxID);              // input depth
//     int iH = input->sizeAt(idxID+1);            // input height
//     int iW = input->sizeAt(idxID+2);            // input width    
//     int oD = gradO->sizeAt(idxID);              // output depth
//     int oH = gradO->sizeAt(idxID+1);            // output height
//     int oW = gradO->sizeAt(idxID+2);            // output width             

//     REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "CUSTOM MAXPOOL3D_BP OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", iD,iH,iW, kD,kH,kW);    
//     REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "CUSTOM MAXPOOL3D_BP OP: pad depth/height/width must not be greater than half of kernel depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", pD,pH,pW, kD,kH,kW);    

//     if(!isNCHW) {
//         input = input ->permute({0, 4, 1, 2, 3});                                                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
//         gradO = gradO ->permute({0, 4, 1, 2, 3});                                                       // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
//         gradI = new NDArray<T>(gradI->ordering(), {bS, iC, iD, iH, iW}, block.getWorkspace());                                  // [bS, iC, iD, iH, iW]

//         input->streamline('c');       
//         gradO->streamline('c');       
//     }
   
//     if(isSameMode)                       // SAME
//         ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, 1, 1, 1);    

//     int* indices = nullptr;
//     ALLOCATE(indices, block.getWorkspace(), bS*iC*oD*oH*oW, int);

//     // input [bS, iC, iD, iH, iW], indices [bS, iC, oD, oH, oW]
//     ConvolutionUtils<T>::maxPool3dIndices(*input, indices, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, 1, 1, 1);
//     ConvolutionUtils<T>::maxPool3dBP(*gradO, indices, *gradI, kD, kH, kW, sD, sH, sW, pD, pH, pW, 1, 1, 1);

//     if(!isNCHW) {              

//         gradI->permutei({0, 2, 3, 4, 1});                                 // [bS, iC, iD, iH, iW] -> [bS, iD, iH, iW, iC]
//         OUTPUT_VARIABLE(0)->assign(gradI);
        
//         delete gradO;
//         delete gradI;
//         delete input;
//     }        

//     RELEASE(indices, block.getWorkspace());

//     return Status::OK();
// }


DECLARE_SHAPE_FN(maxpool3dnew_bp) {

    int* gradIshapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIshapeInfo);
        
    return SHAPELIST(gradIshapeInfo);        
}



}
}

#endif