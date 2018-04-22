//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.03.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_avgpool3dnew)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew, 1, 1, false, 0, 10) {
    
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
    int isSameMode = INT_ARG(9);                                                // 1-SAME,  0-VALID
    int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;         // 1-NDHWC, 0-NCDHW    

    REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM AVGPOOL3D OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());

    int idxID, idxIC;
    if(isNCHW) { idxID = 2; idxIC = 1;}
    else       { idxID = 1; idxIC = 4;}

    int bS = input->sizeAt(0);                  // batch size
    int iC = input->sizeAt(idxIC);              // input channels        
    int iD = input->sizeAt(idxID);              // input depth
    int iH = input->sizeAt(idxID+1);            // input height
    int iW = input->sizeAt(idxID+2);            // input width    
    int oD = output->sizeAt(idxID);             // output depth
    int oH = output->sizeAt(idxID+1);           // output height
    int oW = output->sizeAt(idxID+2);           // output width                
    
    REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "CUSTOM AVGPOOL3D OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", iD,iH,iW, kD,kH,kW);    
    REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "CUSTOM AVGPOOL3D OP: pad depth/height/width must not be greater than half of kernel depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", pD,pH,pW, kD,kH,kW);    

    if(!isNCHW) {
        input = input ->permute({0, 4, 1, 2, 3});                                                       // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
        output = new NDArray<T>(output->ordering(), {bS, iC, oD, oH, oW}, block.getWorkspace());                                // [bS, iC, oD, oH, oW]

        input ->streamline('c');        
    }

    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, 1, 1, 1);    

    int iStride = iC * iD * iH * iW;
    int oStride = iC * oD * oH * oW;
    
    for(int i = 0; i < bS; ++i)   
        ConvolutionUtils<T>::_avgPool3D(input->getBuffer() + i*iStride, output->getBuffer() + i*oStride, iC, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, !isSameMode);
   
    if(!isNCHW) {              

        output->permutei({0, 2, 3, 4, 1});                                 // [bS, iC, oD, oH, oW] -> [bS, oD, oH, oW, iC]
        OUTPUT_VARIABLE(0)->assign(output);
        
        delete input;
        delete output;
    }
        
    return Status::OK();
}


DECLARE_SHAPE_FN(avgpool3dnew) {

    int kD = INT_ARG(0);                                                        // filter(kernel) depth
    int kH = INT_ARG(1);                                                        // filter(kernel) height
    int kW = INT_ARG(2);                                                        // filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int isSameMode = INT_ARG(9);                                                // 1-SAME,  0-VALID;
    int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;        // 1-NDHWC, 0-NCDHW
    
    int* inputShapeInfo = inputShape->at(0);

    int idxID, idxIC;    
    if(isNCHW) { idxID = 2; idxIC = 1;}
    else       { idxID = 1; idxIC = 4;}

    int bS = inputShapeInfo[1];                          // batch size
    int iC = inputShapeInfo[idxIC+1];                    // input channels            
    int iD = inputShapeInfo[idxID+1];                    // input depth
    int iH = inputShapeInfo[idxID+2];                    // input height
    int iW = inputShapeInfo[idxID+3];                    // input width

    int oD, oH, oW;                         // output depth, height, width
    ConvolutionUtils<T>::calcOutSizePool3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, 1, 1, 1, iD, iH, iW, isSameMode);
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    if (isNCHW) {
        outputShapeInfo[0] = 5;
        outputShapeInfo[1] = bS;
        outputShapeInfo[2] = iC;
        outputShapeInfo[3] = oD;
        outputShapeInfo[4] = oH;
        outputShapeInfo[5] = oW;
    } else {
        outputShapeInfo[0] = 5;
        outputShapeInfo[1] = bS;
        outputShapeInfo[2] = oD;
        outputShapeInfo[3] = oH;
        outputShapeInfo[4] = oW;
        outputShapeInfo[5] = iC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}



//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(avgpool3dnew_bp, 2, 1, false, 0, 10) {
    
    NDArray<T> *input = INPUT_VARIABLE(0);                                                  // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    NDArray<T> *gradO = INPUT_VARIABLE(1);                                                  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    
    REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM AVGPOOL3DNEW_BP OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
                                     
    int kD = INT_ARG(0);                                                        // filter(kernel) depth
    int kH = INT_ARG(1);                                                        // filter(kernel) height
    int kW = INT_ARG(2);                                                        // filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int isSameMode = INT_ARG(9);                                                // 1-SAME,  0-VALID
    int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;         // 1-NDHWC, 0-NCDHW    

    int idxID, idxIC;
    if(isNCHW) { idxID = 2; idxIC = 1;}
    else       { idxID = 1; idxIC = 4;}

    int bS = input->sizeAt(0);                  // batch size
    int iC = input->sizeAt(idxIC);              // input channels        
    int iD = input->sizeAt(idxID);              // input depth
    int iH = input->sizeAt(idxID+1);            // input height
    int iW = input->sizeAt(idxID+2);            // input width    
    int oD = gradO->sizeAt(idxID);              // output depth
    int oH = gradO->sizeAt(idxID+1);            // output height
    int oW = gradO->sizeAt(idxID+2);            // output width             

    REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "CUSTOM AVGPOOL3D_BP OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", iD,iH,iW, kD,kH,kW);    
    REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "CUSTOM AVGPOOL3D_BP OP: pad depth/height/width must not be greater than half of kernel depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", pD,pH,pW, kD,kH,kW);    
      
    if(!isNCHW) {
        gradO = gradO ->permute({0, 4, 1, 2, 3});                                                       // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
        gradI = new NDArray<T>(gradI->ordering(), {bS, iC, iD, iH, iW}, block.getWorkspace());                                  // [bS, iC, iD, iH, iW]

        gradO->streamline('c');       
    }
   
    if(isSameMode)                       // SAME
        ConvolutionUtils<T>::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, 1, 1, 1);    

    int iStride = iC * iD * iH * iW;
    int oStride = iC * oD * oH * oW;
    
    for(int i = 0; i < bS; ++i)   
        ConvolutionUtils<T>::_avgPool3D_bp(gradI->getBuffer() + i*iStride, gradO->getBuffer() + i*oStride, iC, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, !isSameMode);

    if(!isNCHW) {              

        gradI->permutei({0, 2, 3, 4, 1});                                 // [bS, iC, iD, iH, iW] -> [bS, iD, iH, iW, iC]
        OUTPUT_VARIABLE(0)->assign(gradI);
        
        delete gradO;
        delete gradI;
    }        

    return Status::OK();
}


DECLARE_SHAPE_FN(avgpool3dnew_bp) {

    int* gradIshapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), gradIshapeInfo);
        
    return SHAPELIST(gradIshapeInfo);        
}



}
}

#endif