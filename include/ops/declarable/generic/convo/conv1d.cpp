//
//  @author raver119@gmail.com and Yurii Shyrma
//  @author Yurii Shyrma


#include <ops/declarable/CustomOperations.h>
#include <declarable/generic/helpers/convolutions.h>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(conv1d, 2, 1, false, 0, 4) {

    NDArray<T> *input   = INPUT_VARIABLE(0);                                    // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                    // [kW, iC, oC] (NWC) or [oC, iC, kW] (NCW)
    NDArray<T> *bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    NDArray<T> *output  = OUTPUT_VARIABLE(0);                                   // [bS, oW, oC] (NWC) or [bS, oC, oW] (NCW)
    
    REQUIRE_TRUE(input->rankOf()   == 3, 0, "CUSTOM CONV1D OP: rank of input array must be equal to 3, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 3, 0, "CUSTOM CONV1D OP: rank of weights array must be equal to 3, but got %i instead !", weights->rankOf());
                                         
    int kW = INT_ARG(0);                                                        // filter(kernel) width
    int sW = INT_ARG(1);                                                        // strides width
    int pW = INT_ARG(2);                                                        // paddings width
    int isSameMode = INT_ARG(3);                                                // 0-VALID, 1-SAME
    int isNCW      = block.getIArguments()->size() > 4 ? !INT_ARG(4) : 1;       // 0-NCH,  1-NHC
    
    std::vector<int> reshapeForInput, reshapeForOutput, reshapeForWeights;
    if(!isNCW) {
        reshapeForInput   = {input->sizeAt(0), 1, input->sizeAt(1), input->sizeAt(2)};                  // [bS, iW, iC] -> [bS, 1, iW, iC]
        reshapeForOutput  = {output->sizeAt(0), 1, output->sizeAt(1), output->sizeAt(2)};               // [bS, oW, oC] -> [bS, 1, oW, oC]
        reshapeForWeights = {1, weights->sizeAt(0), weights->sizeAt(1), weights->sizeAt(2)};            // [kW, iC, oC] -> [1, kW, iC, oC]
    }
    else {
        reshapeForInput   = {input->sizeAt(0),  input->sizeAt(1),  1, input->sizeAt(2)};                // [bS, iC, iW] -> [bS, iC, 1, iW] 
        reshapeForOutput  = {output->sizeAt(0), output->sizeAt(1), 1, output->sizeAt(2)};               // [bS, oC, oW] -> [bS, oC, 1, oW] 
        reshapeForWeights = {weights->sizeAt(0),weights->sizeAt(1),1, weights->sizeAt(2)};              // [oC, iC, kW] -> [oC, iC, 1, kW]
    }
    
    NDArray<T>* inputReshaped   = input  ->reshape(input->ordering(),   reshapeForInput);
    NDArray<T>* outputReshaped  = output ->reshape(output->ordering(),  reshapeForOutput);
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(), reshapeForWeights);

    nd4j::ops::conv2d<T> conv2d;
    const Nd4jStatus status = conv2d.execute({inputReshaped, weightsReshaped, bias}, {outputReshaped}, {}, {1,kW,  1,sW,  0,pW,  1,1,  isSameMode,  !isNCW});             
    if (status != ND4J_STATUS_OK)
        return status;

    delete inputReshaped;
    delete outputReshaped;
    delete weightsReshaped;

    return ND4J_STATUS_OK;
}
        

DECLARE_SHAPE_FN(conv1d) {

    int kW = INT_ARG(0);                                                        // filter(kernel) width
    int sW = INT_ARG(1);                                                        // strides width
    int pW = INT_ARG(2);                                                        // paddings width
    int isSameMode = INT_ARG(3);                                                // 0-VALID, 1-SAME
    int isNCH  = block.getIArguments()->size() > 4 ? !INT_ARG(4) : 1;           // 0-NWC, 1-NCW

    int indIW = isNCH == 1 ? 2 : 1;
    int indIC = isNCH == 1 ? 1 : 2;
    int indOC = isNCH == 1 ? 0 : 2;

    int* inputShapeInfo   = inputShape->at(0);
    int* weightsShapeInfo = inputShape->at(1);

    int bS = inputShapeInfo[1];                         // batch size
    int iW = inputShapeInfo[indIW+1];                   // input width
    int iC = inputShapeInfo[indIC+1];                   // input channels        
    int oC = weightsShapeInfo[indOC+1];                 // output channels

    int oH, oW;                                         // output height, width
    ConvolutionUtils<T>::calcOutSizePool2D(oH,oW,  1,kW,  1,sW,  0,pW,  1,1,  1,iW, isSameMode);    
    
    int* outputShapeInfo = nullptr;
    ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inputShapeInfo), int);

    outputShapeInfo[0] = 3;
    outputShapeInfo[1] = bS;

    if (isNCH) {
        outputShapeInfo[2] = oC;
        outputShapeInfo[3] = oW;
    } else {
        outputShapeInfo[2] = oW;        
        outputShapeInfo[3] = oC;
    }
    
    shape::updateStrides(outputShapeInfo, shape::order(inputShapeInfo));

    return SHAPELIST(outputShapeInfo);
}



////////////////////////////////////////////////////////////////////////// 
CUSTOM_OP_IMPL(conv1d_bp, 3, 2, false, 0, 4) {
    
    NDArray<T> *input   = INPUT_VARIABLE(0);                                                // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW)
    NDArray<T> *weights = INPUT_VARIABLE(1);                                                // [kW, iC, oC] (NWC) or [oC, iC, kW] (NCW)
    NDArray<T> *bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    NDArray<T> *gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oW, oC] (NWC) or [bS, oC, oW] (NCW), epsilon_next
    
    NDArray<T> *gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iW, iC] (NWC) or [bS, iC, iW] (NCW), epsilon
    NDArray<T> *gradW = OUTPUT_VARIABLE(1);                                                 // [kW, iC, oC] (NWC) or [oC, iC, kW] (NCW)
    NDArray<T> *gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]
    
    REQUIRE_TRUE(input->rankOf()   == 3, 0, "CUSTOM CONV1D_BP OP: rank of input array must be equal to 3, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 3, 0, "CUSTOM CONV1D_BP OP: rank of weights array must be equal to 3, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf()   == 3, 0, "CUSTOM CONV1D_BP OP: rank of gradO array must be equal to 3, but got %i instead !", gradO->rankOf());
                                     
    int kW = INT_ARG(0);                                                        // filter(kernel) width
    int sW = INT_ARG(1);                                                        // strides width
    int pW = INT_ARG(2);                                                        // paddings width
    int isSameMode = INT_ARG(3);                                                // 0-VALID, 1-SAME
    int isNCW  = block.getIArguments()->size() > 4 ? !INT_ARG(4) : 1;          // 0-NWC, 1-NCW    

    std::vector<int> reshapeForInput, reshapeForGradO, reshapeForWeights;
    if(!isNCW) {
        reshapeForInput   = {input->sizeAt(0), 1, input->sizeAt(1), input->sizeAt(2)};                  // [bS, iW, iC] -> [bS, 1, iW, iC]
        reshapeForGradO   = {gradO->sizeAt(0), 1, gradO->sizeAt(1), gradO->sizeAt(2)};                  // [bS, oW, oC] -> [bS, 1, oW, oC]
        reshapeForWeights = {1, weights->sizeAt(0), weights->sizeAt(1), weights->sizeAt(2)};            // [kW, iC, oC] -> [1, kW, iC, oC]
    }
    else {
        reshapeForInput   = {input->sizeAt(0), input->sizeAt(1), 1, input->sizeAt(2)};                  // [bS, iC, iW] -> [bS, iC, 1, iW] 
        reshapeForGradO   = {gradO->sizeAt(0), gradO->sizeAt(1), 1, gradO->sizeAt(2)};                  // [bS, oC, oW] -> [bS, oC, 1, oW] 
        reshapeForWeights = {weights->sizeAt(0),weights->sizeAt(1), 1, weights->sizeAt(2)};             // [oC, iC, kW] -> [oC, iC, 1, kW]
    }
    
    NDArray<T>* inputReshaped   = input  ->reshape(input->ordering(),  reshapeForInput);
    NDArray<T>* gradIReshaped   = gradI  ->reshape(gradI->ordering(),  reshapeForInput);
    NDArray<T>* gradOReshaped   = gradO  ->reshape(gradO->ordering(),  reshapeForGradO);
    NDArray<T>* weightsReshaped = weights->reshape(weights->ordering(),reshapeForWeights);
    NDArray<T>* gradWReshaped   = gradW  ->reshape(gradW->ordering(),  reshapeForWeights);

    nd4j::ops::conv2d_bp<T> conv2d_bp;    
    const Nd4jStatus status = conv2d_bp.execute({inputReshaped, weightsReshaped, bias, gradOReshaped}, {gradIReshaped, gradWReshaped, gradB}, {}, {1,kW,  1,sW,  0,pW,  1,1,  isSameMode,  !isNCW});    
    if (status != ND4J_STATUS_OK)
        return status;

    delete inputReshaped;  
    delete gradIReshaped;  
    delete gradOReshaped;  
    delete weightsReshaped;
    delete gradWReshaped;  

    return ND4J_STATUS_OK;
}

DECLARE_SHAPE_FN(conv1d_bp) {


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