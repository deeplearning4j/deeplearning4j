//
// @author Yurii Shyrma, created on 05.04.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/rnn.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(dynamic_rnn, 4, 2, false, 0, 0) {

    NDArray<T>* x  = INPUT_VARIABLE(0);               // input [time x bS x inSize] or [bS x time x inSize], depends on timeMajor parameter
	NDArray<T>* Wx = INPUT_VARIABLE(1);               // input-to-hidden  weights, [inSize  x numUnits] 	
    NDArray<T>* Wh = INPUT_VARIABLE(2);               // hidden-to-hidden weights, [numUnits x numUnits]         
	NDArray<T>* b  = INPUT_VARIABLE(3);               // biases for, [2*numUnits] 

	NDArray<T>* h0          = nullptr;     		      // initial cell output (at time step = 0) [bS x numUnits]	
	NDArray<T>* maxTimeStep = nullptr;			      // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    const int timeMajor = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // if true then [time, bS, ...], else [bS, time, ...]

    if(block.width() == 5) {
        if ((*INPUT_VARIABLE(4)).rankOf() == 2)
            h0 = INPUT_VARIABLE(4);
        else
            maxTimeStep = INPUT_VARIABLE(4);
    }
	else if(block.width() == 6) {
        h0 = INPUT_VARIABLE(4);
        maxTimeStep = INPUT_VARIABLE(5);
    }    
    
    NDArray<T>* h      =  OUTPUT_VARIABLE(0);           // cell outputs [time x bS x numUnits] or [bS x time x numUnits], depends on timeMajor parameter
    NDArray<T>* hFinal =  OUTPUT_VARIABLE(1);           // at the end it will store cell final non-zero output [bS x numUnits]

    REQUIRE_TRUE(x->rankOf() == 3, 0, "DYNAMIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(Wx->rankOf() == 2, 0, "DYNAMIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", Wx->rankOf());    

    const int inRank   = x->rankOf();
    const int time     = timeMajor ? x->sizeAt(0) : x->sizeAt(1);
    const int bS       = timeMajor ? x->sizeAt(1) : x->sizeAt(0);
    const int numUnits = Wx->sizeAt(1);

    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(Wh) == ShapeUtils<T>::shapeAsString({numUnits, numUnits}), 0, "DYNAMIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({numUnits, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(Wh).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(b)  == ShapeUtils<T>::shapeAsString({2*numUnits}), 0, "DYNAMIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnits}).c_str(), ShapeUtils<T>::shapeAsString(b).c_str()); 
    if(h0)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0) == ShapeUtils<T>::shapeAsString({bS, numUnits}), 0, "DYNAMIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(h0).c_str()); 
    if(maxTimeStep)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(maxTimeStep)  == ShapeUtils<T>::shapeAsString({bS}), 0, "DYNAMIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({bS}).c_str(), ShapeUtils<T>::shapeAsString(maxTimeStep).c_str()); 


    if(timeMajor == false) {
        x = x->permute({1, 0, 2});                      // [bS x time x inSize]   -> [time x bS x inSize]
        h = h->permute({1, 0, 2});                      // [bS x time x numUnits] -> [time x bS x numUnits]
    }

    helpers::rnnTimeLoop<T>({x, Wx, Wh, b, h0, maxTimeStep}, h, hFinal);    

    if(timeMajor == false) {
        delete x;
        delete h;
    }
    
    return Status::OK();
}



DECLARE_SHAPE_FN(dynamic_rnn) {    

    auto xShapeInfo  = inputShape->at(0);               // input [time x bS x inSize] or [bS x time x inSize], depends on timeMajor parameter
    auto WxShapeInfo = inputShape->at(1);               // input-to-hidden  weights, [inSize  x numUnits]     
    auto WhShapeInfo = inputShape->at(2);               // hidden-to-hidden weights, [numUnits x numUnits]         
    auto bShapeInfo  = inputShape->at(3);               // biases for, [2*numUnits] 

    Nd4jLong* h0ShapeInfo          = nullptr;                // initial cell output (at time step = 0) [bS x numUnits] 
    Nd4jLong* maxTimeStepShapeInfo = nullptr;                // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    const int timeMajor = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // if true then [time, bS, ...], else [bS, time, ...]

    if(block.width() == 5) {
        if (inputShape->at(4)[0] == 2)
            h0ShapeInfo = inputShape->at(4);
        else
            maxTimeStepShapeInfo = inputShape->at(4);
    }
    else if(block.width() == 6) {
        h0ShapeInfo = inputShape->at(4);
        maxTimeStepShapeInfo = inputShape->at(5);
    }    

    REQUIRE_TRUE(xShapeInfo[0] == 3, 0, "DYNAMIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", xShapeInfo[0]);
    REQUIRE_TRUE(WxShapeInfo[0] == 2, 0, "DYNAMIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", WxShapeInfo[0]);    

    const int inRank   = xShapeInfo[0];
    const int time     = timeMajor ? xShapeInfo[1] : xShapeInfo[2];
    const int bS       = timeMajor ? xShapeInfo[2] : xShapeInfo[1];
    const int numUnits = WxShapeInfo[2];

    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhShapeInfo) == ShapeUtils<T>::shapeAsString({numUnits, numUnits}), 0, "DYNAMIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({numUnits, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(WhShapeInfo).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bShapeInfo)  == ShapeUtils<T>::shapeAsString({2*numUnits}), 0, "DYNAMIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnits}).c_str(), ShapeUtils<T>::shapeAsString(bShapeInfo).c_str()); 
    if(h0ShapeInfo)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0ShapeInfo) == ShapeUtils<T>::shapeAsString({bS, numUnits}), 0, "DYNAMIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnits}).c_str(), ShapeUtils<T>::shapeAsString(h0ShapeInfo).c_str()); 
    if(maxTimeStepShapeInfo)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(maxTimeStepShapeInfo)  == ShapeUtils<T>::shapeAsString({bS}), 0, "DYNAMIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({bS}).c_str(), ShapeUtils<T>::shapeAsString(maxTimeStepShapeInfo).c_str()); 

    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *hPrevShapeInfo(nullptr);
    ALLOCATE(hShapeInfo,     block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    ALLOCATE(hPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), Nd4jLong);
            
    hShapeInfo[0]     = inRank;
    hPrevShapeInfo[0] = inRank-1;
    hShapeInfo[1]     = timeMajor ? time : bS;
    hShapeInfo[2]     = timeMajor ? bS : time;
    hPrevShapeInfo[1] = bS;
    hShapeInfo[3]     = hPrevShapeInfo[2] = numUnits;

    shape::updateStrides(hShapeInfo,     shape::order(xShapeInfo));    
    shape::updateStrides(hPrevShapeInfo, shape::order(xShapeInfo));
         
    return SHAPELIST(hShapeInfo, hPrevShapeInfo);
}   





}
}
