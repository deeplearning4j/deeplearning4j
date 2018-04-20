//
// @author Yurii Shyrma, created on 03.04.2018
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(static_bidirectional_rnn, 7, 3, false, 0, 0) {

    NDArray<T>* x  	  = INPUT_VARIABLE(0);                  // input [time x bS x inSize]
	NDArray<T>* WxFW  = INPUT_VARIABLE(1);                  // input-to-hidden  weights for forward  RNN, [inSize  x numUnitsFW] 	    
    NDArray<T>* WhFW  = INPUT_VARIABLE(2);                  // hidden-to-hidden weights for forward  RNN, [numUnitsFW x numUnitsFW]     
    NDArray<T>* bFW   = INPUT_VARIABLE(3);                  // biases for forward  RNN, [2*numUnitsFW] 
    NDArray<T>* WxBW  = INPUT_VARIABLE(4);                  // input-to-hidden  weights for backward RNN, [inSize  x numUnitsBW] 	    
    NDArray<T>* WhBW  = INPUT_VARIABLE(5);                  // hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]         
	NDArray<T>* bBW   = INPUT_VARIABLE(6);                  // biases for backward RNN, [2*v] 

	NDArray<T>* h0FW = nullptr;								// initial cell output for forward  RNN (at time step = 0) [bS x numUnitsFW]
	NDArray<T>* h0BW = nullptr;								// initial cell output for backward RNN (at time step = 0) [bS x numUnitsBW]
	NDArray<T>* maxTimeStep = nullptr;						// vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

	switch(block.width()) {
		case 8:
			maxTimeStep = INPUT_VARIABLE(7);
			break;
		case 9:
			h0FW = INPUT_VARIABLE(7);
			h0BW = INPUT_VARIABLE(8);
			break;
		case 10:
			h0FW = INPUT_VARIABLE(7);
			h0BW = INPUT_VARIABLE(8);
			maxTimeStep = INPUT_VARIABLE(9);
			break;
	}
    
    NDArray<T>* h        =  OUTPUT_VARIABLE(0);                 // cell outputs [time x bS x (numUnitsFW + numUnitsBW)], that is per each time step    
    NDArray<T>* hFWFinal =  OUTPUT_VARIABLE(1);                 // final cell out for forward  RNN [bS x numUnitsFW]
    NDArray<T>* hBWFinal =  OUTPUT_VARIABLE(2);                 // final cell out for backward RNN [bS x numUnitsBF]    

    REQUIRE_TRUE(x->rankOf() == 3, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input array must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(WxFW->rankOf() == 2, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for forward  RNN) must have rank = 2, but got %i instead !", WxFW->rankOf());    
    REQUIRE_TRUE(WxBW->rankOf() == 2, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for backward RNN) must have rank = 2, but got %i instead !", WxBW->rankOf());    

    const int inRank     = x->rankOf();
    const int time       = x->sizeAt(0);
    const int bS         = x->sizeAt(1);
    const int numUnitsFW = WxFW->sizeAt(1);
    const int numUnitsBW = WxBW->sizeAt(1);

    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhFW) == ShapeUtils<T>::shapeAsString({numUnitsFW, numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({numUnitsFW, numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(WhFW).c_str());     
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhBW) == ShapeUtils<T>::shapeAsString({numUnitsBW, numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for backward RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({numUnitsBW, numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(WhBW).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bFW)  == ShapeUtils<T>::shapeAsString({2*numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for forward  RNN), expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(bFW).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bBW)  == ShapeUtils<T>::shapeAsString({2*numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for backward RNN), expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(bBW).c_str()); 
    if(h0FW)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0FW) == ShapeUtils<T>::shapeAsString({bS, numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(h0FW).c_str());
    if(h0BW)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0BW) == ShapeUtils<T>::shapeAsString({bS, numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for backward RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(h0BW).c_str()); 
    if(maxTimeStep)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(maxTimeStep)  == ShapeUtils<T>::shapeAsString({bS}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of maxTimeStep array, expected is [%i], but got %s instead !", bS, ShapeUtils<T>::shapeAsString(maxTimeStep).c_str()); 

    // forward steps
    nd4j::ops::static_rnn<T> staticRnn;
    nd4j::ResultSet<T>* resultsFW = staticRnn.execute({x, WxFW, WhFW, bFW, h0FW, maxTimeStep}, {}, {});    
    NDArray<T>* hFW = resultsFW->at(0);							// [time x bS x numUnitsFW]
    hFWFinal->assign(resultsFW->at(1));

    NDArray<T>* seqLen = maxTimeStep;
    if(seqLen == nullptr) {    	
    	seqLen = new NDArray<T>(x->ordering(), {x->sizeAt(1)}, block.getWorkspace());	// [bS]
    	*seqLen = (T)x->sizeAt(0);														// set each element of seqLen to be equal to time
    }

    // reverse x 
    nd4j::ops::reverse_sequense<T> reverse;
    ResultSet<T>* resultsIn = reverse.execute({x, seqLen}, {}, {0,1});
    NDArray<T>* revInput = resultsIn->at(0);

    // backward steps    
    nd4j::ResultSet<T>* resultsBW = staticRnn.execute({revInput, WxBW, WhBW, bBW, h0BW, maxTimeStep}, {}, {});    
    NDArray<T>* hBW = resultsBW->at(0);							// [time x bS x numUnitsBW]
    hBWFinal->assign(resultsBW->at(1));

    // reverse hBW 
    ResultSet<T>* resultsOut = reverse.execute({hBW, seqLen}, {}, {0,1});
    hBW = resultsOut->at(0);

    // concatenate hFW and hBW along last third dimension
    NDArrayFactory<T>::concat({hFW, hBW}, 2, h);
    
	delete resultsOut;
	delete resultsBW;
	delete resultsIn;	
    delete resultsFW;    

    if(seqLen != maxTimeStep)
    	delete seqLen;

    return Status::OK();
}



DECLARE_SHAPE_FN(static_bidirectional_rnn) {    

	NDArray<T>* x  	  = INPUT_VARIABLE(0);                  // input [time x bS x inSize]
	NDArray<T>* WxFW  = INPUT_VARIABLE(1);                  // input-to-hidden  weights for forward  RNN, [inSize  x numUnitsFW] 	    
    NDArray<T>* WhFW  = INPUT_VARIABLE(2);                  // hidden-to-hidden weights for forward  RNN, [numUnitsFW x numUnitsFW]     
    NDArray<T>* bFW   = INPUT_VARIABLE(3);                  // biases for forward  RNN, [2*numUnitsFW] 
    NDArray<T>* WxBW  = INPUT_VARIABLE(4);                  // input-to-hidden  weights for backward RNN, [inSize  x numUnitsBW] 	    
    NDArray<T>* WhBW  = INPUT_VARIABLE(5);                  // hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]         
	NDArray<T>* bBW   = INPUT_VARIABLE(6);                  // biases for backward RNN, [2*numUnitsBW] 

	NDArray<T>* h0FW = nullptr;								// initial cell output for forward  RNN (at time step = 0) [bS x numUnitsFW]
	NDArray<T>* h0BW = nullptr;								// initial cell output for backward RNN (at time step = 0) [bS x numUnitsBW]
	NDArray<T>* maxTimeStep = nullptr;						// vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

	switch(block.width()) {
		case 8:
			maxTimeStep = INPUT_VARIABLE(7);
			break;
		case 9:
			h0FW = INPUT_VARIABLE(7);
			h0BW = INPUT_VARIABLE(8);
			break;
		case 10:
			h0FW = INPUT_VARIABLE(7);
			h0BW = INPUT_VARIABLE(8);
			maxTimeStep = INPUT_VARIABLE(9);
			break;
	}

	REQUIRE_TRUE(x->rankOf() == 3, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input array must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(WxFW->rankOf() == 2, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for forward  RNN) must have rank = 2, but got %i instead !", WxFW->rankOf());    
    REQUIRE_TRUE(WxBW->rankOf() == 2, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for backward RNN) must have rank = 2, but got %i instead !", WxBW->rankOf());    

    const int inRank     = x->rankOf();
    const int time       = x->sizeAt(0);
    const int bS         = x->sizeAt(1);
    const int numUnitsFW = WxFW->sizeAt(1);
    const int numUnitsBW = WxBW->sizeAt(1);

    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhFW) == ShapeUtils<T>::shapeAsString({numUnitsFW, numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({numUnitsFW, numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(WhFW).c_str());     
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhBW) == ShapeUtils<T>::shapeAsString({numUnitsBW, numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for backward RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({numUnitsBW, numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(WhBW).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bFW)  == ShapeUtils<T>::shapeAsString({2*numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for forward  RNN), expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(bFW).c_str()); 
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bBW)  == ShapeUtils<T>::shapeAsString({2*numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for backward RNN), expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(bBW).c_str()); 
    if(h0FW)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0FW) == ShapeUtils<T>::shapeAsString({bS, numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(h0FW).c_str());
    if(h0BW)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0BW) == ShapeUtils<T>::shapeAsString({bS, numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for backward RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(h0BW).c_str()); 
    if(maxTimeStep)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(maxTimeStep)  == ShapeUtils<T>::shapeAsString({bS}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of maxTimeStep array, expected is [%i], but got %s instead !", bS, ShapeUtils<T>::shapeAsString(maxTimeStep).c_str()); 

    // evaluate output shapeInfos
    int *hShapeInfo(nullptr), *hFWFinalPrevShapeInfo(nullptr), *hBWFinalPrevShapeInfo(nullptr);
    ALLOCATE(hShapeInfo,            block.getWorkspace(), shape::shapeInfoLength(inRank), int);
    ALLOCATE(hFWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), int);
    ALLOCATE(hBWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), int);

    hShapeInfo[0]     		 = inRank;
    hFWFinalPrevShapeInfo[0] = hBWFinalPrevShapeInfo[0] = inRank-1;
    hShapeInfo[1]     		 = time;
    hShapeInfo[2]     		 = hFWFinalPrevShapeInfo[1] = hBWFinalPrevShapeInfo[1] = bS;
    hShapeInfo[3]     		 = numUnitsFW + numUnitsBW;
    hFWFinalPrevShapeInfo[2] = numUnitsFW;
    hBWFinalPrevShapeInfo[2] = numUnitsBW;

    shape::updateStrides(hShapeInfo,            x->ordering());    
    shape::updateStrides(hFWFinalPrevShapeInfo, x->ordering());
    shape::updateStrides(hBWFinalPrevShapeInfo, x->ordering());
         
    return SHAPELIST(hShapeInfo, hFWFinalPrevShapeInfo, hBWFinalPrevShapeInfo);
}   








}
}

