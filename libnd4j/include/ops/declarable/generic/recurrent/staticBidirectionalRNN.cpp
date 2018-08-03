/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma, created on 03.04.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/rnn.h>
#include<ops/declarable/helpers/reverse.h>
#include<ops/declarable/helpers/transforms.h>

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

    const Nd4jLong inRank     = x->rankOf();
    const Nd4jLong time       = x->sizeAt(0);
    const Nd4jLong bS         = x->sizeAt(1);
    const Nd4jLong numUnitsFW = WxFW->sizeAt(1);
    const Nd4jLong numUnitsBW = WxBW->sizeAt(1);

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
    NDArray<T>* hFW = new NDArray<T>({time, bS, numUnitsFW}, block.getWorkspace());
    helpers::rnnTimeLoop<T>({x, WxFW, WhFW, bFW, h0FW, maxTimeStep}, hFW, hFWFinal);

    NDArray<T>* seqLen = maxTimeStep;
    if(seqLen == nullptr) {    	
    	seqLen = new NDArray<T>(x->ordering(), {x->sizeAt(1)}, block.getWorkspace());	// [bS]
    	*seqLen = (T)x->sizeAt(0);														// set each element of seqLen to be equal to time
    }

    // reverse x 
    NDArray<T>* revOut = new NDArray<T>(x, false, block.getWorkspace());
    helpers::reverseSequence<T>(x, seqLen, revOut, 0, 1);

    // backward steps    
    NDArray<T>* hBW = new NDArray<T>('c', {time, bS, numUnitsBW}, block.getWorkspace());
    helpers::rnnTimeLoop<T>({revOut, WxBW, WhBW, bBW, h0BW, maxTimeStep}, hBW, hBWFinal);

    // reverse hBW 
    NDArray<T>* hBWcopy = new NDArray<T>(*hBW);
    helpers::reverseSequence<T>(hBWcopy, seqLen, hBW, 0, 1);

    // concatenate hFW and hBW along last third dimension
    // NDArrayFactory<T>::concat({hFW, hBW}, 2, h);
    helpers::concat({hFW, hBW}, *h, 2);

    delete hBW;
    delete hFW;
    delete hBWcopy;
    delete revOut;

    if(seqLen != maxTimeStep)
    	delete seqLen;

    return Status::OK();
}



DECLARE_SHAPE_FN(static_bidirectional_rnn) {    

	auto xShapeInfo     = inputShape->at(0);         // input [time x bS x inSize]
	auto WxFWShapeInfo  = inputShape->at(1);         // input-to-hidden  weights for forward  RNN, [inSize  x numUnitsFW]
    auto WhFWShapeInfo  = inputShape->at(2);         // hidden-to-hidden weights for forward  RNN, [numUnitsFW x numUnitsFW]
    auto bFWShapeInfo   = inputShape->at(3);         // biases for forward  RNN, [2*numUnitsFW]
    auto WxBWShapeInfo  = inputShape->at(4);         // input-to-hidden  weights for backward RNN, [inSize  x numUnitsBW]
    auto WhBWShapeInfo  = inputShape->at(5);         // hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]
	auto bBWShapeInfo   = inputShape->at(6);         // biases for backward RNN, [2*numUnitsBW]

	Nd4jLong* h0FWShapeInfo = nullptr;					// initial cell output for forward  RNN (at time step = 0) [bS x numUnitsFW]
	Nd4jLong* h0BWShapeInfo = nullptr;			    	// initial cell output for backward RNN (at time step = 0) [bS x numUnitsBW]
	Nd4jLong* maxTimeStepShapeInfo = nullptr;       		// vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

	switch(block.width()) {
		case 8:
			maxTimeStepShapeInfo = inputShape->at(7);
			break;
		case 9:
			h0FWShapeInfo = inputShape->at(7);
			h0BWShapeInfo = inputShape->at(8);
			break;
		case 10:
			h0FWShapeInfo = inputShape->at(7);
			h0BWShapeInfo = inputShape->at(8);
			maxTimeStepShapeInfo = inputShape->at(9);
			break;
	}

	REQUIRE_TRUE(xShapeInfo[0] == 3, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input array must have rank = 3, but got %i instead !", xShapeInfo[0]);
    REQUIRE_TRUE(WxFWShapeInfo[0] == 2, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for forward  RNN) must have rank = 2, but got %i instead !", WxFWShapeInfo[0]);
    REQUIRE_TRUE(WxBWShapeInfo[0] == 2, 0, "STATIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for backward RNN) must have rank = 2, but got %i instead !", WxBWShapeInfo[0]);

    const int inRank     = xShapeInfo[0];
    const int time       = xShapeInfo[1];
    const int bS         = xShapeInfo[2];
    const int numUnitsFW = WxFWShapeInfo[2];
    const int numUnitsBW = WxBWShapeInfo[2];

    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhFWShapeInfo) == ShapeUtils<T>::shapeAsString({numUnitsFW, numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({numUnitsFW, numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(WhFWShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(WhBWShapeInfo) == ShapeUtils<T>::shapeAsString({numUnitsBW, numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for backward RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({numUnitsBW, numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(WhBWShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bFWShapeInfo)  == ShapeUtils<T>::shapeAsString({2*numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for forward  RNN), expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(bFWShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(bBWShapeInfo)  == ShapeUtils<T>::shapeAsString({2*numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for backward RNN), expected is %s, but got %s instead !", ShapeUtils<T>::shapeAsString({2*numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(bBWShapeInfo).c_str());
    if(h0FWShapeInfo)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0FWShapeInfo) == ShapeUtils<T>::shapeAsString({bS, numUnitsFW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnitsFW}).c_str(), ShapeUtils<T>::shapeAsString(h0FWShapeInfo).c_str());
    if(h0BWShapeInfo)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(h0BWShapeInfo) == ShapeUtils<T>::shapeAsString({bS, numUnitsBW}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for backward RNN), expected is %s but got %s instead !", ShapeUtils<T>::shapeAsString({bS, numUnitsBW}).c_str(), ShapeUtils<T>::shapeAsString(h0BWShapeInfo).c_str());
    if(maxTimeStepShapeInfo)
        REQUIRE_TRUE(ShapeUtils<T>::shapeAsString(maxTimeStepShapeInfo)  == ShapeUtils<T>::shapeAsString({bS}), 0, "STATIC_BIDIRECTIONAL_RNN custom operation: wrong shape of maxTimeStep array, expected is [%i], but got %s instead !", bS, ShapeUtils<T>::shapeAsString(maxTimeStepShapeInfo).c_str());

    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *hFWFinalPrevShapeInfo(nullptr), *hBWFinalPrevShapeInfo(nullptr);
    ALLOCATE(hShapeInfo,            block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    ALLOCATE(hFWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), Nd4jLong);
    ALLOCATE(hBWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), Nd4jLong);

    hShapeInfo[0]     		 = inRank;
    hFWFinalPrevShapeInfo[0] = hBWFinalPrevShapeInfo[0] = inRank-1;
    hShapeInfo[1]     		 = time;
    hShapeInfo[2]     		 = hFWFinalPrevShapeInfo[1] = hBWFinalPrevShapeInfo[1] = bS;
    hShapeInfo[3]     		 = numUnitsFW + numUnitsBW;
    hFWFinalPrevShapeInfo[2] = numUnitsFW;
    hBWFinalPrevShapeInfo[2] = numUnitsBW;

    shape::updateStrides(hShapeInfo,            shape::order(xShapeInfo));
    shape::updateStrides(hFWFinalPrevShapeInfo, shape::order(xShapeInfo));
    shape::updateStrides(hBWFinalPrevShapeInfo, shape::order(xShapeInfo));
         
    return SHAPELIST(hShapeInfo, hFWFinalPrevShapeInfo, hBWFinalPrevShapeInfo);
}   








}
}

