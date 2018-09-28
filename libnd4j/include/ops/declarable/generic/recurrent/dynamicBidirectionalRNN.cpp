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
// @author Yurii Shyrma, created on 05.04.2018
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(dynamic_bidirectional_rnn, 7, 4, false, 0, 0) {
    auto x  	  = INPUT_VARIABLE(0);                  // input [time x bS x inSize] or [bS x time x inSize], shape depends on timeMajor parameter
	auto WxFW  = INPUT_VARIABLE(1);                  // input-to-hidden  weights for forward  RNN, [inSize  x numUnitsFW]
    auto WhFW  = INPUT_VARIABLE(2);                  // hidden-to-hidden weights for forward  RNN, [numUnitsFW x numUnitsFW]
    auto bFW   = INPUT_VARIABLE(3);                  // biases for forward  RNN, [2*numUnitsFW]
    auto WxBW  = INPUT_VARIABLE(4);                  // input-to-hidden  weights for backward RNN, [inSize  x numUnitsBW]
    auto WhBW  = INPUT_VARIABLE(5);                  // hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]
	auto bBW   = INPUT_VARIABLE(6);                  // biases for backward RNN, [2*v]

	NDArray* h0FW = nullptr;								// initial cell output for forward  RNN (at time step = 0) [bS x numUnitsFW]
	NDArray* h0BW = nullptr;								// initial cell output for backward RNN (at time step = 0) [bS x numUnitsBW]
	NDArray* maxTimeStep = nullptr;						// vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    const int timeMajor = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // if non zero then [time, bS, ...], else [bS, time, ...]

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
    
    auto hFW      =  OUTPUT_VARIABLE(0);                 // cell outputs for forward RNN  [time x bS x numUnitsFW] or [bS x time x numUnitsFW], shape depends on timeMajor parameter
    auto hBW      =  OUTPUT_VARIABLE(1);                 // cell outputs for backward RNN [time x bS x numUnitsBW] or [bS x time x numUnitsBW], shape depends on timeMajor parameter
    auto hFWFinal =  OUTPUT_VARIABLE(2);                 // final cell out for forward  RNN [bS x numUnitsFW]
    auto hBWFinal =  OUTPUT_VARIABLE(3);                 // final cell out for backward RNN [bS x numUnitsBF]

    REQUIRE_TRUE(x->rankOf() == 3,    0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: input array must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(WxFW->rankOf() == 2, 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for forward  RNN) must have rank = 2, but got %i instead !", WxFW->rankOf());    
    REQUIRE_TRUE(WxBW->rankOf() == 2, 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for backward RNN) must have rank = 2, but got %i instead !", WxBW->rankOf());    

    const int inRank     = x->rankOf();
    const int time       = timeMajor ? x->sizeAt(0) : x->sizeAt(1);
    const int bS         = timeMajor ? x->sizeAt(1) : x->sizeAt(0);
    const int numUnitsFW = WxFW->sizeAt(1);
    const int numUnitsBW = WxBW->sizeAt(1);

    REQUIRE_TRUE(ShapeUtils::shapeAsString(WhFW) == ShapeUtils::shapeAsString({numUnitsFW, numUnitsFW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({numUnitsFW, numUnitsFW}).c_str(), ShapeUtils::shapeAsString(WhFW).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(WhBW) == ShapeUtils::shapeAsString({numUnitsBW, numUnitsBW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for backward RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({numUnitsBW, numUnitsBW}).c_str(), ShapeUtils::shapeAsString(WhBW).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(bFW)  == ShapeUtils::shapeAsString({2*numUnitsFW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for forward  RNN), expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2*numUnitsFW}).c_str(), ShapeUtils::shapeAsString(bFW).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(bBW)  == ShapeUtils::shapeAsString({2*numUnitsBW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for backward RNN), expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2*numUnitsBW}).c_str(), ShapeUtils::shapeAsString(bBW).c_str());
    if(h0FW)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(h0FW) == ShapeUtils::shapeAsString({bS, numUnitsFW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({bS, numUnitsFW}).c_str(), ShapeUtils::shapeAsString(h0FW).c_str());
    if(h0BW)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(h0BW) == ShapeUtils::shapeAsString({bS, numUnitsBW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for backward RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({bS, numUnitsBW}).c_str(), ShapeUtils::shapeAsString(h0BW).c_str());
    if(maxTimeStep)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(maxTimeStep)  == ShapeUtils::shapeAsString({bS}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of maxTimeStep array, expected is [%i], but got %s instead !", bS, ShapeUtils::shapeAsString(maxTimeStep).c_str());


    // forward steps
    nd4j::ops::dynamic_rnn dynamicRnn;
    auto resultsFW = dynamicRnn.execute({x, WxFW, WhFW, bFW, h0FW, maxTimeStep}, {}, {timeMajor});
    hFW->assign(resultsFW->at(0));                              // [time x bS x numUnitsFW] or [bS x time x numUnitsFW]
    hFWFinal->assign(resultsFW->at(1));

    auto seqLen = maxTimeStep;
    if(seqLen == nullptr) {
        // FIXME: which datatype should be used here?
    	seqLen = new NDArray(x->ordering(), {bS}, x->dataType(), block.getWorkspace());
    	*seqLen = time;                                        // set each element of seqLen to be equal to time
    }

    std::initializer_list<Nd4jLong> dimsForReverse = timeMajor ? std::initializer_list<Nd4jLong>{0,1} : std::initializer_list<Nd4jLong>{1,0};

    // reverse x     
    nd4j::ops::reverse_sequence reverse;
    auto resultsIn = reverse.execute({x, seqLen}, {}, dimsForReverse);
    auto revInput = resultsIn->at(0);

    // backward steps    
    auto resultsBW = dynamicRnn.execute({revInput, WxBW, WhBW, bBW, h0BW, maxTimeStep}, {}, {timeMajor});
    auto hBWtemp = resultsBW->at(0);					           // [time x bS x numUnitsBW] or [ bS x time xnumUnitsBW]
    hBWFinal->assign(resultsBW->at(1));

    // reverse hBWtemp 
    auto resultsOut = reverse.execute({hBWtemp, seqLen}, {}, dimsForReverse);
    hBW->assign(resultsOut->at(0));
    
	delete resultsOut;
	delete resultsBW;
	delete resultsIn;	
    delete resultsFW;    

    if(seqLen != maxTimeStep)
    	delete seqLen;

    return Status::OK();
}



DECLARE_SHAPE_FN(dynamic_bidirectional_rnn) {    

	auto x  	  = INPUT_VARIABLE(0);                  // input [time x bS x inSize] or [bS x time x inSize], shape depends on timeMajor parameter
	auto WxFW  = INPUT_VARIABLE(1);                  // input-to-hidden  weights for forward  RNN, [inSize  x numUnitsFW]
    auto WhFW  = INPUT_VARIABLE(2);                  // hidden-to-hidden weights for forward  RNN, [numUnitsFW x numUnitsFW]
    auto bFW   = INPUT_VARIABLE(3);                  // biases for forward  RNN, [2*numUnitsFW]
    auto WxBW  = INPUT_VARIABLE(4);                  // input-to-hidden  weights for backward RNN, [inSize  x numUnitsBW]
    auto WhBW  = INPUT_VARIABLE(5);                  // hidden-to-hidden weights for backward RNN, [numUnitsBW x numUnitsBW]
	auto bBW   = INPUT_VARIABLE(6);                  // biases for backward RNN, [2*numUnitsBW]

	NDArray* h0FW = nullptr;								// initial cell output for forward  RNN (at time step = 0) [bS x numUnitsFW]
	NDArray* h0BW = nullptr;								// initial cell output for backward RNN (at time step = 0) [bS x numUnitsBW]
	NDArray* maxTimeStep = nullptr;						// vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    const int timeMajor = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // if true then [time, bS, ...], else [bS, time, ...]
	
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

	REQUIRE_TRUE(x->rankOf() == 3,    0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: input array must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(WxFW->rankOf() == 2, 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for forward  RNN) must have rank = 2, but got %i instead !", WxFW->rankOf());    
    REQUIRE_TRUE(WxBW->rankOf() == 2, 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: input-to-hidden weights array (for backward RNN) must have rank = 2, but got %i instead !", WxBW->rankOf());    

    const int inRank     = x->rankOf();
    const int time       = timeMajor ? x->sizeAt(0) : x->sizeAt(1);
    const int bS         = timeMajor ? x->sizeAt(1) : x->sizeAt(0);
    const int numUnitsFW = WxFW->sizeAt(1);
    const int numUnitsBW = WxBW->sizeAt(1);

    REQUIRE_TRUE(ShapeUtils::shapeAsString(WhFW) == ShapeUtils::shapeAsString({numUnitsFW, numUnitsFW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({numUnitsFW, numUnitsFW}).c_str(), ShapeUtils::shapeAsString(WhFW).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(WhBW) == ShapeUtils::shapeAsString({numUnitsBW, numUnitsBW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of hidden-to-hidden weights array (for backward RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({numUnitsBW, numUnitsBW}).c_str(), ShapeUtils::shapeAsString(WhBW).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(bFW)  == ShapeUtils::shapeAsString({2*numUnitsFW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for forward  RNN), expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2*numUnitsFW}).c_str(), ShapeUtils::shapeAsString(bFW).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(bBW)  == ShapeUtils::shapeAsString({2*numUnitsBW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of biases array (for backward RNN), expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2*numUnitsBW}).c_str(), ShapeUtils::shapeAsString(bBW).c_str());
    if(h0FW)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(h0FW) == ShapeUtils::shapeAsString({bS, numUnitsFW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for forward  RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({bS, numUnitsFW}).c_str(), ShapeUtils::shapeAsString(h0FW).c_str());
    if(h0BW)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(h0BW) == ShapeUtils::shapeAsString({bS, numUnitsBW}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of initial cell output array (for backward RNN), expected is %s but got %s instead !", ShapeUtils::shapeAsString({bS, numUnitsBW}).c_str(), ShapeUtils::shapeAsString(h0BW).c_str());
    if(maxTimeStep)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(maxTimeStep)  == ShapeUtils::shapeAsString({bS}), 0, "DYNAMIC_BIDIRECTIONAL_RNN custom operation: wrong shape of maxTimeStep array, expected is [%i], but got %s instead !", bS, ShapeUtils::shapeAsString(maxTimeStep).c_str());

    // evaluate output shapeInfos
    Nd4jLong *hFWShapeInfo(nullptr), *hBWShapeInfo(nullptr), *hFWFinalPrevShapeInfo(nullptr), *hBWFinalPrevShapeInfo(nullptr);
    ALLOCATE(hFWShapeInfo,          block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    ALLOCATE(hBWShapeInfo,          block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    ALLOCATE(hFWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), Nd4jLong);
    ALLOCATE(hBWFinalPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), Nd4jLong);

    hFWShapeInfo[0]          = hBWShapeInfo[0]          = inRank;    
    hFWShapeInfo[1]          = hBWShapeInfo[1]          = timeMajor ? time : bS;
    hFWShapeInfo[2]          = hBWShapeInfo[2]          = timeMajor ? bS   : time;
    hFWShapeInfo[3]          = numUnitsFW;
    hBWShapeInfo[3]          = numUnitsBW;
    hFWFinalPrevShapeInfo[0] = hBWFinalPrevShapeInfo[0] = inRank-1;
    hFWFinalPrevShapeInfo[1] = hBWFinalPrevShapeInfo[1] = bS;    
    hFWFinalPrevShapeInfo[2] = numUnitsFW;
    hBWFinalPrevShapeInfo[2] = numUnitsBW;

    ShapeUtils::updateStirdesAndType(hFWShapeInfo, x->getShapeInfo(), x->ordering());
    ShapeUtils::updateStirdesAndType(hBWShapeInfo, x->getShapeInfo(), x->ordering());
    ShapeUtils::updateStirdesAndType(hFWFinalPrevShapeInfo, x->getShapeInfo(), x->ordering());
    ShapeUtils::updateStirdesAndType(hBWFinalPrevShapeInfo, x->getShapeInfo(), x->ordering());
         
    return SHAPELIST(hFWShapeInfo, hBWShapeInfo, hFWFinalPrevShapeInfo, hBWFinalPrevShapeInfo);
}   








}
}

