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
// @author Yurii Shyrma, created on 02.04.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/rnn.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(static_rnn, 4, 2, false, 0, 0) {

    auto x  = INPUT_VARIABLE(0);               // input [time x bS x inSize]
	auto Wx = INPUT_VARIABLE(1);               // input-to-hidden  weights, [inSize  x numUnits]
    auto Wh = INPUT_VARIABLE(2);               // hidden-to-hidden weights, [numUnits x numUnits]
	auto b  = INPUT_VARIABLE(3);               // biases for, [2*numUnits]

	NDArray* h0          = nullptr;     	   // initial cell output (at time step = 0) [bS x numUnits]
	NDArray* maxTimeStep = nullptr;			   // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

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
    
    auto h      =  OUTPUT_VARIABLE(0);           // cell outputs [time x bS x numUnits]
    auto hFinal =  OUTPUT_VARIABLE(1);           // at the end it will store cell final non-zero output [bS x numUnits]

	REQUIRE_TRUE(x->rankOf() == 3, 0, "STATIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(Wx->rankOf() == 2, 0, "STATIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", Wx->rankOf());    

    const int time     = x->sizeAt(0);
    const int bS       = x->sizeAt(1);
    const int inSize   = x->sizeAt(2);
    const int numUnits = Wx->sizeAt(1);

    REQUIRE_TRUE(ShapeUtils::shapeAsString(Wh) == ShapeUtils::shapeAsString({numUnits, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({numUnits, numUnits}).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(b)  == ShapeUtils::shapeAsString({2*numUnits}), 0, "STATIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2*numUnits}).c_str(), ShapeUtils::shapeAsString(b).c_str());
    if(h0)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(h0) == ShapeUtils::shapeAsString({bS, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils::shapeAsString({bS, numUnits}).c_str(), ShapeUtils::shapeAsString(h0).c_str());
    if(maxTimeStep)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(maxTimeStep)  == ShapeUtils::shapeAsString({bS}), 0, "STATIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS}).c_str(), ShapeUtils::shapeAsString(maxTimeStep).c_str());


    helpers::rnnTimeLoop(x, Wx, Wh, b, h0, maxTimeStep,  h, hFinal);
    
    return Status::OK();
}



DECLARE_SHAPE_FN(static_rnn) {    

    auto xShapeInfo  = inputShape->at(0);               // input [time x bS x inSize]
    auto WxShapeInfo = inputShape->at(1);               // input-to-hidden  weights, [inSize  x numUnits]     
    auto WhShapeInfo = inputShape->at(2);               // hidden-to-hidden weights, [numUnits x numUnits]         
    auto bShapeInfo  = inputShape->at(3);               // biases for, [2*numUnits] 

    Nd4jLong* h0ShapeInfo          = nullptr;                // initial cell output (at time step = 0) [bS x numUnits] 
    Nd4jLong* maxTimeStepShapeInfo = nullptr;                // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

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

    REQUIRE_TRUE(xShapeInfo[0]  == 3, 0, "STATIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", xShapeInfo[0]);
    REQUIRE_TRUE(WxShapeInfo[0] == 2, 0, "STATIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", WxShapeInfo[0]);    

    const int inRank   = xShapeInfo[0];
    const int time     = xShapeInfo[1];
    const int bS       = xShapeInfo[2];
    const int numUnits = WxShapeInfo[2];

    REQUIRE_TRUE(ShapeUtils::shapeAsString(WhShapeInfo) == ShapeUtils::shapeAsString({numUnits, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({numUnits, numUnits}).c_str(), ShapeUtils::shapeAsString(WhShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(bShapeInfo)  == ShapeUtils::shapeAsString({2*numUnits}), 0, "STATIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2*numUnits}).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());
    if(h0ShapeInfo)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(h0ShapeInfo) == ShapeUtils::shapeAsString({bS, numUnits}), 0, "STATIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils::shapeAsString({bS, numUnits}).c_str(), ShapeUtils::shapeAsString(h0ShapeInfo).c_str());
    if(maxTimeStepShapeInfo)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(maxTimeStepShapeInfo)  == ShapeUtils::shapeAsString({bS}), 0, "STATIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS}).c_str(), ShapeUtils::shapeAsString(maxTimeStepShapeInfo).c_str());

    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *hPrevShapeInfo(nullptr);
    ALLOCATE(hShapeInfo,     block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    ALLOCATE(hPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), Nd4jLong);
            
    hShapeInfo[0]            = inRank;
    hPrevShapeInfo[0]        = inRank-1;
    hShapeInfo[1]            = time;
    hShapeInfo[2]            = hPrevShapeInfo[1] = bS;
    hShapeInfo[3]            = hPrevShapeInfo[2] = numUnits;
    ArrayOptions::copyDataType(hShapeInfo, xShapeInfo);
    ArrayOptions::copyDataType(hPrevShapeInfo, xShapeInfo);

    shape::updateStrides(hShapeInfo,     shape::order(xShapeInfo));    
    shape::updateStrides(hPrevShapeInfo, shape::order(xShapeInfo));
         
    return SHAPELIST(hShapeInfo, hPrevShapeInfo);
}   




}
}
