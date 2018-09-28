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
// created by Yurii Shyrma on 15.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_gru)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/gru.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gru, 5, 1, false, 0, 0) {
    auto x  = INPUT_VARIABLE(0);                    // input [time x bS x inSize]
    auto h0 = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [bS x numUnits]
    
    auto Wx  = INPUT_VARIABLE(2);                   // input-to-hidden  weights, [inSize x 3*numUnits]
    auto Wh  = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [numUnits x 3*numUnits]
    auto b   = INPUT_VARIABLE(4);                   // biases, [3*numUnits]
    
    auto h   =  OUTPUT_VARIABLE(0);                 // cell outputs [time x bS x numUnits], that is per each time step

    const int rank     = x->rankOf();              // = 3    
    const int time     = x->sizeAt(0);
    const int bS       = x->sizeAt(1);
    const int inSize   = x->sizeAt(2);
    const int numUnits = h0->sizeAt(1);    

    const std::string h0Shape        = ShapeUtils::shapeAsString(h0);
    const std::string h0CorrectShape = ShapeUtils::shapeAsString({bS, numUnits});
    const std::string wxShape        = ShapeUtils::shapeAsString(Wx);
    const std::string wxCorrectShape = ShapeUtils::shapeAsString({inSize, 3*numUnits});
    const std::string whShape        = ShapeUtils::shapeAsString(Wh);
    const std::string whCorrectShape = ShapeUtils::shapeAsString({numUnits, 3*numUnits});
    const std::string bShape         = ShapeUtils::shapeAsString(b);
    const std::string bCorrectShape  = ShapeUtils::shapeAsString({3*numUnits});
    
    REQUIRE_TRUE(h0Shape == h0CorrectShape, 0, "GRU operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", h0CorrectShape.c_str(), h0Shape.c_str()); 
    REQUIRE_TRUE(wxShape == wxCorrectShape, 0, "GRU operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str()); 
    REQUIRE_TRUE(whShape == whCorrectShape, 0, "GRU operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());     
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "GRU operation: wrong shape of biases array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());     

    helpers::gruTimeLoop(x, h0, Wx, Wh, b, h);
    
    return Status::OK();
}



DECLARE_SHAPE_FN(gru) {    
    const auto xShapeInfo  = inputShape->at(0);                     // input [time x bS x inSize]
    const auto h0ShapeInfo = inputShape->at(1);                     // initial cell output [bS x numUnits], that is at time step t=0
    const auto WxShapeInfo = inputShape->at(2);                     // input-to-hidden weights, [inSize   x 3*numUnits]
    const auto WhShapeInfo = inputShape->at(3);                     // hidden-to-hidden weights, [numUnits x 3*numUnits]
    const auto bShapeInfo  = inputShape->at(4);                     // biases, [3*numUnits]

    const int rank     = shape::rank(xShapeInfo);              // = 3
    const auto time     = xShapeInfo[1];
    const auto bS       = xShapeInfo[2];
    const auto inSize   = xShapeInfo[3];
    const auto numUnits = h0ShapeInfo[2];

    const std::string h0Shape        = ShapeUtils::shapeAsString(h0ShapeInfo);
    const std::string h0CorrectShape = ShapeUtils::shapeAsString({bS, numUnits});
    const std::string wxShape        = ShapeUtils::shapeAsString(WxShapeInfo);
    const std::string wxCorrectShape = ShapeUtils::shapeAsString({inSize, 3*numUnits});
    const std::string whShape        = ShapeUtils::shapeAsString(WhShapeInfo);
    const std::string whCorrectShape = ShapeUtils::shapeAsString({numUnits, 3*numUnits});
    const std::string bShape         = ShapeUtils::shapeAsString(bShapeInfo);
    const std::string bCorrectShape  = ShapeUtils::shapeAsString({3*numUnits});
    
    REQUIRE_TRUE(h0Shape == h0CorrectShape, 0, "GRU operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", h0CorrectShape.c_str(), h0Shape.c_str()); 
    REQUIRE_TRUE(wxShape == wxCorrectShape, 0, "GRU operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str()); 
    REQUIRE_TRUE(whShape == whCorrectShape, 0, "GRU operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());     
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "GRU operation: wrong shape of biases array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());     


    // evaluate output shapeInfo
    Nd4jLong *hShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
            
    hShapeInfo[0] = rank;
    hShapeInfo[1] = time;
    hShapeInfo[2] = bS;
    hShapeInfo[3] = numUnits;

    ShapeUtils::updateStirdesAndType(hShapeInfo, xShapeInfo, shape::order(h0ShapeInfo));    
         
    return SHAPELIST(hShapeInfo);
}   






}
}


#endif