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
// @author Yurii Shyrma, created on 15.02.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstm)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstm.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstm, 8, 2, false, 3, 2) {

    NDArray<T>* x  = INPUT_VARIABLE(0);                    // input [time x bS x inSize]
    NDArray<T>* h0 = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [bS x numProj], in case of projection=false -> numProj == numUnits !!!
    NDArray<T>* c0 = INPUT_VARIABLE(2);                    // initial cell state  (at time step = 0) [bS x numUnits],

    NDArray<T>* Wx  = INPUT_VARIABLE(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    NDArray<T>* Wh  = INPUT_VARIABLE(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    NDArray<T>* Wc  = INPUT_VARIABLE(5);                   // diagonal weights for peephole connections [3*numUnits] 
    NDArray<T>* Wp  = INPUT_VARIABLE(6);                   // projection weights [numUnits x numProj] 
    NDArray<T>* b   = INPUT_VARIABLE(7);                   // biases, [4*numUnits] 
    
    NDArray<T>* h   =  OUTPUT_VARIABLE(0);                 // cell outputs [time x bS x numProj], that is per each time step
    NDArray<T>* c   =  OUTPUT_VARIABLE(1);                 // cell states  [time x bS x numUnits] that is per each time step
    
    const int peephole   = INT_ARG(0);                     // if 1, provide peephole connections
    const int projection = INT_ARG(1);                     // if 1, then projection is performed, if false then numProj==numUnits is mandatory!!!!
    const T clippingCellValue  = T_ARG(0);                 // clipping value for ct, if it is not equal to zero, then cell state is clipped
    const T clippingProjValue  = T_ARG(1);                 // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    const T forgetBias   = T_ARG(2);

    const int rank     = x->rankOf();
    const int time     = x->sizeAt(0);
    const int bS       = x->sizeAt(1);
    const int inSize   = x->sizeAt(2);
    const int numProj  = h0->sizeAt(1);
    const int numUnits = c0->sizeAt(1);

    // input shapes validation
    const std::string h0Shape        = ShapeUtils<T>::shapeAsString(h0);
    const std::string correctH0Shape = ShapeUtils<T>::shapeAsString({bS, numProj});
    const std::string c0Shape        = ShapeUtils<T>::shapeAsString(c0);
    const std::string correctC0Shape = ShapeUtils<T>::shapeAsString({bS, numUnits});
    const std::string WxShape        = ShapeUtils<T>::shapeAsString(Wx);
    const std::string correctWxShape = ShapeUtils<T>::shapeAsString({inSize, 4*numUnits});
    const std::string WhShape        = ShapeUtils<T>::shapeAsString(Wh);
    const std::string correctWhShape = ShapeUtils<T>::shapeAsString({numProj, 4*numUnits});
    const std::string WcShape        = ShapeUtils<T>::shapeAsString(Wc);
    const std::string correctWcShape = ShapeUtils<T>::shapeAsString({3*numUnits});
    const std::string WpShape        = ShapeUtils<T>::shapeAsString(Wp);
    const std::string correctWpShape = ShapeUtils<T>::shapeAsString({numUnits, numProj});
    const std::string bShape         = ShapeUtils<T>::shapeAsString(b);
    const std::string correctBShape  = ShapeUtils<T>::shapeAsString({4*numUnits});

    REQUIRE_TRUE(correctH0Shape == h0Shape, 0, "LSTM operation: wrong shape of initial cell output, expected is %s, but got %s instead !", correctH0Shape.c_str(), h0Shape.c_str());
    REQUIRE_TRUE(correctC0Shape == c0Shape, 0, "LSTM operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", correctC0Shape.c_str(), c0Shape.c_str());
    REQUIRE_TRUE(correctWxShape == WxShape, 0, "LSTM operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", correctWxShape.c_str(), WxShape.c_str());
    REQUIRE_TRUE(correctWhShape == WhShape, 0, "LSTM operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", correctWhShape.c_str(), WhShape.c_str());
    REQUIRE_TRUE(correctWcShape == WcShape, 0, "LSTM operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", correctWcShape.c_str(), WcShape.c_str());
    REQUIRE_TRUE(correctWpShape == WpShape, 0, "LSTM operation: wrong shape of projection weights, expected is %s, but got %s instead !", correctWpShape.c_str(), WpShape.c_str());
    REQUIRE_TRUE(correctBShape  == bShape,  0, "LSTM operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str());
    REQUIRE_TRUE(!(!projection && numUnits != numProj), 0, "LSTM operation: projection option is switched of, and in this case output dimensionality for the projection matrices (numProj) must be equal to number of units in lstmCell !");


    helpers::lstmTimeLoop<T>({x,h0,c0, Wx,Wh,Wc,Wp, b},   {h,c},   {(T)peephole, (T)projection, clippingCellValue, clippingProjValue, forgetBias});

    return Status::OK();
}



DECLARE_SHAPE_FN(lstm) {    

    auto xShapeInfo  = inputShape->at(0);                    // input [time x bS x inSize]
    auto h0ShapeInfo = inputShape->at(1);                    // initial cell output (at time step = 0) [bS x numProj], in case of projection=false -> numProj == numUnits !!!
    auto c0ShapeInfo = inputShape->at(2);                    // initial cell state  (at time step = 0) [bS x numUnits],

    auto WxShapeInfo = inputShape->at(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits]
    auto WhShapeInfo = inputShape->at(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits]
    auto WcShapeInfo = inputShape->at(5);                   // diagonal weights for peephole connections [3*numUnits]
    auto WpShapeInfo = inputShape->at(6);                   // projection weights [numUnits x numProj]
    auto bShapeInfo  = inputShape->at(7);                   // biases, [4*numUnits]



    const int rank     = xShapeInfo[0];
    const int time     = xShapeInfo[1];
    const int bS       = xShapeInfo[2];
    const int inSize   = xShapeInfo[3];
    const int numProj  = h0ShapeInfo[2];
    const int numUnits = c0ShapeInfo[2];
 
    // input shapes validation
    const std::string h0Shape        = ShapeUtils<T>::shapeAsString(h0ShapeInfo);
    const std::string correctH0Shape = ShapeUtils<T>::shapeAsString({bS, numProj});
    const std::string c0Shape        = ShapeUtils<T>::shapeAsString(c0ShapeInfo);
    const std::string correctC0Shape = ShapeUtils<T>::shapeAsString({bS, numUnits});
    const std::string WxShape        = ShapeUtils<T>::shapeAsString(WxShapeInfo);
    const std::string correctWxShape = ShapeUtils<T>::shapeAsString({inSize, 4*numUnits});
    const std::string WhShape        = ShapeUtils<T>::shapeAsString(WhShapeInfo);
    const std::string correctWhShape = ShapeUtils<T>::shapeAsString({numProj, 4*numUnits});
    const std::string WcShape        = ShapeUtils<T>::shapeAsString(WcShapeInfo);
    const std::string correctWcShape = ShapeUtils<T>::shapeAsString({3*numUnits});
    const std::string WpShape        = ShapeUtils<T>::shapeAsString(WpShapeInfo);
    const std::string correctWpShape = ShapeUtils<T>::shapeAsString({numUnits, numProj});
    const std::string bShape         = ShapeUtils<T>::shapeAsString(bShapeInfo);
    const std::string correctBShape  = ShapeUtils<T>::shapeAsString({4*numUnits});

    REQUIRE_TRUE(correctH0Shape == h0Shape, 0, "LSTM operation: wrong shape of initial cell output, expected is %s, but got %s instead !", correctH0Shape.c_str(), h0Shape.c_str()); 
    REQUIRE_TRUE(correctC0Shape == c0Shape, 0, "LSTM operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", correctC0Shape.c_str(), c0Shape.c_str()); 
    REQUIRE_TRUE(correctWxShape == WxShape, 0, "LSTM operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", correctWxShape.c_str(), WxShape.c_str()); 
    REQUIRE_TRUE(correctWhShape == WhShape, 0, "LSTM operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", correctWhShape.c_str(), WhShape.c_str()); 
    REQUIRE_TRUE(correctWcShape == WcShape, 0, "LSTM operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", correctWcShape.c_str(), WcShape.c_str()); 
    REQUIRE_TRUE(correctWpShape == WpShape, 0, "LSTM operation: wrong shape of projection weights, expected is %s, but got %s instead !", correctWpShape.c_str(), WpShape.c_str()); 
    REQUIRE_TRUE(correctBShape  == bShape,  0, "LSTM operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str());     

    
    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [time x bS x numProj]
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [time x bS x numUnits]
            
    hShapeInfo[0] = cShapeInfo[0] = rank;
    hShapeInfo[1] = cShapeInfo[1] = time;
    hShapeInfo[2] = cShapeInfo[2] = bS;
    hShapeInfo[3] = numProj;
    cShapeInfo[3] = numUnits;
    
    shape::updateStrides(hShapeInfo, shape::order(h0ShapeInfo));
    shape::updateStrides(cShapeInfo, shape::order(c0ShapeInfo));
         
    return SHAPELIST(hShapeInfo, cShapeInfo);
}   





}
}

#endif