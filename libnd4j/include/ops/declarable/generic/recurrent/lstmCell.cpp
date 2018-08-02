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
// @author Yurii Shyrma, created on 30.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstmCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstm.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmCell, 8, 2, false, 3, 2) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);                   // input [bS x inSize]
    NDArray<T>* ht_1 = INPUT_VARIABLE(1);                   // previous cell output [bS x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!! 
    NDArray<T>* ct_1 = INPUT_VARIABLE(2);                   // previous cell state  [bS x numUnits], that is at previous time step t-1   

    NDArray<T>* Wx   = INPUT_VARIABLE(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    NDArray<T>* Wh   = INPUT_VARIABLE(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    NDArray<T>* Wc   = INPUT_VARIABLE(5);                   // diagonal weights for peephole connections [3*numUnits] 
    NDArray<T>* Wp   = INPUT_VARIABLE(6);                   // projection weights [numUnits x numProj] 
    NDArray<T>* b    = INPUT_VARIABLE(7);                   // biases, [4*numUnits] 
    
    NDArray<T>* ht   =  OUTPUT_VARIABLE(0);                 // current cell output [bS x numProj], that is at current time step t
    NDArray<T>* ct   =  OUTPUT_VARIABLE(1);                 // current cell state  [bS x numUnits], that is at current time step t
    
    const int peephole   = INT_ARG(0);                            // if 1, provide peephole connections
    const int projection = INT_ARG(1);                            // if 1, then projection is performed, if false then numProj==numUnits is mandatory!!!!
    const T clippingCellValue  = T_ARG(0);                        // clipping value for ct, if it is not equal to zero, then cell state is clipped
    const T clippingProjValue  = T_ARG(1);                        // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    const T forgetBias   = T_ARG(2);

    const int rank     = xt->rankOf();    
    const int bS       = xt->sizeAt(0);
    const int inSize   = xt->sizeAt(1);
    const int numProj  = ht_1->sizeAt(1);
    const int numUnits = ct_1->sizeAt(1);    
 
    // input shapes validation
    const std::string ht_1Shape        = ShapeUtils<T>::shapeAsString(ht_1); 
    const std::string correctHt_1Shape = ShapeUtils<T>::shapeAsString({bS, numProj});
    const std::string ct_1Shape        = ShapeUtils<T>::shapeAsString(ct_1); 
    const std::string correctCt_1Shape = ShapeUtils<T>::shapeAsString({bS, numUnits});
    const std::string WxShape          = ShapeUtils<T>::shapeAsString(Wx); 
    const std::string correctWxShape   = ShapeUtils<T>::shapeAsString({inSize, 4*numUnits});
    const std::string WhShape          = ShapeUtils<T>::shapeAsString(Wh); 
    const std::string correctWhShape   = ShapeUtils<T>::shapeAsString({numProj, 4*numUnits});
    const std::string WcShape          = ShapeUtils<T>::shapeAsString(Wc); 
    const std::string correctWcShape   = ShapeUtils<T>::shapeAsString({3*numUnits});
    const std::string WpShape          = ShapeUtils<T>::shapeAsString(Wp);
    const std::string correctWpShape   = ShapeUtils<T>::shapeAsString({numUnits, numProj});
    const std::string bShape           = ShapeUtils<T>::shapeAsString(b); 
    const std::string correctBShape    = ShapeUtils<T>::shapeAsString({4*numUnits});

    REQUIRE_TRUE(correctHt_1Shape == ht_1Shape, 0, "LSTMCELL operation: wrong shape of initial cell output, expected is %s, but got %s instead !", correctHt_1Shape.c_str(), ht_1Shape.c_str()); 
    REQUIRE_TRUE(correctCt_1Shape == ct_1Shape, 0, "LSTMCELL operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", correctCt_1Shape.c_str(), ct_1Shape.c_str()); 
    REQUIRE_TRUE(correctWxShape == WxShape, 0, "LSTMCELL operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", correctWxShape.c_str(), WxShape.c_str()); 
    REQUIRE_TRUE(correctWhShape == WhShape, 0, "LSTMCELL operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", correctWhShape.c_str(), WhShape.c_str()); 
    REQUIRE_TRUE(correctWcShape == WcShape, 0, "LSTMCELL operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", correctWcShape.c_str(), WcShape.c_str()); 
    REQUIRE_TRUE(correctWpShape == WpShape, 0, "LSTMCELL operation: wrong shape of projection weights, expected is %s, but got %s instead !", correctWpShape.c_str(), WpShape.c_str()); 
    REQUIRE_TRUE(correctBShape  == bShape,  0, "LSTMCELL operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str());     
    REQUIRE_TRUE(!(!projection && numUnits != numProj), 0, "LSTMCELL operation: projection option is switched of, and in this case output dimensionality for the projection matrices (numProj) must be equal to number of units in lstmCell !");
    
    // calculations
    helpers::lstmCell<T>({xt,ht_1,ct_1, Wx,Wh,Wc,Wp, b},   {ht,ct},   {(T)peephole, (T)projection, clippingCellValue, clippingProjValue, forgetBias});
    
    return Status::OK();
}



DECLARE_SHAPE_FN(lstmCell) {    

    auto xtShapeInfo   = inputShape->at(0);                   // input [bS x inSize]
    auto ht_1ShapeInfo = inputShape->at(1);                   // previous cell output [bS x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!! 
    auto ct_1ShapeInfo = inputShape->at(2);                   // previous cell state  [bS x numUnits], that is at previous time step t-1   

    auto WxShapeInfo   = inputShape->at(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits] 
    auto WhShapeInfo   = inputShape->at(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits] 
    auto WcShapeInfo   = inputShape->at(5);                   // diagonal weights for peephole connections [3*numUnits] 
    auto WpShapeInfo   = inputShape->at(6);                   // projection weights [numUnits x numProj] 
    auto bShapeInfo    = inputShape->at(7);                   // biases, [4*numUnits] 
    
    const int rank     = xtShapeInfo[0];    
    const int bS       = xtShapeInfo[1];
    const int inSize   = xtShapeInfo[2];
    const int numProj  = ht_1ShapeInfo[2];
    const int numUnits = ct_1ShapeInfo[2];    
 
    // input shapes validation
    const std::string ht_1Shape        = ShapeUtils<T>::shapeAsString(ht_1ShapeInfo); 
    const std::string correctHt_1Shape = ShapeUtils<T>::shapeAsString({bS, numProj});
    const std::string ct_1Shape        = ShapeUtils<T>::shapeAsString(ct_1ShapeInfo); 
    const std::string correctCt_1Shape = ShapeUtils<T>::shapeAsString({bS, numUnits});
    const std::string WxShape          = ShapeUtils<T>::shapeAsString(WxShapeInfo); 
    const std::string correctWxShape   = ShapeUtils<T>::shapeAsString({inSize, 4*numUnits});
    const std::string WhShape          = ShapeUtils<T>::shapeAsString(WhShapeInfo); 
    const std::string correctWhShape   = ShapeUtils<T>::shapeAsString({numProj, 4*numUnits});
    const std::string WcShape          = ShapeUtils<T>::shapeAsString(WcShapeInfo ); 
    const std::string correctWcShape   = ShapeUtils<T>::shapeAsString({3*numUnits});
    const std::string WpShape          = ShapeUtils<T>::shapeAsString(WpShapeInfo); 
    const std::string correctWpShape   = ShapeUtils<T>::shapeAsString({numUnits, numProj});
    const std::string bShape           = ShapeUtils<T>::shapeAsString(bShapeInfo); 
    const std::string correctBShape    = ShapeUtils<T>::shapeAsString({4*numUnits});

    REQUIRE_TRUE(correctHt_1Shape == ht_1Shape, 0, "LSTMCELL operation: wrong shape of initial cell output, expected is %s, but got %s instead !", correctHt_1Shape.c_str(), ht_1Shape.c_str()); 
    REQUIRE_TRUE(correctCt_1Shape == ct_1Shape, 0, "LSTMCELL operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", correctCt_1Shape.c_str(), ct_1Shape.c_str()); 
    REQUIRE_TRUE(correctWxShape == WxShape, 0, "LSTMCELL operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", correctWxShape.c_str(), WxShape.c_str()); 
    REQUIRE_TRUE(correctWhShape == WhShape, 0, "LSTMCELL operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", correctWhShape.c_str(), WhShape.c_str()); 
    REQUIRE_TRUE(correctWcShape == WcShape, 0, "LSTMCELL operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", correctWcShape.c_str(), WcShape.c_str()); 
    REQUIRE_TRUE(correctWpShape == WpShape, 0, "LSTMCELL operation: wrong shape of projection weights, expected is %s, but got %s instead !", correctWpShape.c_str(), WpShape.c_str()); 
    REQUIRE_TRUE(correctBShape  == bShape,  0, "LSTMCELL operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str());     
    
    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [bS x numProj]
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [bS x numUnits]
            
    hShapeInfo[0] = cShapeInfo[0] = rank;
    hShapeInfo[1] = cShapeInfo[1] = bS;
    hShapeInfo[2] = numProj;
    cShapeInfo[2] = numUnits;    
    
    shape::updateStrides(hShapeInfo, shape::order(ht_1ShapeInfo));
    shape::updateStrides(cShapeInfo, shape::order(ct_1ShapeInfo));
         
    return SHAPELIST(hShapeInfo, cShapeInfo);
}   

}
}

#endif