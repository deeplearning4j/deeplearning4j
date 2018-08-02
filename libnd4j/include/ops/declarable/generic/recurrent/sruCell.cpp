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
//  @author Yurii Shyrma, created on 05.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sruCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/sru.h>


namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sruCell, 4, 2, false, 0, 0) {

    NDArray<T>* xt   = INPUT_VARIABLE(0);               // input [bS x inSize], bS - batch size, inSize - number of features
    NDArray<T>* ct_1 = INPUT_VARIABLE(1);               // previous cell state ct  [bS x inSize], that is at previous time step t-1   
    NDArray<T>* w    = INPUT_VARIABLE(2);               // weights [inSize x 3*inSize]
    NDArray<T>* b    = INPUT_VARIABLE(3);               // biases [1 × 2*inSize]

    NDArray<T>* ht   = OUTPUT_VARIABLE(0);              // current cell output [bS x inSize], that is at current time step t
    NDArray<T>* ct   = OUTPUT_VARIABLE(1);              // current cell state  [bS x inSize], that is at current time step t

    const int rank   = xt->rankOf();
    const int bS     = xt->sizeAt(0);    
    const int inSize = xt->sizeAt(1);                   // inSize - number of features

    // input shapes validation
    const std::string ct_1Shape        = ShapeUtils<T>::shapeAsString(ct_1); 
    const std::string correctCt_1Shape = ShapeUtils<T>::shapeAsString({bS, inSize});
    const std::string WShape           = ShapeUtils<T>::shapeAsString(w); 
    const std::string correctWShape    = ShapeUtils<T>::shapeAsString({inSize, 3*inSize});
    const std::string bShape           = ShapeUtils<T>::shapeAsString(b); 
    const std::string correctBShape    = ShapeUtils<T>::shapeAsString({2*inSize});

    REQUIRE_TRUE(correctCt_1Shape == ct_1Shape, 0, "SRUCELL operation: wrong shape of previous cell state, expected is %s, but got %s instead !", correctCt_1Shape.c_str(), ct_1Shape.c_str()); 
    REQUIRE_TRUE(correctWShape    == WShape,    0, "SRUCELL operation: wrong shape of weights, expected is %s, but got %s instead !", correctWShape.c_str(), WShape.c_str()); 
    REQUIRE_TRUE(correctBShape    == bShape,    0, "SRUCELL operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str()); 

            
    helpers::sruCell<T>({xt, ct_1, w, b}, {ht, ct});
    
    return Status::OK();
}


DECLARE_SHAPE_FN(sruCell) {

    auto xtShapeInfo   = inputShape->at(0);               // input [bS x inSize], bS - batch size, inSize - number of features
    auto ct_1ShapeInfo = inputShape->at(1);               // previous cell state ct  [bS x inSize], that is at previous time step t-1   
    auto wShapeInfo    = inputShape->at(2);               // weights [inSize x 3*inSize]
    auto bShapeInfo    = inputShape->at(3);               // biases [2*inSize]

    const int rank   = xtShapeInfo[0];
    const int bS     = xtShapeInfo[1];    
    const int inSize = xtShapeInfo[2];                   // inSize - number of features

    // input shapes validation
    const std::string ct_1Shape        = ShapeUtils<T>::shapeAsString(ct_1ShapeInfo); 
    const std::string correctCt_1Shape = ShapeUtils<T>::shapeAsString({bS, inSize});
    const std::string WShape           = ShapeUtils<T>::shapeAsString(wShapeInfo); 
    const std::string correctWShape    = ShapeUtils<T>::shapeAsString({inSize, 3*inSize});
    const std::string bShape           = ShapeUtils<T>::shapeAsString(bShapeInfo); 
    const std::string correctBShape    = ShapeUtils<T>::shapeAsString({2*inSize});

    REQUIRE_TRUE(correctCt_1Shape == ct_1Shape, 0, "SRUCELL operation: wrong shape of previous cell state, expected is %s, but got %s instead !", correctCt_1Shape.c_str(), ct_1Shape.c_str()); 
    REQUIRE_TRUE(correctWShape    == WShape,    0, "SRUCELL operation: wrong shape of weights, expected is %s, but got %s instead !", correctWShape.c_str(), WShape.c_str()); 
    REQUIRE_TRUE(correctBShape    == bShape,    0, "SRUCELL operation: wrong shape of biases, expected is %s, but got %s instead !", correctBShape.c_str(), bShape.c_str()); 
    
    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [bS x numProj]
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [bS x numUnits]
            
    hShapeInfo[0] = cShapeInfo[0] = rank;
    hShapeInfo[1] = cShapeInfo[1] = bS;
    hShapeInfo[2] = cShapeInfo[2] = inSize;
    
    shape::updateStrides(hShapeInfo, shape::order(ct_1ShapeInfo));
    shape::updateStrides(cShapeInfo, shape::order(ct_1ShapeInfo));
         
    return SHAPELIST(hShapeInfo, cShapeInfo);
}   




}
}

#endif