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
// created by Yurii Shyrma on 05.12.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_gruCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/gru.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gruCell, 5, 1, false, 0, 0) {
    auto x  = INPUT_VARIABLE(0);                     // input [bS x inSize]
    auto h0 = INPUT_VARIABLE(1);                     // previous cell output [bS x numUnits],  that is at previous time step t-1

    auto Wx   = INPUT_VARIABLE(2);                   // input-to-hidden weights, [inSize   x 3*numUnits]
    auto Wh   = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [numUnits x 3*numUnits]
    auto b    = INPUT_VARIABLE(4);                   // biases, [3*numUnits]
    
    auto h    =  OUTPUT_VARIABLE(0);                  // current cell output [bS x numUnits], that is at current time step t

    const int rank     = x->rankOf();              // = 2
    const auto bS       = x->sizeAt(0);
    const auto inSize   = x->sizeAt(1);
    const auto numUnits = h0->sizeAt(1);

    const std::string h0Shape        = ShapeUtils::shapeAsString(h0);
    const std::string h0CorrectShape = ShapeUtils::shapeAsString({bS, numUnits});
    const std::string wxShape        = ShapeUtils::shapeAsString(Wx);
    const std::string wxCorrectShape = ShapeUtils::shapeAsString({inSize, 3*numUnits});
    const std::string whShape        = ShapeUtils::shapeAsString(Wh);
    const std::string whCorrectShape = ShapeUtils::shapeAsString({numUnits, 3*numUnits});
    const std::string bShape         = ShapeUtils::shapeAsString(b);
    const std::string bCorrectShape  = ShapeUtils::shapeAsString({3*numUnits});
    
    REQUIRE_TRUE(h0Shape == h0CorrectShape, 0, "GRUCELL operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", h0CorrectShape.c_str(), h0Shape.c_str());
    REQUIRE_TRUE(wxShape == wxCorrectShape, 0, "GRUCELL operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str());
    REQUIRE_TRUE(whShape == whCorrectShape, 0, "GRUCELL operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "GRUCELL operation: wrong shape of biases  array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());

    helpers::gruCell(x, h0, Wx, Wh, b, h);

    return Status::OK();
}

DECLARE_TYPES(gruCell) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, nd4j::DataType::ANY)
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedInputTypes(2, {ALL_FLOATS})
        ->setAllowedInputTypes(3, {ALL_FLOATS})
        ->setAllowedInputTypes(4, {ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}


DECLARE_SHAPE_FN(gruCell) {    
    
    const auto xShapeInfo  = inputShape->at(0);                     // input [bS x inSize]
    const auto h0ShapeInfo = inputShape->at(1);                     // previous cell output [bS x numUnits],  that is at previous time step t-1
    const auto WxShapeInfo = inputShape->at(2);                     // input-to-hidden weights, [inSize   x 3*numUnits]
    const auto WhShapeInfo = inputShape->at(3);                     // hidden-to-hidden weights, [numUnits x 3*numUnits]
    const auto bShapeInfo  = inputShape->at(4);                     // biases, [3*numUnits]

    const int rank     = shape::rank(xShapeInfo);              // = 2
    const auto bS       = xShapeInfo[1];
    const auto inSize   = xShapeInfo[2];
    const auto numUnits = h0ShapeInfo[2];

    const std::string h0Shape        = ShapeUtils::shapeAsString(h0ShapeInfo);
    const std::string h0CorrectShape = ShapeUtils::shapeAsString({bS, numUnits});
    const std::string wxShape        = ShapeUtils::shapeAsString(WxShapeInfo);
    const std::string wxCorrectShape = ShapeUtils::shapeAsString({inSize, 3*numUnits});
    const std::string whShape        = ShapeUtils::shapeAsString(WhShapeInfo);
    const std::string whCorrectShape = ShapeUtils::shapeAsString({numUnits, 3*numUnits});
    const std::string bShape         = ShapeUtils::shapeAsString(bShapeInfo);
    const std::string bCorrectShape  = ShapeUtils::shapeAsString({3*numUnits});

    REQUIRE_TRUE(h0Shape == h0CorrectShape, 0, "GRUCELL operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", h0CorrectShape.c_str(), h0Shape.c_str());
    REQUIRE_TRUE(wxShape == wxCorrectShape, 0, "GRUCELL operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str());
    REQUIRE_TRUE(whShape == whCorrectShape, 0, "GRUCELL operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());
    REQUIRE_TRUE(bShape  == bCorrectShape,  0, "GRUCELL operation: wrong shape of biases  array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());
    Nd4jLong *hShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);// [bS x numUnits]

    hShapeInfo[0] = rank;
    hShapeInfo[1] = bS;
    hShapeInfo[2] = numUnits;

    ShapeUtils::updateStridesAndType(hShapeInfo, xShapeInfo, shape::order(h0ShapeInfo));

    return SHAPELIST(hShapeInfo);
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gruCell_bp, 6, 5, false, 0, 0) {

    auto x      = INPUT_VARIABLE(0);                                 // input [bS x iS]
    auto hi     = INPUT_VARIABLE(1);                                 // previous cell output [bS x nU]
    auto Wx     = INPUT_VARIABLE(2);                                 // input-to-hidden  weights, [iS x 3*nU]
    auto Wh     = INPUT_VARIABLE(3);                                 // hidden-to-hidden weights, [nU x 3*nU]
    auto b      = INPUT_VARIABLE(4);                                 // biases, [3*nU]
    auto dLdh   = INPUT_VARIABLE(5);                                 // gradient wrt output, [bS,nU], that is epsilon_next
    auto dLdWxi = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;   // gradient wrt Wx at previous time step, [iS, 3*nU]
    auto dLdWhi = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;   // gradient wrt Wh at previous time step, [nU, 3*nU]
    auto dLdbi  = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;   // gradient wrt b at previous time step,  [3*nU]

    auto dLdx   = OUTPUT_VARIABLE(0);                                // gradient wrt x,  [bS, iS], that is epsilon
    auto dLdhi  = OUTPUT_VARIABLE(1);                                // gradient wrt hi, [bS, nU]
    auto dLdWx  = OUTPUT_VARIABLE(2);                                // gradient wrt Wx, [iS, 3*nU]
    auto dLdWh  = OUTPUT_VARIABLE(3);                                // gradient wrt Wh, [nU, 3*nU]
    auto dLdb   = OUTPUT_VARIABLE(4);                                // gradient wrt biases,  [3*nU]

    const int rank     = x->rankOf();                               // = 2
    const Nd4jLong bS  = x->sizeAt(0);
    const Nd4jLong iS  = x->sizeAt(1);
    const Nd4jLong nU  = hi->sizeAt(1);

    const std::string hiShape          = ShapeUtils::shapeAsString(hi);
    const std::string hiCorrectShape   = ShapeUtils::shapeAsString({bS, nU});
    const std::string wxShape          = ShapeUtils::shapeAsString(Wx);
    const std::string wxCorrectShape   = ShapeUtils::shapeAsString({iS, 3*nU});
    const std::string whShape          = ShapeUtils::shapeAsString(Wh);
    const std::string whCorrectShape   = ShapeUtils::shapeAsString({nU, 3*nU});
    const std::string bShape           = ShapeUtils::shapeAsString(b);
    const std::string bCorrectShape    = ShapeUtils::shapeAsString({3*nU});
    const std::string dLdhShape        = ShapeUtils::shapeAsString(dLdh);
    const std::string dLdhCorrectShape = ShapeUtils::shapeAsString({bS, nU});

    REQUIRE_TRUE(hiShape   == hiCorrectShape,    0, "GRU_CELL_BP op: wrong shape of previous cell output array, expected is %s, but got %s instead !", hiCorrectShape.c_str(), hiShape.c_str());
    REQUIRE_TRUE(wxShape   == wxCorrectShape,    0, "GRU_CELL_BP op: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str());
    REQUIRE_TRUE(whShape   == whCorrectShape,    0, "GRU_CELL_BP op: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());
    REQUIRE_TRUE(bShape    == bCorrectShape,     0, "GRU_CELL_BP op: wrong shape of biases array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());
    REQUIRE_TRUE(dLdhShape == dLdhCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdh array (epsilon_next), expected is %s, but got %s instead !", dLdhCorrectShape.c_str(), dLdhShape.c_str());

    if(dLdWxi != nullptr) {
        const std::string dLdWxiShape        = ShapeUtils::shapeAsString(dLdWxi);
        const std::string dLdWxiCorrectShape = ShapeUtils::shapeAsString({iS, 3*nU});
        REQUIRE_TRUE(dLdWxiShape == dLdWxiCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdWxi array (gradient wrt Wx at previous time step), expected is %s, but got %s instead !", dLdWxiCorrectShape.c_str(), dLdWxiShape.c_str());
    }

    if(dLdWhi != nullptr) {
        const std::string dLdWhiShape        = ShapeUtils::shapeAsString(dLdWhi);
        const std::string dLdWhiCorrectShape = ShapeUtils::shapeAsString({nU, 3*nU});
        REQUIRE_TRUE(dLdWhiShape == dLdWhiCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdWhi array (gradient wrt Wh at previous time step), expected is %s, but got %s instead !", dLdWhiCorrectShape.c_str(), dLdWhiShape.c_str());
    }

    if(dLdbi != nullptr) {
        const std::string dLdbiShape        = ShapeUtils::shapeAsString(dLdbi);
        const std::string dLdbiCorrectShape = ShapeUtils::shapeAsString({3*nU});
        REQUIRE_TRUE(dLdbiShape == dLdbiCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdbi array (gradient wrt biases at previous time step), expected is %s, but got %s instead !", dLdbiCorrectShape.c_str(), dLdbiShape.c_str());
    }

    helpers::gruCellBP(x,  hi, Wx, Wh, b, dLdh, dLdWxi, dLdWhi, dLdbi, dLdx, dLdhi, dLdWx, dLdWh, dLdb);

    return Status::OK();
}

DECLARE_TYPES(gruCell_bp) {
    getOpDescriptor()
        ->setAllowedInputTypes(0, nd4j::DataType::ANY)
        ->setAllowedInputTypes(1, {ALL_FLOATS})
        ->setAllowedInputTypes(2, {ALL_FLOATS})
        ->setAllowedInputTypes(3, {ALL_FLOATS})
        ->setAllowedInputTypes(4, {ALL_FLOATS})
        ->setAllowedInputTypes(5, {ALL_FLOATS})
        ->setAllowedInputTypes(6, {ALL_FLOATS})
        ->setAllowedInputTypes(7, {ALL_FLOATS})
        ->setAllowedInputTypes(8, {ALL_FLOATS})
        ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(gruCell_bp) {

    auto xShapeInfo      = inputShape->at(0);                                              // [bS x iS]
    auto hiShapeInfo     = inputShape->at(1);                                              // [bS x nU]
    auto wxShapeInfo     = inputShape->at(2);                                              // [iS x 3*nU]
    auto whShapeInfo     = inputShape->at(3);                                              // [nU x 3*nU]
    auto bShapeInfo      = inputShape->at(4);                                              // [3*nU]
    auto dLdhShapeInfo   = inputShape->at(5);                                              // [bS x nU]

    const int rank    = xShapeInfo[0];                               // = 2
    const Nd4jLong bS = xShapeInfo[1];
    const Nd4jLong iS = xShapeInfo[2];
    const Nd4jLong nU = hiShapeInfo[2];

    const std::string hiShape          = ShapeUtils::shapeAsString(hiShapeInfo);
    const std::string hiCorrectShape   = ShapeUtils::shapeAsString({bS, nU});
    const std::string wxShape          = ShapeUtils::shapeAsString(wxShapeInfo);
    const std::string wxCorrectShape   = ShapeUtils::shapeAsString({iS, 3*nU});
    const std::string whShape          = ShapeUtils::shapeAsString(whShapeInfo);
    const std::string whCorrectShape   = ShapeUtils::shapeAsString({nU, 3*nU});
    const std::string bShape           = ShapeUtils::shapeAsString(bShapeInfo);
    const std::string bCorrectShape    = ShapeUtils::shapeAsString({3*nU});
    const std::string dLdhShape        = ShapeUtils::shapeAsString(dLdhShapeInfo);
    const std::string dLdhCorrectShape = ShapeUtils::shapeAsString({bS, nU});

    REQUIRE_TRUE(hiShape   == hiCorrectShape,    0, "GRU_CELL_BP op: wrong shape of previous cell output array, expected is %s, but got %s instead !", hiCorrectShape.c_str(), hiShape.c_str());
    REQUIRE_TRUE(wxShape   == wxCorrectShape,    0, "GRU_CELL_BP op: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", wxCorrectShape.c_str(), wxShape.c_str());
    REQUIRE_TRUE(whShape   == whCorrectShape,    0, "GRU_CELL_BP op: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", whCorrectShape.c_str(), whShape.c_str());
    REQUIRE_TRUE(bShape    == bCorrectShape,     0, "GRU_CELL_BP op: wrong shape of biases array, expected is %s, but got %s instead !", bCorrectShape.c_str(), bShape.c_str());
    REQUIRE_TRUE(dLdhShape == dLdhCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdh array (epsilon_next), expected is %s, but got %s instead !", dLdhCorrectShape.c_str(), dLdhShape.c_str());

    if(block.width() > 6) {
        Nd4jLong* dLdWxiShapeInfo = inputShape->at(6);                                              // [iS x 3*nU]
        const std::string dLdWxiShape        = ShapeUtils::shapeAsString(dLdWxiShapeInfo);
        const std::string dLdWxiCorrectShape = ShapeUtils::shapeAsString({iS, 3*nU});
        REQUIRE_TRUE(dLdWxiShape == dLdWxiCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdWxi array (gradient wrt Wx at previous time step), expected is %s, but got %s instead !", dLdWxiCorrectShape.c_str(), dLdWxiShape.c_str());
    }

    if(block.width() > 7) {
        Nd4jLong* dLdWhiShapeInfo = inputShape->at(7);                                              // [nU x 3*nU]
        const std::string dLdWhiShape        = ShapeUtils::shapeAsString(dLdWhiShapeInfo);
        const std::string dLdWhiCorrectShape = ShapeUtils::shapeAsString({nU, 3*nU});
        REQUIRE_TRUE(dLdWhiShape == dLdWhiCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdWhi array (gradient wrt Wh at previous time step), expected is %s, but got %s instead !", dLdWhiCorrectShape.c_str(), dLdWhiShape.c_str());
    }

    if(block.width() > 8) {
        Nd4jLong* dLdbiShapeInfo  = inputShape->at(8);                                              // [3*nU]
        const std::string dLdbiShape        = ShapeUtils::shapeAsString(dLdbiShapeInfo);
        const std::string dLdbiCorrectShape = ShapeUtils::shapeAsString({3*nU});
        REQUIRE_TRUE(dLdbiShape == dLdbiCorrectShape,  0, "GRU_CELL_BP op: wrong shape of dLdbi array (gradient wrt biases at previous time step), expected is %s, but got %s instead !", dLdbiCorrectShape.c_str(), dLdbiShape.c_str());
    }

    Nd4jLong *dLdxShapeInfo = nullptr;
    COPY_SHAPE(xShapeInfo, dLdxShapeInfo);

    Nd4jLong *dLdhiShapeInfo = nullptr;
    COPY_SHAPE(hiShapeInfo, dLdhiShapeInfo);

    Nd4jLong *dLdWxShapeInfo = nullptr;
    COPY_SHAPE(wxShapeInfo, dLdWxShapeInfo);

    Nd4jLong *dLdWhShapeInfo = nullptr;
    COPY_SHAPE(whShapeInfo, dLdWhShapeInfo);

    Nd4jLong *dLdbShapeInfo = nullptr;
    COPY_SHAPE(bShapeInfo, dLdbShapeInfo);

    return SHAPELIST(dLdxShapeInfo, dLdhiShapeInfo, dLdWxShapeInfo, dLdWhShapeInfo, dLdbShapeInfo);

}




}
}

#endif
