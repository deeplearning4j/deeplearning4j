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
// @aurhot Yurii Shyrma
// @author Alex Black
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_gruCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/gru.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gruCell, 6, 4, false, 0, 0) {
    auto x      = INPUT_VARIABLE(0);                   // input [bS x inSize]
    auto hLast  = INPUT_VARIABLE(1);                   // previous cell output [bS x numUnits],  that is at previous time step t-1
    auto Wru    = INPUT_VARIABLE(2);                   // RU weights - [(nIn+nOut), 2*numUnits] - reset and update gates (input/recurrent weights)
    auto Wc     = INPUT_VARIABLE(3);                   // C weights - [(nIn+nOut), numUnits] - cell gate (input/recurrent weights)
    auto bru    = INPUT_VARIABLE(4);                   // reset and update biases, [2*numUnits] - reset and update gates
    auto bc     = INPUT_VARIABLE(5);                   // cell biases, [numUnits]

    auto r    =  OUTPUT_VARIABLE(0);                  // Reset gate output [bS, numUnits]
    auto u    =  OUTPUT_VARIABLE(1);                  // Update gate output [bS, numUnits]
    auto c    =  OUTPUT_VARIABLE(2);                  // Cell gate output [bS, numUnits]
    auto h    =  OUTPUT_VARIABLE(3);                  // current cell output [bS, numUnits]

    REQUIRE_TRUE(x->rankOf()==2 && hLast->rankOf()==2, 0, "gruCell: Input ranks must be 2 for inputs 0 and 1 (x, hLast) - got %i, %i", x->rankOf(), hLast->rankOf());

    const int rank     = x->rankOf();
    const auto bS       = x->sizeAt(0);
    const auto nIn   = x->sizeAt(1);
    const auto nU = hLast->sizeAt(1);

    REQUIRE_TRUE(x->sizeAt(0) == hLast->sizeAt(0), 0, "gruCell: Input minibatch sizes (dimension 0) must be same for x and hLast");
    REQUIRE_TRUE(Wru->rankOf()==2 && Wc->rankOf()==2, 0, "gruCell: weight arrays (Wru, Wc) arrays must be 2, got %i and %i", Wru->rankOf(), Wc->rankOf());
    REQUIRE_TRUE(Wru->sizeAt(0)==(nIn+nU) && Wc->sizeAt(0)==(nIn+nU), 0, "gruCell: Weights size(0) must be equal to inSize + numUnits, got %i", Wru->sizeAt(0));
    REQUIRE_TRUE(Wru->sizeAt(1)==(2*nU), 0, "gruCell: Weights (reset and update) size(1) must be equal to 2*numUnits, got %i", Wru->sizeAt(1));
    REQUIRE_TRUE(Wc->sizeAt(1)==nU, 0, "gruCell: Weights (cell) size(1) must be equal to numUnits, got %i", Wc->sizeAt(1));
    REQUIRE_TRUE(bru->rankOf()==1 && bru->sizeAt(0)==(2*nU), 0, "gruCell: reset/update biases must be rank 1, size 2*numUnits");
    REQUIRE_TRUE(bc->rankOf()==1 && bc->sizeAt(0)==nU, 0, "gruCell: cell biases must be rank 1, size numUnits");
    REQUIRE_TRUE(r->rankOf()==2 && u->rankOf()==2 && c->rankOf()==2 && h->rankOf()==2 &&
                 r->sizeAt(0)==bS && u->sizeAt(0)==bS && c->sizeAt(0)==bS && h->sizeAt(0)==bS &&
                 r->sizeAt(1)==nU && u->sizeAt(1)==nU && c->sizeAt(1)==nU && h->sizeAt(1)==nU,
                 0, "gruCell: Output arrays must all be rank 2 with size(0) == batchSize and size(1) == numUnits");

    helpers::gruCell(block.launchContext(), x, hLast, Wru, Wc, bru, bc, r, u, c, h);

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

    auto x      = inputShape->at(0);                   // input [bS x inSize]
    auto hLast  = inputShape->at(1);                   // previous cell output [bS x numUnits],  that is at previous time step t-1
    auto Wru    = inputShape->at(2);                   // RU weights - [(nIn+nOut), 2*numUnits] - reset and update gates (input/recurrent weights)
    auto Wc     = inputShape->at(3);                   // C weights - [(nIn+nOut), numUnits] - cell gate (input/recurrent weights)
    auto bru    = inputShape->at(4);                   // reset and update biases, [2*numUnits] - reset and update gates
    auto bc     = inputShape->at(5);                   // cell biases, [numUnits]

    REQUIRE_TRUE(shape::rank(x)==2 && shape::rank(hLast)==2, 0, "gruCell: Input ranks must be 2 for inputs 0 and 1 (x, hLast) - got %i, %i", shape::rank(x), shape::rank(hLast));

    const int rank     = x[0];
    const auto bS       = x[1];
    const auto inSize   = x[2];
    const auto numUnits = hLast[2];

    REQUIRE_TRUE(x[1] == hLast[1], 0, "gruCell: Input minibatch sizes (dimension 0) must be same for x and hLast");
    REQUIRE_TRUE(shape::rank(Wru)==2 && shape::rank(Wc)==2, 0, "gruCell: weight arrays (Wru, Wc) arrays must be 2, got %i and %i", shape::rank(Wru), shape::rank(Wc));
    REQUIRE_TRUE(Wru[1]==(inSize+numUnits) && Wc[1]==(inSize+numUnits), 0, "gruCell: Weights size(0) must be equal to inSize + numUnits, got %i and %i", Wru[1], Wc[1]);
    REQUIRE_TRUE(Wru[2]==(2*numUnits), 0, "gruCell: Weights (reset and update) size(1) must be equal to 2*numUnits, got %i", Wru[2]);
    REQUIRE_TRUE(Wc[2]==numUnits, 0, "gruCell: Weights (cell) size(1) must be equal to numUnits, got %i", Wc[2]);
    REQUIRE_TRUE(shape::rank(bru)==1 && bru[1]==(2*numUnits), 0, "gruCell: reset/update biases must be rank 1, size 2*numUnits");
    REQUIRE_TRUE(shape::rank(bc)==1 && bc[1]==numUnits, 0, "gruCell: cell biases must be rank 1, size numUnits");

    Nd4jLong *s0(nullptr);
    ALLOCATE(s0, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);// [bS x numUnits]

    s0[0] = rank;
    s0[1] = bS;
    s0[2] = numUnits;

    ShapeUtils::updateStridesAndType(s0, x, shape::order(hLast));

    Nd4jLong* s1 = ShapeBuilders::copyShapeInfo(s0, true, block.getWorkspace());
    Nd4jLong* s2 = ShapeBuilders::copyShapeInfo(s0, true, block.getWorkspace());
    Nd4jLong* s3 = ShapeBuilders::copyShapeInfo(s0, true, block.getWorkspace());

    //4 output shapes, all [bs, numUnits]
    return SHAPELIST(s0, s1, s2, s3);
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

    helpers::gruCellBP(block.launchContext(), x,  hi, Wx, Wh, b, dLdh, dLdWxi, dLdWhi, dLdbi, dLdx, dLdhi, dLdWx, dLdWh, dLdb);

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
