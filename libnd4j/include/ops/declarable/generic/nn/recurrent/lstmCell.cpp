/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstmCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstm.h>

namespace sd {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmCell, 8, 2, false, 3, 2) {
    auto xt   = INPUT_VARIABLE(0);                   // input [bS x inSize]
    auto ht_1 = INPUT_VARIABLE(1);                   // previous cell output [bS x numProj],  that is at previous time step t-1, in case of projection=false -> numProj=numUnits!!!
    auto ct_1 = INPUT_VARIABLE(2);                   // previous cell state  [bS x numUnits], that is at previous time step t-1

    auto Wx   = INPUT_VARIABLE(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits]
    auto Wh   = INPUT_VARIABLE(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits]
    auto Wc   = INPUT_VARIABLE(5);                   // diagonal weights for peephole connections [3*numUnits]
    auto Wp   = INPUT_VARIABLE(6);                   // projection weights [numUnits x numProj]
    auto b    = INPUT_VARIABLE(7);                   // biases, [4*numUnits]

    auto ht   =  OUTPUT_VARIABLE(0);                 // current cell output [bS x numProj], that is at current time step t
    auto ct   =  OUTPUT_VARIABLE(1);                 // current cell state  [bS x numUnits], that is at current time step t

    const int peephole   = INT_ARG(0);                            // if 1, provide peephole connections
    const int projection = INT_ARG(1);                            // if 1, then projection is performed, if false then numProj==numUnits is mandatory!!!!

    // FIXME: double?
    const double clippingCellValue  = T_ARG(0);                        // clipping value for ct, if it is not equal to zero, then cell state is clipped
    const double clippingProjValue  = T_ARG(1);                        // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    const double forgetBias   = T_ARG(2);

    const int rank     = xt->rankOf();
    const int bS       = xt->sizeAt(0);
    const int inSize   = xt->sizeAt(1);
    const int numProj  = ht_1->sizeAt(1);
    const int numUnits = ct_1->sizeAt(1);

    // input shapes validation
    const std::vector<Nd4jLong> correctHt_1Shape = {bS, numProj};
    const std::vector<Nd4jLong> correctCt_1Shape = {bS, numUnits};
    const std::vector<Nd4jLong> correctWxShape   = {inSize, 4*numUnits};
    const std::vector<Nd4jLong> correctWhShape   = {numProj, 4*numUnits};
    const std::vector<Nd4jLong> correctWcShape   = {3*numUnits};
    const std::vector<Nd4jLong> correctWpShape   = {numUnits, numProj};
    const std::vector<Nd4jLong> correctBShape    = {4*numUnits};

    REQUIRE_TRUE(ht_1->isSameShape(correctHt_1Shape), 0, "LSTMCELL operation: wrong shape of initial cell output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctHt_1Shape).c_str(), ShapeUtils::shapeAsString(ht_1).c_str());
    REQUIRE_TRUE(ct_1->isSameShape(correctCt_1Shape), 0, "LSTMCELL operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctCt_1Shape).c_str(), ShapeUtils::shapeAsString(ct_1).c_str());
    REQUIRE_TRUE(Wx->isSameShape(correctWxShape), 0, "LSTMCELL operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWxShape).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    REQUIRE_TRUE(Wh->isSameShape(correctWhShape), 0, "LSTMCELL operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWhShape).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(Wc->isSameShape(correctWcShape), 0, "LSTMCELL operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWcShape).c_str(), ShapeUtils::shapeAsString(Wc).c_str());
    REQUIRE_TRUE(Wp->isSameShape(correctWpShape), 0, "LSTMCELL operation: wrong shape of projection weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWpShape).c_str(), ShapeUtils::shapeAsString(Wp).c_str());
    REQUIRE_TRUE(b->isSameShape(correctBShape),  0, "LSTMCELL operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctBShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
    REQUIRE_TRUE(!(!projection && numUnits != numProj), 0, "LSTMCELL operation: projection option is switched of, and in this case output dimensionality for the projection matrices (numProj) must be equal to number of units in lstmCell !");

    // calculations
    helpers::lstmCell(block.launchContext(), xt,ht_1,ct_1, Wx,Wh,Wc,Wp, b,   ht,ct,   {(double)peephole, (double)projection, clippingCellValue, clippingProjValue, forgetBias});

    return Status::OK();
}

        DECLARE_TYPES(lstmCell) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
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

    const int rank     = shape::rank(xtShapeInfo);
    const auto bS       = xtShapeInfo[1];
    const auto inSize   = xtShapeInfo[2];
    const auto numProj  = ht_1ShapeInfo[2];
    const auto numUnits = ct_1ShapeInfo[2];

    // input shapes validation
    const std::vector<Nd4jLong> correctHt_1Shape = {bS, numProj};
    const std::vector<Nd4jLong> correctCt_1Shape = {bS, numUnits};
    const std::vector<Nd4jLong> correctWxShape   = {inSize, 4*numUnits};
    const std::vector<Nd4jLong> correctWhShape   = {numProj, 4*numUnits};
    const std::vector<Nd4jLong> correctWcShape   = {3*numUnits};
    const std::vector<Nd4jLong> correctWpShape   = {numUnits, numProj};
    const std::vector<Nd4jLong> correctBShape    = {4*numUnits};

    REQUIRE_TRUE(ShapeUtils::areShapesEqual(ht_1ShapeInfo, correctHt_1Shape), 0, "LSTMCELL operation: wrong shape of initial cell output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctHt_1Shape).c_str(), ShapeUtils::shapeAsString(ht_1ShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(ct_1ShapeInfo, correctCt_1Shape), 0, "LSTMCELL operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctCt_1Shape).c_str(), ShapeUtils::shapeAsString(ct_1ShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WxShapeInfo, correctWxShape), 0, "LSTMCELL operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWxShape).c_str(), ShapeUtils::shapeAsString(WxShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WhShapeInfo, correctWhShape), 0, "LSTMCELL operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWhShape).c_str(), ShapeUtils::shapeAsString(WhShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WcShapeInfo, correctWcShape), 0, "LSTMCELL operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWcShape).c_str(), ShapeUtils::shapeAsString(WcShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WpShapeInfo, correctWpShape), 0, "LSTMCELL operation: wrong shape of projection weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWpShape).c_str(), ShapeUtils::shapeAsString(WpShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(bShapeInfo, correctBShape),  0, "LSTMCELL operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctBShape).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());

    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [bS x numProj]
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [bS x numUnits]

    hShapeInfo[0] = cShapeInfo[0] = rank;
    hShapeInfo[1] = cShapeInfo[1] = bS;
    hShapeInfo[2] = numProj;
    cShapeInfo[2] = numUnits;

    ShapeUtils::updateStridesAndType(hShapeInfo, xtShapeInfo, shape::order(ht_1ShapeInfo));
    ShapeUtils::updateStridesAndType(cShapeInfo, xtShapeInfo, shape::order(ct_1ShapeInfo));

    auto result = SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(hShapeInfo), ConstantShapeHelper::getInstance().createShapeInfo(cShapeInfo));
    RELEASE(hShapeInfo, block.workspace());
    RELEASE(cShapeInfo, block.workspace());
    return result;
}

}
}

#endif