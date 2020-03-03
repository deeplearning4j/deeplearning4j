/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstm)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstm.h>

namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstm, 8, 2, false, 3, 2) {
    auto x  = INPUT_VARIABLE(0);                    // input [time x bS x inSize]
    auto h0 = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [bS x numProj], in case of projection=false -> numProj == numUnits !!!
    auto c0 = INPUT_VARIABLE(2);                    // initial cell state  (at time step = 0) [bS x numUnits],

    auto Wx  = INPUT_VARIABLE(3);                   // input-to-hidden  weights, [inSize  x 4*numUnits]
    auto Wh  = INPUT_VARIABLE(4);                   // hidden-to-hidden weights, [numProj x 4*numUnits]
    auto Wc  = INPUT_VARIABLE(5);                   // diagonal weights for peephole connections [3*numUnits]
    auto Wp  = INPUT_VARIABLE(6);                   // projection weights [numUnits x numProj]
    auto b   = INPUT_VARIABLE(7);                   // biases, [4*numUnits]

    auto h   =  OUTPUT_VARIABLE(0);                 // cell outputs [time x bS x numProj], that is per each time step
    auto c   =  OUTPUT_VARIABLE(1);                 // cell states  [time x bS x numUnits] that is per each time step

    const int peephole   = INT_ARG(0);                     // if 1, provide peephole connections
    const int projection = INT_ARG(1);                     // if 1, then projection is performed, if false then numProj==numUnits is mandatory!!!!

    // FIXME: double
    const double clippingCellValue  = T_ARG(0);                 // clipping value for ct, if it is not equal to zero, then cell state is clipped
    const double clippingProjValue  = T_ARG(1);                 // clipping value for projected ht, if it is not equal to zero, then projected cell output is clipped
    const double forgetBias   = T_ARG(2);

    const int rank     = x->rankOf();
    const int time     = x->sizeAt(0);
    const int bS       = x->sizeAt(1);
    const int inSize   = x->sizeAt(2);
    const int numProj  = h0->sizeAt(1);
    const int numUnits = c0->sizeAt(1);

    // input shapes validation
    const std::vector<Nd4jLong> correctH0Shape = {bS, numProj};
    const std::vector<Nd4jLong> correctC0Shape = {bS, numUnits};
    const std::vector<Nd4jLong> correctWxShape = {inSize, 4*numUnits};
    const std::vector<Nd4jLong> correctWhShape = {numProj, 4*numUnits};
    const std::vector<Nd4jLong> correctWcShape = {3*numUnits};
    const std::vector<Nd4jLong> correctWpShape = {numUnits, numProj};
    const std::vector<Nd4jLong> correctBShape  = {4*numUnits};

    REQUIRE_TRUE(h0->isSameShape(correctH0Shape), 0, "LSTM operation: wrong shape of initial cell output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctH0Shape).c_str(), ShapeUtils::shapeAsString(h0).c_str());
    REQUIRE_TRUE(c0->isSameShape(correctC0Shape), 0, "LSTM operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctC0Shape).c_str(), ShapeUtils::shapeAsString(c0).c_str());
    REQUIRE_TRUE(Wx->isSameShape(correctWxShape), 0, "LSTM operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWxShape).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    REQUIRE_TRUE(Wh->isSameShape(correctWhShape), 0, "LSTM operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWhShape).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(Wc->isSameShape(correctWcShape), 0, "LSTM operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWcShape).c_str(), ShapeUtils::shapeAsString(Wc).c_str());
    REQUIRE_TRUE(Wp->isSameShape(correctWpShape), 0, "LSTM operation: wrong shape of projection weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWpShape).c_str(), ShapeUtils::shapeAsString(Wp).c_str());
    REQUIRE_TRUE(b->isSameShape(correctBShape),  0, "LSTM operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctBShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
    REQUIRE_TRUE(!(!projection && numUnits != numProj), 0, "LSTM operation: projection option is switched of, and in this case output dimensionality for the projection matrices (numProj) must be equal to number of units in lstmCell !");

    helpers::lstmTimeLoop(block.launchContext(), x, h0, c0, Wx, Wh, Wc, Wp, b, h, c, {(double)peephole, (double)projection, clippingCellValue, clippingProjValue, forgetBias});

    return Status::OK();
}

        DECLARE_TYPES(lstm) {
            getOpDescriptor()
                    ->setAllowedInputTypes(sd::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
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
    const std::vector<Nd4jLong> correctH0Shape = {bS, numProj};
    const std::vector<Nd4jLong> correctC0Shape = {bS, numUnits};
    const std::vector<Nd4jLong> correctWxShape = {inSize, 4*numUnits};
    const std::vector<Nd4jLong> correctWhShape = {numProj, 4*numUnits};
    const std::vector<Nd4jLong> correctWcShape = {3*numUnits};
    const std::vector<Nd4jLong> correctWpShape = {numUnits, numProj};
    const std::vector<Nd4jLong> correctBShape  = {4*numUnits};

    REQUIRE_TRUE(ShapeUtils::areShapesEqual(h0ShapeInfo, correctH0Shape), 0, "LSTM operation: wrong shape of initial cell output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctH0Shape).c_str(), ShapeUtils::shapeAsString(h0ShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(c0ShapeInfo, correctC0Shape), 0, "LSTM operation: wrong shape of initial cell state,  expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctC0Shape).c_str(), ShapeUtils::shapeAsString(c0ShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WxShapeInfo, correctWxShape), 0, "LSTM operation: wrong shape of input-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWxShape).c_str(), ShapeUtils::shapeAsString(WxShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WhShapeInfo, correctWhShape), 0, "LSTM operation: wrong shape of hidden-to-hidden weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWhShape).c_str(), ShapeUtils::shapeAsString(WhShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WcShapeInfo, correctWcShape), 0, "LSTM operation: wrong shape of diagonal weights for peephole connections, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWcShape).c_str(), ShapeUtils::shapeAsString(WcShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WpShapeInfo, correctWpShape), 0, "LSTM operation: wrong shape of projection weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctWpShape).c_str(), ShapeUtils::shapeAsString(WpShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(bShapeInfo, correctBShape),  0, "LSTM operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(correctBShape).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());


    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *cShapeInfo(nullptr);
    ALLOCATE(hShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [time x bS x numProj]
    ALLOCATE(cShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);      // [time x bS x numUnits]

    hShapeInfo[0] = cShapeInfo[0] = rank;
    hShapeInfo[1] = cShapeInfo[1] = time;
    hShapeInfo[2] = cShapeInfo[2] = bS;
    hShapeInfo[3] = numProj;
    cShapeInfo[3] = numUnits;

    ShapeUtils::updateStridesAndType(hShapeInfo, xShapeInfo, shape::order(h0ShapeInfo));
    ShapeUtils::updateStridesAndType(cShapeInfo, xShapeInfo, shape::order(c0ShapeInfo));

    return SHAPELIST(CONSTANT(hShapeInfo), CONSTANT(cShapeInfo));
}





}
}

#endif