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
// @author Yurii Shyrma, created on 05.04.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/rnn.h>

namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(dynamic_rnn, 4, 2, false, 0, 0) {

    auto x  = INPUT_VARIABLE(0);               // input [time x bS x inSize] or [bS x time x inSize], depends on timeMajor parameter
	auto Wx = INPUT_VARIABLE(1);               // input-to-hidden  weights, [inSize  x numUnits]
    auto Wh = INPUT_VARIABLE(2);               // hidden-to-hidden weights, [numUnits x numUnits]
	auto b  = INPUT_VARIABLE(3);               // biases for, [2*numUnits]

	NDArray* h0          = nullptr;     		      // initial cell output (at time step = 0) [bS x numUnits]
	NDArray* maxTimeStep = nullptr;			      // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    const int timeMajor = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // if true then [time, bS, ...], else [bS, time, ...]

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

    auto h      =  OUTPUT_VARIABLE(0);           // cell outputs [time x bS x numUnits] or [bS x time x numUnits], depends on timeMajor parameter
    auto hFinal =  OUTPUT_VARIABLE(1);           // at the end it will store cell final non-zero output [bS x numUnits]

    REQUIRE_TRUE(x->rankOf() == 3, 0, "DYNAMIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", x->rankOf());
    REQUIRE_TRUE(Wx->rankOf() == 2, 0, "DYNAMIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", Wx->rankOf());

    const int inRank   = x->rankOf();
    const int time     = timeMajor ? x->sizeAt(0) : x->sizeAt(1);
    const int bS       = timeMajor ? x->sizeAt(1) : x->sizeAt(0);
    const int numUnits = Wx->sizeAt(1);

    std::vector<Nd4jLong> expectedWhShape = {numUnits, numUnits};
    std::vector<Nd4jLong> expectedBShape = {2*numUnits};
    REQUIRE_TRUE(Wh->isSameShape(expectedWhShape), 0, "DYNAMIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWhShape).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(b->isSameShape(expectedBShape), 0, "DYNAMIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedBShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
    if(h0) {
        std::vector<Nd4jLong> expectedh0Shape = {bS, numUnits};
        REQUIRE_TRUE(h0->isSameShape(expectedh0Shape), 0, "DYNAMIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils::shapeAsString(expectedh0Shape).c_str(), ShapeUtils::shapeAsString(h0).c_str());
    }
    if(maxTimeStep) {
        std::vector<Nd4jLong> expectedmaxTimeStepShape = {bS};
        REQUIRE_TRUE(maxTimeStep->isSameShape(expectedmaxTimeStepShape), 0, "DYNAMIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedmaxTimeStepShape).c_str(), ShapeUtils::shapeAsString(maxTimeStep).c_str());
    }

    if(timeMajor == false) {
        x = new NDArray(x->permute({1, 0, 2}));                      // [bS x time x inSize]   -> [time x bS x inSize]
        h = new NDArray(h->permute({1, 0, 2}));                      // [bS x time x numUnits] -> [time x bS x numUnits]
    }

    helpers::rnnTimeLoop(block.launchContext(), x, Wx, Wh, b, h0, maxTimeStep, h, hFinal);

    if(timeMajor == false) {
        delete x;
        delete h;
    }

    return Status::OK();
}


        DECLARE_TYPES(dynamic_rnn) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, sd::DataType::ANY)
                    ->setAllowedInputTypes(1, {ALL_FLOATS})
                    ->setAllowedInputTypes(2, {ALL_FLOATS})
                    ->setAllowedInputTypes(3, {ALL_FLOATS})
                    ->setAllowedInputTypes(4, {ALL_FLOATS, ALL_INTS})
                    ->setAllowedInputTypes(5, {ALL_FLOATS, ALL_INTS})
                    ->setAllowedOutputTypes(0, {ALL_FLOATS})
                    ->setAllowedOutputTypes(1, {ALL_FLOATS});
        }


DECLARE_SHAPE_FN(dynamic_rnn) {

    auto xShapeInfo  = inputShape->at(0);               // input [time x bS x inSize] or [bS x time x inSize], depends on timeMajor parameter
    auto WxShapeInfo = inputShape->at(1);               // input-to-hidden  weights, [inSize  x numUnits]
    auto WhShapeInfo = inputShape->at(2);               // hidden-to-hidden weights, [numUnits x numUnits]
    auto bShapeInfo  = inputShape->at(3);               // biases for, [2*numUnits]

    Nd4jLong const* h0ShapeInfo          = nullptr;                // initial cell output (at time step = 0) [bS x numUnits]
    Nd4jLong const* maxTimeStepShapeInfo = nullptr;                // vector [bS] containing integer values within [0,time), each element of this vector set max time step per each input in batch, this means there are no calculations for time >= maxTimeStep

    const int timeMajor = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;       // if true then [time, bS, ...], else [bS, time, ...]

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

    REQUIRE_TRUE(xShapeInfo[0] == 3, 0, "DYNAMIC_RNN custom operation: input array x must have rank = 3, but got %i instead !", xShapeInfo[0]);
    REQUIRE_TRUE(WxShapeInfo[0] == 2, 0, "DYNAMIC_RNN custom operation: input-to-hidden weights array must have rank = 2, but got %i instead !", WxShapeInfo[0]);

    const int inRank   = xShapeInfo[0];
    const int time     = timeMajor ? xShapeInfo[1] : xShapeInfo[2];
    const int bS       = timeMajor ? xShapeInfo[2] : xShapeInfo[1];
    const int numUnits = WxShapeInfo[2];


    std::vector<Nd4jLong> expectedWhShape = {numUnits, numUnits};
    std::vector<Nd4jLong> expectedBShape  = {2*numUnits};
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(WhShapeInfo, expectedWhShape), 0, "DYNAMIC_RNN custom operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWhShape).c_str(), ShapeUtils::shapeAsString(WhShapeInfo).c_str());
    REQUIRE_TRUE(ShapeUtils::areShapesEqual(bShapeInfo, expectedBShape), 0, "DYNAMIC_RNN custom operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedBShape).c_str(), ShapeUtils::shapeAsString(bShapeInfo).c_str());
    if(h0ShapeInfo) {
        std::vector<Nd4jLong> expectedh0Shape = {bS, numUnits};
        REQUIRE_TRUE(ShapeUtils::areShapesEqual(h0ShapeInfo, expectedh0Shape), 0, "DYNAMIC_RNN custom operation: wrong shape of initial cell output array, expected is %s but got %s instead !", ShapeUtils::shapeAsString(expectedh0Shape).c_str(), ShapeUtils::shapeAsString(h0ShapeInfo).c_str());
    }
    if(maxTimeStepShapeInfo) {
        std::vector<Nd4jLong> expectedmaxTimeStepShape = {bS};
        REQUIRE_TRUE(ShapeUtils::areShapesEqual(maxTimeStepShapeInfo, expectedmaxTimeStepShape), 0, "DYNAMIC_RNN custom operation: wrong shape of maxTimeStep array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedmaxTimeStepShape).c_str(), ShapeUtils::shapeAsString(maxTimeStepShapeInfo).c_str());
    }

    // evaluate output shapeInfos
    Nd4jLong *hShapeInfo(nullptr), *hPrevShapeInfo(nullptr);
    ALLOCATE(hShapeInfo,     block.getWorkspace(), shape::shapeInfoLength(inRank), Nd4jLong);
    ALLOCATE(hPrevShapeInfo, block.getWorkspace(), shape::shapeInfoLength(inRank-1), Nd4jLong);

    hShapeInfo[0]     = inRank;
    hPrevShapeInfo[0] = inRank-1;
    hShapeInfo[1]     = timeMajor ? time : bS;
    hShapeInfo[2]     = timeMajor ? bS : time;
    hPrevShapeInfo[1] = bS;
    hShapeInfo[3]     = hPrevShapeInfo[2] = numUnits;

    ShapeUtils::updateStridesAndType(hShapeInfo, WhShapeInfo, shape::order(xShapeInfo));
    ShapeUtils::updateStridesAndType(hPrevShapeInfo, WhShapeInfo, shape::order(xShapeInfo));

    return SHAPELIST(CONSTANT(hShapeInfo), CONSTANT(hPrevShapeInfo));
}





}
}
