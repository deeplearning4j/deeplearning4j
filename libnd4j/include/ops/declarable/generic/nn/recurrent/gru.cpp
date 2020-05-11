/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2020 Konduit K.K.
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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_gru)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/gru.h>

namespace sd {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gru, 5, 1, false, 0, 0) {
    auto x  = INPUT_VARIABLE(0);                    // input [time, bS, nIn]
    auto hI = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [bS, nOut]

    auto Wx  = INPUT_VARIABLE(2);                   // input-to-hidden  weights, [nIn, 3*nOut]
    auto Wh  = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [nOut, 3*nOut]
    auto b   = INPUT_VARIABLE(4);                   // biases, [3*nOut]

    auto h   =  OUTPUT_VARIABLE(0);                 // cell outputs [time, bS, nOut], that is per each time step

    const int bS   = x->sizeAt(1);
    const int nIn  = x->sizeAt(2);
    const int nOut = hI->sizeAt(1);

    const std::vector<Nd4jLong> h0CorrectShape = {bS, nOut};
    const std::vector<Nd4jLong> wxCorrectShape = {nIn, 3*nOut};
    const std::vector<Nd4jLong> whCorrectShape = {nOut, 3*nOut};
    const std::vector<Nd4jLong> bCorrectShape  = {3*nOut};

    REQUIRE_TRUE(hI->isSameShape(h0CorrectShape), 0, "GRU operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(h0CorrectShape).c_str(), ShapeUtils::shapeAsString(hI).c_str());
    REQUIRE_TRUE(Wx->isSameShape(wxCorrectShape), 0, "GRU operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(wxCorrectShape).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    REQUIRE_TRUE(Wh->isSameShape(whCorrectShape), 0, "GRU operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(whCorrectShape).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(b->isSameShape(bCorrectShape),   0, "GRU operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());

    helpers::gruTimeLoop(block.launchContext(), x, hI, Wx, Wh, b, h);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(gru) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(gru) {

    auto x  = INPUT_VARIABLE(0);                    // input [time, bS, nIn]
    auto hI = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [bS, nOut]

    auto Wx  = INPUT_VARIABLE(2);                   // input-to-hidden  weights, [nIn, 3*nOut]
    auto Wh  = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [nOut, 3*nOut]
    auto b   = INPUT_VARIABLE(4);                   // biases, [3*nOut]

    const int time = x->sizeAt(0);
    const int bS   = x->sizeAt(1);
    const int nIn  = x->sizeAt(2);
    const int nOut = hI->sizeAt(1);

    const std::vector<Nd4jLong> h0CorrectShape = {bS, nOut};
    const std::vector<Nd4jLong> wxCorrectShape = {nIn, 3*nOut};
    const std::vector<Nd4jLong> whCorrectShape = {nOut, 3*nOut};
    const std::vector<Nd4jLong> bCorrectShape  = {3*nOut};

    REQUIRE_TRUE(hI->isSameShape(h0CorrectShape), 0, "GRU operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(h0CorrectShape).c_str(), ShapeUtils::shapeAsString(hI).c_str());
    REQUIRE_TRUE(Wx->isSameShape(wxCorrectShape), 0, "GRU operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(wxCorrectShape).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    REQUIRE_TRUE(Wh->isSameShape(whCorrectShape), 0, "GRU operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(whCorrectShape).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(b->isSameShape(bCorrectShape),   0, "GRU operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());

    auto hShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(hI->dataType(), hI->ordering(), {time, bS, nOut});

    return SHAPELIST(hShapeInfo);
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gru_bp, 6, 5, false, 0, 0) {

    auto x   = INPUT_VARIABLE(0);                   // input [time, bS, nIn]
    auto hI  = INPUT_VARIABLE(1);                   // initial cell output (at time step = 0) [bS, nOut]

    auto Wx   = INPUT_VARIABLE(2);                  // input-to-hidden  weights, [nIn, 3*nOut]
    auto Wh   = INPUT_VARIABLE(3);                  // hidden-to-hidden weights, [nOut, 3*nOut]
    auto b    = INPUT_VARIABLE(4);                  // biases, [3*nOut]

    auto dLdh = INPUT_VARIABLE(5);                  // gradient vs. ff output, [time, bS, nOut]

    auto dLdx  = OUTPUT_VARIABLE(0);                // gradient vs. ff input, [time, bS, nIn]
    auto dLdhI = OUTPUT_NULLIFIED(1);               // gradient vs. initial cell output, [bS, nOut]
    auto dLdWx = OUTPUT_NULLIFIED(2);               // gradient vs. input-to-hidden  weights, [nIn, 3*nOut]
    auto dLdWh = OUTPUT_NULLIFIED(3);               // gradient vs. hidden-to-hidden weights, [nOut, 3*nOut]
    auto dLdb  = OUTPUT_NULLIFIED(4);               // gradient vs. biases [3*nOut]

    const int time = x->sizeAt(0);
    const int bS   = x->sizeAt(1);
    const int nIn  = x->sizeAt(2);
    const int nOut = hI->sizeAt(1);

    const std::vector<Nd4jLong> h0CorrectShape = {bS, nOut};
    const std::vector<Nd4jLong> wxCorrectShape = {nIn, 3*nOut};
    const std::vector<Nd4jLong> whCorrectShape = {nOut, 3*nOut};
    const std::vector<Nd4jLong> bCorrectShape  = {3*nOut};
    const std::vector<Nd4jLong> hCorrectShape  = {time, bS, nOut};

    REQUIRE_TRUE(hI->isSameShape(h0CorrectShape), 0, "GRU_BP operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(h0CorrectShape).c_str(), ShapeUtils::shapeAsString(hI).c_str());
    REQUIRE_TRUE(Wx->isSameShape(wxCorrectShape), 0, "GRU_BP operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(wxCorrectShape).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    REQUIRE_TRUE(Wh->isSameShape(whCorrectShape), 0, "GRU_BP operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(whCorrectShape).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(b->isSameShape(bCorrectShape),   0, "GRU_BP operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
    REQUIRE_TRUE(dLdh->isSameShape(hCorrectShape),0, "GRU_BP operation: wrong shape of gradient vs. ff output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(hCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdh).c_str());

    helpers::gruTimeLoopBp(block.launchContext(), x, hI, Wx, Wh, b, dLdh, dLdx, dLdhI, dLdWx, dLdWh, dLdb);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(gru_bp) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(gru_bp) {

    auto x  = INPUT_VARIABLE(0);                    // input [time, bS, nIn]
    auto hI = INPUT_VARIABLE(1);                    // initial cell output (at time step = 0) [bS, nOut]

    auto Wx   = INPUT_VARIABLE(2);                   // input-to-hidden  weights, [nIn, 3*nOut]
    auto Wh   = INPUT_VARIABLE(3);                   // hidden-to-hidden weights, [nOut, 3*nOut]
    auto b    = INPUT_VARIABLE(4);                   // biases, [3*nOut]

    auto dLdh = INPUT_VARIABLE(5);                  // gradient vs. ff output, [time, bS, nOut]

    const int time = x->sizeAt(0);
    const int bS   = x->sizeAt(1);
    const int nIn  = x->sizeAt(2);
    const int nOut = hI->sizeAt(1);

    const std::vector<Nd4jLong> h0CorrectShape = {bS, nOut};
    const std::vector<Nd4jLong> wxCorrectShape = {nIn, 3*nOut};
    const std::vector<Nd4jLong> whCorrectShape = {nOut, 3*nOut};
    const std::vector<Nd4jLong> bCorrectShape  = {3*nOut};
    const std::vector<Nd4jLong> hCorrectShape  = {time, bS, nOut};

    REQUIRE_TRUE(hI->isSameShape(h0CorrectShape), 0, "GRU_BP operation: wrong shape of previous cell output array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(h0CorrectShape).c_str(), ShapeUtils::shapeAsString(hI).c_str());
    REQUIRE_TRUE(Wx->isSameShape(wxCorrectShape), 0, "GRU_BP operation: wrong shape of input-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(wxCorrectShape).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    REQUIRE_TRUE(Wh->isSameShape(whCorrectShape), 0, "GRU_BP operation: wrong shape of hidden-to-hidden weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(whCorrectShape).c_str(), ShapeUtils::shapeAsString(Wh).c_str());
    REQUIRE_TRUE(b->isSameShape(bCorrectShape),   0, "GRU_BP operation: wrong shape of biases array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(bCorrectShape).c_str(), ShapeUtils::shapeAsString(b).c_str());
    REQUIRE_TRUE(dLdh->isSameShape(hCorrectShape),0, "GRU_BP operation: wrong shape of gradient vs. ff output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(hCorrectShape).c_str(), ShapeUtils::shapeAsString(dLdh).c_str());

    auto  dLdxShapeInfo  = ConstantShapeHelper::getInstance()->createShapeInfo(dLdh->dataType(), x->shapeInfo());
    auto  dLdhIShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(dLdh->dataType(), hI->shapeInfo());
    auto  dLdWxShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(dLdh->dataType(), Wx->shapeInfo());
    auto  dLdWhShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(dLdh->dataType(), Wh->shapeInfo());
    auto  dLdbShapeInfo  = ConstantShapeHelper::getInstance()->createShapeInfo(dLdh->dataType(), b->shapeInfo());

    return SHAPELIST(dLdxShapeInfo, dLdhIShapeInfo, dLdWxShapeInfo, dLdWhShapeInfo, dLdbShapeInfo);
}

}
}


#endif