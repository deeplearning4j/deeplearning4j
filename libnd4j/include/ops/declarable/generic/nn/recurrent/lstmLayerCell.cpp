/*******************************************************************************
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
#if NOT_EXCLUDED(OP_lstmLayerCell)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmLayer.h>

namespace sd {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmLayerCell, 5, 2, false, 1, 3) {

    // equations (no peephole connections)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  bo)
    // ht  = ot ◦ tanh(ct)

    // equations (peephole connections are present)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  Wpi ◦ ct-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  Wpf ◦ ct-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = clip(ft ◦ ct-1 + it ◦ c't)
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  Wpo ◦ ct  +  bo)
    // ht  = ot ◦ tanh(ct)

    // notations:
    // bS - batch size
    // nIn - input size
    // nOut - output size (hidden size)

    // INPUTS:
    // input x:                          [bS, nIn] or [nIn]
    // input weights Wx:                 [nIn, 4*nOut]
    // recurrent weights Wr:             [nOut, 4*nOut]
    // initial (previous) output hI:     [bS, nOut] or [nOut]
    // initial (previous) cell state cI: [bS, nOut] or [nOut]
    // biases b (optional):              [4*nOut]
    // peephole weights Wp (optional):   [3*nOut]

    // OUTPUTS:
    // current output h:     [bS, nOut] or [nOut]
    // current cell state c: [bS, nOut] or [nOut]

    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct       = INT_ARG(0);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct       = INT_ARG(1);    // activation for cell state (c)
    const auto outAct        = INT_ARG(2);    // activation for output (h)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasPH      = B_ARG(1);   // indicates whether peephole connections are present

    const auto gateActHasAlpha = gateAct == 3 || gateAct == 4 || gateAct == 5 || gateAct == 6 || gateAct == 8;
    const auto cellActHasAlpha = cellAct == 3 || cellAct == 4 || cellAct == 5 || cellAct == 6 || cellAct == 8;
    const auto outActHasAlpha  = outAct  == 3 || outAct  == 4 || outAct  == 5 || outAct  == 6 || outAct  == 8;
    const auto gateActHasBeta  = gateAct == 3 || gateAct == 6;
    const auto cellActHasBeta  = cellAct == 3 || cellAct == 6;
    const auto outActHasBeta   = outAct  == 3 || outAct  == 6;

    uint count = 1;
    const auto cellClip = T_ARG(0);                                     // cell clipping value, if it = 0 then do not apply clipping
    const auto gateAlpha = gateActHasAlpha ? T_ARG(count++) : 0;
    const auto gateBeta  = gateActHasBeta  ? T_ARG(count++) : 0;
    const auto cellAlpha = cellActHasAlpha ? T_ARG(count++) : 0;
    const auto cellBeta  = cellActHasBeta  ? T_ARG(count++) : 0;
    const auto outAlpha  = outActHasAlpha  ? T_ARG(count++) : 0;
    const auto outBeta   = outActHasBeta   ? T_ARG(count++) : 0;

    count = 3;
    const auto x  = INPUT_VARIABLE(0);                              // input
    const auto Wx = INPUT_VARIABLE(1);                              // input weights
    const auto Wr = INPUT_VARIABLE(2);                              // recurrent weights
    const auto b  = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto hI = INPUT_VARIABLE(count++);                        // initial output
    const auto cI = INPUT_VARIABLE(count++);                        // initial cell state
    const auto Wp = hasPH ? INPUT_VARIABLE(count) : nullptr;        // peephole weights

    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER_CELL operation: cell clipping value should be nonnegative (>=0) !");

    auto h = OUTPUT_VARIABLE(0);
    auto c = OUTPUT_VARIABLE(1);

    // evaluate dimensions
    const Nd4jLong bS   = x->rankOf() == 1 ? 0 : x->sizeAt(0);
    const Nd4jLong nIn  = x->sizeAt(-1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    // Wx validation
    if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    // Wr validation
    if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
    // initial output/cell validation
    std::vector<Nd4jLong> exphIcIShape = x->rankOf() == 1 ? std::vector<Nd4jLong>{nOut} : std::vector<Nd4jLong>{bS, nOut};
    REQUIRE_TRUE(hI->isSameShape(exphIcIShape), 0, "LSTM_LAYER_CELL operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(exphIcIShape).c_str(), ShapeUtils::shapeAsString(hI).c_str());
    REQUIRE_TRUE(cI->isSameShape(exphIcIShape), 0, "LSTM_LAYER_CELL operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(exphIcIShape).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    // biases validation
    if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
    // peephole weights validation
    if(Wp != nullptr && (Wp->rankOf() != 1 || Wp->sizeAt(0) != 3*nOut))
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL operation: wrong shape of peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp).c_str());

    std::vector<float> params = {static_cast<float>(0)/*ignore*/, static_cast<float>(0)/*ignore*/, static_cast<float>(cellClip),
                                 static_cast<float>(gateAct), static_cast<float>(gateAlpha), static_cast<float>(gateBeta),
                                 static_cast<float>(cellAct), static_cast<float>(cellAlpha), static_cast<float>(cellBeta),
                                 static_cast<float>(outAct), static_cast<float>(outAlpha), static_cast<float>(outBeta)};

    helpers::lstmLayerCell(x, Wx, Wr, b, hI, cI, Wp, params, h, c);

    return Status::OK();
}

DECLARE_TYPES(lstmLayerCell) {
    getOpDescriptor()
        ->setAllowedInputTypes(sd::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}


DECLARE_SHAPE_FN(lstmLayerCell) {

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided

    uint count = hasBiases ? 4 : 3;
    const auto hI = INPUT_VARIABLE(count++);        // initial output
    const auto cI = INPUT_VARIABLE(count);          // initial cell state

    return new ShapeList({hI->getShapeInfo(), cI->getShapeInfo()});
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmLayerCellBp, 7, 5, false, 1, 3) {

    // equations (no peephole connections)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  bo)
    // ht  = ot ◦ tanh(ct)

    // equations (peephole connections are present)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  Wpi ◦ ct-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  Wpf ◦ ct-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = clip(ft ◦ ct-1 + it ◦ c't)
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  Wpo ◦ ct  +  bo)
    // ht  = ot ◦ tanh(ct)

    // notations:
    // bS - batch size
    // nIn - input size
    // nOut - output size (hidden size)

    // INPUTS:
    // input x:                          [bS, nIn] or [nIn]
    // input weights Wx:                 [nIn, 4*nOut]
    // recurrent weights Wr:             [nOut, 4*nOut]
    // initial (previous) output hI:     [bS, nOut] or [nOut]
    // initial (previous) cell state cI: [bS, nOut] or [nOut]
    // gradient wrt output dLdh:         [bS, nOut] or [nOut]
    // gradient wrt cell state dLdc:     [bS, nOut] or [nOut]
    // peephole weights Wp (optional):   [3*nOut]
    // biases b (optional):              [4*nOut]

    // OUTPUTS:
    // gradient wrt x dLdx:              [bS, nIn] or [nIn]
    // gradient wrt Wx dLdWx:            [nIn, 4*nOut]
    // gradient wrt Wr dLdWr:            [nOut, 4*nOut]
    // gradient wrt hI dLdhI:            [bS, nOut] or [nOut]
    // gradient wrt cI dLdcI:            [bS, nOut] or [nOut]
    // gradient wrt b dLdb (optional):   [4*nOut]
    // gradient wrt Wp dLdWp (optional): [3*nOut]


    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct       = INT_ARG(0);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct       = INT_ARG(1);    // activation for cell state (c)
    const auto outAct        = INT_ARG(2);    // activation for output (h)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasPH      = B_ARG(1);   // indicates whether peephole connections are present

    const auto gateActHasAlpha = gateAct == 3 || gateAct == 4 || gateAct == 5 || gateAct == 6 || gateAct == 8;
    const auto cellActHasAlpha = cellAct == 3 || cellAct == 4 || cellAct == 5 || cellAct == 6 || cellAct == 8;
    const auto outActHasAlpha  = outAct  == 3 || outAct  == 4 || outAct  == 5 || outAct  == 6 || outAct  == 8;
    const auto gateActHasBeta  = gateAct == 3 || gateAct == 6;
    const auto cellActHasBeta  = cellAct == 3 || cellAct == 6;
    const auto outActHasBeta   = outAct  == 3 || outAct  == 6;

    uint count = 1;
    const auto cellClip = T_ARG(0);                                     // cell clipping value, if it = 0 then do not apply clipping
    const auto gateAlpha = gateActHasAlpha ? T_ARG(count++) : 0;
    const auto gateBeta  = gateActHasBeta  ? T_ARG(count++) : 0;
    const auto cellAlpha = cellActHasAlpha ? T_ARG(count++) : 0;
    const auto cellBeta  = cellActHasBeta  ? T_ARG(count++) : 0;
    const auto outAlpha  = outActHasAlpha  ? T_ARG(count++) : 0;
    const auto outBeta   = outActHasBeta   ? T_ARG(count++) : 0;

    count = 3;
    const auto x  = INPUT_VARIABLE(0);                              // input
    const auto Wx = INPUT_VARIABLE(1);                              // input weights
    const auto Wr = INPUT_VARIABLE(2);                              // recurrent weights
    const auto b  = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto hI = INPUT_VARIABLE(count++);                        // initial output
    const auto cI = INPUT_VARIABLE(count++);                        // initial cell state
    const auto Wp = hasPH ? INPUT_VARIABLE(count++) : nullptr;      // peephole weights
    const auto dLdh = INPUT_VARIABLE(count);                        // gradient wrt output

    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER_CELL_BP operation: cell clipping value should be nonnegative (>=0) !");

    count = 3;
    auto dLdx  = OUTPUT_VARIABLE(0);
    auto dLdWx = OUTPUT_VARIABLE(1);
    auto dLdWr = OUTPUT_VARIABLE(2);
    auto dLdb  = hasBiases ? OUTPUT_VARIABLE(count++) : nullptr;
    auto dLdhI = OUTPUT_VARIABLE(count++);
    auto dLdcI = OUTPUT_VARIABLE(count++);
    auto dLdWp = hasPH ? OUTPUT_VARIABLE(count) : nullptr;

    // evaluate dimensions
    const Nd4jLong bS   = x->rankOf() == 1 ? 0 : x->sizeAt(0);
    const Nd4jLong nIn  = x->sizeAt(-1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    // Wx validation
    if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL_BP operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
    // Wr validation
    if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL_BP operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
    // initial output/cell validation
    std::vector<Nd4jLong> exphIcIShape = x->rankOf() == 1 ? std::vector<Nd4jLong>{nOut} : std::vector<Nd4jLong>{bS, nOut};
    REQUIRE_TRUE(hI->isSameShape(exphIcIShape), 0, "LSTM_LAYER_CELL_BP operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(exphIcIShape).c_str(), ShapeUtils::shapeAsString(hI).c_str());
    REQUIRE_TRUE(cI->isSameShape(exphIcIShape), 0, "LSTM_LAYER_CELL_BP operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(exphIcIShape).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    REQUIRE_TRUE(dLdh->isSameShape(exphIcIShape), 0, "LSTM_LAYER_CELL_BP operation: wrong shape of dLdh gradient, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(exphIcIShape).c_str(), ShapeUtils::shapeAsString(dLdh).c_str());
    // biases validation
    if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL_BP operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
    if(dLdb != nullptr && (dLdb->rankOf() != 1 || dLdb->sizeAt(0) != 4*nOut))
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL_BP operation: wrong shape of dLdb gradient, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(dLdb).c_str());
    // peephole weights validation
    if(Wp != nullptr && (Wp->rankOf() != 1 || Wp->sizeAt(0) != 3*nOut))
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL_BP operation: wrong shape of peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp).c_str());
    if(dLdWp != nullptr && (dLdWp->rankOf() != 1 || dLdWp->sizeAt(0) != 3*nOut))
        REQUIRE_TRUE(false, 0, "LSTM_LAYER_CELL_BP operation: wrong shape of dLdWp gradient, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({3*nOut}).c_str(), ShapeUtils::shapeAsString(dLdWp).c_str());


    std::vector<float> params = {static_cast<float>(0)/*ignore*/, static_cast<float>(0)/*ignore*/, static_cast<float>(cellClip),
                                 static_cast<float>(gateAct), static_cast<float>(gateAlpha), static_cast<float>(gateBeta),
                                 static_cast<float>(cellAct), static_cast<float>(cellAlpha), static_cast<float>(cellBeta),
                                 static_cast<float>(outAct), static_cast<float>(outAlpha), static_cast<float>(outBeta)};

    std::vector<Nd4jLong> zShape = x->rankOf() == 1 ? std::vector<Nd4jLong>({4*nOut}) : std::vector<Nd4jLong>({bS, 4*nOut});

    NDArray z(x->ordering(), zShape, x->dataType(), block.launchContext());
    NDArray a = z.ulike();
    NDArray h = cI->ulike();
    NDArray c = cI->ulike();

    helpers::lstmLayerCell(x,Wx, Wr, b, hI, cI, Wp, params, &z, &a, &h, &c);

    helpers::lstmLayerCellBp(x, Wx, Wr, b, hI, cI, Wp, dLdh, nullptr, nullptr, &z, &a, &c, params, dLdx, dLdWx, dLdWr, dLdhI, dLdcI, dLdb, dLdWp);

    return Status::OK();
}

DECLARE_TYPES(lstmLayerCellBp) {
    getOpDescriptor()
        ->setAllowedInputTypes(sd::DataType::ANY)
        ->setAllowedOutputTypes({ALL_FLOATS});
}


DECLARE_SHAPE_FN(lstmLayerCellBp) {

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasPH      = B_ARG(1);   // indicates whether peephole connections are present

    uint count = 3;
    const auto x  = INPUT_VARIABLE(0);                              // input
    const auto Wx = INPUT_VARIABLE(1);                              // input weights
    const auto Wr = INPUT_VARIABLE(2);                              // recurrent weights
    const auto b  = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto hI = INPUT_VARIABLE(count++);                        // initial output
    const auto cI = INPUT_VARIABLE(count++);                        // initial cell state
    const auto Wp = hasPH ? INPUT_VARIABLE(count) : nullptr;        // peephole weights

    std::vector<Nd4jLong*> shapes = {x->getShapeInfo(), Wx->getShapeInfo(), Wr->getShapeInfo()};

    if(b != nullptr)
        shapes.push_back(b->getShapeInfo());

    shapes.push_back(hI->getShapeInfo());
    shapes.push_back(cI->getShapeInfo());

    if(Wp != nullptr)
        shapes.push_back(Wp->getShapeInfo());

    return new ShapeList(shapes);
}

}
}

#endif