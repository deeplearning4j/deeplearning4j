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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_lstmLayer)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/lstmLayer.h>


namespace sd {
namespace ops  {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmLayer, 3, 1, false, 1, 5) {

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
    // sL - sequence length, number of time steps
    // nIn - input size
    // nOut - output size (hidden size)

    //     INPUTS:

    // *******
    // input x:
    // 1) [sL, bS, nIn]  when dataFormat == 0
    // 2) [bS, sL, nIn]  when dataFormat == 1
    // 3) [bS, nIn, sL]  when dataFormat == 2

    // *******
    // input weights Wx:
    // 1) [nIn, 4*nOut]    when directionMode <  2
    // 2) [2, nIn, 4*nOut] when directionMode >= 2

    // *******
    // recurrent weights Wr:
    // 1) [nOut, 4*nOut]    when directionMode <  2
    // 2) [2, nOut, 4*nOut] when directionMode >= 2

    // *******
    // peephole weights Wp, optional:
    // 1) [3*nOut]    when directionMode <  2
    // 2) [2, 3*nOut] when directionMode >= 2

    // *******
    // biases b, optional:
    // 1) [4*nOut]    when directionMode <  2
    // 2) [2, 4*nOut] when directionMode >= 2

    // *******
    // sequence length array seqLen, optional:
    // 1) [bS]

    // *******
    // initial output hI, optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // initial cell state cI (same shape as in hI), optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2


    //     OUTPUTS:

    // *******
    // output h, optional:
    // 1) [sL, bS, nOut]    when directionMode <= 2 && dataFormat == 0
    // 2) [bS, sL, nOut]    when directionMode <= 2 && dataFormat == 1
    // 3) [bS, nOut, sL]    when directionMode <= 2 && dataFormat == 2
    // 4) [sL, bS, 2*nOut]  when directionMode == 3 && dataFormat == 0
    // 5) [bS, sL, 2*nOut]  when directionMode == 3 && dataFormat == 1
    // 6) [bS, 2*nOut, sL]  when directionMode == 3 && dataFormat == 2
    // 7) [sL, 2, bS, nOut] when directionMode == 4 && dataFormat == 3

    // *******
    // output at last step hL, optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // cell state at last step cL (same shape as in hL), optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, bS, nIn] && [sL, 2, bS, nOut] (for ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)

    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct       = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct       = INT_ARG(3);    // activation for cell state (c)
    const auto outAct        = INT_ARG(4);    // activation for output (h)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only

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

    const auto x  = INPUT_VARIABLE(0);          // input
    const auto Wx = INPUT_VARIABLE(1);          // input weights
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights

    count = 3;
    const auto b      = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto seqLen = hasSeqLen ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector
    const auto hI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI     = hasInitC  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
    const auto Wp     = hasPH     ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights

    REQUIRE_TRUE(dataFormat < 3 || (dataFormat == 3 && directionMode == 4), 0, "LSTM_LAYER operation: if argument dataFormat = 3, then directionMode = 4, but got dataFormat = %i and directionMode = %i instead !", dataFormat, directionMode);
    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER operation: cell clipping value should be nonnegative (>=0) !");
    REQUIRE_TRUE(retFullSeq || retLastH || retLastC, 0, "LSTM_LAYER operation: please specify what output arrays to produce !");

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;           // output
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;           // output at last step
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;           // cell state at last step

    // evaluate dimensions
    const Nd4jLong sL   = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(1);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(2);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    if(directionMode < 2) {     // no bidirectional

        // Wx validation
        if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 2 || hI->sizeAt(0) != bS || hI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 2 || cI->sizeAt(0) != bS || cI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
        // peephole weights validation
        if(Wp != nullptr && (Wp->rankOf() != 1 || Wp->sizeAt(0) != 3*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp).c_str());
    }
    else {                  // bidirectional
         // Wx validation
        if(Wx->rankOf() != 3 || Wx->sizeAt(0) != 2 || Wx->sizeAt(1) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 3 || Wr->sizeAt(0) != 2 || Wr->sizeAt(1) != nOut || Wr->sizeAt(2) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 2 || b->sizeAt(0) != 2 || b->sizeAt(1) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 3 || hI->sizeAt(0) != 2 || hI->sizeAt(1) != bS || hI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 3 || cI->sizeAt(0) != 2 || cI->sizeAt(1) != bS || cI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
        // peephole weights validation
        if(Wp != nullptr && (Wp->rankOf() != 2 || Wp->sizeAt(0) != 2 || Wp->sizeAt(1) != 3*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER operation: wrong peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp).c_str());
    }

    std::vector<float> params = {static_cast<float>(dataFormat), static_cast<float>(directionMode), static_cast<float>(cellClip),
                                 static_cast<float>(gateAct), static_cast<float>(gateAlpha), static_cast<float>(gateBeta),
                                 static_cast<float>(cellAct), static_cast<float>(cellAlpha), static_cast<float>(cellBeta),
                                 static_cast<float>(outAct), static_cast<float>(outAlpha), static_cast<float>(outBeta)};

    if(directionMode == 0) {   // forward

        helpers::lstmLayerTimeLoop(x, Wx, Wr, b, seqLen, hI, cI, Wp, params, true, h, hL, cL);
    }
    else if(directionMode == 1) {  // backward

        helpers::lstmLayerTimeLoop(x, Wx, Wr, b, seqLen, hI, cI, Wp, params, false, h, hL, cL);
    }
    else {  // bidirectional

        NDArray WxFwd = (*Wx)({0,1, 0,0, 0,0});
        NDArray WxBwd = (*Wx)({1,2, 0,0, 0,0});
        NDArray WrFwd = (*Wr)({0,1, 0,0, 0,0});
        NDArray WrBwd = (*Wr)({1,2, 0,0, 0,0});

        NDArray *WpFwd(nullptr), *WpBwd(nullptr), *bFwd(nullptr), *bBwd(nullptr), *hIFwd(nullptr), *hIBwd(nullptr), *cIFwd(nullptr), *cIBwd(nullptr),
                *hLFwd(nullptr), *hLBwd(nullptr), *cLFwd(nullptr), *cLBwd(nullptr), *hFwd(nullptr), *hBwd(nullptr);

        if(Wp) {
            WpFwd = new NDArray((*Wp)({0,1, 0,0}));
            WpBwd = new NDArray((*Wp)({1,2, 0,0}));
        }
        if(b) {
            bFwd = new NDArray((*b)({0,1, 0,0}));
            bBwd = new NDArray((*b)({1,2, 0,0}));
        }
        if(hI) {
            hIFwd = new NDArray((*hI)({0,1, 0,0, 0,0}));
            hIBwd = new NDArray((*hI)({1,2, 0,0, 0,0}));
        }
        if(cI) {
            cIFwd = new NDArray((*cI)({0,1, 0,0, 0,0}));
            cIBwd = new NDArray((*cI)({1,2, 0,0, 0,0}));
        }
        if(hL) {
            hLFwd = new NDArray((*hL)({0,1, 0,0, 0,0}));
            hLBwd = new NDArray((*hL)({1,2, 0,0, 0,0}));
        }
        if(cL) {
            cLFwd = new NDArray((*cL)({0,1, 0,0, 0,0}));
            cLBwd = new NDArray((*cL)({1,2, 0,0, 0,0}));
        }

        if(h) {
            if(directionMode == 2) {        // sum
                hFwd = h;
                hBwd = new NDArray(h, false, h->getContext());
            }
            else if(directionMode == 3) {   // concat
                hFwd = new NDArray(dataFormat <= 1 ? (*h)({0,0, 0,0,    0,nOut})   : (*h)({0,0,    0,nOut,   0,0}));
                hBwd = new NDArray(dataFormat <= 1 ? (*h)({0,0, 0,0, nOut,2*nOut}) : (*h)({0,0, nOut,2*nOut, 0,0}));
            }
            else {  // directionMode == 4
                hFwd = new NDArray((*h)({0,0, 0,1, 0,0, 0,0}));
                hBwd = new NDArray((*h)({0,0, 1,2, 0,0, 0,0}));
            }
        }

        // FIXME - following two calls are independent and may run in different streams
        helpers::lstmLayerTimeLoop(x, &WxFwd, &WrFwd, bFwd, seqLen, hIFwd, cIFwd, WpFwd, params, true,  hFwd, hLFwd, cLFwd);
        helpers::lstmLayerTimeLoop(x, &WxBwd, &WrBwd, bBwd, seqLen, hIBwd, cIBwd, WpBwd, params, false, hBwd, hLBwd, cLBwd);

        if(h && directionMode == 2)
            *h += *hBwd;

        delete WpFwd; delete WpBwd; delete bFwd;  delete bBwd;  delete hIFwd; delete hIBwd; delete cIFwd;
        delete cIBwd; delete hLFwd; delete hLBwd; delete cLFwd; delete cLBwd; delete hBwd;
        if(hFwd != h)
            delete hFwd;
    }

    return Status::OK();
}

DECLARE_TYPES(lstmLayer) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setAllowedOutputTypes({ALL_FLOATS});
}


DECLARE_SHAPE_FN(lstmLayer) {

    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nIn] (for ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim

    const auto retFullSeq = B_ARG(5);           // indicates whether to return whole h {h_0, h_1, ... , h_sL-1}, if true, format would be [sL,bS,nOut] (exact shape depends on dataFormat argument)
    const auto retLastH   = B_ARG(6);           // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);           // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    const auto x  = INPUT_VARIABLE(0);          // input
    const auto Wx = INPUT_VARIABLE(1);          // input weights
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights

    // evaluate dimensions
    const Nd4jLong sL   = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(1);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(2);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    DataType type;
    if(x->isR())
        type = x->dataType();
    else
        type = sd::DataType::FLOAT32;

    auto shapes = SHAPELIST();

    // evaluate h shape (output)
    if(retFullSeq) {

        std::vector<Nd4jLong> hShape;

        if(directionMode <= 2) {                // single direction or bidirectional with sum
            if(dataFormat == 0)
                hShape = {sL, bS, nOut};
            else if(dataFormat == 1)
                hShape = {bS, sL, nOut};
            else if(dataFormat == 2)
                hShape = {bS, nOut, sL};
        }
        else if(directionMode == 3) {           // bidirectional with concat

            if(dataFormat == 0)
                hShape = {sL, bS, 2*nOut};
            else if(dataFormat == 1)
                hShape = {bS, sL, 2*nOut};
            else if(dataFormat == 2)
                hShape = {bS, 2*nOut, sL};
        }
        else {                                  // bidirectional with extra output dimension equal to 2
            hShape = {sL, 2, bS, nOut};
        }

        shapes->push_back(ConstantShapeHelper::getInstance().createShapeInfo(type, x->ordering(), hShape));
    }

    // evaluate hL shape (output at last step)
    if(retLastH) {

        std::vector<Nd4jLong> hLShape;

        if(directionMode < 2)
            hLShape = {bS, nOut};
        else
            hLShape = {2, bS, nOut};

        shapes->push_back(ConstantShapeHelper::getInstance().createShapeInfo(type, x->ordering(), hLShape));

        if(retLastC)                                // cL and hL have same shapes
            shapes->push_back(shapes->at(shapes->size() - 1));
    }

    // evaluate cL shape (cell state at last step)
    if(retLastC && !retLastH) {

        std::vector<Nd4jLong> cLShape;

        if(directionMode < 2)
            cLShape = {bS, nOut};
        else
            cLShape = {2, bS, nOut};

        shapes->push_back(ConstantShapeHelper::getInstance().createShapeInfo(type, x->ordering(), cLShape));
    }

    return shapes;
}


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(lstmLayer_bp, 4, 1, false, 1, 5) {

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
    // sL - sequence length, number of time steps
    // nIn - input size
    // nOut - output size (hidden size)

    //     INPUTS:

    // *******
    // input x:
    // 1) [sL, bS, nIn]  when dataFormat == 0
    // 2) [bS, sL, nIn]  when dataFormat == 1
    // 3) [bS, nIn, sL]  when dataFormat == 2

    // *******
    // input weights Wx:
    // 1) [nIn, 4*nOut]    when directionMode <  2
    // 2) [2, nIn, 4*nOut] when directionMode >= 2

    // *******
    // recurrent weights Wr:
    // 1) [nOut, 4*nOut]    when directionMode <  2
    // 2) [2, nOut, 4*nOut] when directionMode >= 2

    // *******
    // peephole weights Wp, optional:
    // 1) [3*nOut]    when directionMode <  2
    // 2) [2, 3*nOut] when directionMode >= 2

    // *******
    // biases b, optional:
    // 1) [4*nOut]    when directionMode <  2
    // 2) [2, 4*nOut] when directionMode >= 2

    // *******
    // sequence length array seqLen, optional:
    // 1) [bS]

    // *******
    // initial output hI, optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // initial cell state cI (same shape as in hI), optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // gradient vs. output dLdh, optional:
    // 1) [sL, bS, nOut]    when directionMode <= 2 && dataFormat == 0
    // 2) [bS, sL, nOut]    when directionMode <= 2 && dataFormat == 1
    // 3) [bS, nOut, sL]    when directionMode <= 2 && dataFormat == 2
    // 4) [sL, bS, 2*nOut]  when directionMode == 3 && dataFormat == 0
    // 5) [bS, sL, 2*nOut]  when directionMode == 3 && dataFormat == 1
    // 6) [bS, 2*nOut, sL]  when directionMode == 3 && dataFormat == 2
    // 7) [sL, 2, bS, nOut] when directionMode == 4 && dataFormat == 3

    // *******
    // gradient vs output at last time step dLdhL, optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // gradient vs cell state at last time step dLdcL(same shape as in dLdhL), optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2


    //     OUTPUTS:

    // *******
    // gradient vs. input dLdx:
    // 1) [sL, bS, nIn]  when dataFormat == 0
    // 2) [bS, sL, nIn]  when dataFormat == 1
    // 3) [bS, nIn, sL]  when dataFormat == 2

    // *******
    // gradient vs. input weights dLdWx:
    // 1) [nIn, 4*nOut]    when directionMode <  2
    // 2) [2, nIn, 4*nOut] when directionMode >= 2

    // *******
    // gradient vs. recurrent weights dLdWr:
    // 1) [nOut, 4*nOut]    when directionMode <  2
    // 2) [2, nOut, 4*nOut] when directionMode >= 2

    // *******
    // gradient vs. peephole weights dLdWp, optional:
    // 1) [3*nOut]    when directionMode <  2
    // 2) [2, 3*nOut] when directionMode >= 2

    // *******
    // gradient vs. biases dLdb, optional:
    // 1) [4*nOut]    when directionMode <  2
    // 2) [2, 4*nOut] when directionMode >= 2

    // gradient vs. sequence length array dLdsL, optional (do not calculate it!!!):
    // 1) [bS] always

    // *******
    // gradient vs. initial output dLdhI, optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2

    // *******
    // gradient vs. initial cell state dLdcI (same shape as in dLdhI), optional:
    // 1) [bS, nOut]    when directionMode <  2
    // 2) [2, bS, nOut] when directionMode >= 2


    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, bS, nIn] && [sL, 2, bS, nOut] (for ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)

    // integer numbers corresponding to activations: 0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus
    const auto gateAct       = INT_ARG(2);    // activation for input (i), forget (f) and output (o) gates
    const auto cellAct       = INT_ARG(3);    // activation for cell state (c)
    const auto outAct        = INT_ARG(4);    // activation for output (h)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether gradient vs. outputs is given for whole time sequence dLdh {dLdh_0, dLdh_1, ... , dLdh_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether gradient vs. output at last time step (dLdhL) is given
    const auto retLastC   = B_ARG(7);   // indicates whether gradient vs. cell state at last time step (dLdcL) is given

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

    REQUIRE_TRUE(dataFormat < 3 || (dataFormat == 3 && directionMode == 4), 0, "LSTM_LAYER_BP operation: if argument dataFormat = 3, then directionMode = 4, but got dataFormat = %i and directionMode = %i instead !", dataFormat, directionMode);
    REQUIRE_TRUE(cellClip >= 0 , 0, "LSTM_LAYER_BP operation: cell clipping value should be nonnegative (>=0) !");
    REQUIRE_TRUE(retFullSeq || retLastH || retLastC, 0, "LSTM_LAYER_BP operation: please specify at least one of three input gradient arrays: dLdh, dLdhL or dLdcL !");

    const auto x  = INPUT_VARIABLE(0);          // input
    const auto Wx = INPUT_VARIABLE(1);          // input weights
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights

    count = 3;
    const auto b      = hasBiases  ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto seqLen = hasSeqLen  ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector
    const auto hI     = hasInitH   ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI     = hasInitC   ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
    const auto Wp     = hasPH      ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights
    const auto dLdh   = retFullSeq ? INPUT_VARIABLE(count++) : nullptr;  // gradient vs. output
    const auto dLdhL  = retLastH   ? INPUT_VARIABLE(count++) : nullptr;  // gradient vs. output at last time step
    const auto dLdcL  = retLastC   ? INPUT_VARIABLE(count++) : nullptr;  // gradient vs. cell state at last time step

    count = 3;
    auto dLdx  = OUTPUT_VARIABLE(0);                                 // gradient vs. input
    auto dLdWx = OUTPUT_NULLIFIED(1);                                // gradient vs. input weights
    auto dLdWr = OUTPUT_NULLIFIED(2);                                // gradient vs. recurrent weights
    auto dLdb  = hasBiases ? OUTPUT_NULLIFIED(count++) : nullptr;    // gradient vs. biases
    auto dLdsL = hasSeqLen ? INPUT_VARIABLE(count++)   : nullptr;    // gradient vs. seqLen vector, we don't calculate it !!!
    auto dLdhI = hasInitH  ? OUTPUT_NULLIFIED(count++) : nullptr;    // gradient vs. initial output
    auto dLdcI = hasInitC  ? OUTPUT_NULLIFIED(count++) : nullptr;    // gradient vs. initial cell state
    auto dLdWp = hasPH     ? OUTPUT_NULLIFIED(count)   : nullptr;    // gradient vs. peephole weights

    // evaluate dimensions
    const Nd4jLong sL   = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(1);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(2);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    if(directionMode < 2) {     // no bidirectional

        // Wx validation
        if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 2 || hI->sizeAt(0) != bS || hI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 2 || cI->sizeAt(0) != bS || cI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
        // peephole weights validation
        if(Wp != nullptr && (Wp->rankOf() != 1 || Wp->sizeAt(0) != 3*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp).c_str());
        // gradient vs. output at last time step validation
        if(dLdhL != nullptr && (dLdhL->rankOf() != 2 || dLdhL->sizeAt(0) != bS || dLdhL->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of gradient vs. output at last time step, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(dLdhL).c_str());
        // gradient vs. cell state at last time step validation
        if(dLdcL != nullptr && (dLdcL->rankOf() != 2 || dLdcL->sizeAt(0) != bS || dLdcL->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of gradient vs. cell state at last time, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(dLdcL).c_str());
    }
    else {                  // bidirectional
         // Wx validation
        if(Wx->rankOf() != 3 || Wx->sizeAt(0) != 2 || Wx->sizeAt(1) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 3 || Wr->sizeAt(0) != 2 || Wr->sizeAt(1) != nOut || Wr->sizeAt(2) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 2 || b->sizeAt(0) != 2 || b->sizeAt(1) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 3 || hI->sizeAt(0) != 2 || hI->sizeAt(1) != bS || hI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 3 || cI->sizeAt(0) != 2 || cI->sizeAt(1) != bS || cI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
        // peephole weights validation
        if(Wp != nullptr && (Wp->rankOf() != 2 || Wp->sizeAt(0) != 2 || Wp->sizeAt(1) != 3*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong peephole weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 3*nOut}).c_str(), ShapeUtils::shapeAsString(Wp).c_str());
        // gradient vs. output at last time step validation
        if(dLdhL != nullptr && (dLdhL->rankOf() != 3 || dLdhL->sizeAt(0) != 2 || dLdhL->sizeAt(1) != bS || dLdhL->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of gradient vs. output at last time step, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(dLdhL).c_str());
        // gradient vs. cell state at last time step validation
        if(dLdcL != nullptr && (dLdcL->rankOf() != 3 || dLdcL->sizeAt(0) != 2 || dLdcL->sizeAt(1) != bS || dLdcL->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_BP operation: wrong shape of gradient vs. cell state at last time, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(dLdcL).c_str());
    }

    // gradient vs. output  validation
    if(dLdh) {
        int factor = directionMode <= 2 ? 1 : 2;
        std::vector<Nd4jLong> expdLdhShape;
        if(dataFormat == 0)      expdLdhShape = std::vector<Nd4jLong>{sL, bS, factor*nOut};
        else if(dataFormat == 1) expdLdhShape = std::vector<Nd4jLong>{bS, sL, factor*nOut};
        else if(dataFormat == 2) expdLdhShape = std::vector<Nd4jLong>{bS, factor*nOut, sL};
        else                     expdLdhShape = std::vector<Nd4jLong>{sL, 2, bS, nOut};
        REQUIRE_TRUE(dLdh->isSameShape(expdLdhShape), 0, "LSTM_LAYER_CELL_BP operation: wrong shape of gradient vs. output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expdLdhShape).c_str(), ShapeUtils::shapeAsString(dLdh).c_str());
    }

    std::vector<float> params = {static_cast<float>(dataFormat), static_cast<float>(directionMode), static_cast<float>(cellClip),
                                 static_cast<float>(gateAct), static_cast<float>(gateAlpha), static_cast<float>(gateBeta),
                                 static_cast<float>(cellAct), static_cast<float>(cellAlpha), static_cast<float>(cellBeta),
                                 static_cast<float>(outAct), static_cast<float>(outAlpha), static_cast<float>(outBeta)};

    if(directionMode == 0) {   // forward

        helpers::lstmLayerTimeLoopBp(x, Wx, Wr, b, seqLen, hI, cI, Wp, dLdh, dLdhL, dLdcL, params, true, dLdx, dLdWx, dLdWr, dLdb, dLdhI, dLdcI, dLdWp);
    }
    else if(directionMode == 1) {  // backward

        helpers::lstmLayerTimeLoopBp(x, Wx, Wr, b, seqLen, hI, cI, Wp, dLdh, dLdhL, dLdcL, params, false, dLdx, dLdWx, dLdWr, dLdb, dLdhI, dLdcI, dLdWp);
    }
    else {  // bidirectional

        NDArray WxFwd    = (*Wx)({0,1, 0,0, 0,0});
        NDArray WxBwd    = (*Wx)({1,2, 0,0, 0,0});
        NDArray dLdWxFwd = (*dLdWx)({0,1, 0,0, 0,0});
        NDArray dLdWxBwd = (*dLdWx)({1,2, 0,0, 0,0});

        NDArray WrFwd    = (*Wr)({0,1, 0,0, 0,0});
        NDArray WrBwd    = (*Wr)({1,2, 0,0, 0,0});
        NDArray dLdWrFwd = (*dLdWr)({0,1, 0,0, 0,0});
        NDArray dLdWrBwd = (*dLdWr)({1,2, 0,0, 0,0});

        NDArray *WpFwd(nullptr), *WpBwd(nullptr), *bFwd(nullptr), *bBwd(nullptr), *hIFwd(nullptr), *hIBwd(nullptr), *cIFwd(nullptr), *cIBwd(nullptr),
                *dLdhFwd(nullptr), *dLdhBwd(nullptr), *dLdhLFwd(nullptr), *dLdhLBwd(nullptr), *dLdcLFwd(nullptr), *dLdcLBwd(nullptr),
                *dLdWpFwd(nullptr), *dLdWpBwd(nullptr), *dLdbFwd(nullptr), *dLdbBwd(nullptr),
                *dLdhIFwd(nullptr), *dLdhIBwd(nullptr), *dLdcIFwd(nullptr), *dLdcIBwd(nullptr);

        if(Wp) {
            WpFwd    = new NDArray((*Wp)({0,1, 0,0}));
            WpBwd    = new NDArray((*Wp)({1,2, 0,0}));
            dLdWpFwd = new NDArray((*dLdWp)({0,1, 0,0}));
            dLdWpBwd = new NDArray((*dLdWp)({1,2, 0,0}));
        }
        if(b) {
            bFwd    = new NDArray((*b)({0,1, 0,0}));
            bBwd    = new NDArray((*b)({1,2, 0,0}));
            dLdbFwd = new NDArray((*dLdb)({0,1, 0,0}));
            dLdbBwd = new NDArray((*dLdb)({1,2, 0,0}));
        }
        if(hI) {
            hIFwd    = new NDArray((*hI)({0,1, 0,0, 0,0}));
            hIBwd    = new NDArray((*hI)({1,2, 0,0, 0,0}));
            dLdhIFwd = new NDArray((*dLdhI)({0,1, 0,0, 0,0}));
            dLdhIBwd = new NDArray((*dLdhI)({1,2, 0,0, 0,0}));
        }
        if(cI) {
            cIFwd    = new NDArray((*cI)({0,1, 0,0, 0,0}));
            cIBwd    = new NDArray((*cI)({1,2, 0,0, 0,0}));
            dLdcIFwd = new NDArray((*dLdcI)({0,1, 0,0, 0,0}));
            dLdcIBwd = new NDArray((*dLdcI)({1,2, 0,0, 0,0}));
        }
        if(dLdhL) {
            dLdhLFwd = new NDArray((*dLdhL)({0,1, 0,0, 0,0}));
            dLdhLBwd = new NDArray((*dLdhL)({1,2, 0,0, 0,0}));
        }
        if(dLdcL) {
            dLdcLFwd = new NDArray((*dLdcL)({0,1, 0,0, 0,0}));
            dLdcLBwd = new NDArray((*dLdcL)({1,2, 0,0, 0,0}));
        }

        if(dLdh) {
            if(directionMode == 2) {        // sum
                dLdhFwd = dLdh;
                dLdhBwd = dLdh;
            }
            else if(directionMode == 3) {   // concat
                dLdhFwd = new NDArray(dataFormat <= 1 ? (*dLdh)({0,0, 0,0,    0,nOut})   : (*dLdh)({0,0,    0,nOut,   0,0}));
                dLdhBwd = new NDArray(dataFormat <= 1 ? (*dLdh)({0,0, 0,0, nOut,2*nOut}) : (*dLdh)({0,0, nOut,2*nOut, 0,0}));
            }
            else {  // directionMode == 4
                dLdhFwd = new NDArray((*dLdh)({0,0, 0,1, 0,0, 0,0}));
                dLdhBwd = new NDArray((*dLdh)({0,0, 1,2, 0,0, 0,0}));
            }
        }

        NDArray dLdxBwd = dLdx->ulike();

        // FIXME - following two calls are independent and may run in different streams
        helpers::lstmLayerTimeLoopBp(x, &WxFwd, &WrFwd, bFwd, seqLen, hIFwd, cIFwd, WpFwd, dLdhFwd, dLdhLFwd, dLdcLFwd, params, true,  dLdx,     &dLdWxFwd, &dLdWrFwd, dLdbFwd, dLdhIFwd, dLdcIFwd, dLdWpFwd);
        helpers::lstmLayerTimeLoopBp(x, &WxBwd, &WrBwd, bBwd, seqLen, hIBwd, cIBwd, WpBwd, dLdhBwd, dLdhLBwd, dLdcLBwd, params, false, &dLdxBwd, &dLdWxBwd, &dLdWrBwd, dLdbBwd, dLdhIBwd, dLdcIBwd, dLdWpBwd);

        *dLdx += dLdxBwd;

        delete WpFwd; delete WpBwd; delete bFwd; delete bBwd; delete hIFwd; delete hIBwd; delete cIFwd; delete cIBwd;
        delete dLdhLFwd; delete dLdhLBwd; delete dLdcLFwd; delete dLdcLBwd;
        delete dLdWpFwd; delete dLdWpBwd; delete dLdbFwd; delete dLdbBwd;
        delete dLdhIFwd; delete dLdhIBwd; delete dLdcIFwd; delete dLdcIBwd;

        if(!(dLdh && directionMode == 2)) { delete dLdhFwd; delete dLdhBwd; }
    }

    return Status::OK();
}

DECLARE_TYPES(lstmLayer_bp) {
    getOpDescriptor()
            ->setAllowedInputTypes(sd::DataType::ANY)
            ->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(lstmLayer_bp) {

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present

    int count = 3;
    const auto x  = INPUT_VARIABLE(0);          // input
    const auto Wx = INPUT_VARIABLE(1);          // input weights
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights
    const auto b      = hasBiases  ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto seqLen = hasSeqLen  ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector
    const auto hI     = hasInitH   ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI     = hasInitC   ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
    const auto Wp     = hasPH      ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights

    auto outShapes = SHAPELIST(x->shapeInfo(), Wx->shapeInfo(), Wr->shapeInfo());

    if(b != nullptr)
        outShapes->push_back(b->shapeInfo());
    if(seqLen != nullptr)
        outShapes->push_back(seqLen->shapeInfo());
    if(hI != nullptr)
        outShapes->push_back(hI->shapeInfo());
    if(cI != nullptr)
        outShapes->push_back(cI->shapeInfo());
    if(Wp != nullptr)
        outShapes->push_back(Wp->shapeInfo());

    return outShapes;
}

}
}

#endif