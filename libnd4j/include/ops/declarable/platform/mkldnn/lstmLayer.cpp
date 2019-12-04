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
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/OpRegistrator.h>
#include "mkldnnUtils.h"

using namespace dnnl;

namespace nd4j      {
namespace ops       {
namespace platforms {

static void lstmLayerMKLDNN(const NDArray* x, const NDArray* Wx, const NDArray* Wr,
                            const NDArray* b, const NDArray* hI, const NDArray* cI,
                            const std::vector<float>& params,
                            NDArray* h, NDArray* hL, NDArray* cL) {

    // equations (no peephole connections)
    // it  = σ(Wxi * xt  +  Wri * ht-1  +  bi)
    // ft  = σ(Wxf * xt  +  Wrf * ht-1  +  bf)
    // c't = tanh(Wxc * xt  +  Wrc * ht-1  +  bc)
    // ct  = ft ◦ ct-1 + it ◦ c't
    // ot  = σ(Wxo * xt  +  Wro * ht-1  +  bo)
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

    // *******
    // input weights Wx:
    // 1) [1, 1, nIn, 4*nOut] when directionMode <  2
    // 2) [1, 2, nIn, 4*nOut] when directionMode >= 2

    // *******
    // recurrent weights Wr:
    // 1) [1, 1, nOut, 4*nOut] when directionMode <  2
    // 2) [1, 2, nOut, 4*nOut] when directionMode >= 2

    // *******
    // biases b:
    // 1) [1, 1, 4*nOut] when directionMode <  2
    // 2) [1, 2, 4*nOut] when directionMode >= 2

    // *******
    // initial output hI:
    // 1) [1, 1, bS, nOut] when directionMode <  2
    // 2) [1, 2, bS, nOut] when directionMode >= 2

    // *******
    // initial cell state cI (same shape as in hI):
    // 1) [1, 1, bS, nOut] when directionMode <  2
    // 2) [1, 2, bS, nOut] when directionMode >= 2


    //     OUTPUTS:

    // *******
    // output h:
    // 1) [sL, bS, nOut]    when directionMode <= 2 && dataFormat == 0
    // 2) [sL, bS, 2*nOut]  when directionMode == 3 && dataFormat == 0

    // *******
    // output at last step hL:
    // 1) [1, 1, bS, nOut] when directionMode <  2
    // 2) [1, 2, bS, nOut] when directionMode >= 2

    // *******
    // cell state at last step cL (same shape as in hL):
    // 1) [1, 1, bS, nOut] when directionMode <  2
    // 2) [1, 2, bS, nOut] when directionMode >= 2

    // !!! dimension 4*nOut implies order it, ft, c't, ot
    // !!! dimension 3*nOut implies order it, ft, ot

    // params = {dataFormat, directionMode, cellClip, gateAct, gateAlpha, gateBeta, cellAct, cellAlpha, cellBeta, outAct, outAlpha, outBeta};

    // dataFormat:  0 = [sL, bS, nIn]
    // directionMode:  0 = forward, 1 = backward, 2 = bidirectional sum, 3 = bidirectional concat

    const int dataFormat    = params[0];
    const int directionMode = params[1];

    const int sL   = x->sizeAt(0);      // dataFormat == 0 ?  x->sizeAt(0) : x->sizeAt(1);
    const int bS   = x->sizeAt(1);      // dataFormat == 0 ?  x->sizeAt(1) : x->sizeAt(0);
    const int nIn  = x->sizeAt(-1);
    const int nOut = Wx->sizeAt(-1);

    const int dirDim  = directionMode <  2 ? 1 : 2;     // number of dimensionss, 1 unidirectional, 2 for bidirectional
    const int hDirDim = directionMode <= 2 ? 1 : 2;     // for h array, take into account bidirectional_sum mode (directionMode == 2)

    // evaluate direction
    rnn_direction direction;
    switch (directionMode) {
        case 0:
            direction = rnn_direction::unidirectional_left2right;
            break;
        case 1:
            direction = rnn_direction::unidirectional_right2left;
            break;
        case 2:
            direction = rnn_direction::bidirectional_sum;
            break;
        default:
            direction = rnn_direction::bidirectional_concat;
    }

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    dnnl::memory::desc x_user_md, wx_user_md, wr_user_md, b_user_md, hI_user_md, cI_user_md, h_user_md, hL_user_md, cL_user_md,
                         x_lstm_md, wx_lstm_md, wr_lstm_md, b_lstm_md, hI_lstm_md, cI_lstm_md, h_lstm_md, hL_lstm_md, cL_lstm_md;

    // input type
    dnnl::memory::data_type xType;
    if(x->dataType() == DataType::FLOAT32)
        xType = dnnl::memory::data_type::f32;
    else if(x->dataType() == DataType::HALF)
        xType = dnnl::memory::data_type::f16;
    else
        xType = dnnl::memory::data_type::u8;

    // weights type
    dnnl::memory::data_type wType = xType;
    if(xType == dnnl::memory::data_type::u8)
        wType = dnnl::memory::data_type::s8;

    // bias type
    dnnl::memory::data_type bType = xType;
    if(xType == dnnl::memory::data_type::u8)
        bType = dnnl::memory::data_type::f32;

    // output type
    dnnl::memory::data_type hType;
    if(h->dataType() == DataType::FLOAT32)
        hType = dnnl::memory::data_type::f32;
    else if(h->dataType() == DataType::HALF)
        hType = dnnl::memory::data_type::f16;
    else
        hType = dnnl::memory::data_type::u8;


    // memory descriptors for arrays
    // x
    x_lstm_md = dnnl::memory::desc({sL, bS, nIn}, xType, dnnl::memory::format_tag::any);
    // x_user_md = dataFormat == 0 ? dnnl::memory::desc({sL, bS, nIn}, type, dnnl::memory::format_tag::tnc) : dnnl::memory::desc({bS, sL, nIn}, type, dnnl::memory::format_tag::ntc);
    x_user_md = dnnl::memory::desc({sL, bS, nIn}, xType, dnnl::memory::format_tag::tnc);
    x_user_md.data.format_kind = dnnl_blocked;    // overrides format
    x_user_md.data.format_desc.blocking.strides[0] = x->stridesOf()[0];
    x_user_md.data.format_desc.blocking.strides[1] = x->stridesOf()[1];
    x_user_md.data.format_desc.blocking.strides[2] = x->stridesOf()[2];

    // wx
    wx_lstm_md = dnnl::memory::desc({1,dirDim,nIn,4,nOut}, wType, dnnl::memory::format_tag::any);
    wx_user_md = dnnl::memory::desc({1,dirDim,nIn,4,nOut}, wType, dnnl::memory::format_tag::ldigo);
    wx_user_md.data.format_kind = dnnl_blocked;    // overrides format
    wx_user_md.data.format_desc.blocking.strides[0] = Wx->stridesOf()[0];
    wx_user_md.data.format_desc.blocking.strides[1] = Wx->stridesOf()[1];
    wx_user_md.data.format_desc.blocking.strides[2] = Wx->stridesOf()[2];
    wx_user_md.data.format_desc.blocking.strides[3] = Wx->stridesOf()[3];
    wx_user_md.data.format_desc.blocking.strides[4] = Wx->stridesOf()[4];

    // wr
    wr_lstm_md = dnnl::memory::desc({1,dirDim,nOut,4,nOut}, wType, dnnl::memory::format_tag::any);
    wr_user_md = dnnl::memory::desc({1,dirDim,nOut,4,nOut}, wType, dnnl::memory::format_tag::ldigo);
    wr_user_md.data.format_kind = dnnl_blocked;    // overrides format
    wr_user_md.data.format_desc.blocking.strides[0] = Wr->stridesOf()[0];
    wr_user_md.data.format_desc.blocking.strides[1] = Wr->stridesOf()[1];
    wr_user_md.data.format_desc.blocking.strides[2] = Wr->stridesOf()[2];
    wr_user_md.data.format_desc.blocking.strides[3] = Wr->stridesOf()[3];
    wr_user_md.data.format_desc.blocking.strides[4] = Wr->stridesOf()[4];

    // h
    h_lstm_md = dnnl::memory::desc({sL, bS, hDirDim*nOut}, hType, dnnl::memory::format_tag::any);
    // h_user_md = dataFormat == 0 ? dnnl::memory::desc({sL, bS, hDirDim*nOut}, type, dnnl::memory::format_tag::tnc) : dnnl::memory::desc({bS, sL, hDirDim*nOut}, type, dnnl::memory::format_tag::ntc);
    h_user_md = dnnl::memory::desc({sL, bS, hDirDim*nOut}, hType, dnnl::memory::format_tag::tnc);
    h_user_md.data.format_kind = dnnl_blocked;    // overrides format
    h_user_md.data.format_desc.blocking.strides[0] = h->stridesOf()[0];
    h_user_md.data.format_desc.blocking.strides[1] = h->stridesOf()[1];
    h_user_md.data.format_desc.blocking.strides[2] = h->stridesOf()[2];

    // b
    if(b) {
        b_lstm_md = dnnl::memory::desc({1,dirDim,4,nOut}, bType, dnnl::memory::format_tag::any);
        b_user_md = dnnl::memory::desc({1,dirDim,4,nOut}, bType, dnnl::memory::format_tag::ldgo);
        b_user_md.data.format_kind = dnnl_blocked;    // overrides format
        b_user_md.data.format_desc.blocking.strides[0] = b->stridesOf()[0];
        b_user_md.data.format_desc.blocking.strides[1] = b->stridesOf()[1];
        b_user_md.data.format_desc.blocking.strides[2] = b->stridesOf()[2];
        b_user_md.data.format_desc.blocking.strides[3] = b->stridesOf()[3];
    }

    // hI
    if(hI) {
        hI_lstm_md = dnnl::memory::desc({1,dirDim,bS,nOut}, xType, dnnl::memory::format_tag::any);
        hI_user_md = dnnl::memory::desc({1,dirDim,bS,nOut}, xType, dnnl::memory::format_tag::ldnc);
        hI_user_md.data.format_kind = dnnl_blocked;    // overrides format
        hI_user_md.data.format_desc.blocking.strides[0] = hI->stridesOf()[0];
        hI_user_md.data.format_desc.blocking.strides[1] = hI->stridesOf()[1];
        hI_user_md.data.format_desc.blocking.strides[2] = hI->stridesOf()[2];
        hI_user_md.data.format_desc.blocking.strides[3] = hI->stridesOf()[3];
    }

    // cI
    if(cI) {
        cI_lstm_md = dnnl::memory::desc({1,dirDim,bS,nOut}, xType, dnnl::memory::format_tag::any);
        cI_user_md = dnnl::memory::desc({1,dirDim,bS,nOut}, xType, dnnl::memory::format_tag::ldnc);
        cI_user_md.data.format_kind = dnnl_blocked;    // overrides format
        cI_user_md.data.format_desc.blocking.strides[0] = cI->stridesOf()[0];
        cI_user_md.data.format_desc.blocking.strides[1] = cI->stridesOf()[1];
        cI_user_md.data.format_desc.blocking.strides[2] = cI->stridesOf()[2];
        cI_user_md.data.format_desc.blocking.strides[2] = cI->stridesOf()[3];
    }

    // hL
    if(hL) {
        hL_lstm_md = dnnl::memory::desc({1,dirDim,bS,nOut}, hType, dnnl::memory::format_tag::any);
        hL_user_md = dnnl::memory::desc({1,dirDim,bS,nOut}, hType, dnnl::memory::format_tag::ldnc);
        hL_user_md.data.format_kind = dnnl_blocked;    // overrides format
        hL_user_md.data.format_desc.blocking.strides[0] = hL->stridesOf()[0];
        hL_user_md.data.format_desc.blocking.strides[1] = hL->stridesOf()[1];
        hL_user_md.data.format_desc.blocking.strides[2] = hL->stridesOf()[2];
        hL_user_md.data.format_desc.blocking.strides[3] = hL->stridesOf()[3];
    }

    if(cL) {
        cL_lstm_md = dnnl::memory::desc({1,dirDim,bS,nOut}, hType, dnnl::memory::format_tag::ldnc);
        cL_user_md = dnnl::memory::desc({1,dirDim,bS,nOut}, hType, dnnl::memory::format_tag::ldnc);
        cL_user_md.data.format_kind = dnnl_blocked;    // overrides format
        cL_user_md.data.format_desc.blocking.strides[0] = cL->stridesOf()[0];
        cL_user_md.data.format_desc.blocking.strides[1] = cL->stridesOf()[1];
        cL_user_md.data.format_desc.blocking.strides[2] = cL->stridesOf()[2];
        cL_user_md.data.format_desc.blocking.strides[3] = cL->stridesOf()[3];
    }

    // lstm memory description
    lstm_forward::desc lstm_desc(prop_kind::forward_inference, direction,
                                 x_lstm_md, hI_lstm_md, cI_lstm_md, wx_lstm_md, wr_lstm_md, b_lstm_md,
                                 h_lstm_md, hL_lstm_md, cL_lstm_md);

    dnnl::stream stream(engine);

    // lstm primitive description
    lstm_forward::primitive_desc lstm_prim_desc(lstm_desc, engine);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, dnnl::memory> args;

    // provide memory and check whether reorder is required
    // x
    auto x_user_mem = dnnl::memory(x_user_md, engine, x->getBuffer());
    const bool xReorder = lstm_prim_desc.src_layer_desc() != x_user_mem.get_desc();
    auto x_lstm_mem = xReorder ? dnnl::memory(lstm_prim_desc.src_layer_desc(), engine) : x_user_mem;
    if (xReorder)
        reorder(x_user_mem, x_lstm_mem).execute(stream, x_user_mem, x_lstm_mem);
    args[DNNL_ARG_SRC_LAYER] = x_lstm_mem;

    // wx
    auto wx_user_mem = dnnl::memory(wx_user_md, engine, Wx->getBuffer());
    const bool wxReorder = lstm_prim_desc.weights_layer_desc()!= wx_user_mem.get_desc();
    auto wx_lstm_mem = wxReorder ? dnnl::memory(lstm_prim_desc.weights_layer_desc(), engine) : wx_user_mem;
    if (wxReorder)
        reorder(wx_user_mem, wx_lstm_mem).execute(stream, wx_user_mem, wx_lstm_mem);
    args[DNNL_ARG_WEIGHTS_LAYER] = wx_lstm_mem;

    // wr
    auto wr_user_mem = dnnl::memory(wr_user_md, engine, Wr->getBuffer());
    const bool wrReorder = lstm_prim_desc.weights_iter_desc() != wr_user_mem.get_desc();
    auto wr_lstm_mem = wxReorder ? dnnl::memory(lstm_prim_desc.weights_iter_desc(), engine) : wr_user_mem;
    if (wrReorder)
        reorder(wr_user_mem, wr_lstm_mem).execute(stream, wr_user_mem, wr_lstm_mem);
    args[DNNL_ARG_WEIGHTS_ITER] = wr_lstm_mem;

    // h
    auto h_user_mem = dnnl::memory(h_user_md, engine, h->getBuffer());
    const bool hReorder = lstm_prim_desc.dst_layer_desc() != h_user_mem.get_desc();
    auto h_lstm_mem = hReorder ? dnnl::memory(lstm_prim_desc.dst_layer_desc(), engine) : h_user_mem;
    args[DNNL_ARG_DST_LAYER] = h_lstm_mem;

    // b
    if(b) {
        auto b_user_mem  = dnnl::memory(b_user_md, engine, b->getBuffer());
        const bool bReorder = lstm_prim_desc.bias_desc() != b_user_mem.get_desc();
        auto b_lstm_mem = bReorder ? dnnl::memory(lstm_prim_desc.bias_desc(), engine) : b_user_mem;
        if (bReorder)
            reorder(b_user_mem, b_lstm_mem).execute(stream, b_user_mem, b_lstm_mem);
        args[DNNL_ARG_BIAS] = b_lstm_mem;
    }

    // hI
    if(hI) {
        auto hI_user_mem = dnnl::memory(hI_user_md, engine, hI->getBuffer());
        const bool hIReorder = lstm_prim_desc.src_iter_desc() != hI_user_mem.get_desc();
        auto hI_lstm_mem = hIReorder ? dnnl::memory(lstm_prim_desc.src_iter_desc(), engine) : hI_user_mem;
        if (hIReorder)
            reorder(hI_user_mem, hI_lstm_mem).execute(stream, hI_user_mem, hI_lstm_mem);
        args[DNNL_ARG_SRC_ITER] = hI_lstm_mem;
    }

    // cI
    if(cI) {
        auto cI_user_mem = dnnl::memory(cI_user_md, engine, cI->getBuffer());
        const bool cIReorder = lstm_prim_desc.src_iter_c_desc() != cI_user_mem.get_desc();
        auto cI_lstm_mem = cIReorder ? dnnl::memory(lstm_prim_desc.src_iter_c_desc(), engine) : cI_user_mem;
        if (cIReorder)
            reorder(cI_user_mem, cI_lstm_mem).execute(stream, cI_user_mem, cI_lstm_mem);
        args[DNNL_ARG_SRC_ITER_C] = cI_lstm_mem;
    }

    bool hLReorder(false), cLReorder(false);
    dnnl::memory hL_user_mem, cL_user_mem, hL_lstm_mem, cL_lstm_mem;

    // hL
    if(hL) {
        hL_user_mem = dnnl::memory(hL_user_md, engine, hL->getBuffer());
        hLReorder = lstm_prim_desc.dst_iter_desc() != hL_user_mem.get_desc();
        hL_lstm_mem = hLReorder ? dnnl::memory(lstm_prim_desc.dst_iter_desc(), engine) : hL_user_mem;
        args[DNNL_ARG_DST_ITER] = hL_lstm_mem;
    }

    // cL
    if(cL) {
        cL_user_mem = dnnl::memory(cL_user_md, engine, cL->getBuffer());
        cLReorder = lstm_prim_desc.dst_iter_c_desc() != cL_user_mem.get_desc();
        cL_lstm_mem = cLReorder ? dnnl::memory(lstm_prim_desc.dst_iter_c_desc(), engine) : cL_user_mem;
        args[DNNL_ARG_DST_ITER_C] = cL_lstm_mem;
    }

    // run calculations
    lstm_forward(lstm_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (hReorder)
        reorder(h_lstm_mem, h_user_mem).execute(stream, h_lstm_mem, h_user_mem);
    if(hLReorder)
        reorder(hL_lstm_mem, hL_user_mem).execute(stream, hL_lstm_mem, hL_user_mem);
    if(cLReorder)
        reorder(cL_lstm_mem, cL_user_mem).execute(stream, cL_lstm_mem, cL_user_mem);

    stream.wait();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(lstmLayer) {

    const auto dataFormat    = INT_ARG(0);    // for unidirectional: 0 = [sL, bS, nIn], 1 = [bS, sL ,nIn], 2 = [bS, nIn, sL], for bidirectional: 3 = [sL, 2, bS, nOut] (for ONNX)
    const auto directionMode = INT_ARG(1);    // direction: 0 = fwd, 1 = bwd, 2 = bidirectional sum, 3 = bidirectional concat, 4 = bidirectional extra output dim (in conjunction with format dataFormat = 3)

    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasSeqLen  = B_ARG(1);   // indicates whether seqLen array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto hasPH      = B_ARG(4);   // indicates whether peephole connections are present
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    const auto cellClip = T_ARG(0);                                     // cell clipping value, if it = 0 then do not apply clipping

    const auto x  = INPUT_VARIABLE(0);          // input
    const auto Wx = INPUT_VARIABLE(1);          // input weights
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights

    int count = 3;
    const auto b      = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto seqLen = hasSeqLen ? INPUT_VARIABLE(count++) : nullptr;  // seqLen vector
    const auto hI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI     = hasInitC  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state
    const auto Wp     = hasPH     ? INPUT_VARIABLE(count++) : nullptr;  // peephole weights

    REQUIRE_TRUE(cellClip == 0 , 0, "LSTM_LAYER_MKLDNN operation: cell clipping is not supported currently !");
    REQUIRE_TRUE(retFullSeq, 0, "LSTM_LAYER_MKLDNN operation: option to calculate full time sequence output h should be always true in case of mkl dnn library !");
    REQUIRE_TRUE(hasPH == false , 0, "LSTM_LAYER_MKLDNN operation: mkl dnn library doesn't support peephole connections !");
    REQUIRE_TRUE(hasSeqLen == false, 0, "LSTM_LAYER_MKLDNN operation: mkl dnn library doesn't support array specifying max time step per each example in batch !");
    REQUIRE_TRUE(dataFormat < 2, 0, "LSTM_LAYER_MKLDNN operation: wrong data format, only two formats are allowed for input/output tensors in mkl dnn library: TNC and NTC!");
    REQUIRE_TRUE(directionMode < 4, 0, "LSTM_LAYER_MKLDNN operation: option for bidirectional extra output dimension is not valid in mkl dnn library !");
    REQUIRE_TRUE((retLastH && retLastC) || (!retLastH && !retLastC), 0, "LSTM_LAYER_MKLDNN operation: only two options are present: 1) calculate both output at last time and cell state at last time; 2) do not calculate both !");

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;           // output
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;           // output at last step
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;           // cell state at last step

    // evaluate dimensions
    const Nd4jLong sL   = dataFormat == 3 ?  x->sizeAt(0) : x->sizeAt(dataFormat);
    const Nd4jLong bS   = dataFormat == 1 || dataFormat == 2 ? x->sizeAt(0) : x->sizeAt(-2);
    const Nd4jLong nIn  = dataFormat == 2 ? x->sizeAt(1) : x->sizeAt(-1);
    const Nd4jLong nOut = Wx->sizeAt(-1) / 4;

    // inputs validations
    if(directionMode < 2) {     // no bidirectional

        // Wx validation
        if(Wx->rankOf() != 2 || Wx->sizeAt(0) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 2 || Wr->sizeAt(0) != nOut || Wr->sizeAt(1) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 1 || b->sizeAt(0) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 2 || hI->sizeAt(0) != bS || hI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 2 || cI->sizeAt(0) != bS || cI->sizeAt(1) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    }
    else {                  // bidirectional
         // Wx validation
        if(Wx->rankOf() != 3 || Wx->sizeAt(0) != 2 || Wx->sizeAt(1) != nIn)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of input weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nIn, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wx).c_str());
        // Wr validation
        if(Wr->rankOf() != 3 || Wr->sizeAt(0) != 2 || Wr->sizeAt(1) != nOut || Wr->sizeAt(2) != 4*nOut)
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of recurrent weights, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, nOut, 4*nOut}).c_str(), ShapeUtils::shapeAsString(Wr).c_str());
        // biases validation
        if(b != nullptr && (b->rankOf() != 2 || b->sizeAt(0) != 2 || b->sizeAt(1) != 4*nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of biases, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, 4*nOut}).c_str(), ShapeUtils::shapeAsString(b).c_str());
        // initial output validation
        if(hI != nullptr && (hI->rankOf() != 3 || hI->sizeAt(0) != 2 || hI->sizeAt(1) != bS || hI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of initial output, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(hI).c_str());
        // initial cell  validation
        if(cI != nullptr && (cI->rankOf() != 3 || cI->sizeAt(0) != 2 || cI->sizeAt(1) != bS || cI->sizeAt(2) != nOut))
            REQUIRE_TRUE(false, 0, "LSTM_LAYER_MKLDNN operation: wrong shape of initial cell state, expected is %s, but got %s instead !", ShapeUtils::shapeAsString({2, bS, nOut}).c_str(), ShapeUtils::shapeAsString(cI).c_str());
    }

    std::vector<float> params = {static_cast<float>(dataFormat), static_cast<float>(directionMode), static_cast<float>(cellClip)};

    const int dirDim = directionMode < 2 ? 1 : 2;     // number of dimensions, 1 unidirectional, 2 for bidirectional

    // permut x and h to tnc format if they have ntc format
    NDArray* xP(const_cast<NDArray*>(x)), *hP(h);
    if(dataFormat == 1) {
        xP = new NDArray(x->permute({1,0,2}));      // [bS, sL, nIn] -> [sL, bS, nIn]
        hP = new NDArray(h->permute({1,0,2}));      // [bS, sL, dirDim*nOn] -> [sL, bS, dirDim*nOn]
    }

    // reshape arrays in accordance to mkl allowed formats
    NDArray *WxR(nullptr), *WrR(nullptr), *bR(nullptr), *hIR(nullptr), *cIR(nullptr), *hLR(nullptr), *cLR(nullptr);

    WxR = new NDArray(Wx->reshape(Wx->ordering(), {1,dirDim,nIn,4,nOut}));
    WrR = new NDArray(Wr->reshape(Wr->ordering(), {1,dirDim,nOut,4,nOut}));
    if(b)
        bR  = new NDArray(b->reshape(b->ordering(),  {1,dirDim,4,nOut}));
    if(hI)
        hIR = new NDArray(hI->reshape(hI->ordering(), {1,dirDim,bS,nOut}));
    if(cI)
        cIR = new NDArray(cI->reshape(cI->ordering(), {1,dirDim,bS,nOut}));
    if(hL)
        hLR = new NDArray(hL->reshape(hL->ordering(), {1,dirDim,bS,nOut}));
    if(cL)
        cLR = new NDArray(cL->reshape(cL->ordering(), {1,dirDim,bS,nOut}));

    lstmLayerMKLDNN(xP, WxR, WrR, bR, hIR, cIR, params, hP, hLR, cLR);

    delete WxR;
    delete WrR;
    delete bR;
    delete hIR;
    delete cIR;
    delete hLR;
    delete cLR;

    if(dataFormat == 1) {
        delete xP;
        delete hP;
    }

    return Status::OK();
}

PLATFORM_CHECK(lstmLayer) {
    const auto hasBiases  = B_ARG(0);   // indicates whether biases array is provided
    const auto hasInitH   = B_ARG(2);   // indicates whether initial output is provided
    const auto hasInitC   = B_ARG(3);   // indicates whether initial cell state is provided
    const auto retFullSeq = B_ARG(5);   // indicates whether to return whole time sequence h {h_0, h_1, ... , h_sL-1}
    const auto retLastH   = B_ARG(6);   // indicates whether to return output at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)
    const auto retLastC   = B_ARG(7);   // indicates whether to return cells state at last time step only, in this case shape would be [bS, nOut] (exact shape depends on dataFormat argument)

    const auto x  = INPUT_VARIABLE(0);          // input
    const auto Wx = INPUT_VARIABLE(1);          // input weights
    const auto Wr = INPUT_VARIABLE(2);          // recurrent weights

    int count = 3;
    const auto b      = hasBiases ? INPUT_VARIABLE(count++) : nullptr;  // biases
    const auto hI     = hasInitH  ? INPUT_VARIABLE(count++) : nullptr;  // initial output
    const auto cI     = hasInitC  ? INPUT_VARIABLE(count++) : nullptr;  // initial cell state

    count = 0;
    auto h  = retFullSeq ? OUTPUT_VARIABLE(count++) : nullptr;           // output
    auto hL = retLastH   ? OUTPUT_VARIABLE(count++) : nullptr;           // output at last step
    auto cL = retLastC   ? OUTPUT_VARIABLE(count++) : nullptr;           // cell state at last step

    DataType xType  = x->dataType();
    DataType WxType = Wx->dataType();
    DataType WrType = Wr->dataType();
    DataType bType  = b  != nullptr ? b->dataType() : (xType == DataType::HALF ? xType : DataType::FLOAT32);
    DataType hIType = hI != nullptr ? hI->dataType() : xType;
    DataType cIType = cI != nullptr ? hI->dataType() : xType;
    DataType hType  = h  != nullptr ? h->dataType()  : xType;
    DataType hLType = hL != nullptr ? hL->dataType() : xType;
    DataType cLType = cL != nullptr ? cL->dataType() : xType;

    return block.isUseMKLDNN() && (
            (xType==DataType::FLOAT32 && WxType==DataType::FLOAT32 && WrType==DataType::FLOAT32 && bType==DataType::FLOAT32 && hIType==DataType::FLOAT32 && cIType==DataType::FLOAT32 && hType==DataType::FLOAT32 && hLType==DataType::FLOAT32 && cLType==DataType::FLOAT32) ||
            (xType==DataType::HALF    && WxType==DataType::HALF    && WrType==DataType::HALF    && bType==DataType::HALF    && hIType==DataType::HALF    && cIType==DataType::HALF    && hType==DataType::HALF    && hLType==DataType::HALF    && cLType==DataType::HALF)    ||
            (xType==DataType::UINT8   && WxType==DataType::INT8    && WrType==DataType::INT8    && bType==DataType::FLOAT32 && hIType==DataType::UINT8   && cIType==DataType::UINT8   && (hType==DataType::FLOAT32 && hLType==DataType::FLOAT32 && cLType==DataType::FLOAT32 || hType==DataType::UINT8 && hLType==DataType::UINT8 && cLType==DataType::UINT8))
          );
}



}
}
}
