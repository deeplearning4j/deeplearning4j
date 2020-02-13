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
// @author saudet
// @author raver119@gmail.com
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

using namespace dnnl;

namespace nd4j      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////
static void conv3dMKLDNN(const NDArray *input, const NDArray *weights,
                        const NDArray *bias, NDArray *output,
                        const int kD, const int kH, const int kW,
                        const int sD, const int sH, const int sW,
                        const int pD, const int pH, const int pW,
                        const int dD, const int dH, const int dW,
                        const int paddingMode, const int isNCDHW) {

    // weights [kD, kH, kW, iC, oC], we'll perform permutation since mkl support [oC, iC, kD, kH, kW]

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    // const int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2 : pW;       // dH == 1 for causal mode in conv1d

    dnnl::memory::dims strides   = {sD, sH, sW};
    dnnl::memory::dims padding   = {pD, pH, pW};
    // dnnl::memory::dims padding_r = { (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pWSame };
    dnnl::memory::dims padding_r = {(oD - 1) * sD - iD + kD - pD, (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW};
    dnnl::memory::dims dilation  = {dD-1, dH-1, dW-1};

    auto xzFrmat = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
    dnnl::memory::format_tag wFormat = dnnl::memory::format_tag::oidhw;

    dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
    dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
    dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

    auto type = dnnl::memory::data_type::f32;

    // memory descriptors for arrays

    // input
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFrmat);
    if(input->ews() != 1 || input->ordering() != 'c') {
        x_user_md.data.format_kind = dnnl_blocked;    // overrides format
        x_user_md.data.format_desc.blocking.strides[0] = input->strideAt(0);
        x_user_md.data.format_desc.blocking.strides[1] = input->strideAt(1);
        x_user_md.data.format_desc.blocking.strides[2] = input->strideAt(2);
        x_user_md.data.format_desc.blocking.strides[3] = input->strideAt(3);
        x_user_md.data.format_desc.blocking.strides[4] = input->strideAt(4);
    }

    // weights
    dnnl::memory::desc w_mkl_md  = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, type, wFormat);
    w_user_md.data.format_kind = dnnl_blocked;    // overrides format
    w_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(4);   // permute [kD, kH, kW, iC, oC] -> [oC, iC, kD, kH, kW]
    w_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(3);
    w_user_md.data.format_desc.blocking.strides[2] = weights->strideAt(0);
    w_user_md.data.format_desc.blocking.strides[3] = weights->strideAt(1);
    w_user_md.data.format_desc.blocking.strides[4] = weights->strideAt(2);

    // bias
    dnnl::memory::desc b_mkl_md;
    if(bias != nullptr)
        b_mkl_md = dnnl::memory::desc({oC}, type, dnnl::memory::format_tag::x);

    // output
    dnnl::memory::desc z_mkl_md  = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, type, xzFrmat);
    if(output->ews() != 1 || output->ordering() != 'c') {
        z_user_md.data.format_kind = dnnl_blocked;    // overrides format
        z_user_md.data.format_desc.blocking.strides[0] = output->strideAt(0);
        z_user_md.data.format_desc.blocking.strides[1] = output->strideAt(1);
        z_user_md.data.format_desc.blocking.strides[2] = output->strideAt(2);
        z_user_md.data.format_desc.blocking.strides[3] = output->strideAt(3);
        z_user_md.data.format_desc.blocking.strides[4] = output->strideAt(4);
    }

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // operation primitive description
    dnnl::convolution_forward::desc op_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto, x_mkl_md, w_mkl_md, b_mkl_md, z_mkl_md, strides, dilation, padding, padding_r);
    dnnl::convolution_forward::primitive_desc op_prim_desc(op_desc, engine);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, dnnl::memory> args;

    dnnl::stream stream(engine);

    // provide memory buffers and check whether reorder is required

    // input
    auto x_user_mem = dnnl::memory(x_user_md, engine, input->getBuffer());
    const bool xReorder = op_prim_desc.src_desc() != x_user_mem.get_desc();
    auto x_mkl_mem = xReorder ? dnnl::memory(op_prim_desc.src_desc(), engine) : x_user_mem;
    if (xReorder)
        dnnl::reorder(x_user_mem, x_mkl_mem).execute(stream, x_user_mem, x_mkl_mem);
    args[DNNL_ARG_SRC] = x_mkl_mem;

    // weights
    auto w_user_mem = dnnl::memory(w_user_md, engine, weights->getBuffer());
    const bool wReorder = op_prim_desc.weights_desc() != w_user_mem.get_desc();
    auto w_mkl_mem = wReorder ? dnnl::memory(op_prim_desc.weights_desc(), engine) : w_user_mem;
    if (wReorder)
        dnnl::reorder(w_user_mem, w_mkl_mem).execute(stream, w_user_mem, w_mkl_mem);
    args[DNNL_ARG_WEIGHTS] = w_mkl_mem;

    // bias
    if(bias != nullptr) {
        auto b_mkl_mem = dnnl::memory(b_mkl_md, engine, bias->getBuffer());
        args[DNNL_ARG_BIAS] = b_mkl_mem;
    }

    // output
    auto z_user_mem = dnnl::memory(z_user_md, engine, output->getBuffer());
    const bool zReorder = op_prim_desc.dst_desc() != z_user_mem.get_desc();
    auto z_mkl_mem = zReorder ? dnnl::memory(op_prim_desc.dst_desc(), engine) : z_user_mem;
    args[DNNL_ARG_DST] = z_mkl_mem;

    // run calculations
    dnnl::convolution_forward(op_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (zReorder)
        dnnl::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

    stream.wait();
}

//////////////////////////////////////////////////////////////////////
static void conv3dBpMKLDNN(const NDArray *input, const NDArray *weights, const NDArray *bias, const NDArray *gradO,
                            NDArray *gradI, NDArray *gradW, NDArray *gradB,
                            const int kD, const int kH, const int kW,
                            const int sD, const int sH, const int sW,
                            const int pD, const int pH, const int pW,
                            const int dD, const int dH, const int dW,
                            const int paddingMode, const int isNCDHW) {

    // weights/gradW [kD, kH, kW, iC, oC], we'll perform permutation since mkl support [oC, iC, kD, kH, kW]

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    // const int pWSame = (paddingMode == 2 && dW > 1) ? ((oW - 1) * sW + (kW - 1) * dW + 1 - iW) / 2 : pW;       // dH == 1 for causal mode in conv1d

    dnnl::memory::dims strides   = {sD, sH, sW};
    dnnl::memory::dims padding   = {pD, pH, pW};
    // dnnl::memory::dims padding_r = { (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pWSame };
    dnnl::memory::dims padding_r = {(oD - 1) * sD - iD + kD - pD, (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW};
    dnnl::memory::dims dilation  = {dD-1, dH-1, dW-1};

    auto xzFrmat = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
    dnnl::memory::format_tag wFormat = dnnl::memory::format_tag::oidhw;

    dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
    dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
    dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

    auto type = dnnl::memory::data_type::f32;

    // memory descriptors for arrays

    // input
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFrmat);
    if(input->ews() != 1 || input->ordering() != 'c') {
        x_user_md.data.format_kind = dnnl_blocked;    // overrides format
        x_user_md.data.format_desc.blocking.strides[0] = input->strideAt(0);
        x_user_md.data.format_desc.blocking.strides[1] = input->strideAt(1);
        x_user_md.data.format_desc.blocking.strides[2] = input->strideAt(2);
        x_user_md.data.format_desc.blocking.strides[3] = input->strideAt(3);
        x_user_md.data.format_desc.blocking.strides[4] = input->strideAt(4);
    }

    // weights
    dnnl::memory::desc w_mkl_md  = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, type, wFormat);
    w_user_md.data.format_kind = dnnl_blocked;    // overrides format
    w_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(4);   // permute [kD, kH, kW, iC, oC] -> [oC, iC, kD, kH, kW]
    w_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(3);
    w_user_md.data.format_desc.blocking.strides[2] = weights->strideAt(0);
    w_user_md.data.format_desc.blocking.strides[3] = weights->strideAt(1);
    w_user_md.data.format_desc.blocking.strides[4] = weights->strideAt(2);

    // gradO
    dnnl::memory::desc gradO_mkl_md  = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, type, xzFrmat);
    if(gradO->ews() != 1 || gradO->ordering() != 'c') {
        gradO_user_md.data.format_kind = dnnl_blocked;    // overrides format
        gradO_user_md.data.format_desc.blocking.strides[0] = gradO->strideAt(0);
        gradO_user_md.data.format_desc.blocking.strides[1] = gradO->strideAt(1);
        gradO_user_md.data.format_desc.blocking.strides[2] = gradO->strideAt(2);
        gradO_user_md.data.format_desc.blocking.strides[3] = gradO->strideAt(3);
        gradO_user_md.data.format_desc.blocking.strides[4] = gradO->strideAt(4);
    }

    // gradI
    dnnl::memory::desc gradI_mkl_md  = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, type, xzFrmat);
    if(gradI->ews() != 1 || gradI->ordering() != 'c') {
        gradI_user_md.data.format_kind = dnnl_blocked;    // overrides format
        gradI_user_md.data.format_desc.blocking.strides[0] = gradI->strideAt(0);
        gradI_user_md.data.format_desc.blocking.strides[1] = gradI->strideAt(1);
        gradI_user_md.data.format_desc.blocking.strides[2] = gradI->strideAt(2);
        gradI_user_md.data.format_desc.blocking.strides[3] = gradI->strideAt(3);
        gradI_user_md.data.format_desc.blocking.strides[4] = gradI->strideAt(4);
    }

    // gradW
    dnnl::memory::desc gradW_mkl_md  = dnnl::memory::desc(wDims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradW_user_md = dnnl::memory::desc(wDims, type, wFormat);
    gradW_user_md.data.format_kind = dnnl_blocked;    // overrides format
    gradW_user_md.data.format_desc.blocking.strides[0] = gradW->strideAt(4);   // permute [kD, kH, kW, iC, oC] -> [oC, iC, kD, kH, kW]
    gradW_user_md.data.format_desc.blocking.strides[1] = gradW->strideAt(3);
    gradW_user_md.data.format_desc.blocking.strides[2] = gradW->strideAt(0);
    gradW_user_md.data.format_desc.blocking.strides[3] = gradW->strideAt(1);
    gradW_user_md.data.format_desc.blocking.strides[4] = gradW->strideAt(2);

    // gradB
    dnnl::memory::desc gradB_mkl_md;
    if(gradB != nullptr)
        gradB_mkl_md = dnnl::memory::desc({oC}, type, dnnl::memory::format_tag::x);

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // forward primitive description
    dnnl::convolution_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_auto, x_mkl_md, w_mkl_md, gradB_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::convolution_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

    // backward data primitive description
    dnnl::convolution_backward_data::desc op_data_bp_desc(dnnl::algorithm::convolution_auto, gradI_mkl_md, w_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::convolution_backward_data::primitive_desc op_data_bp_prim_desc(op_data_bp_desc, engine, op_ff_prim_desc);

    // backward weights primitive description
    dnnl::convolution_backward_weights::desc op_weights_bp_desc(dnnl::algorithm::convolution_auto, x_mkl_md, gradW_mkl_md, gradB_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::convolution_backward_weights::primitive_desc op_weights_bp_prim_desc(op_weights_bp_desc, engine, op_ff_prim_desc);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, dnnl::memory> args;

    dnnl::stream stream(engine);

    // provide memory buffers and check whether reorder is required

    // input
    auto x_user_mem = dnnl::memory(x_user_md, engine, input->getBuffer());
    const bool xReorder = op_weights_bp_prim_desc.src_desc() != x_user_mem.get_desc();
    auto x_mkl_mem = xReorder ? dnnl::memory(op_weights_bp_prim_desc.src_desc(), engine) : x_user_mem;
    if (xReorder)
        dnnl::reorder(x_user_mem, x_mkl_mem).execute(stream, x_user_mem, x_mkl_mem);
    args[DNNL_ARG_SRC] = x_mkl_mem;

    // weights
    auto w_user_mem = dnnl::memory(w_user_md, engine, weights->getBuffer());
    const bool wReorder = op_data_bp_prim_desc.weights_desc() != w_user_mem.get_desc();
    auto w_mkl_mem = wReorder ? dnnl::memory(op_data_bp_prim_desc.weights_desc(), engine) : w_user_mem;
    if (wReorder)
        dnnl::reorder(w_user_mem, w_mkl_mem).execute(stream, w_user_mem, w_mkl_mem);
    args[DNNL_ARG_WEIGHTS] = w_mkl_mem;

    // gradO
    auto gradO_user_mem = dnnl::memory(gradO_user_md, engine, gradO->getBuffer());
    const bool gradOReorderW = op_weights_bp_prim_desc.diff_dst_desc() != gradO_user_mem.get_desc();
    const bool gradOReorderD = op_data_bp_prim_desc.diff_dst_desc()    != gradO_user_mem.get_desc();
    auto gradO_mkl_memW = gradOReorderW ? dnnl::memory(op_weights_bp_prim_desc.diff_dst_desc(), engine) : gradO_user_mem;
    auto gradO_mkl_memD = gradOReorderD ? dnnl::memory(op_data_bp_prim_desc.diff_dst_desc(), engine)    : gradO_user_mem;
    if (gradOReorderW)
        dnnl::reorder(gradO_user_mem, gradO_mkl_memW).execute(stream, gradO_user_mem, gradO_mkl_memW);
    if (gradOReorderD)
        dnnl::reorder(gradO_user_mem, gradO_mkl_memD).execute(stream, gradO_user_mem, gradO_mkl_memD);
    args[DNNL_ARG_DIFF_DST] = gradO_mkl_memD;

    // gradI
    auto gradI_user_mem = dnnl::memory(gradI_user_md, engine, gradI->getBuffer());
    const bool gradIReorder = op_data_bp_prim_desc.diff_src_desc() != gradI_user_mem.get_desc();
    auto gradI_mkl_mem = gradIReorder ? dnnl::memory(op_data_bp_prim_desc.diff_src_desc(), engine) : gradI_user_mem;
    args[DNNL_ARG_DIFF_SRC] = gradI_mkl_mem;

    // gradW
    auto gradW_user_mem = dnnl::memory(gradW_user_md, engine, gradW->getBuffer());
    const bool gradWReorder = op_weights_bp_prim_desc.diff_weights_desc() != gradW_user_mem.get_desc();
    auto gradW_mkl_mem = gradWReorder ? dnnl::memory(op_weights_bp_prim_desc.diff_weights_desc(), engine) : gradW_user_mem;
    args[DNNL_ARG_DIFF_WEIGHTS] = gradW_mkl_mem;

    // gradB
    if(gradB != nullptr) {
        auto gradB_mkl_mem = dnnl::memory(gradB_mkl_md, engine, gradB->getBuffer());
        args[DNNL_ARG_DIFF_BIAS] = gradB_mkl_mem;
    }

    // run backward data calculations
    dnnl::convolution_backward_data(op_data_bp_prim_desc).execute(stream, args);

    if(gradOReorderW || gradOReorderD)
        args[DNNL_ARG_DIFF_DST] = gradO_mkl_memW;

    // run backward weights calculations
    dnnl::convolution_backward_weights(op_weights_bp_prim_desc).execute(stream, args);

    // reorder gradI if necessary
    if (gradIReorder)
        dnnl::reorder(gradI_mkl_mem, gradI_user_mem).execute(stream, gradI_mkl_mem, gradI_user_mem);
    if (gradWReorder)
        dnnl::reorder(gradW_mkl_mem, gradW_user_mem).execute(stream, gradW_mkl_mem, gradW_user_mem);

    stream.wait();

    // shape::printArray(z_mkl_mem.map_data<float>(),8);
}


/*
//////////////////////////////////////////////////////////////////////
static void conv3dMKLDNN(nd4j::graph::Context &block,
                        const NDArray *input, const NDArray *weights, const NDArray *bias,
                              NDArray *output,
                        const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, int pD, int pH, int pW, const int dD, const int dH, const int dW,
                        const int paddingMode, const int isNCDHW) {

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    dnnl_memory_desc_t empty;
    dnnl::memory::desc conv_src_md(empty), conv_weights_md(empty), conv_bias_md(empty), conv_dst_md( empty);
    dnnl::memory::desc user_src_md(empty), user_weights_md(empty), user_bias_md(empty), user_dst_md( empty);

    dnnl::memory::dims conv_strides, conv_padding, conv_padding_r, conv_dilation;

    mkldnnUtils::getMKLDNNMemoryDescConv3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode,
                                           isNCDHW,
                                           bS, iC, iD, iH, iW, oC, oD, oH, oW, input, nullptr, weights,
                                           nullptr, bias, output,
                                           &conv_src_md, nullptr, &conv_weights_md, nullptr,
                                           &conv_bias_md, &conv_dst_md,
                                           &user_src_md, nullptr, &user_weights_md, nullptr,
                                           &user_bias_md, &user_dst_md,
                                           conv_strides, conv_padding, conv_padding_r, conv_dilation);
    auto conv_desc = bias != nullptr ? convolution_forward::desc(prop_kind::forward, algorithm::convolution_auto, conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding_r)
                                     : convolution_forward::desc(prop_kind::forward, algorithm::convolution_auto, conv_src_md, conv_weights_md, conv_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding_r);

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
    dnnl::stream stream(engine);

    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, engine);
    auto user_src_memory = dnnl::memory(user_src_md, engine, const_cast<NDArray *>(input)->buffer());
    auto user_weights_memory = dnnl::memory(user_weights_md, engine, const_cast<NDArray *>(weights)->buffer());
    auto user_dst_memory = dnnl::memory(user_dst_md, engine, output->buffer());

    auto conv_src_memory = user_src_memory;
    if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv_src_memory = dnnl::memory(conv_prim_desc.src_desc(), engine);
        reorder(user_src_memory, conv_src_memory).execute(stream, user_src_memory, conv_src_memory);
    }

    auto conv_weights_memory = user_weights_memory;
    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv_weights_memory = dnnl::memory(conv_prim_desc.weights_desc(), engine);
        reorder(user_weights_memory, conv_weights_memory).execute(stream, user_weights_memory, conv_weights_memory);
    }

    auto conv_dst_memory = user_dst_memory;
    if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
        conv_dst_memory = dnnl::memory(conv_prim_desc.dst_desc(), engine);
    }

    if (bias != nullptr) {
        auto conv_bias_memory = dnnl::memory(conv_prim_desc.bias_desc(), engine, bias->getBuffer());
        convolution_forward(conv_prim_desc).execute(stream, {{DNNL_ARG_SRC,     conv_src_memory},
                                                             {DNNL_ARG_WEIGHTS, conv_weights_memory},
                                                             {DNNL_ARG_BIAS,    conv_bias_memory},
                                                             {DNNL_ARG_DST,     conv_dst_memory}});
    }
    else {
        convolution_forward(conv_prim_desc).execute(stream, {{DNNL_ARG_SRC,     conv_src_memory},
                                                             {DNNL_ARG_WEIGHTS, conv_weights_memory},
                                                             {DNNL_ARG_DST,     conv_dst_memory}});
    }

    if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc())
        reorder(conv_dst_memory, user_dst_memory).execute(stream, conv_dst_memory, user_dst_memory);

    stream.wait();
}


//////////////////////////////////////////////////////////////////////
static void conv3dBpMKLDNN(nd4j::graph::Context &block,
                            const NDArray *input, const NDArray *weights, const NDArray *bias, const NDArray *gradO,
                            NDArray *gradI, NDArray *gradW, NDArray *gradB,
                            const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, int pD, int pH, int pW, const int dD, const int dH, const int dW,
                            const int paddingMode, const int isNCDHW) {

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    dnnl_memory_desc_t empty;
    dnnl::memory::desc conv_src_md(empty), conv_diff_src_md(empty), conv_weights_md(empty), conv_diff_weights_md(empty), conv_bias_md(empty), conv_dst_md(empty);
    dnnl::memory::desc user_src_md(empty), user_diff_src_md(empty), user_weights_md(empty), user_diff_weights_md(empty), user_bias_md(empty), user_dst_md(empty);

    dnnl::memory::dims conv_strides, conv_padding, conv_padding_r, conv_dilation;

    mkldnnUtils::getMKLDNNMemoryDescConv3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode,
                                           isNCDHW,
                                           bS, iC, iD, iH, iW, oC, oD, oH, oW, input, gradI, weights,
                                           gradW, gradB, gradO,
                                           &conv_src_md, &conv_diff_src_md, &conv_weights_md,
                                           &conv_diff_weights_md, &conv_bias_md, &conv_dst_md,
                                           &user_src_md, &user_diff_src_md, &user_weights_md,
                                           &user_diff_weights_md, &user_bias_md, &user_dst_md,
                                           conv_strides, conv_padding, conv_padding_r, conv_dilation);

    auto conv_desc = gradB != nullptr ? convolution_forward::desc(prop_kind::forward, algorithm::convolution_auto, conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding_r)
                                      : convolution_forward::desc(prop_kind::forward, algorithm::convolution_auto, conv_src_md, conv_weights_md, conv_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding_r);

    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine()));

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
    dnnl::stream stream(engine);

    if (gradW != nullptr) {

        auto convW_desc = gradB != nullptr ? convolution_backward_weights::desc(algorithm::convolution_auto, conv_src_md, conv_diff_weights_md, conv_bias_md, conv_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding_r)
                                           : convolution_backward_weights::desc(algorithm::convolution_auto, conv_src_md, conv_diff_weights_md, conv_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding_r);        auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

        auto convW_prim_desc = convolution_backward_weights::primitive_desc(convW_desc, engine, conv_prim_desc);

        auto userW_src_memory = dnnl::memory(user_src_md, engine, const_cast<NDArray *>(input)->buffer());
        auto userW_weights_memory = dnnl::memory(user_diff_weights_md, engine, gradW->buffer());
        auto userW_dst_memory = dnnl::memory(user_dst_md, engine, const_cast<NDArray *>(gradO)->buffer());

        auto convW_src_memory = userW_src_memory;
        if (convW_prim_desc.src_desc() != userW_src_memory.get_desc()) {
            convW_src_memory = dnnl::memory(convW_prim_desc.src_desc(), engine);
            reorder(userW_src_memory, convW_src_memory).execute(stream, userW_src_memory, convW_src_memory);
        }

        auto convW_weights_memory = userW_weights_memory;
        if (convW_prim_desc.diff_weights_desc() != userW_weights_memory.get_desc()) {
            convW_weights_memory = dnnl::memory(convW_prim_desc.diff_weights_desc(), engine);
        }

        auto convW_dst_memory = userW_dst_memory;
        if (convW_prim_desc.diff_dst_desc() != userW_dst_memory.get_desc()) {
            convW_dst_memory = dnnl::memory(convW_prim_desc.diff_dst_desc(), engine);
            reorder(userW_dst_memory, convW_dst_memory).execute(stream, userW_dst_memory, convW_dst_memory);
        }

        if (gradB != nullptr) {

            auto convW_bias_memory = dnnl::memory(convW_prim_desc.diff_bias_desc(), engine, gradB->buffer());

            convolution_backward_weights(convW_prim_desc).execute(stream,
                                                                  {{DNNL_ARG_SRC,          convW_src_memory},
                                                                   {DNNL_ARG_DIFF_DST,     convW_dst_memory},
                                                                   {DNNL_ARG_DIFF_WEIGHTS, convW_weights_memory},
                                                                   {DNNL_ARG_DIFF_BIAS,    convW_bias_memory}});
        }
        else {
            convolution_backward_weights(convW_prim_desc).execute(stream,
                                                                  {{DNNL_ARG_SRC,          convW_src_memory},
                                                                   {DNNL_ARG_DIFF_DST,     convW_dst_memory},
                                                                   {DNNL_ARG_DIFF_WEIGHTS, convW_weights_memory}});
        }

        if (convW_prim_desc.diff_weights_desc() != userW_weights_memory.get_desc())
            reorder(convW_weights_memory, userW_weights_memory).execute(stream, convW_weights_memory, userW_weights_memory);

        stream.wait();
    }
    if (gradI != nullptr) {
        auto convI_desc = convolution_backward_data::desc(algorithm::convolution_auto, conv_diff_src_md, conv_weights_md, conv_dst_md, conv_strides, conv_dilation, conv_padding, conv_padding_r);

        auto convI_prim_desc = convolution_backward_data::primitive_desc(convI_desc, engine, conv_prim_desc);
        auto userI_src_memory = dnnl::memory(user_diff_src_md, engine, gradI->buffer());
        auto userI_weights_memory = dnnl::memory(user_weights_md, engine, const_cast<NDArray *>(weights)->buffer());
        auto userI_dst_memory = dnnl::memory(user_dst_md, engine, const_cast<NDArray *>(gradO)->buffer());

        auto convI_src_memory = userI_src_memory;
        if (convI_prim_desc.diff_src_desc() != userI_src_memory.get_desc())
            convI_src_memory = dnnl::memory(convI_prim_desc.diff_src_desc(), engine);

        auto convI_weights_memory = userI_weights_memory;
        if (convI_prim_desc.weights_desc() != userI_weights_memory.get_desc()) {
            convI_weights_memory = dnnl::memory(convI_prim_desc.weights_desc(), engine);
            reorder(userI_weights_memory, convI_weights_memory).execute(stream, userI_weights_memory, convI_weights_memory);
        }

        auto convI_dst_memory = userI_dst_memory;
        if (convI_prim_desc.diff_dst_desc() != userI_dst_memory.get_desc()) {
            convI_dst_memory = dnnl::memory(convI_prim_desc.diff_dst_desc(), engine);
            reorder(userI_dst_memory, convI_dst_memory).execute(stream, userI_dst_memory, convI_dst_memory);
        }

        convolution_backward_data(convI_prim_desc).execute(stream,
                                                           {{DNNL_ARG_DIFF_DST, convI_dst_memory},
                                                            {DNNL_ARG_WEIGHTS,  convI_weights_memory},
                                                            {DNNL_ARG_DIFF_SRC, convI_src_memory}});

        if (convI_prim_desc.diff_src_desc() != userI_src_memory.get_desc())
            reorder(convI_src_memory, userI_src_memory).execute(stream, convI_src_memory, userI_src_memory);
    }
}
*/

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv3dnew, ENGINE_CPU) {

    auto input = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                  // [kD, kH, kW, iC, oC] always
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;       // [oC]
    auto output = OUTPUT_VARIABLE(0);                                  // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM CONV3D MKLDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D MKLDNN OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(weights->sizeAt(2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int paddingMode = INT_ARG(12);                                               // 0-SAME,  1-VALID
    int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;        // INT_ARG(13): 1-NDHWC, 0-NCDHW

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    std::vector<Nd4jLong> expectedWeightsShape = {kD, kH, kW, iC, oC};
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM CONV3D MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if (paddingMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    conv3dMKLDNN(input, weights, bias, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode, isNCDHW);

    return Status::OK();
}

PLATFORM_CHECK(conv3dnew, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC] always
    auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
    auto output = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, weights, bias, output});
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(conv3dnew_bp, ENGINE_CPU) {

    auto input = INPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                               // [kD, kH, kW, iC, oC] always
    auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                    // [oC]
    auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);         // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                // [kD, kH, kW, iC, oC] always
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                  // [oC]

    REQUIRE_TRUE(input->rankOf() == 5, 0, "CUSTOM CONV3D_BP MKLDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM CONV3D_BP MKLDNN OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf() == 5, 0, "CUSTOM CONV3D_BP MKLDNN OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !", gradO->rankOf());

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(weights->sizeAt(2));// filter(kernel) width
    int sD = INT_ARG(3);                                                        // strides depth
    int sH = INT_ARG(4);                                                        // strides height
    int sW = INT_ARG(5);                                                        // strides width
    int pD = INT_ARG(6);                                                        // paddings depth
    int pH = INT_ARG(7);                                                        // paddings height
    int pW = INT_ARG(8);                                                        // paddings width
    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int paddingMode = INT_ARG(12);                                              // 1-SAME,  0-VALID
    int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;        // INT_ARG(13): 1-NDHWC, 0-NCDHW

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWiC, indWoC, indWkD);

    if(paddingMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    int trueoD, trueoH, trueoW;          // true output depth/height/width
    ConvolutionUtils::calcOutSizePool3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, paddingMode);

    std::vector<Nd4jLong> expectedGradOShape = ShapeUtils::composeShapeUsingDimsAndIdx( {bS, oC, trueoD, trueoH, trueoW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2});
    std::vector<Nd4jLong> expectedWeightsShape = {kD, kH, kW, iC, oC};
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0, "CUSTOM CONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    conv3dBpMKLDNN(input, weights, bias, gradO, gradI, gradW, gradB, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode, isNCDHW);

    return Status::OK();
}

PLATFORM_CHECK(conv3dnew_bp, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, iC, oC] always
    auto bias = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), epsilon
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, iC, oC] always
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    return block.isUseMKLDNN() &&
           nd4j::MKLDNNStream::isSupported({input, weights, bias, gradO, gradI, gradW, gradB});
}



}
}
}