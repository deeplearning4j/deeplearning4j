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

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>


namespace sd      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void deconv2dMKLDNN(const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output,
                            const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW,
                            const int paddingMode, const bool isNCHW) {

    // weights [oC, iC, kH, kW] always, mkl doesn't support [kH, kW, oC, iC], so we'll perform permutation

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    dnnl::memory::dims strides   = { sH, sW };
    dnnl::memory::dims padding   = { pH, pW };
    dnnl::memory::dims padding_r = { (iH - 1) * sH - oH + kH - pH, (iW - 1) * sW - oW + kW - pW };
    dnnl::memory::dims dilation  = { dH-1, dW-1 };

    // input type
    dnnl::memory::data_type xType;
    if(input->dataType() == DataType::FLOAT32)
        xType = dnnl::memory::data_type::f32;
    else if(input->dataType() == DataType::HALF)
        xType = dnnl::memory::data_type::f16;
    else if(input->dataType() == DataType::UINT8)
        xType = dnnl::memory::data_type::u8;
    else
        xType = dnnl::memory::data_type::s8;

    // weights type
    dnnl::memory::data_type wType = xType;
    if(xType == dnnl::memory::data_type::u8)
        wType = dnnl::memory::data_type::s8;

    // output and bias type (have the same types)
    dnnl::memory::data_type zType;
    if(output->dataType() == DataType::FLOAT32)
        zType = dnnl::memory::data_type::f32;
    else if(output->dataType() == DataType::HALF)
        zType = dnnl::memory::data_type::f16;
    else if(output->dataType() == DataType::UINT8)
        zType = dnnl::memory::data_type::u8;
    else if(output->dataType() == DataType::INT8)
        zType = dnnl::memory::data_type::s8;
    else
        zType = dnnl::memory::data_type::s32;

    dnnl::memory::format_tag xFormat = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
    dnnl::memory::format_tag wFormat = dnnl::memory::format_tag::oihw;

    dnnl::memory::dims xDims = {bS, iC, iH, iW};
    dnnl::memory::dims wDims = {oC, iC, kH, kW};
    dnnl::memory::dims zDims = {bS, oC, oH, oW};

    // memory descriptors for arrays

    // input
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xDims, xType, dnnl::memory::format_tag::any);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, xType, xFormat);
    if(input->ews() != 1 || input->ordering() != 'c') {
        x_user_md.data.format_kind = dnnl_blocked;    // overrides format
        x_user_md.data.format_desc.blocking.strides[0] = input->strideAt(0);
        x_user_md.data.format_desc.blocking.strides[1] = input->strideAt(1);
        x_user_md.data.format_desc.blocking.strides[2] = input->strideAt(2);
        x_user_md.data.format_desc.blocking.strides[3] = input->strideAt(3);
    }

    // weights
    dnnl::memory::desc w_mkl_md  = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
    dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormat);
    w_user_md.data.format_kind = dnnl_blocked;    // overrides format
    w_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(2);  // [kH, kW, oC, iC] -> [oC, iC, kH, kW]
    w_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(3);
    w_user_md.data.format_desc.blocking.strides[2] = weights->strideAt(0);
    w_user_md.data.format_desc.blocking.strides[3] = weights->strideAt(1);

    // bias
    dnnl::memory::desc b_mkl_md;
    if(bias != nullptr)
        b_mkl_md = dnnl::memory::desc({oC}, zType, dnnl::memory::format_tag::x);

    // output
    dnnl::memory::desc z_mkl_md  = dnnl::memory::desc(zDims, zType, dnnl::memory::format_tag::any);
    dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, zType, xFormat);
    if(output->ews() != 1 || output->ordering() != 'c') {
        z_user_md.data.format_kind = dnnl_blocked;    // overrides format
        z_user_md.data.format_desc.blocking.strides[0] = output->strideAt(0);
        z_user_md.data.format_desc.blocking.strides[1] = output->strideAt(1);
        z_user_md.data.format_desc.blocking.strides[2] = output->strideAt(2);
        z_user_md.data.format_desc.blocking.strides[3] = output->strideAt(3);
    }

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // operation primitive description
    dnnl::deconvolution_forward::desc op_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct,
                                                x_mkl_md, w_mkl_md, b_mkl_md, z_mkl_md, strides, dilation, padding, padding_r);
    dnnl::deconvolution_forward::primitive_desc op_prim_desc(op_desc, engine);

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
    dnnl::deconvolution_forward(op_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (zReorder)
        dnnl::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

    stream.wait();

    // shape::printArray(z_mkl_mem.map_data<float>(),8);
}

//////////////////////////////////////////////////////////////////////////
static void deconv2dBpMKLDNN(const NDArray* input, const NDArray* weights, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB,
                                    const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW,
                                    const int paddingMode, const bool isNCHW) {

    // weights and gradW [oC, iC, kH, kW] always, mkl doesn't support [kH, kW, oC, iC], so we'll perform permutation

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    dnnl::memory::dims strides   = { sH, sW };
    dnnl::memory::dims padding   = { pH, pW };
    dnnl::memory::dims padding_r = { (iH - 1) * sH - oH + kH - pH, (iW - 1) * sW - oW + kW - pW };
    dnnl::memory::dims dilation  = { dH-1, dW-1 };

    // input type
    dnnl::memory::data_type xType = input->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
    // weights type
    dnnl::memory::data_type wType = weights->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
    // gradO type
    dnnl::memory::data_type gradOType = gradO->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
    // gradI type
    dnnl::memory::data_type gradIType = gradI->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
    // gradW type
    dnnl::memory::data_type gradWType = gradW->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16;
    // gradB type
    dnnl::memory::data_type gradBType = gradB != nullptr ? (gradB->dataType() == DataType::FLOAT32 ? dnnl::memory::data_type::f32 : dnnl::memory::data_type::bf16) : dnnl::memory::data_type::f32;

    dnnl::memory::format_tag xFormat = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
    dnnl::memory::format_tag wFormat = dnnl::memory::format_tag::oihw;

    dnnl::memory::dims xDims = {bS, iC, iH, iW};
    dnnl::memory::dims wDims = {oC, iC, kH, kW};
    dnnl::memory::dims zDims = {bS, oC, oH, oW};

    // memory descriptors for arrays

    // input
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xDims, xType, dnnl::memory::format_tag::any);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, xType, xFormat);
    if(input->ews() != 1 || input->ordering() != 'c') {
        x_user_md.data.format_kind = dnnl_blocked;    // overrides format
        x_user_md.data.format_desc.blocking.strides[0] = input->strideAt(0);
        x_user_md.data.format_desc.blocking.strides[1] = input->strideAt(1);
        x_user_md.data.format_desc.blocking.strides[2] = input->strideAt(2);
        x_user_md.data.format_desc.blocking.strides[3] = input->strideAt(3);
    }

    // weights
    dnnl::memory::desc w_mkl_md  = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
    dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormat);
    w_user_md.data.format_kind = dnnl_blocked;    // overrides format
    w_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(2);  // [kH, kW, oC, iC] -> [oC, iC, kH, kW]
    w_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(3);
    w_user_md.data.format_desc.blocking.strides[2] = weights->strideAt(0);
    w_user_md.data.format_desc.blocking.strides[3] = weights->strideAt(1);

    // gradO
    dnnl::memory::desc gradO_mkl_md  = dnnl::memory::desc(zDims, gradOType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, gradOType, xFormat);
    if(gradO->ews() != 1 || gradO->ordering() != 'c') {
        gradO_user_md.data.format_kind = dnnl_blocked;    // overrides format
        gradO_user_md.data.format_desc.blocking.strides[0] = gradO->strideAt(0);
        gradO_user_md.data.format_desc.blocking.strides[1] = gradO->strideAt(1);
        gradO_user_md.data.format_desc.blocking.strides[2] = gradO->strideAt(2);
        gradO_user_md.data.format_desc.blocking.strides[3] = gradO->strideAt(3);
    }

    // gradI
    dnnl::memory::desc gradI_mkl_md  = dnnl::memory::desc(xDims, gradIType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, gradIType, xFormat);
    if(gradI->ews() != 1 || gradI->ordering() != 'c') {
        gradI_user_md.data.format_kind = dnnl_blocked;    // overrides format
        gradI_user_md.data.format_desc.blocking.strides[0] = gradI->strideAt(0);
        gradI_user_md.data.format_desc.blocking.strides[1] = gradI->strideAt(1);
        gradI_user_md.data.format_desc.blocking.strides[2] = gradI->strideAt(2);
        gradI_user_md.data.format_desc.blocking.strides[3] = gradI->strideAt(3);
    }

    // gradW
    dnnl::memory::desc gradW_mkl_md  = dnnl::memory::desc(wDims, gradWType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradW_user_md = dnnl::memory::desc(wDims, gradWType, wFormat);
    gradW_user_md.data.format_kind = dnnl_blocked;    // overrides format
    gradW_user_md.data.format_desc.blocking.strides[0] = gradW->strideAt(2);  // [kH, kW, oC, iC] -> [oC, iC, kH, kW]
    gradW_user_md.data.format_desc.blocking.strides[1] = gradW->strideAt(3);
    gradW_user_md.data.format_desc.blocking.strides[2] = gradW->strideAt(0);
    gradW_user_md.data.format_desc.blocking.strides[3] = gradW->strideAt(1);

    // gradB
    dnnl::memory::desc gradB_mkl_md;
    if(gradB != nullptr)
        gradB_mkl_md = dnnl::memory::desc({oC}, gradBType, dnnl::memory::format_tag::x);

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // forward primitive description
    dnnl::deconvolution_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, dnnl::algorithm::deconvolution_direct, x_mkl_md, w_mkl_md, gradB_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::deconvolution_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

    // backward data primitive description
    dnnl::deconvolution_backward_data::desc op_data_bp_desc(dnnl::algorithm::deconvolution_direct, gradI_mkl_md, w_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::deconvolution_backward_data::primitive_desc op_data_bp_prim_desc(op_data_bp_desc, engine, op_ff_prim_desc);

    // backward weights primitive description
    dnnl::deconvolution_backward_weights::desc op_weights_bp_desc(dnnl::algorithm::deconvolution_direct, x_mkl_md, gradW_mkl_md, gradB_mkl_md, gradO_mkl_md, strides, dilation, padding, padding_r);
    dnnl::deconvolution_backward_weights::primitive_desc op_weights_bp_prim_desc(op_weights_bp_desc, engine, op_ff_prim_desc);

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
    dnnl::deconvolution_backward_data(op_data_bp_prim_desc).execute(stream, args);

    if(gradOReorderW || gradOReorderD)
        args[DNNL_ARG_DIFF_DST] = gradO_mkl_memW;

    // run backward weights calculations
    dnnl::deconvolution_backward_weights(op_weights_bp_prim_desc).execute(stream, args);

    // reorder gradI if necessary
    if (gradIReorder)
        dnnl::reorder(gradI_mkl_mem, gradI_user_mem).execute(stream, gradI_mkl_mem, gradI_user_mem);
    if (gradWReorder)
        dnnl::reorder(gradW_mkl_mem, gradW_user_mem).execute(stream, gradW_mkl_mem, gradW_user_mem);

    stream.wait();

    // shape::printArray(z_mkl_mem.map_data<float>(),8);
}


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv2d, ENGINE_CPU) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, oC, iC] always
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DECONV2D_MKLDNN OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DECONV2D_MKLDNN OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());

    int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) height
    int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW     = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // INT_ARG(9): 0-NCHW,  1-NHWC

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    std::vector<Nd4jLong> expectedWeightsShape = {kH, kW, oC, iC};
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM DECONV2D_MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV2D_MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(paddingMode){                       // SAME
        //Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward pass
        ConvolutionUtils::calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);
    }

    deconv2dMKLDNN(input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW);

    return Status::OK();
}

PLATFORM_CHECK(deconv2d, ENGINE_CPU) {
    auto input   = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

    auto output  = INPUT_VARIABLE(0);

    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                                // 0-VALID, 1-SAME

    const DataType xType = input->dataType();
    const DataType wType = weights->dataType();
    const DataType zType = output->dataType();
    const DataType bType = bias != nullptr ? bias->dataType() : zType;

    return block.isUseMKLDNN() && (dH <= 1 && dW <= 1 && !paddingMode) &&
          (
            (xType==DataType::FLOAT32 && wType==DataType::FLOAT32 && bType==DataType::FLOAT32 && zType==DataType::FLOAT32) ||
            ((xType==DataType::UINT8 || xType==DataType::INT8) && wType==DataType::INT8 && (zType==DataType::UINT8 || zType==DataType::INT8 || zType==DataType::INT32 || zType==DataType::FLOAT32) && bType == zType)
          );
}


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv2d_bp, ENGINE_CPU) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, oC, iC] always
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW), gradI
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, oC, iC] always
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM DECONV2D_MKLDNN_BP OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM DECONV2D_MKLDNN_BP OP: rank of weights array must be equal to 4 , but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf()   == 4, 0, "CUSTOM DECONV2D_MKLDNN_BP OP: rank of output gradients (next epsilon) array must be equal to 4, but got %i instead !", gradO->rankOf());


    int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));// filter(kernel) height
    int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));// filter(kernel) width
    int sH = INT_ARG(2);                                                        // strides height
    int sW = INT_ARG(3);                                                        // strides width
    int pH = INT_ARG(4);                                                        // paddings height
    int pW = INT_ARG(5);                                                        // paddings width
    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
    int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // INT_ARG(9): 1-NHWC, 0-NCHW

    int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
    int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH);

    int trueoH, trueoW;          // true output height, width
    ConvolutionUtils::calcOutSizeDeconv2D(trueoH, trueoW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, paddingMode);

    std::vector<Nd4jLong> expectedGradOShape  = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoH,trueoW,  0,indIOioC,indOoH,indOoH+1});
    std::vector<Nd4jLong> expectedWeightsShape = {kH, kW, oC, iC};
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,  "CUSTOM DECONV2D_MKLDNN_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM DECONV2D_MKLDNN_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV2D_MKLDNN_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(paddingMode){                       // SAME
        //Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward pass
        ConvolutionUtils::calcPadding2D(pH, pW, iH, iW, oH, oW, kH, kW, sH, sW, dH, dW);
    }

    deconv2dBpMKLDNN(input, weights, gradO, gradI, gradW, gradB, kH, kW, sH, sW, pH, pW, dH, dW, paddingMode, isNCHW);

    return Status::OK();
}

PLATFORM_CHECK(deconv2d_bp, ENGINE_CPU) {
    auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, oC, iC] always
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCDHW), gradI
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, oC, iC] always
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    int dH = INT_ARG(6);                                                        // dilations height
    int dW = INT_ARG(7);                                                        // dilations width
    int paddingMode = INT_ARG(8);                                                // 0-VALID, 1-SAME

    const DataType xType = input->dataType();
    const DataType wType = weights->dataType();
    const DataType gradOType = gradO->dataType();

    const DataType gradIType = gradI->dataType();
    const DataType gradWType = gradW->dataType();
    const DataType gradBType = gradB != nullptr ? gradB->dataType() : DataType::FLOAT32;

    return block.isUseMKLDNN() && (dH <= 1 && dW <= 1 && !paddingMode) && ((xType==DataType::FLOAT32 || xType==DataType::BFLOAT16) && (wType==DataType::FLOAT32 || wType==DataType::BFLOAT16) && (gradOType==DataType::FLOAT32 || gradOType==DataType::BFLOAT16) && (gradIType==DataType::FLOAT32 || gradIType==DataType::BFLOAT16) && (gradWType==DataType::FLOAT32 || gradWType==DataType::BFLOAT16) && (gradBType==DataType::FLOAT32 || gradBType==DataType::BFLOAT16) );
}


}
}
}
