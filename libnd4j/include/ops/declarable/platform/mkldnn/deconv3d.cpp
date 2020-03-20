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
static void deconv3dMKLDNN(const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output,
                            const int kD, const int kH, const int kW, const int sD, const int sH, const int sW,
                            const int pD, const int pH, const int pW, const int dD, const int dH, const int dW,
                            const bool isNCDHW, const int wFormat) {

    // mkl supports weights in [oC, iC, kD, kH, kW] only

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWoC, indWiC, indWkD);

    dnnl::memory::dims strides   = { sD, sH, sW };
    dnnl::memory::dims padding   = { pD, pH, pW };
    dnnl::memory::dims padding_r = { (iD - 1) * sD - oD + kD - pD, (iH - 1) * sH - oH + kH - pH, (iW - 1) * sW - oW + kW - pW };
    dnnl::memory::dims dilation  = { dD-1, dH-1, dW-1 };

    uint i0, i1, i2, i3, i4;
    if(0 == wFormat) {
        i0 = 3; i1 = 4; i2 = 0; i3 = 1; i4 = 2;     // [kD, kH, kW, oC, iC] -> [oC, iC, kD, kH, kW]
    }
    else if(1 == wFormat) {
        i0 = 1; i1 = 0; i2 = 2; i3 = 3; i4 = 4;     // [iC, oC, kD, kH, kW] -> [oC, iC, kD, kH, kW]
    }
    else {
        i0 = 4; i1 = 0; i2 = 1; i3 = 2; i4 = 3;     // [iC, kD, kH, kW, oC] -> [oC, iC, kD, kH, kW]
    }

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

    dnnl::memory::format_tag xFormatMkl = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
    dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oidhw;

    dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
    dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
    dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

    // memory descriptors for arrays

    // input
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xDims, xType, dnnl::memory::format_tag::any);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, xType, xFormatMkl);
    mkldnnUtils::setBlockStrides(input, x_user_md);

    // weights
    dnnl::memory::desc w_mkl_md  = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
    dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormatMkl);
    w_user_md.data.format_kind = dnnl_blocked;    // overrides format
    w_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(i0);
    w_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(i1);
    w_user_md.data.format_desc.blocking.strides[2] = weights->strideAt(i2);
    w_user_md.data.format_desc.blocking.strides[3] = weights->strideAt(i3);
    w_user_md.data.format_desc.blocking.strides[4] = weights->strideAt(i4);

    // bias
    dnnl::memory::desc b_mkl_md;
    if(bias != nullptr)
        b_mkl_md = dnnl::memory::desc({oC}, zType, dnnl::memory::format_tag::x);

    // output
    dnnl::memory::desc z_mkl_md  = dnnl::memory::desc(zDims, zType, dnnl::memory::format_tag::any);
    dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, zType, xFormatMkl);
    mkldnnUtils::setBlockStrides(output, z_user_md);

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
    mkldnnUtils::loadDataToMklStream(input, engine, stream, x_user_md,  op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

    // weights
    mkldnnUtils::loadDataToMklStream(weights, engine, stream, w_user_md,  op_prim_desc.weights_desc(), args[DNNL_ARG_WEIGHTS]);

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
static void deconv3dBackPropMKLDNN(const NDArray* input, const NDArray* weights, const NDArray* gradO, NDArray* gradI, NDArray* gradW, NDArray* gradB,
                                    const int kD, const int kH, const int kW,
                                    const int sD, const int sH, const int sW,
                                    const int pD, const int pH, const int pW,
                                    const int dD, const int dH, const int dW,
                                    const bool isNCDHW, const int wFormat) {

    // mkl supports weights/gradW in [oC, iC, kD, kH, kW] format only

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWoC, indWiC, indWkD);

    dnnl::memory::dims strides   = { sD, sH, sW };
    dnnl::memory::dims padding   = { pD, pH, pW };
    dnnl::memory::dims padding_r = { (iD - 1) * sD - oD + kD - pD, (iH - 1) * sH - oH + kH - pH, (iW - 1) * sW - oW + kW - pW };
    dnnl::memory::dims dilation  = { dD-1, dH-1, dW-1 };

    uint i0, i1, i2, i3, i4;
    if(0 == wFormat) {
        i0 = 3; i1 = 4; i2 = 0; i3 = 1; i4 = 2;     // [kD, kH, kW, oC, iC] -> [oC, iC, kD, kH, kW]
    }
    else if(1 == wFormat) {
        i0 = 1; i1 = 0; i2 = 2; i3 = 3; i4 = 4;     // [iC, oC, kD, kH, kW] -> [oC, iC, kD, kH, kW]
    }
    else {
        i0 = 4; i1 = 0; i2 = 1; i3 = 2; i4 = 3;     // [iC, kD, kH, kW, oC] -> [oC, iC, kD, kH, kW]
    }

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

    dnnl::memory::format_tag xFormatMkl = isNCDHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
    dnnl::memory::format_tag wFormatMkl = dnnl::memory::format_tag::oidhw;

    dnnl::memory::dims xDims = {bS, iC, iD, iH, iW};
    dnnl::memory::dims wDims = {oC, iC, kD, kH, kW};
    dnnl::memory::dims zDims = {bS, oC, oD, oH, oW};

    // memory descriptors for arrays

    // input
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(xDims, xType, dnnl::memory::format_tag::any);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, xType, xFormatMkl);
    mkldnnUtils::setBlockStrides(input, x_user_md);

    // weights
    dnnl::memory::desc w_mkl_md  = dnnl::memory::desc(wDims, wType, dnnl::memory::format_tag::any);
    dnnl::memory::desc w_user_md = dnnl::memory::desc(wDims, wType, wFormatMkl);
    w_user_md.data.format_kind = dnnl_blocked;    // overrides format
    w_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(i0);
    w_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(i1);
    w_user_md.data.format_desc.blocking.strides[2] = weights->strideAt(i2);
    w_user_md.data.format_desc.blocking.strides[3] = weights->strideAt(i3);
    w_user_md.data.format_desc.blocking.strides[4] = weights->strideAt(i4);

    // gradO
    dnnl::memory::desc gradO_mkl_md  = dnnl::memory::desc(zDims, gradOType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, gradOType, xFormatMkl);
    mkldnnUtils::setBlockStrides(gradO, gradO_user_md);

    // gradI
    dnnl::memory::desc gradI_mkl_md  = dnnl::memory::desc(xDims, gradIType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, gradIType, xFormatMkl);
    mkldnnUtils::setBlockStrides(gradI, gradI_user_md);

    // gradW
    dnnl::memory::desc gradW_mkl_md  = dnnl::memory::desc(wDims, gradWType, dnnl::memory::format_tag::any);
    dnnl::memory::desc gradW_user_md = dnnl::memory::desc(wDims, gradWType, wFormatMkl);
    gradW_user_md.data.format_kind = dnnl_blocked;    // overrides format
    gradW_user_md.data.format_desc.blocking.strides[0] = gradW->strideAt(i0);
    gradW_user_md.data.format_desc.blocking.strides[1] = gradW->strideAt(i1);
    gradW_user_md.data.format_desc.blocking.strides[2] = gradW->strideAt(i2);
    gradW_user_md.data.format_desc.blocking.strides[3] = gradW->strideAt(i3);
    gradW_user_md.data.format_desc.blocking.strides[4] = gradW->strideAt(i4);

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
    mkldnnUtils::loadDataToMklStream(input, engine, stream, x_user_md,  op_weights_bp_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

    // weights
    mkldnnUtils::loadDataToMklStream(weights, engine, stream, w_user_md, op_data_bp_prim_desc.weights_desc(), args[DNNL_ARG_WEIGHTS]);

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
PLATFORM_IMPL(deconv3d, ENGINE_CPU) {

    auto input   = INPUT_VARIABLE(0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

    auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM DECONV3D_MKLDNN OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM DECONV3D_MKLDNN OP: rank of weights array must be equal to 5, but got %i instead !", weights->rankOf());

    int kD = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0));    // filter(kernel) depth
    int kH = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1));    // filter(kernel) height
    int kW = INT_ARG(2) > 0 ? INT_ARG(2) : static_cast<int>(weights->sizeAt(2));    // filter(kernel) width
    int sD = INT_ARG(3);                                                            // strides depth
    int sH = INT_ARG(4);                                                            // strides height
    int sW = INT_ARG(5);                                                            // strides width
    int pD = INT_ARG(6);                                                            // paddings depth
    int pH = INT_ARG(7);                                                            // paddings height
    int pW = INT_ARG(8);                                                            // paddings width
    int dD = INT_ARG(9);                                                            // dilations depth
    int dH = INT_ARG(10);                                                           // dilations height
    int dW = INT_ARG(11);                                                           // dilations width
    int isSameMode = INT_ARG(12);                                                   // 0-SAME,  1-VALID
    int isNCDHW  = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;           // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;             // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWoC, indWiC, indWkD);

    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM DECONV3D_MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV3D_MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(isSameMode){                       // SAME
        //Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward pass
        ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
    }

    deconv3dMKLDNN(input, weights, bias, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isNCDHW, wFormat);

    return Status::OK();
}

PLATFORM_CHECK(deconv3d, ENGINE_CPU) {
    auto input   = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;

    auto output  = INPUT_VARIABLE(0);

    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID

    const DataType xType = input->dataType();
    const DataType wType = weights->dataType();
    const DataType zType = output->dataType();
    const DataType bType = bias != nullptr ? bias->dataType() : zType;

    return block.isUseMKLDNN() && (dD <= 1 && dH <= 1 && dW <= 1 && !isSameMode) &&
          (
            (xType==DataType::FLOAT32 && wType==DataType::FLOAT32 && bType==DataType::FLOAT32 && zType==DataType::FLOAT32) ||
            ((xType==DataType::UINT8 || xType==DataType::INT8) && wType==DataType::INT8 && (zType==DataType::UINT8 || zType==DataType::INT8 || zType==DataType::INT32 || zType==DataType::FLOAT32) && bType == zType)
          );
}


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(deconv3d_bp, ENGINE_CPU) {

    auto input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW), gradI
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    REQUIRE_TRUE(input->rankOf()   == 5, 0, "CUSTOM DECONV3D_MKLDNN_BP OP: rank of input array must be equal to 5, but got %i instead !", input->rankOf());
    REQUIRE_TRUE(weights->rankOf() == 5, 0, "CUSTOM DECONV3D_MKLDNN_BP OP: rank of weights array must be equal to 5 , but got %i instead !", weights->rankOf());
    REQUIRE_TRUE(gradO->rankOf()   == 5, 0, "CUSTOM DECONV3D_MKLDNN_BP OP: rank of output gradients (next epsilon) array must be equal to 5, but got %i instead !", gradO->rankOf());


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
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID
    int isNCDHW = block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;        // INT_ARG(13): 1-NDHWC, 0-NCDHW
    int wFormat = block.getIArguments()->size() > 14 ? INT_ARG(14) : 0;         // 0 - [kD, kH, kW, oC, iC], 1 - [iC, oC, kD, kH, kW], 2 - [iC, kD, kH, kW, oC]

    int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
    int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
    ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, wFormat, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIOioD, indWoC, indWiC, indWkD);

    int trueoD, trueoH, trueoW;          // true output height, width
    ConvolutionUtils::calcOutSizeDeconv3D(trueoD, trueoH, trueoW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, isSameMode);

    std::vector<Nd4jLong> expectedGradOShape   = ShapeUtils::composeShapeUsingDimsAndIdx({bS,oC,trueoD,trueoH,trueoW,  0,indIOioC,indIOioD,indIOioD+1,indIOioD+2});
    std::vector<Nd4jLong> expectedWeightsShape = ConvolutionUtils::expectWeightsShape(wFormat, kD, kH, kW, oC, iC);
    REQUIRE_TRUE(gradO->isSameShape(expectedGradOShape), 0,  "CUSTOM DECONV3D_MKLDNN_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedGradOShape).c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(weights->isSameShape(expectedWeightsShape), 0, "CUSTOM DECONV3D_MKLDNN_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expectedWeightsShape).c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if(bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM DECONV3D_MKLDNN_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if(isSameMode)               // Note: we're intentionally swapping iH and oH, to calculated the padding for a"normal" conv (not deconv) forward pass
        ConvolutionUtils::calcPadding3D(pD, pH, pW, iD, iH, iW, oD, oH, oW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    deconv3dBackPropMKLDNN(input, weights, gradO, gradI, gradW, gradB, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isNCDHW, wFormat);

    return Status::OK();
}


PLATFORM_CHECK(deconv3d_bp, ENGINE_CPU) {
    auto input   = INPUT_VARIABLE(0);                                                // [bS, iD, iH, iW, iC] (NHWC) or [bS, iD, iC, iH, iW] (NCDHW)
    auto weights = INPUT_VARIABLE(1);                                                // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
    auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
    auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oD, oH, oW, oC] (NHWC) or [bS, oC, oD, oH, oW] (NCDHW), epsilon_next

    auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iD, iH, iW, iC] (NHWC) or [bS, iC, iD, iH, iW] (NCDHW), gradI
    auto gradW = OUTPUT_VARIABLE(1);                                                 // [kD, kH, kW, oC, iC], [iC, oC, kD, kH, kW], [iC, kD, kH, kW, oC]
    auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

    int dD = INT_ARG(9);                                                        // dilations depth
    int dH = INT_ARG(10);                                                       // dilations height
    int dW = INT_ARG(11);                                                       // dilations width
    int isSameMode = INT_ARG(12);                                               // 0-SAME,  1-VALID

    const DataType xType = input->dataType();
    const DataType wType = weights->dataType();
    const DataType gradOType = gradO->dataType();

    const DataType gradIType = gradI->dataType();
    const DataType gradWType = gradW->dataType();
    const DataType gradBType = gradB != nullptr ? gradB->dataType() : DataType::FLOAT32;

    return block.isUseMKLDNN() && (dD <= 1 && dH <= 1 && dW <= 1 && !isSameMode) && ((xType==DataType::FLOAT32 || xType==DataType::BFLOAT16) && (wType==DataType::FLOAT32 || wType==DataType::BFLOAT16) && (gradOType==DataType::FLOAT32 || gradOType==DataType::BFLOAT16) && (gradIType==DataType::FLOAT32 || gradIType==DataType::BFLOAT16) && (gradWType==DataType::FLOAT32 || gradWType==DataType::BFLOAT16) && (gradBType==DataType::FLOAT32 || gradBType==DataType::BFLOAT16) );
}

}
}
}
