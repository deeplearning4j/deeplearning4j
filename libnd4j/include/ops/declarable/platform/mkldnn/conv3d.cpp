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

    std::string expectedWeightsShape = ShapeUtils::shapeAsString({kD, kH, kW, iC, oC});
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM CONV3D MKLDNN OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D MKLDNN OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    if (paddingMode)                       // SAME
        ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);

    conv3dMKLDNN(block, input, weights, bias, output, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode, isNCDHW);

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

    std::string expectedGradOShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx( {bS, oC, trueoD, trueoH, trueoW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2}));
    std::string expectedWeightsShape = ShapeUtils::shapeAsString({kD, kH, kW, iC, oC});
    REQUIRE_TRUE(expectedGradOShape == ShapeUtils::shapeAsString(gradO), 0, "CUSTOM CONV3D_BP OP: wrong shape of output gradients (next epsilon) array, expected is %s, but got %s instead !", expectedGradOShape.c_str(), ShapeUtils::shapeAsString(gradO).c_str());
    REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0, "CUSTOM CONV3D_BP OP: wrong shape of weights array, expected is %s, but got %s instead !", expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
    if (bias)
        REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0, "CUSTOM CONV3D_BP OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !", oC, bias->rankOf(), bias->lengthOf());

    conv3dBpMKLDNN(block, input, weights, bias, gradO, gradI, gradW, gradB, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, paddingMode, isNCDHW);

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