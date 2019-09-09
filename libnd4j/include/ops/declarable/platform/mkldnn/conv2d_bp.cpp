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

using namespace mkldnn;

namespace nd4j {
    namespace ops {
        PLATFORM_IMPL(conv2d_bp) {
            auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC] always
            auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
            auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

            auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
            auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, oC] always
            auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]

            int kH = INT_ARG(0);                                                        // filter(kernel) height
            int kW = INT_ARG(1);                                                        // filter(kernel) width
            int sH = INT_ARG(2);                                                        // strides height
            int sW = INT_ARG(3);                                                        // strides width
            int pH = INT_ARG(4);                                                        // paddings height
            int pW = INT_ARG(5);                                                        // paddings width
            int dH = INT_ARG(6);                                                        // dilations height
            int dW = INT_ARG(7);                                                        // dilations width
            int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
            int isNCHW  = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;          // INT_ARG(9): 0-NCHW, 1-NHWC

            REQUIRE_TRUE(input->rankOf()   == 4, 0, "CUSTOM CONV2D_BP OP: rank of input array must be equal to 4, but got %i instead !", input->rankOf());
            REQUIRE_TRUE(weights->rankOf() == 4, 0, "CUSTOM CONV2D_BP OP: rank of weights array must be equal to 4, but got %i instead !", weights->rankOf());
            REQUIRE_TRUE(gradO->rankOf() == 4, 0, "CUSTOM CONV2D_BP OP: rank of output's gradients (next epsilon) array must be equal to 4, but got %i instead !", gradO->rankOf());

            int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
            int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);

            if(isSameMode)                       // SAME
                ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

            std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
            if (streams.empty()) {
                streams.emplace_back(MKLDNNStream("conv2d_bp_weights"));
                streams.emplace_back(MKLDNNStream("conv2d_bp_data"));
            }

            bool resetW = streams[0].checkAndReset({input, weights, bias, gradO}, {gradI, gradW, gradB}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW});
            bool resetI = streams[1].checkAndReset({input, weights, bias, gradO}, {gradI, gradW, gradB}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW});
            if (resetW || resetI) {
                mkldnn_memory_desc_t empty;
                mkldnn::memory::desc conv_src_md(empty), conv_diff_src_md(empty), conv_weights_md(empty),
                        conv_diff_weights_md(empty), conv_bias_md(empty), conv_dst_md(empty);
                mkldnn::memory::desc user_src_md(empty), user_diff_src_md(empty), user_weights_md(empty),
                        user_diff_weights_md(empty), user_bias_md(empty), user_dst_md(empty);
                mkldnn::memory::dims conv_strides, conv_padding, conv_padding_r;

                mkldnnUtils::getMKLDNNMemoryDescConv2d(kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW,
                                                            bS, iC, iH, iW, oC, oH, oW, input, gradI, weights, gradW, gradB, gradO,
                                                            &conv_src_md, &conv_diff_src_md, &conv_weights_md, &conv_diff_weights_md, &conv_bias_md, &conv_dst_md,
                                                            &user_src_md, &user_diff_src_md, &user_weights_md, &user_diff_weights_md, &user_bias_md, &user_dst_md,
                                                            conv_strides, conv_padding, conv_padding_r);

                auto conv_desc = gradB != nullptr
                                 ? convolution_forward::desc(prop_kind::forward,
                                                             algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                                                             conv_dst_md, conv_strides, conv_padding, conv_padding_r)
                                 : convolution_forward::desc(prop_kind::forward,
                                                             algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                                             conv_dst_md, conv_strides, conv_padding, conv_padding_r);

                auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, streams[0].getEngine());

                if (gradW != nullptr) {
                    auto convW_desc = gradB != nullptr
                                      ? convolution_backward_weights::desc(
                                    algorithm::convolution_direct, conv_src_md, conv_diff_weights_md, conv_bias_md,
                                    conv_dst_md, conv_strides, conv_padding, conv_padding_r)
                                      : convolution_backward_weights::desc(
                                    algorithm::convolution_direct, conv_src_md, conv_diff_weights_md,
                                    conv_dst_md, conv_strides, conv_padding, conv_padding_r);

                    auto engine = streams[0].getEngine();
                    mkldnn::stream stream(engine);
                    auto convW_prim_desc = convolution_backward_weights::primitive_desc(convW_desc, engine, conv_prim_desc);
                    auto userW_src_memory = mkldnn::memory(user_src_md, engine, const_cast<NDArray*>(input)->buffer());
                    auto userW_weights_memory = mkldnn::memory(user_diff_weights_md, engine, gradW->buffer());
                    auto userW_dst_memory = mkldnn::memory(user_dst_md, engine, const_cast<NDArray*>(gradO)->buffer());

                    auto convW_src_memory = userW_src_memory;
                    if (convW_prim_desc.src_desc() != userW_src_memory.get_desc()) {
                        convW_src_memory = mkldnn::memory(convW_prim_desc.src_desc(), engine);
                        reorder(userW_src_memory, convW_src_memory).execute(stream, userW_src_memory, convW_src_memory);
                    }

                    auto convW_weights_memory = userW_weights_memory;
                    if (convW_prim_desc.diff_weights_desc() != userW_weights_memory.get_desc()) {
                        convW_weights_memory = mkldnn::memory(convW_prim_desc.diff_weights_desc(), engine);
                    }

                    auto convW_dst_memory = userW_dst_memory;
                    if (convW_prim_desc.diff_dst_desc() != userW_dst_memory.get_desc()) {
                        convW_dst_memory = mkldnn::memory(convW_prim_desc.diff_dst_desc(), engine);
                        reorder(userW_dst_memory, convW_dst_memory).execute(stream, userW_dst_memory, convW_dst_memory);
                    }

                    if (gradB != nullptr) {
                        auto convW_bias_memory = mkldnn::memory(convW_prim_desc.diff_bias_desc(), engine, gradB->buffer());
                        convolution_backward_weights(convW_prim_desc).execute(stream, {{MKLDNN_ARG_SRC, convW_src_memory}, {MKLDNN_ARG_DIFF_DST, convW_dst_memory}, {MKLDNN_ARG_DIFF_WEIGHTS, convW_weights_memory}, {MKLDNN_ARG_DIFF_BIAS, convW_bias_memory}});
                    } else {
                        convolution_backward_weights(convW_prim_desc).execute(stream, {{MKLDNN_ARG_SRC, convW_src_memory}, {MKLDNN_ARG_DIFF_DST, convW_dst_memory}, {MKLDNN_ARG_DIFF_WEIGHTS, convW_weights_memory}});
                    }

                    if (convW_prim_desc.diff_weights_desc() != userW_weights_memory.get_desc()) {
                        reorder(convW_weights_memory, userW_weights_memory).execute(stream, convW_weights_memory, userW_weights_memory);
                    }

                    stream.wait();
                }

                nd4j_printf("Finished weights step\n", "");

                if (gradI != nullptr) {
                    auto convI_desc =
                            convolution_backward_data::desc(algorithm::convolution_direct, conv_diff_src_md, conv_weights_md, conv_dst_md, conv_strides, conv_padding, conv_padding_r);

                    auto engine = streams[1].getEngine();
                    mkldnn::stream stream(engine);
                    auto convI_prim_desc = convolution_backward_data::primitive_desc(convI_desc, engine, conv_prim_desc);
                    auto userI_src_memory = mkldnn::memory(user_diff_src_md, engine, gradI->buffer());
                    auto userI_weights_memory = mkldnn::memory(user_weights_md, engine, const_cast<NDArray*>(weights)->buffer());
                    auto userI_dst_memory = mkldnn::memory(user_dst_md, engine, const_cast<NDArray*>(gradO)->buffer());

                    auto convI_src_memory = userI_src_memory;
                    if (convI_prim_desc.diff_src_desc() != userI_src_memory.get_desc()) {
                        convI_src_memory = mkldnn::memory(convI_prim_desc.diff_src_desc(), engine);
                    }

                    auto convI_weights_memory = userI_weights_memory;
                    if (convI_prim_desc.weights_desc() != userI_weights_memory.get_desc()) {
                        convI_weights_memory = mkldnn::memory(convI_prim_desc.weights_desc(), engine);
                        reorder(userI_weights_memory, convI_weights_memory).execute(stream, userI_weights_memory, convI_weights_memory);
                    }

                    auto convI_dst_memory = userI_dst_memory;
                    if (convI_prim_desc.diff_dst_desc() != userI_dst_memory.get_desc()) {
                        convI_dst_memory = mkldnn::memory(convI_prim_desc.diff_dst_desc(), engine);
                        reorder(userI_dst_memory, convI_dst_memory).execute(stream, userI_dst_memory, convI_dst_memory);
                    }

                    convolution_backward_data(convI_prim_desc).execute(stream, {{MKLDNN_ARG_DIFF_DST, convI_dst_memory}, {MKLDNN_ARG_WEIGHTS, convI_weights_memory}, {MKLDNN_ARG_DIFF_SRC, convI_src_memory}});

                    if (convI_prim_desc.diff_src_desc() != userI_src_memory.get_desc()) {
                        reorder(convI_src_memory, userI_src_memory).execute(stream, convI_src_memory, userI_src_memory);
                    }

                    stream.wait();
                }
            }

            return Status::OK();
        }

        PLATFORM_CHECK(conv2d_bp) {
            auto input   = INPUT_VARIABLE(0);                                                // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            auto weights = INPUT_VARIABLE(1);                                                // [kH, kW, iC, oC] always
            auto bias    = block.width() > 3 ? INPUT_VARIABLE(2) : nullptr;                  // [oC]
            auto gradO   = block.width() > 3 ? INPUT_VARIABLE(3) : INPUT_VARIABLE(2);        // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW), epsilon_next

            auto gradI = OUTPUT_VARIABLE(0);                                                 // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW), epsilon
            auto gradW = OUTPUT_VARIABLE(1);                                                 // [kH, kW, iC, oC] always
            auto gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;                   // [oC]


            return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, weights, bias, gradO, gradI, gradW, gradB});
        }
    }
}
