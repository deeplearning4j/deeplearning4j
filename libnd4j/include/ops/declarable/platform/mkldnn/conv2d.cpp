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

        static void conv2d_mkldnn(nd4j::graph::Context& block, const NDArray* input, const NDArray* weights, const NDArray* bias, NDArray* output, const int kH, const int kW, const int sH, const int sW, int pH, int pW, const int dH, const int dW, const int isSameMode, const int isNCHW) {
                std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
                if (streams.empty()) {
                    streams.emplace_back(MKLDNNStream("conv2d"));
                }

            int bS, iC, iH, iW, oC, oH, oW;                             // batch size, input channels, input height/width, output channels, output height/width;
            int indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;       // corresponding indexes
            ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH, indWiC, indWoC, indWkH, indOoH);


            if (streams[0].checkAndReset({input, weights, bias}, {output}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW})) {
                    mkldnn_memory_desc_t empty;
                    mkldnn::memory::desc conv_src_md(empty), conv_weights_md(empty), conv_bias_md(empty), conv_dst_md(empty);
                    mkldnn::memory::desc user_src_md(empty), user_weights_md(empty), user_bias_md(empty), user_dst_md(empty);
                    mkldnn::memory::dims conv_strides, conv_padding, conv_padding_r;

                    mkldnnUtils::getMKLDNNMemoryDescConv2d(kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW,
                                                                bS, iC, iH, iW, oC, oH, oW, input, nullptr, weights, nullptr, bias, output,
                                                                &conv_src_md, nullptr, &conv_weights_md, nullptr, &conv_bias_md, &conv_dst_md,
                                                                &user_src_md, nullptr, &user_weights_md, nullptr, &user_bias_md, &user_dst_md,
                                                                conv_strides, conv_padding, conv_padding_r);

                    auto conv_desc = bias != nullptr
                                     ? convolution_forward::desc(prop_kind::forward,
                                                                 algorithm::convolution_direct, conv_src_md, conv_weights_md, conv_bias_md,
                                                                 conv_dst_md, conv_strides, conv_padding, conv_padding_r)
                                     : convolution_forward::desc(prop_kind::forward,
                                                                 algorithm::convolution_direct, conv_src_md, conv_weights_md,
                                                                 conv_dst_md, conv_strides, conv_padding, conv_padding_r);

                    auto engine = streams[0].getEngine();
                    mkldnn::stream stream(engine);
                    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, engine);
                    auto user_src_memory = mkldnn::memory(user_src_md, engine, const_cast<NDArray*>(input)->buffer());
                    auto user_weights_memory = mkldnn::memory(user_weights_md, engine, const_cast<NDArray*>(weights)->buffer());
                    auto user_dst_memory = mkldnn::memory(user_dst_md, engine, output->buffer());

                    auto conv_src_memory = user_src_memory;
                    if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
                        conv_src_memory = mkldnn::memory(conv_prim_desc.src_desc(), engine);
                        reorder(user_src_memory, conv_src_memory).execute(stream, user_src_memory, conv_src_memory);
                    }

                    auto conv_weights_memory = user_weights_memory;
                    if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
                        conv_weights_memory = mkldnn::memory(conv_prim_desc.weights_desc(), engine);
                        reorder(user_weights_memory, conv_weights_memory).execute(stream, user_weights_memory, conv_weights_memory);
                    }

                    auto conv_dst_memory = user_dst_memory;
                    if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                        conv_dst_memory = mkldnn::memory(conv_prim_desc.dst_desc(), engine);
                    }

                    if (bias != nullptr) {
                        auto conv_bias_memory = mkldnn::memory(conv_prim_desc.bias_desc(), engine, const_cast<NDArray*>(bias)->buffer());
                        convolution_forward(conv_prim_desc).execute(stream, {{MKLDNN_ARG_SRC, conv_src_memory}, {MKLDNN_ARG_WEIGHTS, conv_weights_memory}, {MKLDNN_ARG_BIAS, conv_bias_memory}, {MKLDNN_ARG_DST, conv_dst_memory}});
                    } else {
                        convolution_forward(conv_prim_desc).execute(stream, {{MKLDNN_ARG_SRC, conv_src_memory}, {MKLDNN_ARG_WEIGHTS, conv_weights_memory}, {MKLDNN_ARG_DST, conv_dst_memory}});
                    }

                    if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                        reorder(conv_dst_memory, user_dst_memory).execute(stream, conv_dst_memory, user_dst_memory);
                    }

                    stream.wait();
                }
        }

        PLATFORM_IMPL(conv2d) {
            auto input   = INPUT_VARIABLE(0);                                    // [bS, iH, iW, iC] (NHWC) or [bS, iC, iH, iW] (NCHW)
            auto weights = INPUT_VARIABLE(1);                                    // [kH, kW, iC, oC] always
            auto bias    = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]

            auto output  = OUTPUT_VARIABLE(0);                                   // [bS, oH, oW, oC] (NHWC) or [bS, oC, oH, oW] (NCHW)

            int sH = INT_ARG(2);                                                        // strides height
            int sW = INT_ARG(3);                                                        // strides width
            int pH = INT_ARG(4);                                                        // paddings height
            int pW = INT_ARG(5);                                                        // paddings width
            int dH = INT_ARG(6);                                                        // dilations height
            int dW = INT_ARG(7);                                                        // dilations width
            int isSameMode = INT_ARG(8);                                                // 0-VALID, 1-SAME
            bool isNCHW    = block.getIArguments()->size() > 9 ? !INT_ARG(9) : 1;       // INT_ARG(9): 0-NCHW,  1-NHWC

            int kH = INT_ARG(0) > 0 ? INT_ARG(0) : static_cast<int>(weights->sizeAt(0)); // filter(kernel) height
            int kW = INT_ARG(1) > 0 ? INT_ARG(1) : static_cast<int>(weights->sizeAt(1)); // filter(kernel) width

            conv2d_mkldnn(block, input, weights, bias, output, kH, kW, sH, sW, pH, pW, dH, dW, isSameMode, isNCHW);

            return Status::OK();
        }

        PLATFORM_CHECK(conv2d) {
            auto input = INPUT_VARIABLE(0);
            auto weights = INPUT_VARIABLE(1);

            // conv2d is only available for float32 dtype
            return block.isUseMKLDNN() && input->dataType() == nd4j::DataType::FLOAT32 && weights->dataType() == nd4j::DataType::FLOAT32;
        }
    }
}
