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
        namespace platforms {
            PLATFORM_IMPL(conv3dnew) {
                auto input = INPUT_VARIABLE(
                        0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
                auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC] always
                auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
                auto output = OUTPUT_VARIABLE(
                        0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

                REQUIRE_TRUE(input->rankOf() == 5, 0,
                             "CUSTOM CONV3D OP: rank of input array must be equal to 5, but got %i instead !",
                             input->rankOf());
                REQUIRE_TRUE(weights->rankOf() == 5, 0,
                             "CUSTOM CONV3D OP: rank of weights array must be equal to 5, but got %i instead !",
                             weights->rankOf());

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
                int isNCDHW =
                        block.getIArguments()->size() > 13 ? !INT_ARG(13) : 1;       // INT_ARG(13): 1-NDHWC, 0-NCDHW

                int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
                int indIOioC, indIOioD, indWoC, indWiC, indWkD;             // corresponding indexes
                ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                                           indIOioC, indIOioD, indWiC, indWoC, indWkD);

                std::string expectedWeightsShape = ShapeUtils::shapeAsString({kD, kH, kW, iC, oC});
                REQUIRE_TRUE(expectedWeightsShape == ShapeUtils::shapeAsString(weights), 0,
                             "CUSTOM CONV3D OP: wrong shape of weights array, expected is %s, but got %s instead !",
                             expectedWeightsShape.c_str(), ShapeUtils::shapeAsString(weights).c_str());
                if (bias)
                    REQUIRE_TRUE(bias->rankOf() <= 2 && oC == bias->lengthOf(), 0,
                                 "CUSTOM CONV3D OP: wrong shape of array with biases, expected rank, length: <=2, %i, but got %i, %i instead !",
                                 oC, bias->rankOf(), bias->lengthOf());

                if (isSameMode)                       // SAME
                    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);


                mkldnn_memory_desc_t empty;
                mkldnn::memory::desc conv_src_md(empty), conv_weights_md(empty), conv_bias_md(empty), conv_dst_md(
                        empty);
                mkldnn::memory::desc user_src_md(empty), user_weights_md(empty), user_bias_md(empty), user_dst_md(
                        empty);
                mkldnn::memory::dims conv_strides, conv_padding, conv_padding_r;
                mkldnnUtils::getMKLDNNMemoryDescConv3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, isSameMode,
                                                       isNCDHW,
                                                       bS, iC, iD, iH, iW, oC, oD, oH, oW, input, nullptr, weights,
                                                       nullptr, bias, output,
                                                       &conv_src_md, nullptr, &conv_weights_md, nullptr,
                                                       &conv_bias_md, &conv_dst_md,
                                                       &user_src_md, nullptr, &user_weights_md, nullptr,
                                                       &user_bias_md, &user_dst_md,
                                                       conv_strides, conv_padding, conv_padding_r);
                auto conv_desc = bias != nullptr
                                 ? convolution_forward::desc(prop_kind::forward,
                                                             algorithm::convolution_direct, conv_src_md,
                                                             conv_weights_md, conv_bias_md,
                                                             conv_dst_md, conv_strides, conv_padding,
                                                             conv_padding_r)
                                 : convolution_forward::desc(prop_kind::forward,
                                                             algorithm::convolution_direct, conv_src_md,
                                                             conv_weights_md,
                                                             conv_dst_md, conv_strides, conv_padding,
                                                             conv_padding_r);
                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
                mkldnn::stream stream(engine);
                auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, engine);
                auto user_src_memory = mkldnn::memory(user_src_md, engine, const_cast<NDArray *>(input)->buffer());
                auto user_weights_memory = mkldnn::memory(user_weights_md, engine,
                                                          const_cast<NDArray *>(weights)->buffer());
                auto user_dst_memory = mkldnn::memory(user_dst_md, engine, output->buffer());
                auto conv_src_memory = user_src_memory;
                if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
                    conv_src_memory = mkldnn::memory(conv_prim_desc.src_desc(), engine);
                    reorder(user_src_memory, conv_src_memory).execute(stream, user_src_memory, conv_src_memory);
                }
                auto conv_weights_memory = user_weights_memory;
                if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
                    conv_weights_memory = mkldnn::memory(conv_prim_desc.weights_desc(), engine);
                    reorder(user_weights_memory, conv_weights_memory).execute(stream, user_weights_memory,
                                                                              conv_weights_memory);
                }
                auto conv_dst_memory = user_dst_memory;
                if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                    conv_dst_memory = mkldnn::memory(conv_prim_desc.dst_desc(), engine);
                }
                if (bias != nullptr) {
                    auto conv_bias_memory = mkldnn::memory(conv_prim_desc.bias_desc(), engine, bias->buffer());
                    convolution_forward(conv_prim_desc).execute(stream, {{MKLDNN_ARG_SRC,     conv_src_memory},
                                                                         {MKLDNN_ARG_WEIGHTS, conv_weights_memory},
                                                                         {MKLDNN_ARG_BIAS,    conv_bias_memory},
                                                                         {MKLDNN_ARG_DST,     conv_dst_memory}});
                } else {
                    convolution_forward(conv_prim_desc).execute(stream, {{MKLDNN_ARG_SRC,     conv_src_memory},
                                                                         {MKLDNN_ARG_WEIGHTS, conv_weights_memory},
                                                                         {MKLDNN_ARG_DST,     conv_dst_memory}});
                }
                if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                    reorder(conv_dst_memory, user_dst_memory).execute(stream, conv_dst_memory, user_dst_memory);
                }
                stream.wait();

                return Status::OK();
            }

            PLATFORM_CHECK(conv3dnew) {
                auto input = INPUT_VARIABLE(
                        0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
                auto weights = INPUT_VARIABLE(1);                                    // [kD, kH, kW, iC, oC] always
                auto bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;      // [oC]
                auto output = OUTPUT_VARIABLE(
                        0);                                   // [bS, oD, oH, oW, oC] (NDHWC) or [bS, oC, oD, oH, oW] (NCDHW)

                return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, weights, bias, output});
            }
        }
    }
}