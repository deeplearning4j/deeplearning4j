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
        namespace platforms {
            PLATFORM_IMPL(maxpool3dnew) {
                auto input = INPUT_VARIABLE(
                        0);                                    // [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
                auto output = OUTPUT_VARIABLE(
                        0);                                   // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)

                int kD = INT_ARG(0);                                                        // filter(kernel) depth
                int kH = INT_ARG(1);                                                        // filter(kernel) height
                int kW = INT_ARG(2);                                                        // filter(kernel) width
                int sD = INT_ARG(3);                                                        // strides depth
                int sH = INT_ARG(4);                                                        // strides height
                int sW = INT_ARG(5);                                                        // strides width
                int pD = INT_ARG(6);                                                        // paddings depth
                int pH = INT_ARG(7);                                                        // paddings height
                int pW = INT_ARG(8);                                                        // paddings width
                int dD = INT_ARG(9);                                                        // dilations depth
                int dH = INT_ARG(10);                                                       // dilations height
                int dW = INT_ARG(11);                                                       // dilations width
                int isSameMode = INT_ARG(12);                                               // 1-SAME,  0-VALID
                // int extraParam0 = INT_ARG(13);                                           // unnecessary for max case, required only for avg and pnorm cases
                int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;       // 1-NDHWC, 0-NCDHW

                REQUIRE_TRUE(input->rankOf() == 5, 0,
                             "MAXPOOL3DNEW OP: rank of input array must be equal to 5, but got %i instead !",
                             input->rankOf());
                REQUIRE_TRUE(dD != 0 && dH != 0 && dW != 0, 0,
                             "MAXPOOL3DNEW op: dilation must not be zero, but got instead {%i, %i, %i}", dD, dH, dW);

                int bS, iC, iD, iH, iW, oC, oD, oH, oW;                     // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
                int indIOioC, indIOioD, indWoC, indWiC, indWkD;       // corresponding indexes
                ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                                           indIOioC, indIOioD, indWiC, indWoC, indWkD);

                std::string expectedOutputShape = ShapeUtils::shapeAsString(ShapeUtils::composeShapeUsingDimsAndIdx(
                        {bS, iC, oD, oH, oW, 0, indIOioC, indIOioD, indIOioD + 1, indIOioD + 2}));
                REQUIRE_TRUE(expectedOutputShape == ShapeUtils::shapeAsString(output), 0,
                             "MAXPOOL3D op: wrong shape of output array, expected is %s, but got %s instead !",
                             expectedOutputShape.c_str(), ShapeUtils::shapeAsString(output).c_str());
                // REQUIRE_TRUE(iD   >= kD && iH   >= kH && iW   >= kW, 0, "MAXPOOL3D OP: the input depth/height/width must be greater or equal to kernel(filter) depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", iD,iH,iW, kD,kH,kW);
                // REQUIRE_TRUE(kD/2 >= pD && kH/2 >= pH && kW/2 >= pW, 0, "MAXPOOL3D OP: pad depth/height/width must not be greater than half of kernel depth/height/width, but got [%i, %i, %i] and [%i, %i, %i] correspondingly !", pD,pH,pW, kD,kH,kW);

                if (!isNCDHW) {
                    input = new NDArray(
                            input->permute({0, 4, 1, 2, 3}));          // [bS, iD, iH, iW, iC] -> [bS, iC, iD, iH, iW]
                    output = new NDArray(
                            output->permute({0, 4, 1, 2, 3}));         // [bS, oD, oH, oW, iC] -> [bS, iC, oD, oH, oW]
                }

                if (isSameMode)                       // SAME
                    ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH,
                                                    dW);


                std::vector<nd4j::MKLDNNStream> &streams = block.getMKLDNNStreams();
                if (streams.empty()) {
                    streams.push_back(MKLDNNStream("pooling3d"));
                }

                auto poolingMode = PoolingType::MAX_POOL;
                auto extraParam0 = 1;

                if (streams[0].checkAndReset({input}, {output}, {},
                                             {kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode,
                                              extraParam0})) {
                    mkldnn_memory_desc_t empty;
                    mkldnn::memory::desc pool_src_md(empty), pool_dst_md(empty);
                    mkldnn::memory::desc user_src_md(empty), user_dst_md(empty);
                    mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
                    mkldnn::algorithm algorithm;

                    mkldnnUtils::getMKLDNNMemoryDescPool3d(kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, poolingMode,
                                                           extraParam0, true,
                                                           bS, iC, iD, iH, iW, oC, oD, oH, oW, input, nullptr, output,
                                                           algorithm,
                                                           &pool_src_md, nullptr, &pool_dst_md, &user_src_md, nullptr,
                                                           &user_dst_md,
                                                           pool_strides, pool_kernel, pool_padding, pool_padding_r);

                    auto pool_desc = pooling_forward::desc(prop_kind::forward_inference, algorithm, pool_src_md,
                                                           pool_dst_md, pool_strides, pool_kernel, pool_padding,
                                                           pool_padding_r);

                    auto engine = streams[0].getEngine();
                    mkldnn::stream stream(engine);
                    auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);
                    auto user_src_memory = mkldnn::memory(user_src_md, engine, input->buffer());
                    auto user_dst_memory = mkldnn::memory(user_dst_md, engine, output->buffer());

                    auto pool_src_memory = user_src_memory;
                    if (pool_prim_desc.src_desc() != user_src_memory.get_desc()) {
                        pool_src_memory = mkldnn::memory(pool_prim_desc.src_desc(), engine);
                        reorder(user_src_memory, pool_src_memory).execute(stream, user_src_memory, pool_src_memory);
                    }

                    auto pool_dst_memory = user_dst_memory;
                    if (pool_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                        pool_dst_memory = mkldnn::memory(pool_prim_desc.dst_desc(), engine);
                    }

                    pooling_forward(pool_prim_desc).execute(stream, {{MKLDNN_ARG_SRC, pool_src_memory},
                                                                     {MKLDNN_ARG_DST, pool_dst_memory}});

                    if (pool_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                        reorder(pool_dst_memory, user_dst_memory).execute(stream, pool_dst_memory, user_dst_memory);
                    }

                    stream.wait();
                }

                if (!isNCDHW) {
                    delete input;
                    delete output;
                }

                return Status::OK();
            }

            PLATFORM_CHECK(maxpool3dnew) {
                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);

                return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output});
            }
        }
    }
}