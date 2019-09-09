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
        PLATFORM_IMPL(maxpool2d) {
            auto input = INPUT_VARIABLE(0);

            REQUIRE_TRUE(input->rankOf() == 4, 0, "Input should have rank of 4, but got %i instead", input->rankOf());

            // 0,1 - kernel Height/Width; 2,3 - stride Height/Width; 4,5 - pad Height/Width; 6,7 - dilation Height/Width; 8 - same mode;
            auto argI = *(block.getIArguments());
            auto output = OUTPUT_VARIABLE(0);

            const auto kH = INT_ARG(0);
            const auto kW = INT_ARG(1);
            const auto sH = INT_ARG(2);
            const auto sW = INT_ARG(3);
            int pH = INT_ARG(4);
            int pW = INT_ARG(5);
            const auto dH = INT_ARG(6);
            const auto dW = INT_ARG(7);
            const auto isSameMode = static_cast<bool>(INT_ARG(8));

            REQUIRE_TRUE(dH != 0 && dW != 0, 0, "AVGPOOL2D op: dilation must not be zero, but got instead {%i, %i}", dH, dW);

            int oH = 0;
            int oW = 0;

            int isNCHW  = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;       // INT_ARG(10): 0-NCHW, 1-NHWC

            const int iH = static_cast<int>(isNCHW ? input->sizeAt(2) : input->sizeAt(1));
            const int iW = static_cast<int>(isNCHW ? input->sizeAt(3) : input->sizeAt(2));

            if(!isNCHW) {
                input  = new NDArray(input->permute({0, 3, 1, 2}));                // [bS, iH, iW, iC] -> [bS, iC, iH, iW]
                output = new NDArray(output->permute({0, 3, 1, 2}));               // [bS, oH, oW, iC] -> [bS, iC, oH, oW]
            }

            ConvolutionUtils::calcOutSizePool2D(oH, oW, kH, kW, sH, sW, pH, pW, dH, dW, iH, iW, isSameMode);

            if (isSameMode)
                ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);

            const int bS = input->sizeAt(0);
            const int iC = input->sizeAt(1);
            const int oC = output->sizeAt(1);

            std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
            if (streams.empty()) {
                streams.emplace_back(MKLDNNStream("pooling2d"));
            }


            auto poolingMode = PoolingType::MAX_POOL;
            int extraParam0 = 1;

            if (streams[0].checkAndReset({input}, {output}, {}, {kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0})) {
                mkldnn_memory_desc_t empty;
                mkldnn::memory::desc pool_src_md(empty), pool_dst_md(empty);
                mkldnn::memory::desc user_src_md(empty), user_dst_md(empty);
                mkldnn::memory::dims pool_strides, pool_kernel, pool_padding, pool_padding_r;
                mkldnn::algorithm algorithm;

                mkldnnUtils::getMKLDNNMemoryDescPool2d(kH, kW, sH, sW, pH, pW, dH, dW, poolingMode, extraParam0, true,
                                                       bS, iC, iH, iW, oC, oH, oW, input, nullptr, output, algorithm,
                                                       &pool_src_md, nullptr, &pool_dst_md, &user_src_md, nullptr, &user_dst_md,
                                                       pool_strides, pool_kernel, pool_padding, pool_padding_r);

                auto pool_desc = pooling_forward::desc(prop_kind::forward_inference, algorithm, pool_src_md, pool_dst_md,
                                                       pool_strides, pool_kernel, pool_padding, pool_padding_r);

                auto engine = streams[0].getEngine();
                auto pool_prim_desc = pooling_forward::primitive_desc(pool_desc, engine);
                auto user_src_memory = mkldnn::memory(user_src_md, engine, input->buffer());
                auto user_dst_memory = mkldnn::memory(user_dst_md, engine, output->buffer());

                auto pool_src_memory = user_src_memory;
                mkldnn::stream stream(engine);
                if (pool_prim_desc.src_desc() != user_src_memory.get_desc()) {
                    pool_src_memory = mkldnn::memory(pool_prim_desc.src_desc(), engine);
                    reorder(user_src_memory, pool_src_memory).execute(stream, user_src_memory, pool_src_memory);
                }

                auto pool_dst_memory = user_dst_memory;
                if (pool_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                    pool_dst_memory = mkldnn::memory(pool_prim_desc.dst_desc(), engine);
                }

                pooling_forward(pool_prim_desc).execute(stream, {{MKLDNN_ARG_SRC, pool_src_memory}, {MKLDNN_ARG_DST, pool_dst_memory}});

                if (pool_prim_desc.dst_desc()  != user_dst_memory.get_desc()) {
                    reorder(pool_dst_memory, user_dst_memory).execute(stream, pool_dst_memory, user_dst_memory);
                }

                stream.wait();
            }

            if(!isNCHW) {
                delete input;
                delete output;
            }

            return Status::OK();
        }

        PLATFORM_CHECK(maxpool2d) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output});
        }
    }
}