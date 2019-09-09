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
        PLATFORM_IMPL(lrn) {
                std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
                if (streams.empty()) {
                    streams.push_back(MKLDNNStream("lrn"));
                }

                if (streams[0].checkAndReset({input}, {output}, {(float)bias, (float)alpha, (float)beta}, {depth})) {
                    mkldnn_memory_desc_t empty;
                    mkldnn::memory::desc lrn_src_md(empty), lrn_dst_md(empty), user_src_md(empty), user_dst_md(empty);

                    getMKLDNNMemoryDescLrn(input, nullptr, output, &lrn_src_md, nullptr, &lrn_dst_md, &user_src_md, nullptr, &user_dst_md, input->rankOf() - 1);

                    auto lrn_desc = lrn_forward::desc(prop_kind::forward_inference, lrn_across_channels, lrn_src_md, (2 * depth + 1), alpha * (2 * depth + 1), beta, bias);

                    auto engine = streams[0].getEngine();
                    auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, engine);
                    auto user_src_memory = mkldnn::memory({user_src_md, engine}, input->buffer());
                    auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());

                    auto lrn_src_memory = user_src_memory;
                    streams[0].addMemory(user_src_memory);
                    if (mkldnn::memory::primitive_desc(lrn_prim_desc.src_primitive_desc())
                        != user_src_memory.get_primitive_desc()) {
                        lrn_src_memory = mkldnn::memory(lrn_prim_desc.src_primitive_desc());
                        streams[0].addMemory(lrn_src_memory);
                        streams[0].addOperation(reorder(user_src_memory, lrn_src_memory));
                    }

                    auto lrn_dst_memory = user_dst_memory;
                    streams[0].addMemory(user_dst_memory);
                    if (mkldnn::memory::primitive_desc(lrn_prim_desc.dst_primitive_desc())
                        != user_dst_memory.get_primitive_desc()) {
                        lrn_dst_memory = mkldnn::memory(lrn_prim_desc.dst_primitive_desc());
                        streams[0].addMemory(lrn_dst_memory);
                    }

                    streams[0].addOperation(lrn_forward(lrn_prim_desc, lrn_src_memory, lrn_dst_memory));

                    if (mkldnn::memory::primitive_desc(lrn_prim_desc.dst_primitive_desc())
                        != user_dst_memory.get_primitive_desc()) {
                        streams[0].addOperation(reorder(lrn_dst_memory, user_dst_memory));
                    }
                }

                streams[0].submitAndWait();

            return Status::OK();
        };

        PLATFORM_CHECK(lrn) {
            return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, output});
        }
    }
}