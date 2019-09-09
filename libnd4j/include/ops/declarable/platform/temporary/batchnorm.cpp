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
        PLATFORM_IMPL(batchnorm) {
                std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
                if (streams.empty()) {
                    streams.push_back(MKLDNNStream("batchnorm_new"));
                }

                std::vector<Nd4jLong> shape({2, mean->lengthOf()});
                NDArray weights = NDArrayFactory::create<float>('c', shape, block.launchContext());
                weights({0, 1, 0, 0}).assign(1.0f);
                weights({1, 2, 0, 0}).assign(0.0f);

                if (streams[0].checkAndReset({input, mean, variance, gamma, beta}, {output}, {(float)epsilon}, axes)) {
                    mkldnn_memory_desc_t empty;
                    mkldnn::memory::desc batchnorm_src_md(empty), batchnorm_dst_md(empty), user_src_md(empty), user_dst_md(empty);

                    getMKLDNNMemoryDescBatchNorm(input, nullptr, output,
                                                 &batchnorm_src_md, nullptr, &batchnorm_dst_md,
                                                 &user_src_md, nullptr, &user_dst_md, axes[0]);

                    auto batchnorm_desc = batch_normalization_forward::desc(prop_kind::forward_inference, batchnorm_src_md, epsilon,
                                                                            use_global_stats | (applyScale || applyOffset ? use_scale_shift : 0));

                    auto engine = streams[0].getEngine();
                    auto batchnorm_prim_desc = batch_normalization_forward::primitive_desc(batchnorm_desc, engine);
                    auto user_src_memory = mkldnn::memory({user_src_md, engine}, input->buffer());
                    auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());
                    auto batchnorm_mean_memory = mkldnn::memory(batchnorm_prim_desc.mean_primitive_desc(), mean->buffer());
                    auto batchnorm_variance_memory = mkldnn::memory(batchnorm_prim_desc.variance_primitive_desc(), variance->buffer());

                    auto batchnorm_src_memory = user_src_memory;
                    streams[0].addMemory(user_src_memory);
                    if (mkldnn::memory::primitive_desc({batchnorm_src_md, engine})
                        != user_src_memory.get_primitive_desc()) {
                        batchnorm_src_memory = mkldnn::memory({batchnorm_src_md, engine});
                        streams[0].addMemory(batchnorm_src_memory);
                        streams[0].addOperation(reorder(user_src_memory, batchnorm_src_memory));
                    }

                    auto batchnorm_dst_memory = user_dst_memory;
                    streams[0].addMemory(user_dst_memory);
                    if (mkldnn::memory::primitive_desc(batchnorm_prim_desc.dst_primitive_desc())
                        != user_dst_memory.get_primitive_desc()) {
                        batchnorm_dst_memory = mkldnn::memory(batchnorm_prim_desc.dst_primitive_desc());
                        streams[0].addMemory(batchnorm_dst_memory);
                    }

                    streams[0].addMemory(batchnorm_mean_memory);
                    streams[0].addMemory(batchnorm_variance_memory);

                    if (applyScale || applyOffset) {
                        auto batchnorm_weights_memory = mkldnn::memory(batchnorm_prim_desc.weights_primitive_desc(), weights.buffer());
                        streams[0].addMemory(batchnorm_weights_memory);
                        streams[0].addOperation(batch_normalization_forward(batchnorm_prim_desc, (mkldnn::primitive::at)batchnorm_src_memory,
                                                                            (mkldnn::primitive::at)batchnorm_mean_memory, (mkldnn::primitive::at)batchnorm_variance_memory, (mkldnn::primitive::at)batchnorm_weights_memory, batchnorm_dst_memory));
                    } else {
                        streams[0].addOperation(batch_normalization_forward(batchnorm_prim_desc, (mkldnn::primitive::at)batchnorm_src_memory,
                                                                            (mkldnn::primitive::at)batchnorm_mean_memory, (mkldnn::primitive::at)batchnorm_variance_memory, batchnorm_dst_memory));
                    }

                    if (mkldnn::memory::primitive_desc(batchnorm_prim_desc.dst_primitive_desc())
                        != user_dst_memory.get_primitive_desc()) {
                        streams[0].addOperation(reorder(batchnorm_dst_memory, user_dst_memory));
                    }
                }

                if (applyScale || applyOffset) {
                    if (gamma != nullptr) {
                        weights({0, 1, 0, 0}).assign(gamma);
                    }
                    if (beta != nullptr) {
                        weights({1, 2, 0, 0}).assign(beta);
                    }
                }
                streams[0].submitAndWait();
                return Status::OK();
        }

        PLATFORM_CHECK(batchnorm) {
            return block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, mean, variance, gamma, beta, output}) && numOfAxes == 1;
        }
    }
}