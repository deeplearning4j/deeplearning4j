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
#include <NDArrayFactory.h>

using namespace mkldnn;

namespace nd4j {
    namespace ops {
        namespace platforms {
            PLATFORM_IMPL(batchnorm_new) {
                auto input = INPUT_VARIABLE(0);
                auto mean = INPUT_VARIABLE(1);
                auto variance = INPUT_VARIABLE(2);
                NDArray *gamma = nullptr;
                NDArray *beta = nullptr;

                auto output = OUTPUT_VARIABLE(0);

                const bool applyScale = (bool) INT_ARG(0);
                const bool applyOffset = (bool) INT_ARG(1);
                const double epsilon = T_ARG(0);

                if (applyScale)
                    gamma = INPUT_VARIABLE(3);
                if (applyOffset)
                    beta = INPUT_VARIABLE(3 + static_cast<int>(applyScale));

                std::vector<int> axes;
                if (block.numI() > 2)
                    for (int i = 2; i < block.numI(); ++i)
                        axes.push_back(INT_ARG(i));
                else
                    axes.push_back(input->rankOf() - 1);

                std::vector<Nd4jLong> shape({2, mean->lengthOf()});
                NDArray weights = NDArrayFactory::create<float>('c', shape, block.launchContext());
                weights({0, 1, 0, 0}).assign(1.0f);
                weights({1, 2, 0, 0}).assign(0.0f);

                mkldnn_memory_desc_t empty;
                mkldnn::memory::desc batchnorm_src_md(empty), batchnorm_dst_md(empty), user_src_md(
                        empty), user_dst_md(empty);

                auto norm_flag = normalization_flags::use_global_stats;
                if (applyScale || applyOffset)
                    norm_flag |= normalization_flags::use_scale_shift;

                mkldnnUtils::getMKLDNNMemoryDescBatchNorm(input, nullptr, output,
                                                          &batchnorm_src_md, nullptr, &batchnorm_dst_md,
                                                          &user_src_md, nullptr, &user_dst_md, axes[0]);

                auto batchnorm_desc = batch_normalization_forward::desc(prop_kind::forward_inference, batchnorm_src_md, epsilon, norm_flag);

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
                mkldnn::stream stream(engine);
                auto batchnorm_prim_desc = batch_normalization_forward::primitive_desc(batchnorm_desc, engine);
                auto user_src_memory = mkldnn::memory(user_src_md, engine, input->buffer());
                auto user_dst_memory = mkldnn::memory(user_dst_md, engine, output->buffer());
                auto batchnorm_mean_memory = mkldnn::memory(batchnorm_prim_desc.mean_desc(), engine,
                                                            mean->buffer());
                auto batchnorm_variance_memory = mkldnn::memory(batchnorm_prim_desc.variance_desc(), engine,
                                                                variance->buffer());
                auto batchnorm_src_memory = user_src_memory;
                mkldnn::memory m(batchnorm_src_md, engine);
                if (m.get_desc() != user_src_memory.get_desc()) {
                    batchnorm_src_memory = mkldnn::memory(batchnorm_src_md, engine);
                    reorder(user_src_memory, batchnorm_src_memory).execute(stream, user_src_memory,
                                                                           batchnorm_src_memory);
                }
                auto batchnorm_dst_memory = user_dst_memory;
                if (batchnorm_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                    batchnorm_dst_memory = mkldnn::memory(batchnorm_prim_desc.dst_desc(), engine);
                }
                if (applyScale || applyOffset) {
                    if (gamma != nullptr) {
                        weights({0, 1, 0, 0}).assign(gamma);
                    }
                    if (beta != nullptr) {
                        weights({1, 2, 0, 0}).assign(beta);
                    }

                    auto batchnorm_weights_memory = mkldnn::memory(batchnorm_prim_desc.weights_desc(), engine, weights.buffer());
                    batch_normalization_forward(batchnorm_prim_desc).execute(stream,
                                                                             {{MKLDNN_ARG_SRC,      batchnorm_src_memory},
                                                                              {MKLDNN_ARG_MEAN,     batchnorm_mean_memory},
                                                                              {MKLDNN_ARG_VARIANCE, batchnorm_variance_memory},
                                                                              {MKLDNN_ARG_WEIGHTS,  batchnorm_weights_memory},
                                                                              {MKLDNN_ARG_DST,      batchnorm_dst_memory}});
                } else {
                    batch_normalization_forward(batchnorm_prim_desc).execute(stream,
                                                                             {{MKLDNN_ARG_SRC,      batchnorm_src_memory},
                                                                              {MKLDNN_ARG_MEAN,     batchnorm_mean_memory},
                                                                              {MKLDNN_ARG_VARIANCE, batchnorm_variance_memory},
                                                                              {MKLDNN_ARG_DST,      batchnorm_dst_memory}});
                }
                if (batchnorm_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                    reorder(batchnorm_dst_memory, user_dst_memory).execute(stream, batchnorm_dst_memory,
                                                                           user_dst_memory);
                }
                stream.wait();

                return Status::OK();
            }

            PLATFORM_CHECK(batchnorm_new) {
                auto input = INPUT_VARIABLE(0);
                auto mean = INPUT_VARIABLE(1);
                auto variance = INPUT_VARIABLE(2);
                NDArray *gamma = nullptr;
                NDArray *beta = nullptr;

                auto output = OUTPUT_VARIABLE(0);

                const bool applyScale = (bool) INT_ARG(0);
                const bool applyOffset = (bool) INT_ARG(1);
                const double epsilon = T_ARG(0);

                if (applyScale)
                    gamma = INPUT_VARIABLE(3);
                if (applyOffset)
                    beta = INPUT_VARIABLE(3 + static_cast<int>(applyScale));

                std::vector<int> axes;
                if (block.numI() > 2)
                    for (int i = 2; i < block.numI(); ++i)
                        axes.push_back(INT_ARG(i));
                else
                    axes.push_back(input->rankOf() - 1);

                return block.isUseMKLDNN() &&
                       nd4j::MKLDNNStream::isSupported({input, mean, variance, gamma, beta, output}) &&
                       axes.size() == 1;
            }
        }
    }
}