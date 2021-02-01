/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <system/platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

using namespace dnnl;

namespace sd {
    namespace ops {
        namespace platforms {
            PLATFORM_IMPL(lrn, ENGINE_CPU) {
                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);

                REQUIRE_TRUE(input->rankOf() == 4, 0, "lrn: Input rank of 4 expected, but got %i instead",
                             input->rankOf());

                double alpha = T_ARG(1);
                double beta = T_ARG(2);
                double bias = T_ARG(0);
                int depth = INT_ARG(0);

                dnnl_memory_desc_t empty;
                dnnl::memory::desc lrn_src_md(empty), lrn_dst_md(empty), user_src_md(empty), user_dst_md(empty);

                mkldnnUtils::getMKLDNNMemoryDescLrn(input, nullptr, output, &lrn_src_md, nullptr, &lrn_dst_md,
                                                        &user_src_md, nullptr, &user_dst_md, input->rankOf() - 1);

                auto lrn_desc = lrn_forward::desc(prop_kind::forward_inference, algorithm::lrn_across_channels,
                                                      lrn_src_md, (2 * depth + 1), alpha * (2 * depth + 1), beta, bias);

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
                dnnl::stream stream(engine);
                auto lrn_prim_desc = lrn_forward::primitive_desc(lrn_desc, engine);
                auto user_src_memory = dnnl::memory(user_src_md, engine, input->buffer());
                auto user_dst_memory = dnnl::memory(user_dst_md, engine, output->buffer());

                auto lrn_src_memory = user_src_memory;
                if (lrn_prim_desc.src_desc() != user_src_memory.get_desc()) {
                    lrn_src_memory = dnnl::memory(lrn_prim_desc.src_desc(), engine);
                    reorder(user_src_memory, lrn_src_memory).execute(stream, user_src_memory, lrn_src_memory);
                }

                auto lrn_dst_memory = user_dst_memory;
                if (lrn_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                    lrn_dst_memory = dnnl::memory(lrn_prim_desc.dst_desc(), engine);
                }

                lrn_forward(lrn_prim_desc).execute(stream, {{DNNL_ARG_SRC, lrn_src_memory},
                                                                {DNNL_ARG_DST, lrn_dst_memory}});

                if (lrn_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
                    reorder(lrn_dst_memory, user_dst_memory).execute(stream, lrn_dst_memory, user_dst_memory);
                }

                stream.wait();

                return Status::OK();
            };

            PLATFORM_CHECK(lrn, ENGINE_CPU) {
                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);

                return block.isUseMKLDNN() && sd::MKLDNNStream::isSupported({input, output});
            }
        }
    }
}