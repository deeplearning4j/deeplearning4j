/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
 //  @author Oleg Semeniv <oleg.semeniv@gmail.com>
 //
 //

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"

using namespace dnnl;

namespace sd {
    namespace ops {
        namespace platforms {

            //////////////////////////////////////////////////////////////////////
            static void tanhMKLDNN(const NDArray* x, NDArray* z) {

                const auto xRank = x->rankOf();

                std::vector<int64_t> dimsX(xRank), dimsZ(xRank);
                for (auto i = 0; i < xRank; i++) {
                    dimsX[i] = x->sizeAt(i);
                    dimsZ[i] = z->sizeAt(i);
                }

                dnnl::memory::dims xShape = dnnl::memory::dims(dimsX);
                dnnl::memory::dims zShape = dnnl::memory::dims(dimsZ);

                dnnl::memory::format_tag format = dnnl::memory::format_tag::a;
                if (2 == xRank) {
                    format = dnnl::memory::format_tag::ab;
                }
                else if (3 == xRank) {
                    format = dnnl::memory::format_tag::abc;
                }
                else if (4 == xRank) {
                    format = dnnl::memory::format_tag::abcd;
                }
                else if (5 == xRank) {
                    format = dnnl::memory::format_tag::abcde;
                }
                else if (6 == xRank) {
                    format = dnnl::memory::format_tag::abcdef;
                }

                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);

                if (x->ews() != 1 || x->ordering() != 'c') {
                    x_user_md.data.format_kind = dnnl_blocked;    // overrides format
                    for (auto i = 0; i < xRank; ++i) {
                        x_user_md.data.format_desc.blocking.strides[i] = x->strideAt(i);
                    }
                }

                // z
                dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc z_user_md = dnnl::memory::desc(zShape, dnnl::memory::data_type::f32, format);
                if (z->ews() != 1 || z->ordering() != 'c') {
                    z_user_md.data.format_kind = dnnl_blocked;    // overrides format
                    for (auto i = 0; i < xRank; ++i) {
                        z_user_md.data.format_desc.blocking.strides[i] = z->strideAt(i);
                    }
                }

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

                // Create attributes (to handle alpha and beta if necessary)
                dnnl::primitive_attr attr; // it is empty since we have usual values for alpha (=1) and beta (=0)

                // operation primitive description
                dnnl::eltwise_forward::desc op_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_tanh, x_mkl_md, 0, 0);

                dnnl::eltwise_forward::primitive_desc op_prim_desc(op_desc, attr, engine);

                // arguments (memory buffers) necessary for calculations
                std::unordered_map<int, dnnl::memory> args;

                dnnl::stream stream(engine);

                // provide memory buffers and check whether reorder is required
                // input
                auto x_user_mem = dnnl::memory(x_user_md, engine, x->getBuffer());
                const bool xReorder = op_prim_desc.src_desc() != x_user_mem.get_desc();
                auto x_mkl_mem = xReorder ? dnnl::memory(op_prim_desc.src_desc(), engine) : x_user_mem;
                if (xReorder)
                    dnnl::reorder(x_user_mem, x_mkl_mem).execute(stream, x_user_mem, x_mkl_mem);
                args[DNNL_ARG_SRC] = x_mkl_mem;

                // z
                auto z_user_mem = dnnl::memory(z_user_md, engine, z->getBuffer());
                const bool zReorder = op_prim_desc.dst_desc() != z_user_mem.get_desc();
                auto z_mkl_mem = zReorder ? dnnl::memory(op_prim_desc.dst_desc(), engine) : z_user_mem;
                args[DNNL_ARG_DST] = z_mkl_mem;

                // run calculations
                dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

                // reorder outputs if necessary
                if (zReorder)
                    dnnl::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

                stream.wait();
            }


            PLATFORM_IMPL(tanh, ENGINE_CPU) {

                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);
                const int rank = input->rankOf();
                REQUIRE_TRUE(rank <= 6, 0, "TANH_MKLDNN OP: the rank of input must be less or qual 6, but got rank = %i instead !", rank);

                // mkldnnTanh
                tanhMKLDNN(input, output);

                return Status::OK();
            }

            PLATFORM_CHECK(tanh, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto z = OUTPUT_VARIABLE(0);

                const DataType xType = x->dataType();
                const DataType zType = z->dataType();

                const int xRank = x->rankOf();
                bool bSupportedRanks = xRank < 7;
                /*
                Source     Destination
                f32 	    f32
                */
                return !x->isEmpty() && block.isUseMKLDNN() && bSupportedRanks && (xType == DataType::FLOAT32 && zType == DataType::FLOAT32);
            }

        }
    }
}
