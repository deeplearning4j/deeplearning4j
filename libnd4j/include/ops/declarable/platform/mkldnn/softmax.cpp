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
            static void softmaxMKLDNN(const NDArray* x, NDArray* z, const int axis) {

                const auto xRank = x->rankOf();
                const auto zRank = z->rankOf();

                std::vector<int64_t> dimsX(xRank), dimsZ(zRank);
                for (auto i = 0; i < xRank; i++) {
                    dimsX[i] = x->sizeAt(i);
                    dimsZ[i] = z->sizeAt(i);
                }

                dnnl::memory::dims xShape = dnnl::memory::dims(dimsX);
                dnnl::memory::dims zShape = dnnl::memory::dims(dimsZ);

                dnnl::memory::format_tag format = dnnl::memory::format_tag::a; // 1 == xRank
                if (2 == xRank && 1 == axis) {
                    format = dnnl::memory::format_tag::ab;
                }
                else if (2 == xRank && 0 == axis) {
                    format = dnnl::memory::format_tag::ba;
                }
                else if (3 == xRank) {
                    format = dnnl::memory::format_tag::abc;
                }
                else if (4 == xRank && 3 == axis) {
                    format = dnnl::memory::format_tag::abcd;
                }
                else if (4 == xRank && 1 == axis && dimsX[2] * dimsX[3] > 1) {
                    format = dnnl::memory::format_tag::acdb;
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

                dnnl::memory::data_type xType = dnnl::memory::data_type::f32;
                dnnl::memory::data_type zType = dnnl::memory::data_type::f32;

                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, xType, format);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, xType, format);

                if (x->ews() != 1 || x->ordering() != 'c') {
                    x_user_md.data.format_kind = dnnl_blocked;    // overrides format
                    for (auto i = 0; i < xRank; ++i) {
                        x_user_md.data.format_desc.blocking.strides[i] = x->strideAt(i);
                    }
                }

                // z
                dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zShape, zType, dnnl::memory::format_tag::any);
                dnnl::memory::desc z_user_md = dnnl::memory::desc(zShape, zType, format);
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
                // todo check this
                dnnl::softmax_forward::desc op_desc(dnnl::prop_kind::forward_inference, x_mkl_md, axis);

                dnnl::softmax_forward::primitive_desc op_prim_desc(op_desc, attr, engine);

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
                dnnl::softmax_forward(op_prim_desc).execute(stream, args);

                // reorder outputs if necessary
                if (zReorder)
                    dnnl::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

                stream.wait();
            }


            PLATFORM_IMPL(softmax, ENGINE_CPU) {

                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);

                const int rank = input->rankOf();
                int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

                if (dim < 0) {
                    dim += rank;
                }

                REQUIRE_TRUE(dim < rank && dim >= 0, 0, "SOFTMAX_MKLDNN OP: the value of input integer parameter (dimension) must be less than input array rank %i, but got dimension = %i instead !", rank, dim);

                REQUIRE_TRUE(rank <= 6, 0, "SOFTMAX_MKLDNN OP: the rank of input must be less or qual 4, but got rank = %i instead !", rank);

                // mkldnnSoftMax
                softmaxMKLDNN(input, output, dim);

                return Status::OK();
            }

            PLATFORM_CHECK(softmax, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto z = OUTPUT_VARIABLE(0);

                const DataType xType = x->dataType();
                const DataType zType = z->dataType();

                const int xRank = x->rankOf();
                bool bSupportedRanks = (xRank > 2 && xRank < 7);
                /*
                Source     Destination
                f32 	    f32
                */
                return  block.isUseMKLDNN() && bSupportedRanks && (xType == DataType::FLOAT32 && zType == DataType::FLOAT32);

            }

        }
    }
}
