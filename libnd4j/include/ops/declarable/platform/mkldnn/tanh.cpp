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
                dnnl::memory::dims xShape, zShape;

                mkldnnUtils::getDims(x, xRank, xShape);
                mkldnnUtils::getDims(z, xRank, zShape);

                dnnl::memory::format_tag format = mkldnnUtils::getFormat(xRank);

                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(x, x_user_md);

                // z
                dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc z_user_md = dnnl::memory::desc(zShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(z, z_user_md);

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
                mkldnnUtils::loadDataToMklStream(x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

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
                bool bSupportedRanks = !x->isEmpty() && xRank < 7 && xRank > 0 && (xType == DataType::FLOAT32 && zType == DataType::FLOAT32);
                /*
                Source     Destination
                f32 	    f32
                */
                return  block.isUseMKLDNN() && bSupportedRanks;
            }


            //////////////////////////////////////////////////////////////////////
            static void tanhBpMKLDNN(const NDArray* x, const NDArray* dLdz, NDArray* dLdx) {

                const auto xRank = x->rankOf();
                dnnl::memory::dims xShape, dLdzShape, dLdxShape;

                mkldnnUtils::getDims(x, xRank, xShape);
                mkldnnUtils::getDims(dLdz, xRank, dLdzShape);
                mkldnnUtils::getDims(dLdx, xRank, dLdxShape);

                dnnl::memory::format_tag format = mkldnnUtils::getFormat(xRank);

                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(x, x_user_md);

                // dLdz
                dnnl::memory::desc dLdz_mkl_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc dLdz_user_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(dLdz, dLdz_user_md);

                // dLdx
                dnnl::memory::desc dLdx_mkl_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc dLdx_user_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(dLdx, dLdx_user_md);

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

                // arguments (memory buffers) necessary for calculations
                std::unordered_map<int, dnnl::memory> args;

                dnnl::stream stream(engine);

                // operation primitive description
                // forward
                dnnl::eltwise_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_tanh, x_mkl_md, 0, 0);
                dnnl::eltwise_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

                // backward description
                dnnl::eltwise_backward::desc op_desc(algorithm::eltwise_tanh, dLdz_mkl_md, x_mkl_md, 0, 0);
                dnnl::eltwise_backward::primitive_desc op_prim_desc(op_desc, engine, op_ff_prim_desc);

                // provide memory buffers and check whether reorder is required for forward
                // input
                mkldnnUtils::loadDataToMklStream(x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

                // dLdz
                mkldnnUtils::loadDataToMklStream(dLdz, engine, stream, dLdz_user_md, op_prim_desc.diff_dst_desc(), args[DNNL_ARG_DIFF_DST]);

                // dLdx
                auto dLdx_user_mem = dnnl::memory(dLdx_user_md, engine, dLdx->getBuffer());
                const bool dLdxReorder = op_prim_desc.diff_src_desc() != dLdx_user_mem.get_desc();
                auto dLdx_mkl_mem = dLdxReorder ? dnnl::memory(op_prim_desc.diff_src_desc(), engine) : dLdx_user_mem;
                args[DNNL_ARG_DIFF_SRC] = dLdx_mkl_mem;

                // run calculations backward
                dnnl::eltwise_backward(op_prim_desc).execute(stream, args);

                // reorder outputs if necessary
                if (dLdxReorder)
                    dnnl::reorder(dLdx_mkl_mem, dLdx_user_mem).execute(stream, dLdx_mkl_mem, dLdx_user_mem);

                stream.wait();
            }


            PLATFORM_IMPL(tanh_bp, ENGINE_CPU) {

                auto input = INPUT_VARIABLE(0);
                auto dLdz = INPUT_VARIABLE(1);
                auto dLdx = OUTPUT_VARIABLE(0);

                const int rank = input->rankOf();
                const int dLdzRank = dLdz->rankOf();

                REQUIRE_TRUE(rank <= 6 && dLdzRank <= 6, 0, "TANH_BP_MKLDNN OP: the rank of input and dLdz must be less or qual 6, but got input rank = %i and dLdz rank rank = %i instead !", rank, dLdzRank);

                // mkldnnSoftMax
                tanhBpMKLDNN(input, dLdz, dLdx);

                return Status::OK();
            }

            PLATFORM_CHECK(tanh_bp, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto dLdz = INPUT_VARIABLE(1);
                auto dLdx = OUTPUT_VARIABLE(0);

                const DataType xType = x->dataType();
                const DataType dLdzType = dLdz->dataType();
                const DataType dLdxType = dLdx->dataType();

                const int xRank = x->rankOf();
                const int dLdzRank = dLdz->rankOf();

                bool bSupportedRanks = xRank < 7 && xRank > 0 && dLdzRank == xRank && (!x->isEmpty() && !dLdz->isEmpty());
                bSupportedRanks &= (xType == DataType::FLOAT32 && dLdzType == DataType::FLOAT32 && dLdxType == DataType::FLOAT32);

                if (bSupportedRanks) {
                    for (int i = 0; i < xRank; i++) {
                        if (x->sizeAt(i) != dLdz->sizeAt(i)) {
                            bSupportedRanks = false;
                            break;
                        }
                    }
                }

                //Source     Destination
                //f32 	    f32
                return block.isUseMKLDNN() && bSupportedRanks;
            }

        }
    }
}
