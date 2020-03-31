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
            static void xwPlusBiasMKLDNN(const NDArray* x, const NDArray* weights, const NDArray* bias, NDArray* z, const bool bShouldTransp) {

                // mkl works with following
                // [M,K]     x [N,K]^T  + [N]   = [M,N]
                const auto xRank = x->rankOf();

                // [M,K] x [K,N] = [M,N]
                const int M = x->sizeAt(0);
                const int K = x->sizeAt(1); // K == wK
                const int N = z->sizeAt(1);

                dnnl::memory::dims xShape = dnnl::memory::dims({ M, K });
                dnnl::memory::dims wShape = dnnl::memory::dims({ N, K });
                dnnl::memory::dims zShape = dnnl::memory::dims({ M, N });
                dnnl::memory::dims bShape = dnnl::memory::dims({ N });

                dnnl::memory::format_tag format = dnnl::memory::format_tag::ab;

                // x type
                dnnl::memory::data_type xType = dnnl::memory::data_type::f32;
                if (x->dataType() == DataType::UINT8)
                    xType = dnnl::memory::data_type::u8;
                else if (x->dataType() == DataType::INT8)
                    xType = dnnl::memory::data_type::s8;

                // weights type
                dnnl::memory::data_type wType = (weights->dataType() == DataType::FLOAT32) ?
                    wType = dnnl::memory::data_type::f32 : wType = dnnl::memory::data_type::s8;

                // bias type need add description for bias
                dnnl::memory::data_type bType = dnnl::memory::data_type::f32;
                if (bias->dataType() == DataType::INT32)
                    bType = dnnl::memory::data_type::s32;
                else if (bias->dataType() == DataType::UINT8)
                    bType = dnnl::memory::data_type::u8;
                else if (bias->dataType() == DataType::INT8)
                    bType = dnnl::memory::data_type::s8;

                // z type
                dnnl::memory::data_type zType = dnnl::memory::data_type::f32;
                if (z->dataType() == DataType::INT32)
                    zType = dnnl::memory::data_type::s32;
                else if (z->dataType() == DataType::UINT8)
                    zType = dnnl::memory::data_type::u8;
                else if (z->dataType() == DataType::INT8)
                    zType = dnnl::memory::data_type::s8;

                // memory descriptors for arrays
                // x
                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, xType, dnnl::memory::format_tag::any);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, xType, format);
                mkldnnUtils::setBlockStrides(x, x_user_md);

                // weights
                dnnl::memory::desc weights_mkl_md = dnnl::memory::desc(wShape, wType, dnnl::memory::format_tag::any);
                dnnl::memory::desc weights_user_md = dnnl::memory::desc(wShape, wType, format);
                if (weights->ews() != 1 || weights->ordering() != 'c' || bShouldTransp) {

                    weights_user_md.data.format_kind = dnnl_blocked;    // overrides format
                    if (bShouldTransp) {
                        weights_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(1);
                        weights_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(0);
                    }
                    else {
                        weights_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(0);
                        weights_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(1);
                    }
                }
                // bias
                dnnl::memory::desc bias_mkl_md = dnnl::memory::desc(bShape, bType, dnnl::memory::format_tag::x);
                dnnl::memory::desc bias_user_md = dnnl::memory::desc(bShape, bType, dnnl::memory::format_tag::x);
                mkldnnUtils::setBlockStrides(bias, bias_user_md);

                // z
                dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zShape, zType, dnnl::memory::format_tag::any);
                dnnl::memory::desc z_user_md = dnnl::memory::desc(zShape, zType, format);
                mkldnnUtils::setBlockStrides(z, z_user_md);

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

                // operation primitive description
                dnnl::inner_product_forward::desc op_desc(dnnl::prop_kind::forward_inference, x_mkl_md, weights_mkl_md, bias_mkl_md, z_mkl_md);

                dnnl::inner_product_forward::primitive_desc op_prim_desc(op_desc, engine);

                // arguments (memory buffers) necessary for calculations
                std::unordered_map<int, dnnl::memory> args;

                dnnl::stream stream(engine);

                // provide memory buffers and check whether reorder is required

                // input
                mkldnnUtils::loadDataToMklStream(x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

                // weights
                mkldnnUtils::loadDataToMklStream(weights, engine, stream, weights_user_md, op_prim_desc.weights_desc(), args[DNNL_ARG_WEIGHTS]);

                // bias
                auto bias_mkl_mem = dnnl::memory(bias_mkl_md, engine, bias->getBuffer());
                args[DNNL_ARG_BIAS] = bias_mkl_mem;

                // z
                auto z_user_mem = dnnl::memory(z_user_md, engine, z->getBuffer());
                const bool zReorder = op_prim_desc.dst_desc() != z_user_mem.get_desc();
                auto z_mkl_mem = zReorder ? dnnl::memory(op_prim_desc.dst_desc(), engine) : z_user_mem;
                args[DNNL_ARG_DST] = z_mkl_mem;

                // run calculations
                dnnl::inner_product_forward(op_prim_desc).execute(stream, args);

                // reorder outputs if necessary
                if (zReorder)
                    dnnl::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

                stream.wait();
            }

            //////////////////////////////////////////////////////////////////////
            static void xwPlusBiasBp(const NDArray* x, const NDArray* weights, const NDArray* bias, const NDArray* dLdz,
                NDArray* dLdx, NDArray* dLdw, NDArray* dLdb, const bool bShouldTransp) {

                // mkl works with following
                // [M,K]     x [N,K]^T  + [N]   = [M,N]
                const auto xRank = x->rankOf();

                // [M,K] x [K,N] = [M,N]
                const int M = x->sizeAt(0);
                const int K = x->sizeAt(1); // K == wK               
                const int N = dLdz->sizeAt(1);
                // input dims
                dnnl::memory::dims xShape = dnnl::memory::dims({ M, K });
                dnnl::memory::dims wShape = dnnl::memory::dims({ N, K });
                dnnl::memory::dims dLdzShape = dnnl::memory::dims({ M, N });

                dnnl::memory::dims bShape = dnnl::memory::dims({ N });
                // output dims
                dnnl::memory::dims dLdxShape = xShape;
                dnnl::memory::dims dLdwShape = wShape;

                dnnl::memory::format_tag format = dnnl::memory::format_tag::ab;
                dnnl::memory::data_type dataType = dnnl::memory::data_type::f32;

                // memory descriptors for arrays
                // x
                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, dataType, dnnl::memory::format_tag::any);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, dataType, format);
                mkldnnUtils::setBlockStrides(x, x_user_md);

                // weights
                dnnl::memory::desc weights_mkl_md = dnnl::memory::desc(wShape, dataType, dnnl::memory::format_tag::any);
                dnnl::memory::desc weights_user_md = dnnl::memory::desc(wShape, dataType, format);
                if (weights->ews() != 1 || weights->ordering() != 'c' || bShouldTransp) {

                    weights_user_md.data.format_kind = dnnl_blocked;    // overrides format
                    if (bShouldTransp) {
                        weights_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(1);
                        weights_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(0);
                    }
                    else {
                        weights_user_md.data.format_desc.blocking.strides[0] = weights->strideAt(0);
                        weights_user_md.data.format_desc.blocking.strides[1] = weights->strideAt(1);
                    }
                }
                // bias
                dnnl::memory::desc bias_mkl_md = dnnl::memory::desc(bShape, dataType, dnnl::memory::format_tag::x);
                dnnl::memory::desc bias_user_md = dnnl::memory::desc(bShape, dataType, dnnl::memory::format_tag::x);
                mkldnnUtils::setBlockStrides(bias, bias_user_md);

                // dLdz
                dnnl::memory::desc dLdz_mkl_md = dnnl::memory::desc(dLdzShape, dataType, dnnl::memory::format_tag::any);
                dnnl::memory::desc dLdz_user_md = dnnl::memory::desc(dLdzShape, dataType, format);
                mkldnnUtils::setBlockStrides(dLdz, dLdz_user_md);

                // dLdw
                dnnl::memory::desc dLdw_mkl_md = dnnl::memory::desc(wShape, dataType, format);
                dnnl::memory::desc dLdw_user_md = dnnl::memory::desc(wShape, dataType, format);
                if (dLdw->ews() != 1 || dLdw->ordering() != 'c' || bShouldTransp) {

                    dLdw_user_md.data.format_kind = dnnl_blocked;    // overrides format
                    if (bShouldTransp) {
                        dLdw_user_md.data.format_desc.blocking.strides[0] = dLdw->strideAt(1);
                        dLdw_user_md.data.format_desc.blocking.strides[1] = dLdw->strideAt(0);
                    }
                    else {
                        dLdw_user_md.data.format_desc.blocking.strides[0] = dLdw->strideAt(0);
                        dLdw_user_md.data.format_desc.blocking.strides[1] = dLdw->strideAt(1);
                    }
                }

                // dLdb
                dnnl::memory::desc dLdb_mkl_md = dnnl::memory::desc(bShape, dataType, dnnl::memory::format_tag::x);
                dnnl::memory::desc dLdb_user_md = dnnl::memory::desc(bShape, dataType, dnnl::memory::format_tag::x);
                mkldnnUtils::setBlockStrides(dLdb, dLdb_user_md);

                // dLdx
                dnnl::memory::desc dLdx_mkl_md = dnnl::memory::desc(xShape, dataType, dnnl::memory::format_tag::any);
                dnnl::memory::desc dLdx_user_md = dnnl::memory::desc(xShape, dataType, format);
                mkldnnUtils::setBlockStrides(dLdx, dLdx_user_md);

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
                // forward
                // operation primitive description
                dnnl::inner_product_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, x_mkl_md, weights_mkl_md, bias_mkl_md, dLdz_mkl_md);
                dnnl::inner_product_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

                // backprob
                // dLdw
                auto op_bpdw_desc = inner_product_backward_weights::desc(x_mkl_md, dLdw_mkl_md, dLdb_mkl_md, dLdz_mkl_md);
                auto op_bpdw_prim_desc = inner_product_backward_weights::primitive_desc(op_bpdw_desc, engine, op_ff_prim_desc);

                // backprob
                // dLdx
                auto op_bpdx_desc = inner_product_backward_data::desc(dLdx_mkl_md, weights_mkl_md, dLdz_mkl_md);
                auto op_bpdx_prim_desc = inner_product_backward_data::primitive_desc(op_bpdx_desc, engine, op_ff_prim_desc);

                // arguments (memory buffers) necessary for calculations
                std::unordered_map<int, dnnl::memory> argsDw, argsDx;

                dnnl::stream stream(engine);

                // dLdz dw
                mkldnnUtils::loadDataToMklStream(dLdz, engine, stream, dLdz_user_md, op_bpdw_prim_desc.diff_dst_desc(), argsDw[DNNL_ARG_DIFF_DST]);

                // dLdz - dx
                mkldnnUtils::loadDataToMklStream(dLdz, engine, stream, dLdz_user_md, op_bpdx_prim_desc.diff_dst_desc(), argsDx[DNNL_ARG_DIFF_DST]);

                // input x for dw
                mkldnnUtils::loadDataToMklStream(x, engine, stream, x_user_md, op_bpdw_prim_desc.src_desc(), argsDw[DNNL_ARG_SRC]);

                // weights - dx
                mkldnnUtils::loadDataToMklStream(weights, engine, stream, weights_user_md, op_bpdx_prim_desc.weights_desc(), argsDx[DNNL_ARG_WEIGHTS]);

                // dLdw 
                auto dLdw_user_mem = dnnl::memory(dLdw_user_md, engine, dLdw->getBuffer());
                const bool dLdwReorder = op_bpdw_prim_desc.diff_weights_desc() != dLdw_user_mem.get_desc();
                auto dLdw_mkl_mem = dLdwReorder ? dnnl::memory(op_bpdw_prim_desc.diff_weights_desc(), engine) : dLdw_user_mem;
                argsDw[DNNL_ARG_DIFF_WEIGHTS] = dLdw_mkl_mem;

                // dLdx 
                auto dLdx_user_mem = dnnl::memory(dLdx_user_md, engine, dLdx->getBuffer());
                const bool dLdxReorder = op_bpdx_prim_desc.diff_src_desc() != dLdx_user_mem.get_desc();
                auto dLdx_mkl_mem = dLdxReorder ? dnnl::memory(op_bpdx_prim_desc.diff_src_desc(), engine) : dLdx_user_mem;
                argsDx[DNNL_ARG_DIFF_SRC] = dLdx_mkl_mem;

                // dLdb
                auto dLdb_user_mem = dnnl::memory(dLdb_user_md, engine, dLdb->getBuffer());
                const bool dLdbReorder = op_bpdw_prim_desc.diff_bias_desc() != dLdb_user_mem.get_desc();
                auto dLdb_mkl_mem = dLdbReorder ? dnnl::memory(op_bpdw_prim_desc.diff_bias_desc(), engine) : dLdb_user_mem;
                argsDw[DNNL_ARG_DIFF_BIAS] = dLdb_mkl_mem;

                // run calculations dw
                dnnl::inner_product_backward_weights(op_bpdw_prim_desc).execute(stream, argsDw);
                // run calculations dx
                dnnl::inner_product_backward_data(op_bpdx_prim_desc).execute(stream, argsDx);

                // reorder outputs if necessary
                if (dLdxReorder)
                    dnnl::reorder(dLdx_mkl_mem, dLdx_user_mem).execute(stream, dLdx_mkl_mem, dLdx_user_mem);

                if (dLdwReorder)
                    dnnl::reorder(dLdw_mkl_mem, dLdw_user_mem).execute(stream, dLdw_mkl_mem, dLdw_user_mem);

                if (dLdbReorder)
                    dnnl::reorder(dLdb_mkl_mem, dLdb_user_mem).execute(stream, dLdb_mkl_mem, dLdb_user_mem);

                stream.wait();
            }

            PLATFORM_IMPL(xw_plus_b, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto w = INPUT_VARIABLE(1);
                auto b = INPUT_VARIABLE(2);
                auto z = OUTPUT_VARIABLE(0);

                if (x->isEmpty() || w->isEmpty() || b->isEmpty())
                    return Status::OK();

                const int xRank = x->rankOf();
                const int wRank = w->rankOf();
                const int zRank = z->rankOf();

                const bool bShouldTransp = block.getIArguments()->size() > 0 ? (1 != INT_ARG(0)) : true; // [M,K] * [K,N] -> [M, N], mkl -> [M,K] * [N, K]^T -> [M, N] 

                REQUIRE_TRUE(xRank == 2, 0, "xw_plus_b MKL: Input x array should have rank equal 2, but got instead %i!", xRank);
                REQUIRE_TRUE(wRank == 2, 0, "xw_plus_b MKL: Input weights array should have rank equal 2, but got instead %i!", wRank);
                REQUIRE_TRUE(zRank == 2, 0, "xw_plus_b MKL: Output array should have rank equal 2, but got instead %i!", zRank);

                REQUIRE_TRUE(1 == b->rankOf() && b->lengthOf() == z->sizeAt(-1), 0, "xw_plus_b MKL: Input bias vector should be 1D and have proper dimension 1x%i."
                    " But got rank %i, and got length %i instead %i.", z->sizeAt(-1), b->rankOf(), b->lengthOf(), z->sizeAt(-1));

                // mkldnnInerPorductss
                xwPlusBiasMKLDNN(x, w, b, z, bShouldTransp);

                return Status::OK();
            }

            PLATFORM_CHECK(xw_plus_b, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto w = INPUT_VARIABLE(1);
                auto b = INPUT_VARIABLE(2);
                auto z = OUTPUT_VARIABLE(0);

                const DataType xType = x->dataType();
                const DataType wType = w->dataType();
                const DataType bType = b->dataType();
                const DataType zType = z->dataType();

                /*
                Source    Weights   Destination         Bias
                f32 	    f32 	f32 	            f32
                u8, s8  	s8 	    u8, s8, s32, f32 	u8, s8, s32, f32
                */
                return block.isUseMKLDNN() &&
                    ((xType == DataType::FLOAT32 && wType == DataType::FLOAT32 && bType == DataType::FLOAT32 && zType == DataType::FLOAT32) ||
                        ( // x
                            (xType == DataType::UINT8 || xType == DataType::INT8) &&
                            // w
                            (wType == DataType::UINT8 || wType == DataType::INT8) &&
                            // b
                            (bType == DataType::UINT8 || bType == DataType::INT8 || bType == DataType::INT32 || bType == DataType::FLOAT32) &&
                            // z
                            (zType == DataType::UINT8 || zType == DataType::INT8 || zType == DataType::INT32 || zType == DataType::FLOAT32)
                            ));
            }

            PLATFORM_IMPL(xw_plus_b_bp, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto w = INPUT_VARIABLE(1);
                auto b = INPUT_VARIABLE(2);
                auto dLdz = INPUT_VARIABLE(3);

                auto dLdx = OUTPUT_VARIABLE(0);
                auto dLdw = OUTPUT_VARIABLE(1);
                auto dLdb = OUTPUT_VARIABLE(2);

                if (x->isEmpty() || w->isEmpty() || b->isEmpty() || dLdz->isEmpty())
                    return Status::OK();

                const int xRank = x->rankOf();
                const int wRank = w->rankOf();
                const int dLdzRank = dLdz->rankOf();

                const bool bShouldTransp = block.getIArguments()->size() > 0 ? (1 != INT_ARG(0)) : true; // [M,K] * [K,N] -> [M, N], mkl -> [M,K] * [N, K]^T -> [M, N] 

                REQUIRE_TRUE(x->rankOf() == 2, 0, "xw_plus_b BP MKL: Input x array should have rank equal 2, but got instead %i!", x->rankOf());
                REQUIRE_TRUE(w->rankOf() == 2, 0, "xw_plus_b BP MKL: Input weights array should have rank equal 2, but got instead %i!", w->rankOf());
                REQUIRE_TRUE(dLdz->rankOf() == 2, 0, "xw_plus_b BP MKL: Output array should have rank equal 2, but got instead %i!", dLdz->rankOf());
                REQUIRE_TRUE(1 == b->rankOf() && b->lengthOf() == dLdz->sizeAt(1), 0, "xw_plus_b BP MKL: Input bias vector should be 1D and have proper dimension 1x%i."
                    " But got rank %i, and got length %i instead %i.", dLdz->sizeAt(1), b->rankOf(), b->lengthOf(), dLdz->sizeAt(1));

                xwPlusBiasBp(x, w, b, dLdz, dLdx, dLdw, dLdb, bShouldTransp);

                return Status::OK();
            }

            PLATFORM_CHECK(xw_plus_b_bp, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto w = INPUT_VARIABLE(1);
                auto b = INPUT_VARIABLE(2);
                auto dLdz = INPUT_VARIABLE(3);

                auto dLdx = OUTPUT_VARIABLE(0);
                auto dLdw = OUTPUT_VARIABLE(1);
                auto dLdb = OUTPUT_VARIABLE(2);

                const DataType xType = x->dataType();
                const DataType wType = w->dataType();
                const DataType bType = b->dataType();
                const DataType dLdzType = dLdz->dataType();
                const DataType dLdxType = dLdx->dataType();
                const DataType dLdwType = dLdw->dataType();
                const DataType dLdbType = dLdb->dataType();

                /*
                Source    Weights   Destination         Bias
                f32 	    f32 	f32 	            f32
                */
                return block.isUseMKLDNN() &&
                    (xType == DataType::FLOAT32 && wType == DataType::FLOAT32 &&
                        bType == DataType::FLOAT32 && dLdzType == DataType::FLOAT32 &&
                        dLdbType == DataType::FLOAT32 && dLdxType == DataType::FLOAT32 &&
                        dLdwType == DataType::FLOAT32);
            }

        }
    }
}
