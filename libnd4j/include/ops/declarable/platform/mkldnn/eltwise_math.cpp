/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// OneDNN implementation of additional math eltwise operations
// Includes operations that can be composed from OneDNN primitives
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
// Helper for generic eltwise backward pass
static void eltwiseBpMKLDNN(NDArray* x, NDArray* dLdz, NDArray* dLdx, dnnl::algorithm alg, float alpha = 0.0f, float beta = 0.0f) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, dLdx_mkl_md, dLdx_user_md, dLdz_mkl_md, dLdz_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  dLdz_user_md = dLdz_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdz));
  onednnUtils::setBlockStrides(*dLdz, dLdz_user_md);

  dLdx_user_md = dLdx_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdx));
  onednnUtils::setBlockStrides(*dLdx, dLdx_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // OneDNN 3.x API for forward hint: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training,
                                                         alg, x_mkl_md, x_mkl_md, alpha, beta);

  // OneDNN 3.x API for backward: primitive_desc(engine, algorithm, diff_src_md, diff_dst_md, data_md, alpha, beta, hint_fwd_pd)
  dnnl::eltwise_backward::primitive_desc op_prim_desc(engine, alg,
                                                       dLdx_mkl_md, dLdz_mkl_md, x_mkl_md, alpha, beta, op_ff_prim_desc);

  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  onednnUtils::loadDataToMklStream(*dLdz, engine, stream, dLdz_user_md, op_prim_desc.diff_dst_desc(),
                                   args[DNNL_ARG_DIFF_DST]);

  auto dLdx_user_mem = onednnUtils::loadDataToMklStream(*dLdx, engine, stream, dLdx_user_md,
                                                        op_prim_desc.diff_src_desc(), args[DNNL_ARG_DIFF_SRC]);

  dnnl::eltwise_backward(op_prim_desc).execute(stream, args);

  if (op_prim_desc.diff_src_desc() != dLdx_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_SRC], dLdx_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], dLdx_user_mem);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// SQUARE BACKWARD
PLATFORM_IMPL(square_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "SQUARE_BP_MKLDNN OP: the rank must be <= 6, but got rank = %i", rank);

  eltwiseBpMKLDNN(input, dLdz, dLdx, algorithm::eltwise_square);

  return sd::Status::OK;
}

PLATFORM_CHECK(square_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SQUARE_BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// EXP BACKWARD
PLATFORM_IMPL(Exp_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "EXP_BP_MKLDNN OP: the rank must be <= 6, but got rank = %i", rank);

  eltwiseBpMKLDNN(input, dLdz, dLdx, algorithm::eltwise_exp);

  return sd::Status::OK;
}

PLATFORM_CHECK(Exp_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN EXP_BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// LOG BACKWARD
PLATFORM_IMPL(Log_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "LOG_BP_MKLDNN OP: the rank must be <= 6, but got rank = %i", rank);

  eltwiseBpMKLDNN(input, dLdz, dLdx, algorithm::eltwise_log);

  return sd::Status::OK;
}

PLATFORM_CHECK(Log_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN LOG_BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// SQRT BACKWARD
PLATFORM_IMPL(Sqrt_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "SQRT_BP_MKLDNN OP: the rank must be <= 6, but got rank = %i", rank);

  eltwiseBpMKLDNN(input, dLdz, dLdx, algorithm::eltwise_sqrt);

  return sd::Status::OK;
}

PLATFORM_CHECK(Sqrt_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SQRT_BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
