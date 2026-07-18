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
// OneDNN implementation of extended eltwise operations:
// expm1 (exp(x) - 1), log1p (log(1 + x)), gelu_tanh
//
// NOTE: expm1 and log1p are multi-step pipelines using f32 intermediate
// buffers.  They remain f32-only intentionally — widening would require
// allocating intermediate bf16/f16 tensors which is significantly more
// complex and the performance benefit on those paths is marginal.
// gelu_tanh is a single primitive and uses the dtype-aware macro.
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"
#include "mkldnnEltwise.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
// EXPM1: exp(x) - 1  (f32 only — multi-pass with f32 intermediate)
static void expm1MKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Step 1: exp(x)
  dnnl::eltwise_forward::primitive_desc exp_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                       algorithm::eltwise_exp, x_mkl_md, x_mkl_md, 0.f, 0.f);

  dnnl::memory exp_mem(exp_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> exp_args;
  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, exp_prim_desc.src_desc(), exp_args[DNNL_ARG_SRC]);
  exp_args[DNNL_ARG_DST] = exp_mem;

  dnnl::eltwise_forward(exp_prim_desc).execute(stream, exp_args);

  // Step 2: exp(x) - 1 using linear: y = 1*x + (-1)
  dnnl::eltwise_forward::primitive_desc sub_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                       algorithm::eltwise_linear, exp_prim_desc.dst_desc(), z_mkl_md, 1.0f, -1.0f);

  std::unordered_map<int, dnnl::memory> sub_args;
  sub_args[DNNL_ARG_SRC] = exp_mem;

  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, sub_prim_desc.dst_desc(), sub_args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(sub_prim_desc).execute(stream, sub_args);

  if (sub_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(sub_args[DNNL_ARG_DST], z_user_mem).execute(stream, sub_args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(expm1, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "EXPM1_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  expm1MKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(expm1, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN EXPM1 OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// LOG1P: log(1 + x)  (f32 only — multi-pass with f32 intermediate)
static void log1pMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Step 1: 1 + x using linear: y = 1*x + 1
  dnnl::eltwise_forward::primitive_desc add_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                       algorithm::eltwise_linear, x_mkl_md, x_mkl_md, 1.0f, 1.0f);

  dnnl::memory add_mem(add_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> add_args;
  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, add_prim_desc.src_desc(), add_args[DNNL_ARG_SRC]);
  add_args[DNNL_ARG_DST] = add_mem;

  dnnl::eltwise_forward(add_prim_desc).execute(stream, add_args);

  // Step 2: log(1 + x)
  dnnl::eltwise_forward::primitive_desc log_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                       algorithm::eltwise_log, add_prim_desc.dst_desc(), z_mkl_md, 0.f, 0.f);

  std::unordered_map<int, dnnl::memory> log_args;
  log_args[DNNL_ARG_SRC] = add_mem;

  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, log_prim_desc.dst_desc(), log_args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(log_prim_desc).execute(stream, log_args);

  if (log_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(log_args[DNNL_ARG_DST], z_user_mem).execute(stream, log_args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(log1p, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "LOG1P_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  log1pMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(log1p, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN LOG1P OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// GELU_TANH — single primitive, use dtype-aware macro
DEFINE_MKLDNN_ELTWISE_FWD(gelu_tanh, "GELU_TANH", dnnl::algorithm::eltwise_gelu_tanh, 0.f, 0.f)

}  // namespace platforms
}  // namespace ops
}  // namespace sd
