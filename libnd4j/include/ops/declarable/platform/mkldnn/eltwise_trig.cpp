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
// OneDNN-compatible implementations for trigonometric and hyperbolic functions
// Note: OneDNN doesn't have native trig support, but we provide optimized implementations
// for functions that can be composed from available primitives
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
// Helper for generic eltwise forward pass
static void eltwiseMKLDNN(NDArray* x, NDArray* z, dnnl::algorithm alg, float alpha = 0.0f, float beta = 0.0f) {
  dnnl::memory::dims shape = x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  dnnl::primitive_attr attr;

  dnnl::eltwise_forward::desc op_desc(dnnl::prop_kind::forward_inference, alg, x_mkl_md, alpha, beta);

  dnnl::eltwise_forward::primitive_desc op_prim_desc(op_desc, attr, engine);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// COSH: (exp(x) + exp(-x)) / 2
// Implemented using exp operations
static void coshMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // Step 1: exp(x)
  dnnl::eltwise_forward::desc exp_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_exp, x_md, 0, 0);
  dnnl::eltwise_forward::primitive_desc exp_prim_desc(exp_desc, attr, engine);

  dnnl::memory exp_x_mem(exp_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> exp_args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  exp_args[DNNL_ARG_SRC] = x_mem;
  exp_args[DNNL_ARG_DST] = exp_x_mem;

  dnnl::eltwise_forward(exp_prim_desc).execute(stream, exp_args);

  // Step 2: -x using linear: y = -1*x + 0
  dnnl::eltwise_forward::desc neg_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear, x_md, -1.0f, 0.0f);
  dnnl::eltwise_forward::primitive_desc neg_prim_desc(neg_desc, attr, engine);

  dnnl::memory neg_x_mem(neg_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> neg_args;
  neg_args[DNNL_ARG_SRC] = x_mem;
  neg_args[DNNL_ARG_DST] = neg_x_mem;

  dnnl::eltwise_forward(neg_prim_desc).execute(stream, neg_args);

  // Step 3: exp(-x)
  dnnl::memory exp_neg_x_mem(exp_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> exp_neg_args;
  exp_neg_args[DNNL_ARG_SRC] = neg_x_mem;
  exp_neg_args[DNNL_ARG_DST] = exp_neg_x_mem;

  dnnl::eltwise_forward(exp_prim_desc).execute(stream, exp_neg_args);

  // Step 4: exp(x) + exp(-x) using binary add
  dnnl::binary::desc add_desc(dnnl::algorithm::binary_add, exp_prim_desc.dst_desc(),
                               exp_prim_desc.dst_desc(), exp_prim_desc.dst_desc());
  dnnl::binary::primitive_desc add_prim_desc(add_desc, engine);

  dnnl::memory sum_mem(add_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> add_args;
  add_args[DNNL_ARG_SRC_0] = exp_x_mem;
  add_args[DNNL_ARG_SRC_1] = exp_neg_x_mem;
  add_args[DNNL_ARG_DST] = sum_mem;

  dnnl::binary(add_prim_desc).execute(stream, add_args);

  // Step 5: divide by 2 using linear: y = 0.5*x + 0
  dnnl::eltwise_forward::desc div_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear,
                                        add_prim_desc.dst_desc(), 0.5f, 0.0f);
  dnnl::eltwise_forward::primitive_desc div_prim_desc(div_desc, attr, engine);

  std::unordered_map<int, dnnl::memory> div_args;
  div_args[DNNL_ARG_SRC] = sum_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  div_args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(div_prim_desc).execute(stream, div_args);

  stream.wait();
}

PLATFORM_IMPL(cosh, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "COSH_MKLDNN OP: the rank must be <= 6, but got rank = %i", rank);

  coshMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(cosh, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN COSH OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// SINH: (exp(x) - exp(-x)) / 2
static void sinhMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // Step 1: exp(x)
  dnnl::eltwise_forward::desc exp_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_exp, x_md, 0, 0);
  dnnl::eltwise_forward::primitive_desc exp_prim_desc(exp_desc, attr, engine);

  dnnl::memory exp_x_mem(exp_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> exp_args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  exp_args[DNNL_ARG_SRC] = x_mem;
  exp_args[DNNL_ARG_DST] = exp_x_mem;

  dnnl::eltwise_forward(exp_prim_desc).execute(stream, exp_args);

  // Step 2: -x using linear
  dnnl::eltwise_forward::desc neg_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear, x_md, -1.0f, 0.0f);
  dnnl::eltwise_forward::primitive_desc neg_prim_desc(neg_desc, attr, engine);

  dnnl::memory neg_x_mem(neg_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> neg_args;
  neg_args[DNNL_ARG_SRC] = x_mem;
  neg_args[DNNL_ARG_DST] = neg_x_mem;

  dnnl::eltwise_forward(neg_prim_desc).execute(stream, neg_args);

  // Step 3: exp(-x)
  dnnl::memory exp_neg_x_mem(exp_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> exp_neg_args;
  exp_neg_args[DNNL_ARG_SRC] = neg_x_mem;
  exp_neg_args[DNNL_ARG_DST] = exp_neg_x_mem;

  dnnl::eltwise_forward(exp_prim_desc).execute(stream, exp_neg_args);

  // Step 4: exp(x) - exp(-x) using binary sub
  dnnl::binary::desc sub_desc(dnnl::algorithm::binary_sub, exp_prim_desc.dst_desc(),
                               exp_prim_desc.dst_desc(), exp_prim_desc.dst_desc());
  dnnl::binary::primitive_desc sub_prim_desc(sub_desc, engine);

  dnnl::memory diff_mem(sub_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> sub_args;
  sub_args[DNNL_ARG_SRC_0] = exp_x_mem;
  sub_args[DNNL_ARG_SRC_1] = exp_neg_x_mem;
  sub_args[DNNL_ARG_DST] = diff_mem;

  dnnl::binary(sub_prim_desc).execute(stream, sub_args);

  // Step 5: divide by 2
  dnnl::eltwise_forward::desc div_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear,
                                        sub_prim_desc.dst_desc(), 0.5f, 0.0f);
  dnnl::eltwise_forward::primitive_desc div_prim_desc(div_desc, attr, engine);

  std::unordered_map<int, dnnl::memory> div_args;
  div_args[DNNL_ARG_SRC] = diff_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  div_args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(div_prim_desc).execute(stream, div_args);

  stream.wait();
}

PLATFORM_IMPL(sinh, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "SINH_MKLDNN OP: the rank must be <= 6, but got rank = %i", rank);

  sinhMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(sinh, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SINH OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
