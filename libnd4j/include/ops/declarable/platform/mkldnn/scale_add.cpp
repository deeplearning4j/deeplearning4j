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
// OneDNN implementation of scale and add operations
// Includes: axpy (a*x + y), scalaradd (x + scalar), scalarmul (x * scalar)
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
// SCALAR ADD: x + scalar
static void scalarAddMKLDNN(NDArray* x, float scalar, NDArray* z) {
  dnnl::memory::dims shape = x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // x + scalar using linear: y = 1*x + scalar
  dnnl::eltwise_forward::desc op_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear, x_md, 1.0f, scalar);
  dnnl::eltwise_forward::primitive_desc op_prim_desc(op_desc, attr, engine);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(add_scalar, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  float scalar = T_ARG(0);

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "ADD_SCALAR_MKLDNN OP: rank must be <= 6, but got rank = %i",
               input->rankOf());

  scalarAddMKLDNN(input, scalar, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(add_scalar, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN ADD_SCALAR OP");
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
// SCALAR MULTIPLY: x * scalar
static void scalarMulMKLDNN(NDArray* x, float scalar, NDArray* z) {
  dnnl::memory::dims shape = x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // x * scalar using linear: y = scalar*x + 0
  dnnl::eltwise_forward::desc op_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear, x_md, scalar, 0.0f);
  dnnl::eltwise_forward::primitive_desc op_prim_desc(op_desc, attr, engine);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(multiply_scalar, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  float scalar = T_ARG(0);

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "MULTIPLY_SCALAR_MKLDNN OP: rank must be <= 6, but got rank = %i",
               input->rankOf());

  scalarMulMKLDNN(input, scalar, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(multiply_scalar, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN MULTIPLY_SCALAR OP");
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
// AXPY: alpha * x + y
static void axpyMKLDNN(NDArray* x, NDArray* y, float alpha, NDArray* z) {
  dnnl::memory::dims shape = x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc y_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*y));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Step 1: alpha * x using linear
  dnnl::primitive_attr attr;
  dnnl::eltwise_forward::desc scale_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear, x_md, alpha, 0.0f);
  dnnl::eltwise_forward::primitive_desc scale_prim_desc(scale_desc, attr, engine);

  dnnl::memory scaled_x_mem(scale_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> scale_args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  scale_args[DNNL_ARG_SRC] = x_mem;
  scale_args[DNNL_ARG_DST] = scaled_x_mem;

  dnnl::eltwise_forward(scale_prim_desc).execute(stream, scale_args);

  // Step 2: scaled_x + y using binary add
  dnnl::binary::desc add_desc(dnnl::algorithm::binary_add, scale_prim_desc.dst_desc(), y_md, z_md);
  dnnl::binary::primitive_desc add_prim_desc(add_desc, engine);

  std::unordered_map<int, dnnl::memory> add_args;
  add_args[DNNL_ARG_SRC_0] = scaled_x_mem;

  dnnl::memory y_mem(y_md, engine, y->buffer());
  add_args[DNNL_ARG_SRC_1] = y_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  add_args[DNNL_ARG_DST] = z_mem;

  dnnl::binary(add_prim_desc).execute(stream, add_args);

  stream.wait();
}

PLATFORM_IMPL(axpy, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  float alpha = T_ARG(0);

  REQUIRE_TRUE(x->rankOf() <= 6, 0, "AXPY_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());
  REQUIRE_TRUE(x->isSameShape(y), 0, "AXPY_MKLDNN OP: x and y must have same shape");

  axpyMKLDNN(x, y, alpha, z);

  return sd::Status::OK;
}

PLATFORM_CHECK(axpy, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN AXPY OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(x->isSameShape(y), "SAME_SHAPE"), "x and y must have same shape") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(y->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(y->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
