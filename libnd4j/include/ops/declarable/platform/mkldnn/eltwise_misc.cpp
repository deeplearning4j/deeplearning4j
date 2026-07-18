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
// OneDNN implementation of miscellaneous eltwise operations:
// squared_difference, rsqrt, reciprocal, etc.
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
// SQUARED_DIFFERENCE: (x - y)^2
static void squaredDifferenceMKLDNN(NDArray* x, NDArray* y, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc y_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*y));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // Step 1: x - y using binary sub
  // OneDNN 3.x API: binary::primitive_desc(engine, algorithm, src0_md, src1_md, dst_md)
  dnnl::binary::primitive_desc sub_prim_desc(engine, dnnl::algorithm::binary_sub, x_md, y_md, x_md);

  dnnl::memory diff_mem(sub_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> sub_args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  dnnl::memory y_mem(y_md, engine, y->buffer());
  sub_args[DNNL_ARG_SRC_0] = x_mem;
  sub_args[DNNL_ARG_SRC_1] = y_mem;
  sub_args[DNNL_ARG_DST] = diff_mem;

  dnnl::binary(sub_prim_desc).execute(stream, sub_args);

  // Step 2: (x - y)^2 using eltwise square
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc sq_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_square, sub_prim_desc.dst_desc(), z_md, 0.f, 0.f);

  std::unordered_map<int, dnnl::memory> sq_args;
  sq_args[DNNL_ARG_SRC] = diff_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  sq_args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(sq_prim_desc).execute(stream, sq_args);

  stream.wait();
}

PLATFORM_IMPL(squaredsubtract, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(x->isSameShape(y), 0, "SQUAREDSUBTRACT_MKLDNN OP: x and y must have same shape");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "SQUAREDSUBTRACT_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());

  squaredDifferenceMKLDNN(x, y, z);

  return sd::Status::OK;
}

PLATFORM_CHECK(squaredsubtract, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SQUAREDSUBTRACT OP");
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

//////////////////////////////////////////////////////////////////////
// RSQRT: 1 / sqrt(x)
static void rsqrtMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // Step 1: sqrt(x)
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc sqrt_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                        algorithm::eltwise_sqrt, x_md, x_md, 0.f, 0.f);

  dnnl::memory sqrt_mem(sqrt_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> sqrt_args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  sqrt_args[DNNL_ARG_SRC] = x_mem;
  sqrt_args[DNNL_ARG_DST] = sqrt_mem;

  dnnl::eltwise_forward(sqrt_prim_desc).execute(stream, sqrt_args);

  // Step 2: 1 / sqrt(x) using pow with exponent -1
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc inv_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                       algorithm::eltwise_pow, sqrt_prim_desc.dst_desc(), z_md, 1.0f, -1.0f);

  std::unordered_map<int, dnnl::memory> inv_args;
  inv_args[DNNL_ARG_SRC] = sqrt_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  inv_args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(inv_prim_desc).execute(stream, inv_args);

  stream.wait();
}

PLATFORM_IMPL(rsqrt, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "RSQRT_MKLDNN OP: rank must be <= 6, but got rank = %i", input->rankOf());

  rsqrtMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(rsqrt, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN RSQRT OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// RECIPROCAL: 1 / x
static void reciprocalMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // 1/x using pow with exponent -1
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_pow, x_md, z_md, 1.0f, -1.0f);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(Reciprocal, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "RECIPROCAL_MKLDNN OP: rank must be <= 6, but got rank = %i", input->rankOf());

  reciprocalMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(Reciprocal, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN RECIPROCAL OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// CUBE: x^3
static void cubeMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // x^3 using pow with exponent 3
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_pow, x_md, z_md, 1.0f, 3.0f);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(cube, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "CUBE_MKLDNN OP: rank must be <= 6, but got rank = %i", input->rankOf());

  cubeMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(cube, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN CUBE OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// IDENTITY / COPY
static void identityMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Identity using reorder
  dnnl::memory x_mem(x_md, engine, x->buffer());
  dnnl::memory z_mem(z_md, engine, z->buffer());

  dnnl::reorder(x_mem, z_mem).execute(stream, x_mem, z_mem);

  stream.wait();
}

PLATFORM_IMPL(identity, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "IDENTITY_MKLDNN OP: rank must be <= 6, but got rank = %i", input->rankOf());

  identityMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(identity, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN IDENTITY OP");
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
