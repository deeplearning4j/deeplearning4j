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
// OneDNN implementation of comparison operations
// Uses binary primitives for element-wise comparisons
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
// Generic binary comparison using OneDNN
static void binaryComparisonMKLDNN(NDArray* x, NDArray* y, NDArray* z, dnnl::algorithm alg) {
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims yDims = *y->getShapeAsFlatVector();
  dnnl::memory::dims zDims = *z->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc y_md = dnnl::memory::desc(yDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*y));
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // OneDNN 3.x API
  dnnl::binary::primitive_desc op_prim_desc(engine, alg, x_md, y_md, z_md);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC_0] = x_mem;

  dnnl::memory y_mem(y_md, engine, y->buffer());
  args[DNNL_ARG_SRC_1] = y_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  dnnl::binary(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
static bool shapesCompatible(NDArray* x, NDArray* y, NDArray* z) {
  if (x->rankOf() != y->rankOf() || x->rankOf() != z->rankOf()) {
    return false;
  }
  for (int i = 0; i < x->rankOf(); i++) {
    if (x->sizeAt(i) != y->sizeAt(i) || x->sizeAt(i) != z->sizeAt(i)) {
      return false;
    }
  }
  return true;
}

//////////////////////////////////////////////////////////////////////
// GREATER (x > y)
PLATFORM_IMPL(greater, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "GREATER_MKLDNN OP: shapes must be equal");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "GREATER_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());

  binaryComparisonMKLDNN(x, y, z, dnnl::algorithm::binary_gt);

  return sd::Status::OK;
}

PLATFORM_CHECK(greater, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN GREATER OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(shapesCompatible(x, y, z), "SHAPES_COMPATIBLE"), "Shapes must be equal") &&
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
// GREATER_EQUAL (x >= y)
PLATFORM_IMPL(greater_equal, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "GREATER_EQUAL_MKLDNN OP: shapes must be equal");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "GREATER_EQUAL_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());

  binaryComparisonMKLDNN(x, y, z, dnnl::algorithm::binary_ge);

  return sd::Status::OK;
}

PLATFORM_CHECK(greater_equal, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN GREATER_EQUAL OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(shapesCompatible(x, y, z), "SHAPES_COMPATIBLE"), "Shapes must be equal") &&
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
// LESS (x < y)
PLATFORM_IMPL(less, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "LESS_MKLDNN OP: shapes must be equal");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "LESS_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());

  binaryComparisonMKLDNN(x, y, z, dnnl::algorithm::binary_lt);

  return sd::Status::OK;
}

PLATFORM_CHECK(less, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN LESS OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(shapesCompatible(x, y, z), "SHAPES_COMPATIBLE"), "Shapes must be equal") &&
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
// LESS_EQUAL (x <= y)
PLATFORM_IMPL(less_equal, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "LESS_EQUAL_MKLDNN OP: shapes must be equal");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "LESS_EQUAL_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());

  binaryComparisonMKLDNN(x, y, z, dnnl::algorithm::binary_le);

  return sd::Status::OK;
}

PLATFORM_CHECK(less_equal, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN LESS_EQUAL OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(shapesCompatible(x, y, z), "SHAPES_COMPATIBLE"), "Shapes must be equal") &&
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
// EQUALS (x == y)
PLATFORM_IMPL(equals, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "EQUALS_MKLDNN OP: shapes must be equal");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "EQUALS_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());

  binaryComparisonMKLDNN(x, y, z, dnnl::algorithm::binary_eq);

  return sd::Status::OK;
}

PLATFORM_CHECK(equals, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN EQUALS OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(shapesCompatible(x, y, z), "SHAPES_COMPATIBLE"), "Shapes must be equal") &&
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
// NOT_EQUALS (x != y)
PLATFORM_IMPL(not_equals, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "NOT_EQUALS_MKLDNN OP: shapes must be equal");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "NOT_EQUALS_MKLDNN OP: rank must be <= 6, but got rank = %i", x->rankOf());

  binaryComparisonMKLDNN(x, y, z, dnnl::algorithm::binary_ne);

  return sd::Status::OK;
}

PLATFORM_CHECK(not_equals, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN NOT_EQUALS OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(shapesCompatible(x, y, z), "SHAPES_COMPATIBLE"), "Shapes must be equal") &&
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
