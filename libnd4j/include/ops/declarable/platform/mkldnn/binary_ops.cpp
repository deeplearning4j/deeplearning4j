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
// OneDNN implementation of binary operations: add, subtract, multiply, divide
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
// Generic binary operation using OneDNN
static void binaryOpMKLDNN(NDArray* x, NDArray* y, NDArray* z, dnnl::algorithm alg) {
  // Get shapes - OneDNN requires same shapes for binary ops (broadcasting handled elsewhere)
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims yDims = *y->getShapeAsFlatVector();
  dnnl::memory::dims zDims = *z->getShapeAsFlatVector();

  // Create memory descriptors
  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc y_md = dnnl::memory::desc(yDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*y));
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Create binary primitive descriptor
  // OneDNN 3.x API: binary::primitive_desc(engine, algorithm, src0_md, src1_md, dst_md)
  dnnl::binary::primitive_desc op_prim_desc(engine, alg, x_md, y_md, z_md);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memories
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC_0] = x_mem;

  dnnl::memory y_mem(y_md, engine, y->buffer());
  args[DNNL_ARG_SRC_1] = y_mem;

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Execute binary operation
  dnnl::binary(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// Check if shapes are compatible for OneDNN binary operations
static bool shapesCompatible(NDArray* x, NDArray* y, NDArray* z) {
  // For simplest case: all shapes must match exactly
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
// ADD
PLATFORM_IMPL(add, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "ADD_MKLDNN OP: shapes must be equal for OneDNN implementation");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "ADD_MKLDNN OP: the rank must be <= 6, but got rank = %i", x->rankOf());

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_add);

  return sd::Status::OK;
}

PLATFORM_CHECK(add, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN ADD OP");
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
// SUBTRACT
PLATFORM_IMPL(subtract, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "SUBTRACT_MKLDNN OP: shapes must be equal for OneDNN implementation");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "SUBTRACT_MKLDNN OP: the rank must be <= 6, but got rank = %i", x->rankOf());

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_sub);

  return sd::Status::OK;
}

PLATFORM_CHECK(subtract, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SUBTRACT OP");
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
// MULTIPLY
PLATFORM_IMPL(multiply, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "MULTIPLY_MKLDNN OP: shapes must be equal for OneDNN implementation");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "MULTIPLY_MKLDNN OP: the rank must be <= 6, but got rank = %i", x->rankOf());

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_mul);

  return sd::Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN MULTIPLY OP");
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
// DIVIDE
PLATFORM_IMPL(divide, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "DIVIDE_MKLDNN OP: shapes must be equal for OneDNN implementation");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "DIVIDE_MKLDNN OP: the rank must be <= 6, but got rank = %i", x->rankOf());

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_div);

  return sd::Status::OK;
}

PLATFORM_CHECK(divide, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN DIVIDE OP");
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
// MAXIMUM
PLATFORM_IMPL(maximum, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "MAXIMUM_MKLDNN OP: shapes must be equal for OneDNN implementation");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "MAXIMUM_MKLDNN OP: the rank must be <= 6, but got rank = %i", x->rankOf());

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_max);

  return sd::Status::OK;
}

PLATFORM_CHECK(maximum, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN MAXIMUM OP");
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
// MINIMUM
PLATFORM_IMPL(minimum, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(shapesCompatible(x, y, z), 0, "MINIMUM_MKLDNN OP: shapes must be equal for OneDNN implementation");
  REQUIRE_TRUE(x->rankOf() <= 6, 0, "MINIMUM_MKLDNN OP: the rank must be <= 6, but got rank = %i", x->rankOf());

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_min);

  return sd::Status::OK;
}

PLATFORM_CHECK(minimum, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN MINIMUM OP");
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
