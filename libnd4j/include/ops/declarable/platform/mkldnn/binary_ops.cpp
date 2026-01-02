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
// Supports broadcasting and multiple data types
//

#include <helpers/MKLDNNStream.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/Environment.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
// Get OneDNN data type from NDArray
static dnnl::memory::data_type getOneDnnDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32:
      return dnnl::memory::data_type::f32;
    case DataType::BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case DataType::HALF:
      return dnnl::memory::data_type::f16;
    default:
      return dnnl::memory::data_type::f32;
  }
}

//////////////////////////////////////////////////////////////////////
// Check if data type is supported by OneDNN binary ops
static bool isSupportedType(DataType dt) {
  return dt == DataType::FLOAT32 || dt == DataType::BFLOAT16 || dt == DataType::HALF;
}

//////////////////////////////////////////////////////////////////////
// Create broadcast-compatible dimensions for OneDNN
// OneDNN requires dimensions to be padded to match for broadcasting
static dnnl::memory::dims getBroadcastDims(NDArray* arr, int targetRank) {
  dnnl::memory::dims dims(targetRank, 1);
  int arrRank = arr->rankOf();
  int offset = targetRank - arrRank;

  for (int i = 0; i < arrRank; i++) {
    dims[offset + i] = arr->sizeAt(i);
  }
  return dims;
}

//////////////////////////////////////////////////////////////////////
// Get plain format tag based on rank
static dnnl::memory::format_tag getPlainFormat(int rank) {
  switch (rank) {
    case 1: return dnnl::memory::format_tag::a;
    case 2: return dnnl::memory::format_tag::ab;
    case 3: return dnnl::memory::format_tag::abc;
    case 4: return dnnl::memory::format_tag::abcd;
    case 5: return dnnl::memory::format_tag::abcde;
    case 6: return dnnl::memory::format_tag::abcdef;
    default: return dnnl::memory::format_tag::a;
  }
}

//////////////////////////////////////////////////////////////////////
// Generic binary operation using OneDNN with broadcasting support
static void binaryOpMKLDNN(NDArray* x, NDArray* y, NDArray* z, dnnl::algorithm alg) {
  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  auto dType = getOneDnnDataType(z->dataType());

  // Get output dimensions (result of broadcast)
  dnnl::memory::dims zDims = *z->getShapeAsFlatVector();
  int maxRank = z->rankOf();

  // Create broadcast-compatible dimensions
  dnnl::memory::dims xDims = getBroadcastDims(x, maxRank);
  dnnl::memory::dims yDims = getBroadcastDims(y, maxRank);

  // Create memory descriptors - use plain format for broadcasting
  auto plainFormat = getPlainFormat(maxRank);
  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dType, plainFormat);
  dnnl::memory::desc y_md = dnnl::memory::desc(yDims, dType, plainFormat);
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dType, plainFormat);

  // Create binary primitive descriptor
  dnnl::binary::primitive_desc op_prim_desc(engine, alg, x_md, y_md, z_md);

  std::unordered_map<int, dnnl::memory> args;

  // Handle case where input needs to be broadcast - may need to reshape view
  dnnl::memory x_mem(x_md, engine, const_cast<void*>(x->buffer()));
  dnnl::memory y_mem(y_md, engine, const_cast<void*>(y->buffer()));
  dnnl::memory z_mem(z_md, engine, z->buffer());

  args[DNNL_ARG_SRC_0] = x_mem;
  args[DNNL_ARG_SRC_1] = y_mem;
  args[DNNL_ARG_DST] = z_mem;

  // Execute binary operation
  dnnl::binary(op_prim_desc).execute(stream, args);
  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// Check if shapes are broadcast-compatible
static bool canBroadcast(NDArray* x, NDArray* y, NDArray* z) {
  // Output shape should match the broadcast result
  auto xShape = x->shapeOf();
  auto yShape = y->shapeOf();
  auto zShape = z->shapeOf();

  int xRank = x->rankOf();
  int yRank = y->rankOf();
  int zRank = z->rankOf();

  // z rank should be max of x and y ranks
  if (zRank != std::max(xRank, yRank)) {
    return false;
  }

  // Check each dimension from the right
  for (int i = 0; i < zRank; i++) {
    int xIdx = xRank - 1 - i;
    int yIdx = yRank - 1 - i;
    int zIdx = zRank - 1 - i;

    sd::LongType xDim = (xIdx >= 0) ? xShape[xIdx] : 1;
    sd::LongType yDim = (yIdx >= 0) ? yShape[yIdx] : 1;
    sd::LongType zDim = zShape[zIdx];

    // Check broadcast compatibility
    if (xDim != yDim && xDim != 1 && yDim != 1) {
      return false;
    }

    // Check output matches broadcast result
    sd::LongType expectedDim = std::max(xDim, yDim);
    if (zDim != expectedDim) {
      return false;
    }
  }

  return true;
}

//////////////////////////////////////////////////////////////////////
// Common check for binary operations
static Requirements binaryOpCheck(const char* opName, graph::Context& block,
                                   NDArray* x, NDArray* y, NDArray* z) {
  Requirements req(opName);
  req.expectTrue(makeInfoVariable(sd::Environment::getInstance().helpersAllowed(), "HELPERS_ALLOWED"), EXPECTED_TRUE) &&
      req.expectTrue(makeInfoVariable(canBroadcast(x, y, z), "BROADCAST_COMPATIBLE"),
                     "Shapes must be broadcast-compatible") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(y->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLessEq(makeInfoVariable(z->rankOf(), RANK_MSG_OUTPUT), 6) &&
      req.expectGreater(makeInfoVariable(z->rankOf(), RANK_MSG_OUTPUT), 0) &&
      // Require contiguous (EWS=1) arrays for correct memory layout
      req.expectEq(makeInfoVariable(x->ews(), "X_EWS"), 1) &&
      req.expectEq(makeInfoVariable(y->ews(), "Y_EWS"), 1) &&
      req.expectEq(makeInfoVariable(z->ews(), "Z_EWS"), 1) &&
      req.expectTrue(makeInfoVariable(isSupportedType(x->dataType()), TYPE_MSG_INPUT0),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(isSupportedType(y->dataType()), TYPE_MSG_INPUT1),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(isSupportedType(z->dataType()), TYPE_MSG_OUTPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      // All types should match for OneDNN binary ops
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT)) &&
      req.expectEq(makeInfoVariable(y->dataType(), TYPE_MSG_INPUT1),
                   makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// ADD
PLATFORM_IMPL(add, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_add);
  return sd::Status::OK;
}

PLATFORM_CHECK(add, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  return binaryOpCheck("ONEDNN ADD OP", block, x, y, z);
}

//////////////////////////////////////////////////////////////////////
// SUBTRACT
PLATFORM_IMPL(subtract, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_sub);
  return sd::Status::OK;
}

PLATFORM_CHECK(subtract, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  return binaryOpCheck("ONEDNN SUBTRACT OP", block, x, y, z);
}

//////////////////////////////////////////////////////////////////////
// MULTIPLY
PLATFORM_IMPL(multiply, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_mul);
  return sd::Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  return binaryOpCheck("ONEDNN MULTIPLY OP", block, x, y, z);
}

//////////////////////////////////////////////////////////////////////
// DIVIDE
PLATFORM_IMPL(divide, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  binaryOpMKLDNN(x, y, z, dnnl::algorithm::binary_div);
  return sd::Status::OK;
}

PLATFORM_CHECK(divide, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);
  return binaryOpCheck("ONEDNN DIVIDE OP", block, x, y, z);
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
