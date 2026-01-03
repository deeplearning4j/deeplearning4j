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
// OneDNN implementation of PReLU (Parametric ReLU)
// f(x) = x if x > 0, alpha * x otherwise
// where alpha is a learned parameter (can be per-channel)
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
static void preluMKLDNN(NDArray* x, NDArray* alpha, NDArray* z) {
  // Get dimensions
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims alphaDims = *alpha->getShapeAsFlatVector();

  // For OneDNN PReLU, alpha needs to be broadcastable to x
  // Create memory descriptors
  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  // Alpha memory descriptor - needs to match prelu weight format
  // For per-channel PReLU, alpha shape should be [1, C, 1, 1, ...] or just [C]
  dnnl::memory::desc alpha_md = dnnl::memory::desc(alphaDims, dnnl::memory::data_type::f32,
                                                    onednnUtils::getFormat(*alpha));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Create PReLU primitive descriptor (OneDNN 3.x API) - requires dst_desc
  dnnl::prelu_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference, x_md, alpha_md, z_md);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memory
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  // Weights (alpha) memory
  dnnl::memory alpha_mem(alpha_md, engine, alpha->buffer());
  args[DNNL_ARG_WEIGHTS] = alpha_mem;

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Execute PReLU
  dnnl::prelu_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// Check if alpha shape is compatible with OneDNN PReLU
static bool isPreluCompatible(NDArray* x, NDArray* alpha) {
  // OneDNN PReLU supports:
  // 1. Scalar alpha (single value applied to all)
  // 2. Per-channel alpha where alpha broadcasts correctly

  if (alpha->lengthOf() == 1) {
    return true;  // Scalar case
  }

  // For multi-element alpha, check if it can broadcast
  // The alpha should have shape that broadcasts with x
  auto xRank = x->rankOf();
  auto alphaRank = alpha->rankOf();

  if (alphaRank > xRank) {
    return false;
  }

  // Check trailing dimensions match or are 1
  for (int i = 0; i < alphaRank; i++) {
    auto xDim = x->sizeAt(xRank - alphaRank + i);
    auto alphaDim = alpha->sizeAt(i);
    if (alphaDim != xDim && alphaDim != 1 && xDim != 1) {
      return false;
    }
  }

  return true;
}

PLATFORM_IMPL(prelu, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto alpha = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "PRELU_MKLDNN OP: the rank of input must be <= 6, but got rank = %i", rank);
  REQUIRE_TRUE(isPreluCompatible(input, alpha), 0, "PRELU_MKLDNN OP: alpha shape not compatible for OneDNN");

  preluMKLDNN(input, alpha, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(prelu, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto alpha = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN PRELU OP");
  req.expectTrue(makeInfoVariable(isPreluCompatible(x, alpha), "PRELU_COMPATIBLE"), "Alpha shape must be compatible") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(alpha->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(alpha->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
static void preluBpMKLDNN(NDArray* x, NDArray* alpha, NDArray* dLdz, NDArray* dLdx, NDArray* dLdAlpha) {
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims alphaDims = *alpha->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc alpha_md = dnnl::memory::desc(alphaDims, dnnl::memory::data_type::f32,
                                                    onednnUtils::getFormat(*alpha));
  dnnl::memory::desc dLdz_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdz));
  dnnl::memory::desc dLdx_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdx));
  dnnl::memory::desc dLdAlpha_md = dnnl::memory::desc(alphaDims, dnnl::memory::data_type::f32,
                                                       onednnUtils::getFormat(*dLdAlpha));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Forward output has same shape as input
  dnnl::memory::desc dst_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));

  // Forward descriptor needed for backward (OneDNN 3.x API) - requires dst_desc
  dnnl::prelu_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training, x_md, alpha_md, dst_md);

  // Backward descriptor (OneDNN 3.x API)
  dnnl::prelu_backward::primitive_desc op_prim_desc(engine, x_md, alpha_md, dLdz_md, dLdx_md, dLdAlpha_md, op_ff_prim_desc);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memories
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  dnnl::memory alpha_mem(alpha_md, engine, alpha->buffer());
  args[DNNL_ARG_WEIGHTS] = alpha_mem;

  dnnl::memory dLdz_mem(dLdz_md, engine, dLdz->buffer());
  args[DNNL_ARG_DIFF_DST] = dLdz_mem;

  // Destination memories
  dnnl::memory dLdx_mem(dLdx_md, engine, dLdx->buffer());
  args[DNNL_ARG_DIFF_SRC] = dLdx_mem;

  dnnl::memory dLdAlpha_mem(dLdAlpha_md, engine, dLdAlpha->buffer());
  args[DNNL_ARG_DIFF_WEIGHTS] = dLdAlpha_mem;

  // Execute backward
  dnnl::prelu_backward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(prelu_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto alpha = INPUT_VARIABLE(1);
  auto dLdz = INPUT_VARIABLE(2);
  auto dLdx = OUTPUT_VARIABLE(0);
  auto dLdAlpha = OUTPUT_VARIABLE(1);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "PRELU_BP_MKLDNN OP: the rank of input must be <= 6, but got rank = %i", rank);
  REQUIRE_TRUE(isPreluCompatible(input, alpha), 0, "PRELU_BP_MKLDNN OP: alpha shape not compatible for OneDNN");

  preluBpMKLDNN(input, alpha, dLdz, dLdx, dLdAlpha);

  return sd::Status::OK;
}

PLATFORM_CHECK(prelu_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto alpha = INPUT_VARIABLE(1);
  auto dLdz = INPUT_VARIABLE(2);
  auto dLdx = OUTPUT_VARIABLE(0);
  auto dLdAlpha = OUTPUT_VARIABLE(1);

  Requirements req("ONEDNN PRELU BP OP");
  req.expectTrue(makeInfoVariable(isPreluCompatible(x, alpha), "PRELU_COMPATIBLE"), "Alpha shape must be compatible") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(alpha->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(alpha->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdAlpha->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
