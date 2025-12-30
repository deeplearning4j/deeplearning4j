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
// OneDNN implementation of dense/fully-connected layer using inner_product primitive
// Computes: y = x * W^T + b (or y = x * W^T if no bias)
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
static void denseLayerMKLDNN(NDArray* x, NDArray* weights, NDArray* bias, NDArray* z, bool hasBias) {
  // x: [batch, in_features] or [batch, ..., in_features]
  // weights: [out_features, in_features]
  // bias: [out_features]
  // z: [batch, out_features] or [batch, ..., out_features]

  auto xRank = x->rankOf();
  auto batch = x->sizeAt(0);
  auto inFeatures = x->sizeAt(-1);
  auto outFeatures = weights->sizeAt(0);

  // Flatten x if needed (for multi-dimensional input)
  LongType flattenedBatch = 1;
  for (int i = 0; i < xRank - 1; i++) {
    flattenedBatch *= x->sizeAt(i);
  }

  dnnl::memory::dims xDims = {flattenedBatch, inFeatures};
  dnnl::memory::dims wDims = {outFeatures, inFeatures};
  dnnl::memory::dims bDims = {outFeatures};
  dnnl::memory::dims zDims = {flattenedBatch, outFeatures};

  // Memory descriptors
  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);
  dnnl::memory::desc w_md = dnnl::memory::desc(wDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oi);
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nc);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  std::unordered_map<int, dnnl::memory> args;

  if (hasBias && bias != nullptr) {
    dnnl::memory::desc b_md = dnnl::memory::desc(bDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

    dnnl::inner_product_forward::desc op_desc(dnnl::prop_kind::forward_inference, x_md, w_md, b_md, z_md);
    dnnl::inner_product_forward::primitive_desc op_prim_desc(op_desc, engine);

    // Source memory
    dnnl::memory x_mem(x_md, engine, x->buffer());
    args[DNNL_ARG_SRC] = x_mem;

    // Weights memory
    dnnl::memory w_mem(w_md, engine, weights->buffer());
    args[DNNL_ARG_WEIGHTS] = w_mem;

    // Bias memory
    dnnl::memory b_mem(b_md, engine, bias->buffer());
    args[DNNL_ARG_BIAS] = b_mem;

    // Destination memory
    dnnl::memory z_mem(z_md, engine, z->buffer());
    args[DNNL_ARG_DST] = z_mem;

    dnnl::inner_product_forward(op_prim_desc).execute(stream, args);
  } else {
    dnnl::inner_product_forward::desc op_desc(dnnl::prop_kind::forward_inference, x_md, w_md, z_md);
    dnnl::inner_product_forward::primitive_desc op_prim_desc(op_desc, engine);

    // Source memory
    dnnl::memory x_mem(x_md, engine, x->buffer());
    args[DNNL_ARG_SRC] = x_mem;

    // Weights memory
    dnnl::memory w_mem(w_md, engine, weights->buffer());
    args[DNNL_ARG_WEIGHTS] = w_mem;

    // Destination memory
    dnnl::memory z_mem(z_md, engine, z->buffer());
    args[DNNL_ARG_DST] = z_mem;

    dnnl::inner_product_forward(op_prim_desc).execute(stream, args);
  }

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(dense, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  bool hasBias = bias != nullptr;

  REQUIRE_TRUE(input->rankOf() >= 2, 0, "DENSE_MKLDNN OP: input rank must be >= 2, but got rank = %i",
               input->rankOf());
  REQUIRE_TRUE(weights->rankOf() == 2, 0, "DENSE_MKLDNN OP: weights rank must be 2, but got rank = %i",
               weights->rankOf());

  denseLayerMKLDNN(input, weights, bias, output, hasBias);

  return sd::Status::OK;
}

PLATFORM_CHECK(dense, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto w = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN DENSE OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(w->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectGreaterEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 2) &&
      req.expectEq(makeInfoVariable(w->rankOf(), RANK_MSG_INPUT1), 2) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(w->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
