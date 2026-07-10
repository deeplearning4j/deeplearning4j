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
// OneDNN implementation of resampling operations (bilinear/nearest neighbor interpolation)
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
static void resampleMKLDNN(NDArray* x, NDArray* z, dnnl::algorithm alg) {
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims zDims = *z->getShapeAsFlatVector();

  // Create memory descriptors
  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Create resampling primitive descriptor (OneDNN 3.x API)
  dnnl::resampling_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference, alg, x_md, z_md);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memory
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Execute resampling
  dnnl::resampling_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
static void resampleBpMKLDNN(NDArray* x, NDArray* dLdz, NDArray* dLdx, dnnl::algorithm alg) {
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims dLdzDims = *dLdz->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc dLdz_md = dnnl::memory::desc(dLdzDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdz));
  dnnl::memory::desc dLdx_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdx));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Forward descriptor for hint (OneDNN 3.x API)
  dnnl::resampling_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training, alg, x_md, dLdz_md);

  // Backward descriptor (OneDNN 3.x API)
  dnnl::resampling_backward::primitive_desc op_prim_desc(engine, alg, dLdz_md, dLdx_md, op_ff_prim_desc);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Diff destination (gradient from output)
  dnnl::memory dLdz_mem(dLdz_md, engine, dLdz->buffer());
  args[DNNL_ARG_DIFF_DST] = dLdz_mem;

  // Diff source (gradient to input)
  dnnl::memory dLdx_mem(dLdx_md, engine, dLdx->buffer());
  args[DNNL_ARG_DIFF_SRC] = dLdx_mem;

  // Execute backward
  dnnl::resampling_backward(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// RESIZE_BILINEAR
PLATFORM_IMPL(resize_bilinear, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() >= 3 && input->rankOf() <= 5, 0,
               "RESIZE_BILINEAR_MKLDNN OP: rank must be 3, 4, or 5, but got rank = %i", input->rankOf());

  resampleMKLDNN(input, output, dnnl::algorithm::resampling_linear);

  return sd::Status::OK;
}

PLATFORM_CHECK(resize_bilinear, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN RESIZE_BILINEAR OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectGreaterEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 3) &&
      req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 5) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(x->sizeAt(0), "INPUT_BATCH_DIM"), makeInfoVariable(z->sizeAt(0), "OUTPUT_BATCH_DIM")) &&
      // OneDNN always interprets rank-4 tensors as NCHW (via abcd format tag), so the channel
      // is always at dim 1.  For NHWC input [N,H,W,C]→[N,H',W',C] the oneDNN "channel" (dim 1)
      // is H≠H', causing primitive descriptor creation to fail.  Checking sizeAt(1) equality
      // correctly rejects NHWC inputs (H≠H') and accepts NCHW inputs (C==C).
      req.expectEq(makeInfoVariable(x->sizeAt(1), "INPUT_CHANNEL_DIM"), makeInfoVariable(z->sizeAt(1), "OUTPUT_CHANNEL_DIM")) &&
      // OneDNN resampling does not support input spatial dimensions of 1; fall back to generic impl
      req.expectGreater(makeInfoVariable(x->sizeAt(1), "INPUT_HEIGHT_DIM"), (sd::LongType)1) &&
      req.expectGreater(makeInfoVariable(x->sizeAt(2), "INPUT_WIDTH_DIM"), (sd::LongType)1);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// RESIZE_NEAREST_NEIGHBOR
PLATFORM_IMPL(resize_nearest_neighbor, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->rankOf() >= 3 && input->rankOf() <= 5, 0,
               "RESIZE_NEAREST_MKLDNN OP: rank must be 3, 4, or 5, but got rank = %i", input->rankOf());

  resampleMKLDNN(input, output, dnnl::algorithm::resampling_nearest);

  return sd::Status::OK;
}

PLATFORM_CHECK(resize_nearest_neighbor, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN RESIZE_NEAREST OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectGreaterEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 3) &&
      req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 5) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(x->sizeAt(0), "INPUT_BATCH_DIM"), makeInfoVariable(z->sizeAt(0), "OUTPUT_BATCH_DIM")) &&
      req.expectEq(makeInfoVariable(x->sizeAt(-1), "INPUT_CHANNEL_DIM"), makeInfoVariable(z->sizeAt(-1), "OUTPUT_CHANNEL_DIM"));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
