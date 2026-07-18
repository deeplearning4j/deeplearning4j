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
// OneDNN implementation of channel shuffle operation
// Used in ShuffleNet and similar architectures
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
static void shuffleChannelMKLDNN(NDArray* x, NDArray* z, int groups, int axis) {
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Create shuffle descriptor
  // OneDNN shuffle operates on the specified axis with given group_size
  int axisSize = x->sizeAt(axis);
  int groupSize = axisSize / groups;

  // OneDNN 3.x API - requires both src_desc and dst_desc
  dnnl::shuffle_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference, x_md, z_md, axis, groupSize);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memory
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Execute shuffle
  dnnl::shuffle_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(shuffle_channel, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int groups = INT_ARG(0);
  int axis = block.numI() > 1 ? INT_ARG(1) : 1;  // Default channel axis is 1

  REQUIRE_TRUE(input->rankOf() >= 2 && input->rankOf() <= 6, 0,
               "SHUFFLE_CHANNEL_MKLDNN OP: input rank must be 2-6, but got rank = %i", input->rankOf());
  REQUIRE_TRUE(input->sizeAt(axis) % groups == 0, 0,
               "SHUFFLE_CHANNEL_MKLDNN OP: channel dimension must be divisible by groups");

  shuffleChannelMKLDNN(input, output, groups, axis);

  return sd::Status::OK;
}

PLATFORM_CHECK(shuffle_channel, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  int groups = INT_ARG(0);
  int axis = block.numI() > 1 ? INT_ARG(1) : 1;

  bool divisible = x->sizeAt(axis) % groups == 0;

  Requirements req("ONEDNN SHUFFLE_CHANNEL OP");
  req.expectTrue(makeInfoVariable(divisible, "DIVISIBLE_BY_GROUPS"), "Channels must be divisible by groups") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectGreaterEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 2) &&
      req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 6) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
static void shuffleChannelBpMKLDNN(NDArray* dLdz, NDArray* dLdx, int groups, int axis) {
  dnnl::memory::dims dims = *dLdz->getShapeAsFlatVector();

  dnnl::memory::desc dLdz_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdz));
  dnnl::memory::desc dLdx_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdx));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  int axisSize = dLdz->sizeAt(axis);
  int groupSize = axisSize / groups;

  // Forward descriptor for hint (OneDNN 3.x API) - requires both src_desc and dst_desc
  dnnl::shuffle_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training, dLdx_md, dLdz_md, axis, groupSize);

  // Backward descriptor (OneDNN 3.x API) - requires diff_src_desc and diff_dst_desc
  dnnl::shuffle_backward::primitive_desc op_prim_desc(engine, dLdx_md, dLdz_md, axis, groupSize, op_ff_prim_desc);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Diff destination
  dnnl::memory dLdz_mem(dLdz_md, engine, dLdz->buffer());
  args[DNNL_ARG_DIFF_DST] = dLdz_mem;

  // Diff source
  dnnl::memory dLdx_mem(dLdx_md, engine, dLdx->buffer());
  args[DNNL_ARG_DIFF_SRC] = dLdx_mem;

  // Execute backward
  dnnl::shuffle_backward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(shuffle_channel_bp, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  int groups = INT_ARG(0);
  int axis = block.numI() > 1 ? INT_ARG(1) : 1;

  REQUIRE_TRUE(dLdz->rankOf() >= 2 && dLdz->rankOf() <= 6, 0,
               "SHUFFLE_CHANNEL_BP_MKLDNN OP: rank must be 2-6, but got rank = %i", dLdz->rankOf());

  shuffleChannelBpMKLDNN(dLdz, dLdx, groups, axis);

  return sd::Status::OK;
}

PLATFORM_CHECK(shuffle_channel_bp, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  int groups = INT_ARG(0);
  int axis = block.numI() > 1 ? INT_ARG(1) : 1;

  bool divisible = x->sizeAt(axis) % groups == 0;

  Requirements req("ONEDNN SHUFFLE_CHANNEL_BP OP");
  req.expectTrue(makeInfoVariable(divisible, "DIVISIBLE_BY_GROUPS"), "Channels must be divisible by groups") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectGreaterEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 2) &&
      req.expectLessEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 6) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
