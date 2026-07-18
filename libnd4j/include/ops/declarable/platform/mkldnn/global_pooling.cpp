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
// OneDNN implementation of global pooling operations (global average/max pooling)
// These pool across the entire spatial dimensions
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
// Global pooling using OneDNN pooling with kernel size = input spatial size
static void globalPoolingMKLDNN(NDArray* x, NDArray* z, dnnl::algorithm alg, bool isNCHW) {
  auto xRank = x->rankOf();

  // Expected format: [N, C, H, W] for 4D or [N, C, D, H, W] for 5D (NCHW/NCDHW)
  // or [N, H, W, C] for 4D or [N, D, H, W, C] for 5D (NHWC/NDHWC)

  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims zDims = *z->getShapeAsFlatVector();

  // Determine spatial dimensions based on format
  dnnl::memory::dims kernel, strides, padding;

  if (xRank == 4) {
    // 2D global pooling
    int hIdx = isNCHW ? 2 : 1;
    int wIdx = isNCHW ? 3 : 2;
    kernel = {x->sizeAt(hIdx), x->sizeAt(wIdx)};
    strides = {1, 1};
    padding = {0, 0};
  } else if (xRank == 5) {
    // 3D global pooling
    int dIdx = isNCHW ? 2 : 1;
    int hIdx = isNCHW ? 3 : 2;
    int wIdx = isNCHW ? 4 : 3;
    kernel = {x->sizeAt(dIdx), x->sizeAt(hIdx), x->sizeAt(wIdx)};
    strides = {1, 1, 1};
    padding = {0, 0, 0};
  } else {
    return;  // Unsupported rank
  }

  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Create pooling descriptor (OneDNN 3.x API)
  // For global pooling, dilation is typically not used, so we pass zeros
  dnnl::memory::dims dilation(kernel.size(), 0);
  dnnl::pooling_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference, alg,
                                                      x_md, z_md, strides, kernel, dilation, padding, padding);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memory
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Execute pooling
  dnnl::pooling_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// GLOBAL_MAXPOOL2D - note: maxpool2d already exists, this adds global variant support
PLATFORM_IMPL(global_max_pooling_2d, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // Determine format
  bool isNCHW = true;  // Default, can be overridden by argument

  REQUIRE_TRUE(input->rankOf() == 4, 0, "GLOBAL_MAXPOOL2D_MKLDNN OP: input rank must be 4, but got rank = %i",
               input->rankOf());

  globalPoolingMKLDNN(input, output, dnnl::algorithm::pooling_max, isNCHW);

  return sd::Status::OK;
}

PLATFORM_CHECK(global_max_pooling_2d, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN GLOBAL_MAXPOOL2D OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 4) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// GLOBAL_AVGPOOL2D
PLATFORM_IMPL(global_avg_pooling_2d, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  bool isNCHW = true;  // Default

  REQUIRE_TRUE(input->rankOf() == 4, 0, "GLOBAL_AVGPOOL2D_MKLDNN OP: input rank must be 4, but got rank = %i",
               input->rankOf());

  globalPoolingMKLDNN(input, output, dnnl::algorithm::pooling_avg_exclude_padding, isNCHW);

  return sd::Status::OK;
}

PLATFORM_CHECK(global_avg_pooling_2d, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN GLOBAL_AVGPOOL2D OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 4) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
