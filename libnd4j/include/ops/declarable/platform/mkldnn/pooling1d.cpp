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
// OneDNN implementation of 1D pooling operations (max pooling and average pooling)
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
static void pooling1dMKLDNN(NDArray* input, NDArray* output,
                             int kW, int sW, int pW, int dW,
                             bool isNCW, dnnl::algorithm alg) {
  // Input: [N, C, W] (NCW) or [N, W, C] (NWC)
  int bS, iC, iW, oW;

  if (isNCW) {
    bS = input->sizeAt(0);
    iC = input->sizeAt(1);
    iW = input->sizeAt(2);
    oW = output->sizeAt(2);
  } else {
    bS = input->sizeAt(0);
    iW = input->sizeAt(1);
    iC = input->sizeAt(2);
    oW = output->sizeAt(1);
  }

  dnnl::memory::dims srcDims = {bS, iC, iW};
  dnnl::memory::dims dstDims = {bS, iC, oW};
  dnnl::memory::dims kernel = {kW};
  dnnl::memory::dims strides = {sW};
  dnnl::memory::dims padding = {pW};
  dnnl::memory::dims dilation = {dW - 1};

  auto srcFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;
  auto dstFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;

  dnnl::memory::desc src_md = dnnl::memory::desc(srcDims, dnnl::memory::data_type::f32, srcFormat);
  dnnl::memory::desc dst_md = dnnl::memory::desc(dstDims, dnnl::memory::data_type::f32, dstFormat);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // OneDNN 3.x API
  dnnl::pooling_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference, alg,
                                                      src_md, dst_md, strides, kernel, dilation, padding, padding);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::memory src_mem(src_md, engine, input->buffer());
  args[DNNL_ARG_SRC] = src_mem;

  dnnl::memory dst_mem(dst_md, engine, output->buffer());
  args[DNNL_ARG_DST] = dst_mem;

  dnnl::pooling_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// MAXPOOL1D
PLATFORM_IMPL(maxpool1d, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = block.numI() > 3 ? INT_ARG(3) : 1;
  bool isNCW = block.numI() > 4 ? (INT_ARG(4) == 1) : true;

  REQUIRE_TRUE(input->rankOf() == 3, 0, "MAXPOOL1D_MKLDNN OP: input rank must be 3, but got rank = %i",
               input->rankOf());

  pooling1dMKLDNN(input, output, kW, sW, pW, dW, isNCW, dnnl::algorithm::pooling_max);

  return sd::Status::OK;
}

PLATFORM_CHECK(maxpool1d, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN MAXPOOL1D OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 3) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// AVGPOOL1D
PLATFORM_IMPL(avgpool1d, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = block.numI() > 3 ? INT_ARG(3) : 1;
  bool isNCW = block.numI() > 4 ? (INT_ARG(4) == 1) : true;
  bool countIncludePad = block.numI() > 5 ? (INT_ARG(5) == 1) : false;

  REQUIRE_TRUE(input->rankOf() == 3, 0, "AVGPOOL1D_MKLDNN OP: input rank must be 3, but got rank = %i",
               input->rankOf());

  auto alg = countIncludePad ? dnnl::algorithm::pooling_avg_include_padding
                              : dnnl::algorithm::pooling_avg_exclude_padding;

  pooling1dMKLDNN(input, output, kW, sW, pW, dW, isNCW, alg);

  return sd::Status::OK;
}

PLATFORM_CHECK(avgpool1d, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN AVGPOOL1D OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 3) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// Pooling 1D Backward
static void pooling1dBpMKLDNN(NDArray* input, NDArray* gradO, NDArray* gradI,
                               int kW, int sW, int pW, int dW,
                               bool isNCW, dnnl::algorithm alg) {
  int bS, iC, iW, oW;

  if (isNCW) {
    bS = input->sizeAt(0);
    iC = input->sizeAt(1);
    iW = input->sizeAt(2);
    oW = gradO->sizeAt(2);
  } else {
    bS = input->sizeAt(0);
    iW = input->sizeAt(1);
    iC = input->sizeAt(2);
    oW = gradO->sizeAt(1);
  }

  dnnl::memory::dims srcDims = {bS, iC, iW};
  dnnl::memory::dims dstDims = {bS, iC, oW};
  dnnl::memory::dims kernel = {kW};
  dnnl::memory::dims strides = {sW};
  dnnl::memory::dims padding = {pW};

  auto srcFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;
  auto dstFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;

  dnnl::memory::desc src_md = dnnl::memory::desc(srcDims, dnnl::memory::data_type::f32, srcFormat);
  dnnl::memory::desc dst_md = dnnl::memory::desc(dstDims, dnnl::memory::data_type::f32, dstFormat);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Forward descriptor for hint (OneDNN 3.x API)
  dnnl::memory::dims dilation = {0};  // No dilation for backward
  dnnl::pooling_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training, alg,
                                                         src_md, dst_md, strides, kernel, dilation, padding, padding);

  // Backward descriptor (OneDNN 3.x API)
  dnnl::pooling_backward::primitive_desc op_prim_desc(engine, alg, src_md, dst_md, strides, kernel,
                                                       dilation, padding, padding, op_ff_prim_desc);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::memory diff_dst_mem(dst_md, engine, gradO->buffer());
  args[DNNL_ARG_DIFF_DST] = diff_dst_mem;

  dnnl::memory diff_src_mem(src_md, engine, gradI->buffer());
  args[DNNL_ARG_DIFF_SRC] = diff_src_mem;

  // For max pooling, we need workspace from forward pass
  if (alg == dnnl::algorithm::pooling_max) {
    // Execute forward to get workspace
    dnnl::memory src_mem(src_md, engine, input->buffer());
    dnnl::memory dst_mem(dst_md, engine);

    std::unordered_map<int, dnnl::memory> ff_args;
    ff_args[DNNL_ARG_SRC] = src_mem;
    ff_args[DNNL_ARG_DST] = dst_mem;

    if (op_ff_prim_desc.workspace_desc().get_size() > 0) {
      dnnl::memory workspace(op_ff_prim_desc.workspace_desc(), engine);
      ff_args[DNNL_ARG_WORKSPACE] = workspace;

      dnnl::pooling_forward(op_ff_prim_desc).execute(stream, ff_args);

      args[DNNL_ARG_WORKSPACE] = workspace;
    }
  }

  dnnl::pooling_backward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(maxpool1d_bp, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = block.numI() > 3 ? INT_ARG(3) : 1;
  bool isNCW = block.numI() > 4 ? (INT_ARG(4) == 1) : true;

  REQUIRE_TRUE(input->rankOf() == 3, 0, "MAXPOOL1D_BP_MKLDNN OP: input rank must be 3, but got rank = %i",
               input->rankOf());

  pooling1dBpMKLDNN(input, gradO, gradI, kW, sW, pW, dW, isNCW, dnnl::algorithm::pooling_max);

  return sd::Status::OK;
}

PLATFORM_CHECK(maxpool1d_bp, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN MAXPOOL1D_BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 3) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

PLATFORM_IMPL(avgpool1d_bp, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = block.numI() > 3 ? INT_ARG(3) : 1;
  bool isNCW = block.numI() > 4 ? (INT_ARG(4) == 1) : true;
  bool countIncludePad = block.numI() > 5 ? (INT_ARG(5) == 1) : false;

  REQUIRE_TRUE(input->rankOf() == 3, 0, "AVGPOOL1D_BP_MKLDNN OP: input rank must be 3, but got rank = %i",
               input->rankOf());

  auto alg = countIncludePad ? dnnl::algorithm::pooling_avg_include_padding
                              : dnnl::algorithm::pooling_avg_exclude_padding;

  pooling1dBpMKLDNN(input, gradO, gradI, kW, sW, pW, dW, isNCW, alg);

  return sd::Status::OK;
}

PLATFORM_CHECK(avgpool1d_bp, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto gradO = INPUT_VARIABLE(1);
  auto gradI = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN AVGPOOL1D_BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 3) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
