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
// OneDNN implementation of 1D convolution operations
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
static void conv1dMKLDNN(NDArray* input, NDArray* weights, NDArray* bias, NDArray* output,
                          int kW, int sW, int pW, int dW, bool isNCW, bool hasBias) {
  // Input: [N, C, W] (NCW) or [N, W, C] (NWC)
  // Weights: [oC, iC, kW]
  // Output: [N, oC, oW] (NCW) or [N, oW, oC] (NWC)

  int bS, iC, iW, oC, oW;

  if (isNCW) {
    bS = input->sizeAt(0);
    iC = input->sizeAt(1);
    iW = input->sizeAt(2);
    oC = output->sizeAt(1);
    oW = output->sizeAt(2);
  } else {
    bS = input->sizeAt(0);
    iW = input->sizeAt(1);
    iC = input->sizeAt(2);
    oW = output->sizeAt(1);
    oC = output->sizeAt(2);
  }

  // OneDNN uses NCW format internally for 1D conv
  dnnl::memory::dims srcDims = {bS, iC, iW};
  dnnl::memory::dims wDims = {oC, iC, kW};
  dnnl::memory::dims bDims = {oC};
  dnnl::memory::dims dstDims = {bS, oC, oW};
  dnnl::memory::dims strides = {sW};
  dnnl::memory::dims padding = {pW};
  dnnl::memory::dims dilation = {dW - 1};  // OneDNN uses dilation-1

  auto srcFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;
  auto dstFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;

  dnnl::memory::desc src_md = dnnl::memory::desc(srcDims, dnnl::memory::data_type::f32, srcFormat);
  dnnl::memory::desc w_md = dnnl::memory::desc(wDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oiw);
  dnnl::memory::desc dst_md = dnnl::memory::desc(dstDims, dnnl::memory::data_type::f32, dstFormat);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  std::unordered_map<int, dnnl::memory> args;

  if (hasBias && bias != nullptr) {
    dnnl::memory::desc b_md = dnnl::memory::desc(bDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

    // OneDNN 3.x API: primitive_desc takes all parameters directly
    dnnl::convolution_forward::primitive_desc op_prim_desc(engine,
                                             dnnl::prop_kind::forward_inference,
                                             dnnl::algorithm::convolution_direct,
                                             src_md, w_md, b_md, dst_md,
                                             strides, dilation, padding, padding);

    dnnl::memory src_mem(src_md, engine, input->buffer());
    args[DNNL_ARG_SRC] = src_mem;

    dnnl::memory w_mem(w_md, engine, weights->buffer());
    args[DNNL_ARG_WEIGHTS] = w_mem;

    dnnl::memory b_mem(b_md, engine, bias->buffer());
    args[DNNL_ARG_BIAS] = b_mem;

    dnnl::memory dst_mem(dst_md, engine, output->buffer());
    args[DNNL_ARG_DST] = dst_mem;

    dnnl::convolution_forward(op_prim_desc).execute(stream, args);
  } else {
    // OneDNN 3.x API: primitive_desc takes all parameters directly (no bias version)
    dnnl::convolution_forward::primitive_desc op_prim_desc(engine,
                                             dnnl::prop_kind::forward_inference,
                                             dnnl::algorithm::convolution_direct,
                                             src_md, w_md, dst_md,
                                             strides, dilation, padding, padding);

    dnnl::memory src_mem(src_md, engine, input->buffer());
    args[DNNL_ARG_SRC] = src_mem;

    dnnl::memory w_mem(w_md, engine, weights->buffer());
    args[DNNL_ARG_WEIGHTS] = w_mem;

    dnnl::memory dst_mem(dst_md, engine, output->buffer());
    args[DNNL_ARG_DST] = dst_mem;

    dnnl::convolution_forward(op_prim_desc).execute(stream, args);
  }

  stream.wait();
}

PLATFORM_IMPL(conv1d, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  NDArray* bias = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
  auto output = OUTPUT_VARIABLE(0);

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = INT_ARG(3);
  bool isNCW = block.numI() > 4 ? (INT_ARG(4) == 1) : true;

  bool hasBias = bias != nullptr;

  REQUIRE_TRUE(input->rankOf() == 3, 0, "CONV1D_MKLDNN OP: input rank must be 3, but got rank = %i",
               input->rankOf());

  conv1dMKLDNN(input, weights, bias, output, kW, sW, pW, dW, isNCW, hasBias);

  return sd::Status::OK;
}

PLATFORM_CHECK(conv1d, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto w = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN CONV1D OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(w->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 3) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(w->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// 1D Convolution Backward
static void conv1dBpMKLDNN(NDArray* input, NDArray* weights, NDArray* gradO,
                            NDArray* gradI, NDArray* gradW, NDArray* gradB,
                            int kW, int sW, int pW, int dW, bool isNCW) {
  int bS, iC, iW, oC, oW;

  if (isNCW) {
    bS = input->sizeAt(0);
    iC = input->sizeAt(1);
    iW = input->sizeAt(2);
    oC = gradO->sizeAt(1);
    oW = gradO->sizeAt(2);
  } else {
    bS = input->sizeAt(0);
    iW = input->sizeAt(1);
    iC = input->sizeAt(2);
    oW = gradO->sizeAt(1);
    oC = gradO->sizeAt(2);
  }

  dnnl::memory::dims srcDims = {bS, iC, iW};
  dnnl::memory::dims wDims = {oC, iC, kW};
  dnnl::memory::dims bDims = {oC};
  dnnl::memory::dims dstDims = {bS, oC, oW};
  dnnl::memory::dims strides = {sW};
  dnnl::memory::dims padding = {pW};
  dnnl::memory::dims dilation = {dW - 1};

  auto srcFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;
  auto dstFormat = isNCW ? dnnl::memory::format_tag::ncw : dnnl::memory::format_tag::nwc;

  dnnl::memory::desc src_md = dnnl::memory::desc(srcDims, dnnl::memory::data_type::f32, srcFormat);
  dnnl::memory::desc w_md = dnnl::memory::desc(wDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oiw);
  dnnl::memory::desc dst_md = dnnl::memory::desc(dstDims, dnnl::memory::data_type::f32, dstFormat);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Forward descriptor for hint (OneDNN 3.x API)
  dnnl::convolution_forward::primitive_desc op_ff_prim_desc(engine,
                                              dnnl::prop_kind::forward_training,
                                              dnnl::algorithm::convolution_direct,
                                              src_md, w_md, dst_md,
                                              strides, dilation, padding, padding);

  // Backward data
  if (gradI != nullptr) {
    // OneDNN 3.x API: primitive_desc takes all parameters directly
    dnnl::convolution_backward_data::primitive_desc op_data_prim_desc(engine,
                                                        dnnl::algorithm::convolution_direct,
                                                        src_md, w_md, dst_md,
                                                        strides, dilation, padding, padding,
                                                        op_ff_prim_desc);

    std::unordered_map<int, dnnl::memory> data_args;

    dnnl::memory w_mem(w_md, engine, weights->buffer());
    data_args[DNNL_ARG_WEIGHTS] = w_mem;

    dnnl::memory diff_dst_mem(dst_md, engine, gradO->buffer());
    data_args[DNNL_ARG_DIFF_DST] = diff_dst_mem;

    dnnl::memory diff_src_mem(src_md, engine, gradI->buffer());
    data_args[DNNL_ARG_DIFF_SRC] = diff_src_mem;

    dnnl::convolution_backward_data(op_data_prim_desc).execute(stream, data_args);
  }

  // Backward weights
  if (gradW != nullptr) {
    dnnl::memory::desc diff_w_md = dnnl::memory::desc(wDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oiw);

    if (gradB != nullptr) {
      dnnl::memory::desc b_md = dnnl::memory::desc(bDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

      // OneDNN 3.x API: primitive_desc takes all parameters directly
      dnnl::convolution_backward_weights::primitive_desc op_weights_prim_desc(engine,
                                                                dnnl::algorithm::convolution_direct,
                                                                src_md, diff_w_md, b_md, dst_md,
                                                                strides, dilation, padding, padding,
                                                                op_ff_prim_desc);

      std::unordered_map<int, dnnl::memory> weights_args;

      dnnl::memory src_mem(src_md, engine, input->buffer());
      weights_args[DNNL_ARG_SRC] = src_mem;

      dnnl::memory diff_dst_mem(dst_md, engine, gradO->buffer());
      weights_args[DNNL_ARG_DIFF_DST] = diff_dst_mem;

      dnnl::memory diff_w_mem(diff_w_md, engine, gradW->buffer());
      weights_args[DNNL_ARG_DIFF_WEIGHTS] = diff_w_mem;

      dnnl::memory diff_b_mem(b_md, engine, gradB->buffer());
      weights_args[DNNL_ARG_DIFF_BIAS] = diff_b_mem;

      dnnl::convolution_backward_weights(op_weights_prim_desc).execute(stream, weights_args);
    } else {
      // OneDNN 3.x API: primitive_desc takes all parameters directly (no bias version)
      dnnl::convolution_backward_weights::primitive_desc op_weights_prim_desc(engine,
                                                                dnnl::algorithm::convolution_direct,
                                                                src_md, diff_w_md, dst_md,
                                                                strides, dilation, padding, padding,
                                                                op_ff_prim_desc);

      std::unordered_map<int, dnnl::memory> weights_args;

      dnnl::memory src_mem(src_md, engine, input->buffer());
      weights_args[DNNL_ARG_SRC] = src_mem;

      dnnl::memory diff_dst_mem(dst_md, engine, gradO->buffer());
      weights_args[DNNL_ARG_DIFF_DST] = diff_dst_mem;

      dnnl::memory diff_w_mem(diff_w_md, engine, gradW->buffer());
      weights_args[DNNL_ARG_DIFF_WEIGHTS] = diff_w_mem;

      dnnl::convolution_backward_weights(op_weights_prim_desc).execute(stream, weights_args);
    }
  }

  stream.wait();
}

PLATFORM_IMPL(conv1d_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto gradO = INPUT_VARIABLE(2);

  auto gradI = OUTPUT_VARIABLE(0);
  auto gradW = OUTPUT_VARIABLE(1);
  NDArray* gradB = block.width() > 3 ? OUTPUT_VARIABLE(2) : nullptr;

  int kW = INT_ARG(0);
  int sW = INT_ARG(1);
  int pW = INT_ARG(2);
  int dW = INT_ARG(3);
  bool isNCW = block.numI() > 4 ? (INT_ARG(4) == 1) : true;

  REQUIRE_TRUE(input->rankOf() == 3, 0, "CONV1D_BP_MKLDNN OP: input rank must be 3, but got rank = %i",
               input->rankOf());

  conv1dBpMKLDNN(input, weights, gradO, gradI, gradW, gradB, kW, sW, pW, dW, isNCW);

  return sd::Status::OK;
}

PLATFORM_CHECK(conv1d_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto w = INPUT_VARIABLE(1);
  auto gradO = INPUT_VARIABLE(2);
  auto gradI = OUTPUT_VARIABLE(0);
  auto gradW = OUTPUT_VARIABLE(1);

  Requirements req("ONEDNN CONV1D_BP OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(w->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 3) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(w->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradO->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradI->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gradW->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
