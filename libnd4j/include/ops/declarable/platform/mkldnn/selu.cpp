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
// OneDNN implementation of SELU (Scaled Exponential Linear Unit)
// SELU(x) = scale * (max(0,x) + min(0, alpha*(exp(x)-1)))
// where scale = 1.0507009873554804934193349852946 and alpha = 1.6732632423543772848170429916717
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

// SELU constants
static constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
static constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

//////////////////////////////////////////////////////////////////////
static void seluMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  dnnl::primitive_attr attr;

  // SELU: scale * elu(x, alpha)
  // Using eltwise_elu with alpha parameter, then scaling
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_elu, x_mkl_md, x_mkl_md, SELU_ALPHA, 0.f);

  // Create intermediate result for ELU
  auto elu_md = op_prim_desc.dst_desc();
  dnnl::memory elu_mem(elu_md, engine);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  args[DNNL_ARG_DST] = elu_mem;

  // Execute ELU
  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  // Now apply scaling using linear: y = scale * x + 0
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc scale_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                         algorithm::eltwise_linear, elu_md, z_mkl_md, SELU_SCALE, 0.f);

  std::unordered_map<int, dnnl::memory> scale_args;
  scale_args[DNNL_ARG_SRC] = elu_mem;

  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, scale_prim_desc.dst_desc(), scale_args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(scale_prim_desc).execute(stream, scale_args);

  if (scale_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(scale_args[DNNL_ARG_DST], z_user_mem).execute(stream, scale_args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(selu, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "SELU_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  seluMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(selu, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SELU OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
static void seluBpMKLDNN(NDArray* x, NDArray* dLdz, NDArray* dLdx) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, dLdx_mkl_md, dLdx_user_md, dLdz_mkl_md, dLdz_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  dLdz_user_md = dLdz_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdz));
  onednnUtils::setBlockStrides(*dLdz, dLdz_user_md);

  dLdx_user_md = dLdx_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdx));
  onednnUtils::setBlockStrides(*dLdx, dLdx_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // SELU backward: d/dx[selu(x)] = scale * d/dx[elu(x, alpha)]
  // For ELU: d/dx = 1 if x > 0, alpha * exp(x) if x <= 0
  // For SELU: multiply by scale

  // OneDNN 3.x API for forward hint: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training,
                                                         algorithm::eltwise_elu, x_mkl_md, x_mkl_md, SELU_ALPHA, 0.f);

  // OneDNN 3.x API for backward: primitive_desc(engine, algorithm, diff_src_md, diff_dst_md, data_md, alpha, beta, hint_fwd_pd)
  dnnl::eltwise_backward::primitive_desc op_prim_desc(engine, algorithm::eltwise_elu,
                                                       dLdx_mkl_md, dLdz_mkl_md, x_mkl_md, SELU_ALPHA, 0.f, op_ff_prim_desc);

  // Create intermediate for ELU gradient
  auto elu_grad_md = op_prim_desc.diff_src_desc();
  dnnl::memory elu_grad_mem(elu_grad_md, engine);

  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  onednnUtils::loadDataToMklStream(*dLdz, engine, stream, dLdz_user_md, op_prim_desc.diff_dst_desc(),
                                   args[DNNL_ARG_DIFF_DST]);

  args[DNNL_ARG_DIFF_SRC] = elu_grad_mem;

  dnnl::eltwise_backward(op_prim_desc).execute(stream, args);

  // Scale the gradient by SELU_SCALE
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc scale_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                         algorithm::eltwise_linear, elu_grad_md, dLdx_mkl_md, SELU_SCALE, 0.f);

  std::unordered_map<int, dnnl::memory> scale_args;
  scale_args[DNNL_ARG_SRC] = elu_grad_mem;

  auto dLdx_user_mem = onednnUtils::loadDataToMklStream(*dLdx, engine, stream, dLdx_user_md,
                                                        scale_prim_desc.dst_desc(), scale_args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(scale_prim_desc).execute(stream, scale_args);

  if (scale_prim_desc.dst_desc() != dLdx_user_mem.get_desc())
    dnnl::reorder(scale_args[DNNL_ARG_DST], dLdx_user_mem).execute(stream, scale_args[DNNL_ARG_DST], dLdx_user_mem);

  stream.wait();
}

PLATFORM_IMPL(selu_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  const sd::LongType dLdzRank = dLdz->rankOf();

  REQUIRE_TRUE(rank <= 6 && dLdzRank <= 6, 0,
               "SELU_BP_MKLDNN OP: the rank of input and dLdz must be less or equal 6");

  seluBpMKLDNN(input, dLdz, dLdx);

  return sd::Status::OK;
}

PLATFORM_CHECK(selu_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SELU BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
