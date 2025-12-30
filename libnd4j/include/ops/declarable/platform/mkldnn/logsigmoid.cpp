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
// OneDNN implementation of log sigmoid: log(sigmoid(x)) = -softplus(-x) = x - softplus(x)
// Using the identity: log(sigmoid(x)) = x - log(1 + exp(x))
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
static void logSigmoidMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // Step 1: Compute -x using eltwise_linear with alpha=-1, beta=0
  dnnl::eltwise_forward::desc neg_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear, x_mkl_md, -1.0f, 0.0f);
  dnnl::eltwise_forward::primitive_desc neg_prim_desc(neg_desc, attr, engine);

  dnnl::memory neg_x_mem(neg_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> neg_args;
  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, neg_prim_desc.src_desc(), neg_args[DNNL_ARG_SRC]);
  neg_args[DNNL_ARG_DST] = neg_x_mem;

  dnnl::eltwise_forward(neg_prim_desc).execute(stream, neg_args);

  // Step 2: Compute softplus(-x) = log(1 + exp(-x))
  dnnl::eltwise_forward::desc softplus_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_soft_relu, neg_prim_desc.dst_desc(), 0, 0);
  dnnl::eltwise_forward::primitive_desc softplus_prim_desc(softplus_desc, attr, engine);

  dnnl::memory softplus_mem(softplus_prim_desc.dst_desc(), engine);

  std::unordered_map<int, dnnl::memory> softplus_args;
  softplus_args[DNNL_ARG_SRC] = neg_x_mem;
  softplus_args[DNNL_ARG_DST] = softplus_mem;

  dnnl::eltwise_forward(softplus_prim_desc).execute(stream, softplus_args);

  // Step 3: Compute -softplus(-x) to get log(sigmoid(x))
  dnnl::eltwise_forward::desc neg2_desc(dnnl::prop_kind::forward_inference, algorithm::eltwise_linear, softplus_prim_desc.dst_desc(), -1.0f, 0.0f);
  dnnl::eltwise_forward::primitive_desc neg2_prim_desc(neg2_desc, attr, engine);

  std::unordered_map<int, dnnl::memory> neg2_args;
  neg2_args[DNNL_ARG_SRC] = softplus_mem;

  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, neg2_prim_desc.dst_desc(), neg2_args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(neg2_prim_desc).execute(stream, neg2_args);

  if (neg2_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(neg2_args[DNNL_ARG_DST], z_user_mem).execute(stream, neg2_args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(logsigmoid, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "LOGSIGMOID_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  logSigmoidMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(logsigmoid, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN LOGSIGMOID OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
