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
// OneDNN implementation of thresholded ReLU: x if x > theta, 0 otherwise
// Implemented using eltwise_relu with alpha=theta
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "mkldnnEltwise.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
static void thresholdedReluMKLDNN(NDArray* x, NDArray* z, float theta) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  auto dataType = onednnUtils::toDnnlDataType(x->dataType());

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dataType, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dataType, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  dnnl::eltwise_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_relu, x_mkl_md, z_mkl_md, theta, 0.f);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(thresholdedrelu, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "THRESHOLDEDRELU_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  float theta = block.numT() > 0 ? static_cast<float>(T_ARG(0)) : 1.0f;

  thresholdedReluMKLDNN(input, output, theta);

  return sd::Status::OK;
}

PLATFORM_CHECK(thresholdedrelu, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN THRESHOLDEDRELU OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectTrue(makeInfoVariable(onednnUtils::isSupportedEltwiseType(x->dataType()), TYPE_MSG_INPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(onednnUtils::isSupportedEltwiseType(z->dataType()), TYPE_MSG_OUTPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT),
                   makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
static void thresholdedReluBpMKLDNN(NDArray* x, NDArray* dLdz, NDArray* dLdx, float theta) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  auto dataType = onednnUtils::toDnnlDataType(x->dataType());

  dnnl::memory::desc x_mkl_md, x_user_md, dLdx_mkl_md, dLdx_user_md, dLdz_mkl_md, dLdz_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dataType, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  dLdz_user_md = dLdz_mkl_md = dnnl::memory::desc(shape, dataType, onednnUtils::getFormat(*dLdz));
  onednnUtils::setBlockStrides(*dLdz, dLdz_user_md);

  dLdx_user_md = dLdx_mkl_md = dnnl::memory::desc(shape, dataType, onednnUtils::getFormat(*dLdx));
  onednnUtils::setBlockStrides(*dLdx, dLdx_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  dnnl::eltwise_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training,
                                                         algorithm::eltwise_relu, x_mkl_md, x_mkl_md, theta, 0.f);

  dnnl::eltwise_backward::primitive_desc op_prim_desc(engine, algorithm::eltwise_relu,
                                                       dLdx_mkl_md, dLdz_mkl_md, x_mkl_md, theta, 0.f, op_ff_prim_desc);

  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  onednnUtils::loadDataToMklStream(*dLdz, engine, stream, dLdz_user_md, op_prim_desc.diff_dst_desc(),
                                   args[DNNL_ARG_DIFF_DST]);

  auto dLdx_user_mem = onednnUtils::loadDataToMklStream(*dLdx, engine, stream, dLdx_user_md,
                                                        op_prim_desc.diff_src_desc(), args[DNNL_ARG_DIFF_SRC]);

  dnnl::eltwise_backward(op_prim_desc).execute(stream, args);

  if (op_prim_desc.diff_src_desc() != dLdx_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_SRC], dLdx_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], dLdx_user_mem);

  stream.wait();
}

PLATFORM_IMPL(thresholdedrelu_bp, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  const sd::LongType dLdzRank = dLdz->rankOf();

  REQUIRE_TRUE(rank <= 6 && dLdzRank <= 6, 0,
               "THRESHOLDEDRELU_BP_MKLDNN OP: the rank of input and dLdz must be less or equal 6");

  float theta = block.numT() > 0 ? static_cast<float>(T_ARG(0)) : 1.0f;

  thresholdedReluBpMKLDNN(input, dLdz, dLdx, theta);

  return sd::Status::OK;
}

PLATFORM_CHECK(thresholdedrelu_bp, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN THRESHOLDEDRELU BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectTrue(makeInfoVariable(onednnUtils::isSupportedEltwiseType(x->dataType()), TYPE_MSG_INPUT0),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(onednnUtils::isSupportedEltwiseType(dLdz->dataType()), TYPE_MSG_INPUT1),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(onednnUtils::isSupportedEltwiseType(dLdx->dataType()), TYPE_MSG_OUTPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1)) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
