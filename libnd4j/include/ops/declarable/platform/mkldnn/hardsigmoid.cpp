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
static void hardsigmoidMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  // z
  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Create attributes
  dnnl::primitive_attr attr;

  // operation primitive description - hardsigmoid: max(0, min(1, alpha*x + beta))
  // Standard hardsigmoid: alpha = 0.2, beta = 0.5
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_hardsigmoid, x_mkl_md, z_mkl_md, 0.2f, 0.5f);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // provide memory buffers and check whether reorder is required
  // input
  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  // z
  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  // run calculations
  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  // reorder outputs if necessary
  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(hardsigmoid, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "HARDSIGMOID_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  hardsigmoidMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(hardsigmoid, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN HARDSIGMOID OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
static void hardsigmoidBpMKLDNN(NDArray* x, NDArray* dLdz, NDArray* dLdx) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, dLdx_mkl_md, dLdx_user_md, dLdz_mkl_md, dLdz_user_md;

  // x
  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  // dLdz
  dLdz_user_md = dLdz_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdz));
  onednnUtils::setBlockStrides(*dLdz, dLdz_user_md);

  // dLdx
  dLdx_user_md = dLdx_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*dLdx));
  onednnUtils::setBlockStrides(*dLdx, dLdx_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // operation primitive description
  // OneDNN 3.x API for forward hint: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training,
                                                         algorithm::eltwise_hardsigmoid, x_mkl_md, x_mkl_md, 0.2f, 0.5f);

  // OneDNN 3.x API for backward: primitive_desc(engine, algorithm, diff_src_md, diff_dst_md, data_md, alpha, beta, hint_fwd_pd)
  dnnl::eltwise_backward::primitive_desc op_prim_desc(engine, algorithm::eltwise_hardsigmoid,
                                                       dLdx_mkl_md, dLdz_mkl_md, x_mkl_md, 0.2f, 0.5f, op_ff_prim_desc);

  // provide memory buffers and check whether reorder is required for forward
  // input
  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  // dLdz
  onednnUtils::loadDataToMklStream(*dLdz, engine, stream, dLdz_user_md, op_prim_desc.diff_dst_desc(),
                                   args[DNNL_ARG_DIFF_DST]);

  // dLdx
  auto dLdx_user_mem = onednnUtils::loadDataToMklStream(*dLdx, engine, stream, dLdx_user_md,
                                                        op_prim_desc.diff_src_desc(), args[DNNL_ARG_DIFF_SRC]);

  // run calculations backward
  dnnl::eltwise_backward(op_prim_desc).execute(stream, args);

  // reorder outputs if necessary
  if (op_prim_desc.diff_src_desc() != dLdx_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_SRC], dLdx_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], dLdx_user_mem);

  stream.wait();
}

PLATFORM_IMPL(hardsigmoid_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  const sd::LongType dLdzRank = dLdz->rankOf();

  REQUIRE_TRUE(rank <= 6 && dLdzRank <= 6, 0,
               "HARDSIGMOID_BP_MKLDNN OP: the rank of input and dLdz must be less or equal 6, but got input rank = %i and dLdz "
               "rank = %i instead !",
               rank, dLdzRank);

  hardsigmoidBpMKLDNN(input, dLdz, dLdx);

  return sd::Status::OK;
}

PLATFORM_CHECK(hardsigmoid_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto dLdx = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN HARDSIGMOID BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(dLdx->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expect(
          makeShapeInfoVariable(x, SHAPE_MSG_INPUT0), makeShapeInfoVariable(dLdz, SHAPE_MSG_INPUT1),
          [](const decltype(x)& l, const decltype(dLdz)& r) {
            for (int i = 0; i < l->rankOf(); i++) {
              if (l->sizeAt(i) != r->sizeAt(i)) {
                return false;
              }
            }
            return true;
          },
          EXPECTED_EQ_MSG);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
