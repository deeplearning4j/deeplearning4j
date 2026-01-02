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
// OneDNN implementation of power function: x^alpha (with scalar exponent)
// Note: This is for scalar exponent only. For element-wise power, use the binary pow op.
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
static void powScalarMKLDNN(NDArray* x, NDArray* z, float alpha, float beta, float scale) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  dnnl::primitive_attr attr;

  // pow: scale * (alpha * x + beta)^exponent
  // For simple x^exponent: scale=1, alpha=1, beta=0
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_pow, x_mkl_md, z_mkl_md, alpha, beta);

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

// Note: Pow is a binary op with two array inputs - OneDNN binary operations handle this differently
// This implementation is for the scalar power case

PLATFORM_IMPL(Pow, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);  // exponent array
  auto z = OUTPUT_VARIABLE(0);

  // Only handle scalar exponent case with OneDNN
  if (!y->isScalar()) {
    // Fall back to default implementation for non-scalar case
    return sd::Status::KERNEL_FAILURE;
  }

  const sd::LongType rank = x->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "POW_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  float exponent = y->e<float>(0);
  powScalarMKLDNN(x, z, exponent, 0.0f, 1.0f);

  return sd::Status::OK;
}

PLATFORM_CHECK(Pow, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto y = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN POW OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(y->isScalar(), "SCALAR_EXPONENT"), "Exponent must be scalar for OneDNN") &&
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
