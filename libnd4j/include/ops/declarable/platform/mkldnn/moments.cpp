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
// OneDNN implementation of statistical moments (mean, variance)
//

#include <helpers/MKLDNNStream.h>
#include <array/NDArrayFactory.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
// Compute variance using: var = E[x^2] - E[x]^2
static void varianceMKLDNN(NDArray* x, NDArray* z, const std::vector<LongType>& axes, bool keepDims) {
  if (x->rankOf() != z->rankOf()) {
    return;  // Need keepDims for OneDNN reduction
  }

  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims zDims = *z->getShapeAsFlatVector();

  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // Step 1: Calculate mean (E[x])
  // OneDNN 3.x API: reduction::primitive_desc(engine, algorithm, src_md, dst_md, p, eps)
  dnnl::reduction::primitive_desc mean_prim_desc(engine, dnnl::algorithm::reduction_mean, x_md, z_md, 0.f, 0.f);

  dnnl::memory mean_mem(z_md, engine);

  std::unordered_map<int, dnnl::memory> mean_args;
  dnnl::memory x_mem(x_md, engine, x->buffer());
  mean_args[DNNL_ARG_SRC] = x_mem;
  mean_args[DNNL_ARG_DST] = mean_mem;

  dnnl::reduction(mean_prim_desc).execute(stream, mean_args);

  // Step 2: Calculate x^2
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc sq_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_square, x_md, x_md, 0.f, 0.f);

  dnnl::memory x_sq_mem(x_md, engine);

  std::unordered_map<int, dnnl::memory> sq_args;
  sq_args[DNNL_ARG_SRC] = x_mem;
  sq_args[DNNL_ARG_DST] = x_sq_mem;

  dnnl::eltwise_forward(sq_prim_desc).execute(stream, sq_args);

  // Step 3: Calculate E[x^2]
  dnnl::memory mean_sq_mem(z_md, engine);

  std::unordered_map<int, dnnl::memory> mean_sq_args;
  mean_sq_args[DNNL_ARG_SRC] = x_sq_mem;
  mean_sq_args[DNNL_ARG_DST] = mean_sq_mem;

  dnnl::reduction(mean_prim_desc).execute(stream, mean_sq_args);

  // Step 4: Calculate mean^2
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc mean_sq_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                           algorithm::eltwise_square, z_md, z_md, 0.f, 0.f);

  dnnl::memory mean_squared_mem(z_md, engine);

  std::unordered_map<int, dnnl::memory> mean_sq2_args;
  mean_sq2_args[DNNL_ARG_SRC] = mean_mem;
  mean_sq2_args[DNNL_ARG_DST] = mean_squared_mem;

  dnnl::eltwise_forward(mean_sq_prim_desc).execute(stream, mean_sq2_args);

  // Step 5: var = E[x^2] - mean^2 using binary sub
  // OneDNN 3.x API: binary::primitive_desc(engine, algorithm, src0_md, src1_md, dst_md)
  dnnl::binary::primitive_desc sub_prim_desc(engine, dnnl::algorithm::binary_sub, z_md, z_md, z_md);

  std::unordered_map<int, dnnl::memory> sub_args;
  sub_args[DNNL_ARG_SRC_0] = mean_sq_mem;
  sub_args[DNNL_ARG_SRC_1] = mean_squared_mem;

  dnnl::memory z_mem(z_md, engine, z->buffer());
  sub_args[DNNL_ARG_DST] = z_mem;

  dnnl::binary(sub_prim_desc).execute(stream, sub_args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
static bool isVarianceSuitable(NDArray* x, NDArray* z) {
  return x->rankOf() == z->rankOf();
}

PLATFORM_IMPL(reduce_variance, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  }

  bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "REDUCE_VARIANCE_MKLDNN OP: rank must be <= 6, but got rank = %i",
               input->rankOf());
  REQUIRE_TRUE(isVarianceSuitable(input, output), 0,
               "REDUCE_VARIANCE_MKLDNN OP: keepDims must be true for OneDNN");

  varianceMKLDNN(input, output, axes, keepDims);

  return sd::Status::OK;
}

PLATFORM_CHECK(reduce_variance, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN REDUCE_VARIANCE OP");
  req.expectTrue(makeInfoVariable(isVarianceSuitable(x, z), "KEEP_DIMS"), "keepDims must be true") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// REDUCE_STDEV: sqrt(variance)
PLATFORM_IMPL(reduce_stdev, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  }

  bool keepDims = block.numB() > 0 ? B_ARG(0) : false;

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "REDUCE_STDEV_MKLDNN OP: rank must be <= 6, but got rank = %i",
               input->rankOf());
  REQUIRE_TRUE(isVarianceSuitable(input, output), 0,
               "REDUCE_STDEV_MKLDNN OP: keepDims must be true for OneDNN");

  // First compute variance
  auto tempVar = NDArrayFactory::create_<float>(output->ordering(), *output->getShapeAsVector(),
                                                 LaunchContext::defaultContext());
  varianceMKLDNN(input, tempVar, axes, keepDims);

  // Then sqrt
  dnnl::memory::dims shape = *tempVar->getShapeAsFlatVector();
  dnnl::memory::desc md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*tempVar));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::primitive_attr attr;
  dnnl::stream stream(engine);

  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc sqrt_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                        algorithm::eltwise_sqrt, md, md, 0.f, 0.f);

  std::unordered_map<int, dnnl::memory> sqrt_args;
  dnnl::memory var_mem(md, engine, tempVar->buffer());
  sqrt_args[DNNL_ARG_SRC] = var_mem;

  dnnl::memory out_mem(md, engine, output->buffer());
  sqrt_args[DNNL_ARG_DST] = out_mem;

  dnnl::eltwise_forward(sqrt_prim_desc).execute(stream, sqrt_args);

  stream.wait();
  delete tempVar;

  return sd::Status::OK;
}

PLATFORM_CHECK(reduce_stdev, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN REDUCE_STDEV OP");
  req.expectTrue(makeInfoVariable(isVarianceSuitable(x, z), "KEEP_DIMS"), "keepDims must be true") &&
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
