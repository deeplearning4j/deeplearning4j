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
// OneDNN implementation of reduction operations: reduce_sum, reduce_mean, reduce_max, reduce_min
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
// Generic reduction operation using OneDNN
static void reductionMKLDNN(NDArray* x, NDArray* z, const std::vector<LongType>& axes, dnnl::algorithm alg) {
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims zDims = *z->getShapeAsFlatVector();

  // Create source memory descriptor
  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(zDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Create reduction primitive descriptor (OneDNN 3.x API)
  dnnl::reduction::primitive_desc op_prim_desc(engine, alg, x_md, z_md, 0.f, 0.f);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memory
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Execute reduction
  dnnl::reduction(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// Check if reduction is suitable for OneDNN (only reducing last dimensions continuously)
static bool isReductionSuitable(NDArray* x, NDArray* z, const std::vector<LongType>& axes) {
  // OneDNN reduction works best when output shape is compatible
  // Check that all dimensions match or are reduced to 1
  if (x->rankOf() != z->rankOf()) {
    return false;  // Keep-dims must be true for OneDNN reduction
  }

  // Verify shapes are compatible
  for (int i = 0; i < x->rankOf(); i++) {
    if (x->sizeAt(i) != z->sizeAt(i) && z->sizeAt(i) != 1) {
      return false;
    }
  }

  return true;
}

//////////////////////////////////////////////////////////////////////
// REDUCE_SUM
PLATFORM_IMPL(reduce_sum, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      axes.push_back(INT_ARG(i));
    }
  }

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "REDUCE_SUM_MKLDNN OP: the rank must be <= 6, but got rank = %i", input->rankOf());
  REQUIRE_TRUE(isReductionSuitable(input, output, axes), 0, "REDUCE_SUM_MKLDNN OP: shapes not suitable for OneDNN");

  reductionMKLDNN(input, output, axes, dnnl::algorithm::reduction_sum);

  return sd::Status::OK;
}

PLATFORM_CHECK(reduce_sum, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  }

  Requirements req("ONEDNN REDUCE_SUM OP");
  req.expectTrue(makeInfoVariable(isReductionSuitable(x, z, axes), "REDUCTION_SUITABLE"), "Reduction must keep dims") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// REDUCE_MEAN
PLATFORM_IMPL(reduce_mean, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      axes.push_back(INT_ARG(i));
    }
  }

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "REDUCE_MEAN_MKLDNN OP: the rank must be <= 6, but got rank = %i", input->rankOf());
  REQUIRE_TRUE(isReductionSuitable(input, output, axes), 0, "REDUCE_MEAN_MKLDNN OP: shapes not suitable for OneDNN");

  reductionMKLDNN(input, output, axes, dnnl::algorithm::reduction_mean);

  return sd::Status::OK;
}

PLATFORM_CHECK(reduce_mean, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  }

  // oneDNN reduction primitive can fail to create descriptors for rank-3
  // tensors when reducing a middle dimension (e.g. [1,4,16] -> [1,1,16]).
  // Restrict to rank 2, 4, 5, 6 where oneDNN is well-tested.
  // Rank 1 and rank 3 fall through to the generic CPU implementation.
  bool rankSupported = (x->rankOf() == 2 || x->rankOf() >= 4);

  Requirements req("ONEDNN REDUCE_MEAN OP");
  req.expectTrue(makeInfoVariable(rankSupported, "RANK_SUPPORTED"), "OneDNN reduce_mean supports rank 2 or >= 4") &&
      req.expectTrue(makeInfoVariable(isReductionSuitable(x, z, axes), "REDUCTION_SUITABLE"), "Reduction must keep dims") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// REDUCE_MAX
PLATFORM_IMPL(reduce_max, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      axes.push_back(INT_ARG(i));
    }
  }

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "REDUCE_MAX_MKLDNN OP: the rank must be <= 6, but got rank = %i", input->rankOf());
  REQUIRE_TRUE(isReductionSuitable(input, output, axes), 0, "REDUCE_MAX_MKLDNN OP: shapes not suitable for OneDNN");

  reductionMKLDNN(input, output, axes, dnnl::algorithm::reduction_max);

  return sd::Status::OK;
}

PLATFORM_CHECK(reduce_max, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  }

  Requirements req("ONEDNN REDUCE_MAX OP");
  req.expectTrue(makeInfoVariable(isReductionSuitable(x, z, axes), "REDUCTION_SUITABLE"), "Reduction must keep dims") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// REDUCE_MIN
PLATFORM_IMPL(reduce_min, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      axes.push_back(INT_ARG(i));
    }
  }

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "REDUCE_MIN_MKLDNN OP: the rank must be <= 6, but got rank = %i", input->rankOf());
  REQUIRE_TRUE(isReductionSuitable(input, output, axes), 0, "REDUCE_MIN_MKLDNN OP: shapes not suitable for OneDNN");

  reductionMKLDNN(input, output, axes, dnnl::algorithm::reduction_min);

  return sd::Status::OK;
}

PLATFORM_CHECK(reduce_min, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  }

  Requirements req("ONEDNN REDUCE_MIN OP");
  req.expectTrue(makeInfoVariable(isReductionSuitable(x, z, axes), "REDUCTION_SUITABLE"), "Reduction must keep dims") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// REDUCE_PROD - using multiplication reduction
PLATFORM_IMPL(reduce_prod, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      axes.push_back(INT_ARG(i));
    }
  }

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "REDUCE_PROD_MKLDNN OP: the rank must be <= 6, but got rank = %i", input->rankOf());
  REQUIRE_TRUE(isReductionSuitable(input, output, axes), 0, "REDUCE_PROD_MKLDNN OP: shapes not suitable for OneDNN");

  reductionMKLDNN(input, output, axes, dnnl::algorithm::reduction_mul);

  return sd::Status::OK;
}

PLATFORM_CHECK(reduce_prod, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  std::vector<LongType> axes;
  if (block.width() > 1) {
    auto axesArr = INPUT_VARIABLE(1);
    for (sd::LongType i = 0; i < axesArr->lengthOf(); i++) {
      axes.push_back(axesArr->e<LongType>(i));
    }
  }

  Requirements req("ONEDNN REDUCE_PROD OP");
  req.expectTrue(makeInfoVariable(isReductionSuitable(x, z, axes), "REDUCTION_SUITABLE"), "Reduction must keep dims") &&
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
