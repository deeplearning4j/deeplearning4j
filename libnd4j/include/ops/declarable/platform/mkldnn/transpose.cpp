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
// OneDNN implementation of transpose/permute operations
// Uses reorder primitive for efficient memory layout transformations
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
static void transposeMKLDNN(NDArray* x, NDArray* z, const std::vector<LongType>& permutation) {
  auto xRank = x->rankOf();

  // Build source and destination dimensions
  dnnl::memory::dims srcDims = x->getShapeAsFlatVector();
  dnnl::memory::dims dstDims(xRank);

  for (int i = 0; i < xRank; i++) {
    dstDims[i] = srcDims[permutation[i]];
  }

  // Create memory descriptors
  // Source uses standard format
  dnnl::memory::desc src_md = dnnl::memory::desc(srcDims, dnnl::memory::data_type::f32,
                                                  onednnUtils::getFormat(*x));

  // Destination uses standard format based on output shape
  dnnl::memory::desc dst_md = dnnl::memory::desc(dstDims, dnnl::memory::data_type::f32,
                                                  onednnUtils::getFormat(*z));

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Create source memory
  dnnl::memory src_mem(src_md, engine, x->buffer());

  // Create destination memory
  dnnl::memory dst_mem(dst_md, engine, z->buffer());

  // Create reorder primitive with permutation
  // For transpose, we need to create a memory descriptor with permuted strides
  dnnl::memory::dims srcStrides(xRank);
  dnnl::memory::dims dstStrides(xRank);

  // Calculate source strides
  srcStrides[xRank - 1] = 1;
  for (int i = xRank - 2; i >= 0; i--) {
    srcStrides[i] = srcStrides[i + 1] * srcDims[i + 1];
  }

  // Calculate permuted strides for the source view
  dnnl::memory::dims permutedStrides(xRank);
  for (int i = 0; i < xRank; i++) {
    permutedStrides[i] = srcStrides[permutation[i]];
  }

  // Create source descriptor with permuted strides (this is our "transposed view")
  dnnl::memory::desc src_permuted_md = dnnl::memory::desc(dstDims, dnnl::memory::data_type::f32, permutedStrides);

  // Create source memory with permuted view
  dnnl::memory src_permuted_mem(src_permuted_md, engine, x->buffer());

  // Reorder from permuted source to contiguous destination
  dnnl::reorder reorder_prim(src_permuted_mem, dst_mem);
  reorder_prim.execute(stream, src_permuted_mem, dst_mem);

  stream.wait();
}

PLATFORM_IMPL(transpose, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto xRank = input->rankOf();
  REQUIRE_TRUE(xRank <= 6, 0, "TRANSPOSE_MKLDNN OP: rank must be <= 6, but got rank = %i", xRank);

  // Get permutation
  std::vector<LongType> permutation;
  if (block.width() > 1) {
    auto permArr = INPUT_VARIABLE(1);
    for (LongType i = 0; i < permArr->lengthOf(); i++) {
      permutation.push_back(permArr->e<LongType>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      permutation.push_back(INT_ARG(i));
    }
  } else {
    // Default: reverse dimensions
    for (int i = xRank - 1; i >= 0; i--) {
      permutation.push_back(i);
    }
  }

  transposeMKLDNN(input, output, permutation);

  return sd::Status::OK;
}

PLATFORM_CHECK(transpose, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN TRANSPOSE OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// PERMUTE - similar to transpose but with explicit permutation order
PLATFORM_IMPL(permute, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  auto xRank = input->rankOf();
  REQUIRE_TRUE(xRank <= 6, 0, "PERMUTE_MKLDNN OP: rank must be <= 6, but got rank = %i", xRank);

  // Get permutation from arguments
  std::vector<LongType> permutation;
  if (block.width() > 1) {
    auto permArr = INPUT_VARIABLE(1);
    for (LongType i = 0; i < permArr->lengthOf(); i++) {
      permutation.push_back(permArr->e<LongType>(i));
    }
  } else {
    for (int i = 0; i < block.numI(); i++) {
      permutation.push_back(INT_ARG(i));
    }
  }

  REQUIRE_TRUE(permutation.size() == xRank, 0, "PERMUTE_MKLDNN OP: permutation size must equal rank");

  transposeMKLDNN(input, output, permutation);

  return sd::Status::OK;
}

PLATFORM_CHECK(permute, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN PERMUTE OP");
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
