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
// OneDNN implementation of multi-tensor sum: z = x1 + x2 + ... + xn
// Useful for residual connections and similar patterns
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
// Sum multiple tensors using OneDNN sum primitive
static void multiSumMKLDNN(std::vector<NDArray*>& inputs, NDArray* z) {
  if (inputs.empty()) return;

  dnnl::memory::dims shape = *inputs[0]->getShapeAsFlatVector();

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // Create memory descriptors for all inputs
  std::vector<dnnl::memory::desc> src_mds;
  std::vector<float> scales;

  for (auto* input : inputs) {
    src_mds.push_back(dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*input)));
    scales.push_back(1.0f);  // All inputs weighted equally
  }

  // Output memory descriptor
  dnnl::memory::desc z_md = dnnl::memory::desc(shape, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  // Create sum primitive descriptor - OneDNN 3.x API: engine first
  dnnl::sum::primitive_desc op_prim_desc(engine, z_md, scales, src_mds);

  std::unordered_map<int, dnnl::memory> args;

  // Add all source memories
  for (size_t i = 0; i < inputs.size(); i++) {
    dnnl::memory src_mem(src_mds[i], engine, inputs[i]->buffer());
    args[DNNL_ARG_MULTIPLE_SRC + i] = src_mem;
  }

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Execute sum
  dnnl::sum(op_prim_desc).execute(stream, args);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
// Check if all input shapes match
static bool allShapesMatch(std::vector<NDArray*>& inputs) {
  if (inputs.size() < 2) return true;

  auto* first = inputs[0];
  for (size_t i = 1; i < inputs.size(); i++) {
    if (first->rankOf() != inputs[i]->rankOf()) return false;
    for (int j = 0; j < first->rankOf(); j++) {
      if (first->sizeAt(j) != inputs[i]->sizeAt(j)) return false;
    }
  }
  return true;
}

PLATFORM_IMPL(mergeadd, ENGINE_CPU) {
  std::vector<NDArray*> inputs;
  for (int i = 0; i < block.width(); i++) {
    inputs.push_back(INPUT_VARIABLE(i));
  }

  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(!inputs.empty(), 0, "MERGEADD_MKLDNN OP: at least one input required");
  REQUIRE_TRUE(allShapesMatch(inputs), 0, "MERGEADD_MKLDNN OP: all input shapes must match");
  REQUIRE_TRUE(inputs[0]->rankOf() <= 6, 0, "MERGEADD_MKLDNN OP: rank must be <= 6, but got rank = %i",
               inputs[0]->rankOf());

  multiSumMKLDNN(inputs, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(mergeadd, ENGINE_CPU) {
  std::vector<NDArray*> inputs;
  for (int i = 0; i < block.width(); i++) {
    inputs.push_back(INPUT_VARIABLE(i));
  }
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN MERGEADD OP");
  req.expectTrue(makeInfoVariable(!inputs.empty(), "HAS_INPUTS"), "At least one input required") &&
      req.expectTrue(makeInfoVariable(allShapesMatch(inputs), "SHAPES_MATCH"), "All shapes must match");

  if (!inputs.empty()) {
    req.expectFalse(makeInfoVariable(inputs[0]->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
        req.expectLess(makeInfoVariable(inputs[0]->rankOf(), RANK_MSG_INPUT), 7) &&
        req.expectGreater(makeInfoVariable(inputs[0]->rankOf(), RANK_MSG_INPUT), 0) &&
        req.expectEq(makeInfoVariable(inputs[0]->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
        req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);

    // Check all inputs are FLOAT32
    for (size_t i = 1; i < inputs.size(); i++) {
      req.expectEq(makeInfoVariable(inputs[i]->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32);
    }
  }

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
