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
// OneDNN implementation of split operations
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
// SPLIT: Split tensor into multiple outputs along an axis
static void splitMKLDNN(NDArray* input, std::vector<NDArray*>& outputs, int axis) {
  if (axis < 0) axis += input->rankOf();

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  dnnl::memory::dims inputDims = *input->getShapeAsFlatVector();
  dnnl::memory::desc input_md = dnnl::memory::desc(inputDims, dnnl::memory::data_type::f32,
                                                    onednnUtils::getFormat(*input));
  dnnl::memory input_mem(input_md, engine, input->buffer());

  // For each output, create a view (submemory) and copy using reorder
  LongType offset = 0;
  for (size_t i = 0; i < outputs.size(); i++) {
    NDArray* out = outputs[i];
    dnnl::memory::dims outDims = *out->getShapeAsFlatVector();
    dnnl::memory::desc out_md = dnnl::memory::desc(outDims, dnnl::memory::data_type::f32,
                                                    onednnUtils::getFormat(*out));

    // Create offset array for submemory
    dnnl::memory::dims offsets(input->rankOf(), 0);
    offsets[axis] = offset;

    // Create submemory descriptor
    auto submem_md = input_md.submemory_desc(outDims, offsets);

    // Create memory objects
    dnnl::memory submem(submem_md, engine, input->buffer());
    dnnl::memory out_mem(out_md, engine, out->buffer());

    // Reorder from submemory to output
    dnnl::reorder(submem, out_mem).execute(stream, submem, out_mem);

    offset += out->sizeAt(axis);
  }

  stream.wait();
}

PLATFORM_IMPL(split, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);

  int axis = 0;
  if (block.getIArguments()->size() > 0) {
    axis = INT_ARG(0);
  }

  if (axis < 0) axis += input->rankOf();

  std::vector<NDArray*> outputs;
  for (int i = 0; i < block.outputWidth(); i++) {
    outputs.push_back(OUTPUT_VARIABLE(i));
  }

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "SPLIT_MKLDNN OP: rank must be <= 6, but got rank = %i", input->rankOf());

  splitMKLDNN(input, outputs, axis);

  return sd::Status::OK;
}

PLATFORM_CHECK(split, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("ONEDNN SPLIT OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// SPLIT_V: Split tensor with variable sizes along an axis
PLATFORM_IMPL(split_v, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto sizeSplits = INPUT_VARIABLE(1);

  int axis = 0;
  if (block.getIArguments()->size() > 0) {
    axis = INT_ARG(0);
  }

  if (axis < 0) axis += input->rankOf();

  std::vector<NDArray*> outputs;
  for (int i = 0; i < block.outputWidth(); i++) {
    outputs.push_back(OUTPUT_VARIABLE(i));
  }

  REQUIRE_TRUE(input->rankOf() <= 6, 0, "SPLIT_V_MKLDNN OP: rank must be <= 6, but got rank = %i", input->rankOf());

  splitMKLDNN(input, outputs, axis);

  return sd::Status::OK;
}

PLATFORM_CHECK(split_v, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);

  Requirements req("ONEDNN SPLIT_V OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
