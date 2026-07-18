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
// OneDNN implementation of reshape, squeeze, unsqueeze, flatten operations
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
// RESHAPE: Reshape tensor to new shape (using reorder for contiguous copy)
static void reshapeMKLDNN(NDArray* input, NDArray* output) {
  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // For reshape, we simply copy data as the underlying buffer layout doesn't change
  // Use OneDNN reorder for efficient memory copy
  dnnl::memory::dims inputDims = *input->getShapeAsFlatVector();
  dnnl::memory::dims outputDims = *output->getShapeAsFlatVector();

  // Flatten both to 1D for simple copy
  dnnl::memory::dims flatDims = {static_cast<dnnl::memory::dim>(input->lengthOf())};

  dnnl::memory::desc flat_md = dnnl::memory::desc(flatDims, dnnl::memory::data_type::f32,
                                                   dnnl::memory::format_tag::a);

  dnnl::memory in_mem(flat_md, engine, input->buffer());
  dnnl::memory out_mem(flat_md, engine, output->buffer());

  dnnl::reorder(in_mem, out_mem).execute(stream, in_mem, out_mem);

  stream.wait();
}

PLATFORM_IMPL(reshape, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // DSP view path: output shares input's buffer with reshaped shape info.
  if (input->dataBuffer() == output->dataBuffer()) return sd::Status::OK;

  REQUIRE_TRUE(input->lengthOf() == output->lengthOf(), 0,
               "RESHAPE_MKLDNN OP: input and output must have same number of elements");

  reshapeMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(reshape, ENGINE_ONEDNN) {
  // Disable OneDNN for reshape - the generic implementation has a fast path
  // for same-buffer views that avoids copying entirely, which is faster than
  // OneDNN reorder. Reshape should ideally be a zero-cost view change.
  Requirements req("ONEDNN RESHAPE OP");
  req.expectFalse(makeInfoVariable(true, "DISABLED"), "OneDNN reshape disabled - generic has view optimization");
  return req;
}

//////////////////////////////////////////////////////////////////////
// SQUEEZE: Remove dimensions of size 1
PLATFORM_IMPL(squeeze, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // DSP view path: output shares input's buffer with squeezed shape info.
  if (input->dataBuffer() == output->dataBuffer()) return sd::Status::OK;

  REQUIRE_TRUE(input->lengthOf() == output->lengthOf(), 0,
               "SQUEEZE_MKLDNN OP: input and output must have same number of elements");

  reshapeMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(squeeze, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SQUEEZE OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// UNSQUEEZE / EXPAND_DIMS: Add dimension of size 1
PLATFORM_IMPL(expand_dims, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->lengthOf() == output->lengthOf(), 0,
               "EXPAND_DIMS_MKLDNN OP: input and output must have same number of elements");

  reshapeMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(expand_dims, ENGINE_ONEDNN) {
  // Disable OneDNN for expand_dims - the generic implementation can use views
  // to avoid copying entirely, which is faster than OneDNN reorder.
  Requirements req("ONEDNN EXPAND_DIMS OP");
  req.expectFalse(makeInfoVariable(true, "DISABLED"), "OneDNN expand_dims disabled - generic has view optimization");
  return req;
}

//////////////////////////////////////////////////////////////////////
// FLATTEN: Flatten tensor to 1D or 2D
PLATFORM_IMPL(flatten, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->lengthOf() == output->lengthOf(), 0,
               "FLATTEN_MKLDNN OP: input and output must have same number of elements");

  reshapeMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(flatten, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN FLATTEN OP");
  // MKLDNN flatten only handles single-input case; multi-input flatten must use generic implementation
  req.expectTrue(makeInfoVariable(block.width() == 1, "SINGLE_INPUT"), EXPECTED_TRUE) &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// FLATTEN_2D: Flatten tensor to 2D
PLATFORM_IMPL(flatten_2d, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  REQUIRE_TRUE(input->lengthOf() == output->lengthOf(), 0,
               "FLATTEN_2D_MKLDNN OP: input and output must have same number of elements");

  reshapeMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(flatten_2d, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN FLATTEN_2D OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
