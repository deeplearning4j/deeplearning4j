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
// OneDNN implementation of Layer Normalization
// Normalizes over the last dimension by default
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
static void layerNormMKLDNN(NDArray* x, NDArray* gain, NDArray* bias, NDArray* z, float epsilon) {
  const int rank = x->rankOf();
  const int lastDim = x->sizeAt(-1);

  // Get shapes
  dnnl::memory::dims xDims = *x->getShapeAsFlatVector();
  dnnl::memory::dims statsDims = xDims;
  statsDims[rank - 1] = 1;  // Stats are computed over the last dimension

  // Create memory descriptors
  dnnl::memory::desc x_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*x));
  dnnl::memory::desc z_md = dnnl::memory::desc(xDims, dnnl::memory::data_type::f32, onednnUtils::getFormat(*z));

  // Scale and shift (gain and bias) descriptors - 1D arrays of size lastDim
  dnnl::memory::dims scaleDims = {lastDim};
  dnnl::memory::desc scale_md = dnnl::memory::desc(scaleDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
  dnnl::memory::desc shift_md = dnnl::memory::desc(scaleDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // Layer norm flags
  dnnl::normalization_flags flags = dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift;

  // Create layer norm primitive descriptor (OneDNN 3.x API)
  dnnl::layer_normalization_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                                  x_md, z_md, epsilon, flags);

  std::unordered_map<int, dnnl::memory> args;
  dnnl::stream stream(engine);

  // Source memory
  dnnl::memory x_mem(x_md, engine, x->buffer());
  args[DNNL_ARG_SRC] = x_mem;

  // Destination memory
  dnnl::memory z_mem(z_md, engine, z->buffer());
  args[DNNL_ARG_DST] = z_mem;

  // Scale (gain) memory
  dnnl::memory scale_mem(scale_md, engine, gain->buffer());
  args[DNNL_ARG_SCALE] = scale_mem;

  // Shift (bias) memory
  if (bias != nullptr) {
    dnnl::memory shift_mem(shift_md, engine, bias->buffer());
    args[DNNL_ARG_SHIFT] = shift_mem;
  }

  // Execute layer normalization
  dnnl::layer_normalization_forward(op_prim_desc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(layer_norm, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto output = OUTPUT_VARIABLE(0);

  NDArray* bias = nullptr;
  if (block.width() > 2) {
    bias = INPUT_VARIABLE(2);
  }

  // Get axis from integer arguments - default is last dimension
  std::vector<sd::LongType> axis = *block.getIArguments();

  const int rank = input->rankOf();

  // Check if normalizing over last dimension only (the OneDNN-optimized case)
  bool lastDimOnly = (axis.size() == 1 && axis[0] == rank - 1) || axis.empty();

  if (!lastDimOnly) {
    // Fall back to default implementation for non-last-dimension normalization
    return sd::Status::KERNEL_FAILURE;
  }

  REQUIRE_TRUE(rank >= 2 && rank <= 6, 0,
               "LAYER_NORM_MKLDNN OP: the rank of input must be between 2 and 6, but got rank = %i instead !",
               rank);

  const float epsilon = 1e-5f;

  layerNormMKLDNN(input, gain, bias, output, epsilon);

  return sd::Status::OK;
}

PLATFORM_CHECK(layer_norm, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto gain = INPUT_VARIABLE(1);
  auto z = OUTPUT_VARIABLE(0);

  // Check if normalizing over last dimension only
  std::vector<sd::LongType> axis = *block.getIArguments();
  const int rank = x->rankOf();
  bool lastDimOnly = (axis.size() == 1 && axis[0] == rank - 1) || axis.empty();

  Requirements req("ONEDNN LAYER_NORM OP");
  req.expectTrue(block.isUseONEDNN(), IS_USE_ONEDNN_MSG) &&
      req.expectTrue(makeInfoVariable(lastDimOnly, "LAST_DIM_NORM"), "Only last dimension normalization supported") &&
      req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 1) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(gain->rankOf(), "GAIN_RANK"), 1) &&
      req.expectEq(makeInfoVariable(gain->sizeAt(0), "GAIN_SIZE"), x->sizeAt(-1));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
