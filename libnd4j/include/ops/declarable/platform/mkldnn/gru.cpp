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
// OneDNN implementation of GRU (Gated Recurrent Unit) operations
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
static void gruMKLDNN(NDArray* x, NDArray* hI, NDArray* Wx, NDArray* Wh, NDArray* b,
                      NDArray* h, NDArray* hL, int seqLen, int batchSize, int inputSize, int hiddenSize) {
  // x: [seqLen, batchSize, inputSize]
  // hI: [1, batchSize, hiddenSize] - initial hidden state
  // Wx: [1, 3*hiddenSize, inputSize] - input weights
  // Wh: [1, 3*hiddenSize, hiddenSize] - recurrent weights
  // b: [1, 3*hiddenSize] - biases
  // h: [seqLen, batchSize, hiddenSize] - output (all hidden states)
  // hL: [1, batchSize, hiddenSize] - last hidden state

  const int numDirections = 1;  // unidirectional
  const int numLayers = 1;
  const int numGates = 3;  // GRU has 3 gates: reset, update, new

  dnnl::memory::dims srcLayerDims = {seqLen, batchSize, inputSize};
  dnnl::memory::dims srcIterDims = {numLayers, numDirections, batchSize, hiddenSize};
  dnnl::memory::dims weightLayerDims = {numLayers, numDirections, inputSize, numGates, hiddenSize};
  dnnl::memory::dims weightIterDims = {numLayers, numDirections, hiddenSize, numGates, hiddenSize};
  dnnl::memory::dims biasDims = {numLayers, numDirections, numGates, hiddenSize};
  dnnl::memory::dims dstLayerDims = {seqLen, batchSize, hiddenSize};
  dnnl::memory::dims dstIterDims = {numLayers, numDirections, batchSize, hiddenSize};

  // Memory descriptors
  auto srcLayerMd = dnnl::memory::desc(srcLayerDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::tnc);
  auto srcIterMd = dnnl::memory::desc(srcIterDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldnc);
  auto weightLayerMd = dnnl::memory::desc(weightLayerDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldigo);
  auto weightIterMd = dnnl::memory::desc(weightIterDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldigo);
  auto biasMd = dnnl::memory::desc(biasDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldgo);
  auto dstLayerMd = dnnl::memory::desc(dstLayerDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::tnc);
  auto dstIterMd = dnnl::memory::desc(dstIterDims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ldnc);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // GRU primitive descriptor (OneDNN 3.x API)
  dnnl::gru_forward::primitive_desc gruPrimDesc(engine, dnnl::prop_kind::forward_inference,
                                                 dnnl::rnn_direction::unidirectional_left2right,
                                                 srcLayerMd, srcIterMd, weightLayerMd, weightIterMd, biasMd,
                                                 dstLayerMd, dstIterMd);

  // Create memory objects
  std::unordered_map<int, dnnl::memory> args;

  dnnl::memory srcLayerMem(srcLayerMd, engine, x->buffer());
  args[DNNL_ARG_SRC_LAYER] = srcLayerMem;

  dnnl::memory srcIterMem(srcIterMd, engine, hI->buffer());
  args[DNNL_ARG_SRC_ITER] = srcIterMem;

  dnnl::memory weightLayerMem(weightLayerMd, engine, Wx->buffer());
  args[DNNL_ARG_WEIGHTS_LAYER] = weightLayerMem;

  dnnl::memory weightIterMem(weightIterMd, engine, Wh->buffer());
  args[DNNL_ARG_WEIGHTS_ITER] = weightIterMem;

  dnnl::memory biasMem(biasMd, engine, b->buffer());
  args[DNNL_ARG_BIAS] = biasMem;

  dnnl::memory dstLayerMem(dstLayerMd, engine, h->buffer());
  args[DNNL_ARG_DST_LAYER] = dstLayerMem;

  dnnl::memory dstIterMem(dstIterMd, engine, hL->buffer());
  args[DNNL_ARG_DST_ITER] = dstIterMem;

  // Execute
  dnnl::gru_forward(gruPrimDesc).execute(stream, args);

  stream.wait();
}

PLATFORM_IMPL(gruCell, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);   // input
  auto hI = INPUT_VARIABLE(1);  // initial hidden state
  auto Wx = INPUT_VARIABLE(2);  // input weights
  auto Wh = INPUT_VARIABLE(3);  // recurrent weights
  auto b = INPUT_VARIABLE(4);   // biases

  auto h = OUTPUT_VARIABLE(0);  // output hidden states
  auto hL = OUTPUT_VARIABLE(1); // last hidden state

  int seqLen = x->sizeAt(0);
  int batchSize = x->sizeAt(1);
  int inputSize = x->sizeAt(2);
  int hiddenSize = hI->sizeAt(2);

  gruMKLDNN(x, hI, Wx, Wh, b, h, hL, seqLen, batchSize, inputSize, hiddenSize);

  return sd::Status::OK;
}

PLATFORM_CHECK(gruCell, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto hI = INPUT_VARIABLE(1);

  Requirements req("ONEDNN GRU OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 3) &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0), DataType::FLOAT32) &&
      req.expectEq(makeInfoVariable(hI->dataType(), TYPE_MSG_INPUT1), DataType::FLOAT32);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
