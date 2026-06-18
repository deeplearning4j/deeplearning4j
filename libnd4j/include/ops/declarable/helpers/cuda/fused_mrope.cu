/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Adam Gibson
//
// CUDA implementation of Multimodal Rotary Position Embedding (M-RoPE).
// Used in Qwen3-VL and Qwen2.5-VL for video/image+text transformer models.
//

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <array/NDArray.h>
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/fused_llm_ops.h>
#include <system/op_boilerplate.h>
#include <system/type_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Contiguous-section M-RoPE kernel.
 *
 * Each thread processes one (batch, seq, head, half-dim-within-section) tuple.
 * The three sections are: temporal [0, halfT), height [halfT, halfT+halfH),
 * width [halfT+halfH, halfDim).  Within each section, element d is paired
 * with element d + sectionSize/2.
 *
 * Thread index encodes: d in [0, halfDim), h in [0, heads), flattened b*seq.
 */
template <typename T, typename P>
SD_KERNEL static void fusedMRoPEKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const P* __restrict__ posT,
    const P* __restrict__ posH,
    const P* __restrict__ posW,
    int batch, int seq, int heads, int headDim,
    int sectionT, int sectionH, int sectionW,
    float freqBase) {

  // Each thread covers one (b*s, h, d) triple where d is in [0, halfDim).
  const int halfDim = headDim / 2;
  const int bsLen   = batch * seq;
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  const int total   = bsLen * heads * halfDim;

  if (tid >= total) return;

  // Decompose linear index (column-major over d, h, b*s)
  int rem = tid;
  const int d   = rem % halfDim;  rem /= halfDim;
  const int h   = rem % heads;    rem /= heads;
  const int bs  = rem;             // flattened batch*seq index

  // Map d to its section and select the corresponding position
  const int halfT = sectionT / 2;
  const int halfH = sectionH / 2;
  // halfW = sectionW / 2 (implicit: halfDim = halfT + halfH + halfW)

  float pos;
  int sectionLocalD;
  int sectionSize;

  if (d < halfT) {
    pos           = static_cast<float>(posT[bs]);
    sectionLocalD = d;
    sectionSize   = sectionT;
  } else if (d < halfT + halfH) {
    pos           = static_cast<float>(posH[bs]);
    sectionLocalD = d - halfT;
    sectionSize   = sectionH;
  } else {
    pos           = static_cast<float>(posW[bs]);
    sectionLocalD = d - halfT - halfH;
    sectionSize   = sectionW;
  }

  // Compute rotation for this (position, section-local-d) pair
  const float freq   = 1.0f / __powf(freqBase, (2.0f * sectionLocalD) / static_cast<float>(sectionSize));
  const float angle  = pos * freq;
  float cosVal, sinVal;
  __sincosf(angle, &sinVal, &cosVal);

  // Index into the flat [batch, seq, heads, headDim] buffer
  const int base = (bs * heads + h) * headDim;
  const int idx1 = base + d;
  const int idx2 = base + d + halfDim;

  const float x0 = static_cast<float>(input[idx1]);
  const float x1 = static_cast<float>(input[idx2]);

  output[idx1] = static_cast<T>(x0 * cosVal - x1 * sinVal);
  output[idx2] = static_cast<T>(x0 * sinVal + x1 * cosVal);
}

/**
 * Interleaved M-RoPE kernel.
 *
 * Frequency slot d is assigned to sections round-robin:
 *   d % 3 == 0 -> temporal
 *   d % 3 == 1 -> height
 *   d % 3 == 2 -> width
 *
 * The local frequency within the section is d / 3.
 */
template <typename T, typename P>
SD_KERNEL static void fusedMRoPEInterleavedKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const P* __restrict__ posT,
    const P* __restrict__ posH,
    const P* __restrict__ posW,
    int batch, int seq, int heads, int headDim) {

  const int halfDim   = headDim / 2;
  const int bsLen     = batch * seq;
  const int tid       = blockIdx.x * blockDim.x + threadIdx.x;
  const int total     = bsLen * heads * halfDim;

  if (tid >= total) return;

  int rem = tid;
  const int d  = rem % halfDim;  rem /= halfDim;
  const int h  = rem % heads;    rem /= heads;
  const int bs = rem;

  float pos;
  switch (d % 3) {
    case 0:  pos = static_cast<float>(posT[bs]); break;
    case 1:  pos = static_cast<float>(posH[bs]); break;
    default: pos = static_cast<float>(posW[bs]); break;
  }

  // Approximate section size for consistent frequency computation
  const int sectionSize = (headDim + 2) / 3;
  const int localD      = d / 3;

  const float freq  = 1.0f / __powf(10000.0f, (2.0f * localD) / static_cast<float>(sectionSize));
  const float angle = pos * freq;
  float cosVal, sinVal;
  __sincosf(angle, &sinVal, &cosVal);

  const int base = (bs * heads + h) * headDim;
  const int idx1 = base + d;
  const int idx2 = base + d + halfDim;

  const float x0 = static_cast<float>(input[idx1]);
  const float x1 = static_cast<float>(input[idx2]);

  output[idx1] = static_cast<T>(x0 * cosVal - x1 * sinVal);
  output[idx2] = static_cast<T>(x0 * sinVal + x1 * cosVal);
}

template <typename T, typename P>
static void fusedMRoPE_(
    NDArray* input,
    NDArray* posT, NDArray* posH, NDArray* posW,
    NDArray* output,
    int sectionT, int sectionH, int sectionW,
    bool interleaved, float freqBase,
    LaunchContext* context) {

  const int batch   = static_cast<int>(input->sizeAt(0));
  const int seq     = static_cast<int>(input->sizeAt(1));
  const int heads   = static_cast<int>(input->sizeAt(2));
  const int headDim = static_cast<int>(input->sizeAt(3));
  const int halfDim = headDim / 2;

  if (posT->dataType() != posH->dataType() || posT->dataType() != posW->dataType()) {
    THROW_EXCEPTION("fusedMRoPE requires position tensors to have matching data types");
  }

  auto* stream = context->getCudaStream();

  const int totalElements = batch * seq * heads * halfDim;
  const dim3 dims         = getLaunchDims("fused_mrope");
  const int blockSize     = static_cast<int>(dims.y);
  const int gridSize      = (totalElements + blockSize - 1) / blockSize;

  if (interleaved) {
    fusedMRoPEInterleavedKernel<T, P><<<gridSize, blockSize, 0, *stream>>>(
        input->bufferAsT<T>(), output->bufferAsT<T>(),
        posT->bufferAsT<P>(),
        posH->bufferAsT<P>(),
        posW->bufferAsT<P>(),
        batch, seq, heads, headDim);
  } else {
    fusedMRoPEKernel<T, P><<<gridSize, blockSize, 0, *stream>>>(
        input->bufferAsT<T>(), output->bufferAsT<T>(),
        posT->bufferAsT<P>(),
        posH->bufferAsT<P>(),
        posW->bufferAsT<P>(),
        batch, seq, heads, headDim,
        sectionT, sectionH, sectionW,
        freqBase);
  }

  DebugHelper::checkGlobalErrorCode("fusedMRoPEKernel failed");
}

void fusedMRoPE(
    NDArray* input,
    NDArray* posT,
    NDArray* posH,
    NDArray* posW,
    NDArray* output,
    int sectionT,
    int sectionH,
    int sectionW,
    bool interleaved,
    float freqBase,
    LaunchContext* context) {

  NDArray::prepareSpecialUse({output}, {input, posT, posH, posW});

  BUILD_DOUBLE_SELECTOR(input->dataType(), posT->dataType(), fusedMRoPE_,
                        (input, posT, posH, posW, output,
                         sectionT, sectionH, sectionW, interleaved, freqBase, context),
                        SD_FLOAT_TYPES, SD_NUMERIC_TYPES);

  NDArray::registerSpecialUse({output}, {input, posT, posH, posW});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
