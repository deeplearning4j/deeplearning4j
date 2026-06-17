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
// CPU implementation of Multimodal Rotary Position Embedding (M-RoPE).
// Used in Qwen3-VL and Qwen2.5-VL for video/image+text transformer models.
//
// Pairing convention (matches CUDA kernel):
//   The head_dim is split into two halves.  Element at index d (d < halfDim)
//   is paired with element at index d + halfDim.  The first halfDim indices
//   are split into three contiguous section groups:
//     temporal: d in [0,       halfT)
//     height:   d in [halfT,   halfT + halfH)
//     width:    d in [halfT+halfH, halfDim)
//   where halfT = sectionT/2, halfH = sectionH/2.
//
//   For each pair (d, d+halfDim), a sinusoidal angle is computed using
//   the position for d's section and the frequency:
//     freq = 1 / freqBase^(2 * sectionLocalD / sectionSize)
//   where sectionLocalD = d's offset within its section.
//

#include <ops/declarable/helpers/fused_llm_ops.h>
#include <array/NDArray.h>
#include <execution/Threads.h>
#include <system/op_boilerplate.h>
#include <system/type_boilerplate.h>
#include <cmath>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Contiguous-section M-RoPE (non-interleaved).
 *
 * Iterates over d in [0, halfDim) and applies the rotation to the pair
 * (input[base+d], input[base+d+halfDim]).  The position and frequency
 * formula are determined by which section d belongs to.
 */
template <typename T, typename P>
static void applyMRoPEContiguous(
    const T* input, T* output,
    const P* posT,
    const P* posH,
    const P* posW,
    int batch, int seq, int heads, int headDim,
    int sectionT, int sectionH, int sectionW,
    float freqBase) {

  const int halfDim = headDim / 2;
  const int halfT   = sectionT / 2;
  const int halfH   = sectionH / 2;
  // halfW = sectionW / 2  (implicit: halfT + halfH + halfW == halfDim)

  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < seq; s++) {
      for (int h = 0; h < heads; h++) {
        const float posTval = static_cast<float>(posT[b * seq + s]);
        const float posHval = static_cast<float>(posH[b * seq + s]);
        const float posWval = static_cast<float>(posW[b * seq + s]);

        const int base = ((b * seq + s) * heads + h) * headDim;

        for (int d = 0; d < halfDim; d++) {
          float pos;
          int   sectionLocalD;
          int   sectionSize;

          if (d < halfT) {
            pos           = posTval;
            sectionLocalD = d;
            sectionSize   = sectionT;
          } else if (d < halfT + halfH) {
            pos           = posHval;
            sectionLocalD = d - halfT;
            sectionSize   = sectionH;
          } else {
            pos           = posWval;
            sectionLocalD = d - halfT - halfH;
            sectionSize   = sectionW;
          }

          const float freq   = 1.0f / std::pow(freqBase,
                                                (2.0f * sectionLocalD) / static_cast<float>(sectionSize));
          const float angle  = pos * freq;
          const float cosVal = std::cos(angle);
          const float sinVal = std::sin(angle);

          const int idx1 = base + d;
          const int idx2 = base + d + halfDim;

          const float x0 = static_cast<float>(input[idx1]);
          const float x1 = static_cast<float>(input[idx2]);

          output[idx1] = static_cast<T>(x0 * cosVal - x1 * sinVal);
          output[idx2] = static_cast<T>(x0 * sinVal + x1 * cosVal);
        }
      }
    }
  }
}

/**
 * Interleaved M-RoPE.
 *
 * Frequency slot d is assigned round-robin to sections:
 *   d % 3 == 0 -> temporal
 *   d % 3 == 1 -> height
 *   d % 3 == 2 -> width
 *
 * The local frequency within the section is d / 3.
 * Section size for frequency computation is (headDim + 2) / 3.
 */
template <typename T, typename P>
static void applyMRoPEInterleaved(
    const T* input, T* output,
    const P* posT,
    const P* posH,
    const P* posW,
    int batch, int seq, int heads, int headDim) {

  const int halfDim    = headDim / 2;
  const int sectionSize = (headDim + 2) / 3;

  PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
  for (int b = 0; b < batch; b++) {
    for (int s = 0; s < seq; s++) {
      for (int h = 0; h < heads; h++) {
        const float posTval = static_cast<float>(posT[b * seq + s]);
        const float posHval = static_cast<float>(posH[b * seq + s]);
        const float posWval = static_cast<float>(posW[b * seq + s]);

        const int base = ((b * seq + s) * heads + h) * headDim;

        for (int d = 0; d < halfDim; d++) {
          float pos;
          switch (d % 3) {
            case 0:  pos = posTval; break;
            case 1:  pos = posHval; break;
            default: pos = posWval; break;
          }

          const int   localD  = d / 3;
          const float freq    = 1.0f / std::pow(10000.0f,
                                                 (2.0f * localD) / static_cast<float>(sectionSize));
          const float angle   = pos * freq;
          const float cosVal  = std::cos(angle);
          const float sinVal  = std::sin(angle);

          const int idx1 = base + d;
          const int idx2 = base + d + halfDim;

          const float x0 = static_cast<float>(input[idx1]);
          const float x1 = static_cast<float>(input[idx2]);

          output[idx1] = static_cast<T>(x0 * cosVal - x1 * sinVal);
          output[idx2] = static_cast<T>(x0 * sinVal + x1 * cosVal);
        }
      }
    }
  }
}

template <typename T, typename P>
static void fusedMRoPE_(
    NDArray* input, NDArray* posT, NDArray* posH, NDArray* posW,
    NDArray* output,
    int sectionT, int sectionH, int sectionW,
    bool interleaved, float freqBase) {

  const int batch   = static_cast<int>(input->sizeAt(0));
  const int seq     = static_cast<int>(input->sizeAt(1));
  const int heads   = static_cast<int>(input->sizeAt(2));
  const int headDim = static_cast<int>(input->sizeAt(3));

  if (posT->dataType() != posH->dataType() || posT->dataType() != posW->dataType()) {
    THROW_EXCEPTION("fusedMRoPE requires position tensors to have matching data types");
  }

  const auto* inBuf   = input->bufferAsT<T>();
  auto*       outBuf  = output->bufferAsT<T>();
  const auto* posTBuf = posT->bufferAsT<P>();
  const auto* posHBuf = posH->bufferAsT<P>();
  const auto* posWBuf = posW->bufferAsT<P>();

  if (interleaved) {
    applyMRoPEInterleaved<T, P>(inBuf, outBuf, posTBuf, posHBuf, posWBuf,
                                batch, seq, heads, headDim);
  } else {
    applyMRoPEContiguous<T, P>(inBuf, outBuf, posTBuf, posHBuf, posWBuf,
                               batch, seq, heads, headDim,
                               sectionT, sectionH, sectionW, freqBase);
  }
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

  BUILD_DOUBLE_SELECTOR(input->dataType(), posT->dataType(), fusedMRoPE_,
                        (input, posT, posH, posW, output,
                         sectionT, sectionH, sectionW, interleaved, freqBase),
                        SD_FLOAT_TYPES, SD_NUMERIC_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
