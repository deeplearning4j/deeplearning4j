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

#ifndef LIBND4J_DECODER_MASKED_MHA_H
#define LIBND4J_DECODER_MASKED_MHA_H

#include <ops/declarable/helpers/helpers.h>
#include <execution/LaunchContext.h>

namespace sd {
namespace ops {
namespace helpers {

struct DecoderMhaConfig {
    int numHeads;       // Query heads
    int numKvHeads;     // KV heads (GQA)
    int headDim;        // Dimension per head
    int cachePosition;  // Current position in KV cache
    int ropeType;       // 0=none, 1=standard, 2=neox
    float attScale;     // Attention scale (0 = auto)
    float ropeFreqBase; // RoPE frequency base
};

// Single unified declaration — CPU impl in helpers/cpu/, CUDA impl in helpers/cuda/
SD_LIB_HIDDEN void decoderMaskedMha(
    NDArray* hiddenStates,   // [B, 1, H]
    NDArray* qkvWeight,      // [H, 3*H]
    NDArray* oWeight,        // [H, H]
    NDArray* keyCache,       // [B, numKvHeads, maxSeq, headDim]
    NDArray* valueCache,     // [B, numKvHeads, maxSeq, headDim]
    NDArray* mask,           // [B, 1, 1, seqLen] optional
    NDArray* cosCache,       // [maxSeq, headDim/2] optional
    NDArray* sinCache,       // [maxSeq, headDim/2] optional
    NDArray* output,         // [B, 1, H]
    NDArray* updatedKeyCache,
    NDArray* updatedValueCache,
    const DecoderMhaConfig& config,
    LaunchContext* context = nullptr);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_DECODER_MASKED_MHA_H
