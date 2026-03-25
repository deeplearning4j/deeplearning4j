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
// Fused LLM operations helper declarations
// These operations implement mega-kernel fused ops for transformer models.
//

#ifndef LIBND4J_FUSED_LLM_OPS_H
#define LIBND4J_FUSED_LLM_OPS_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Fused RMS Norm + SwiGLU operation
 * Computes: silu(rms_norm(x) @ W_gate) * (rms_norm(x) @ W_up)
 *
 * This fuses the FFN computation in LLaMA-style models into a single kernel,
 * avoiding multiple memory passes.
 *
 * @param input Input tensor [batch, seq_len, hidden_dim]
 * @param gamma RMS norm scale [hidden_dim]
 * @param wGate Gate projection weights [hidden_dim, intermediate_dim]
 * @param wUp Up projection weights [hidden_dim, intermediate_dim]
 * @param output Output tensor [batch, seq_len, intermediate_dim]
 * @param epsilon RMS norm epsilon
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedRmsNormSwiGLU(
    NDArray* input,
    NDArray* gamma,
    NDArray* wGate,
    NDArray* wUp,
    NDArray* output,
    float epsilon,
    LaunchContext* context);

/**
 * Fused Rotary Position Embedding (RoPE)
 * Applies rotary position embeddings with optimized sin/cos computation.
 *
 * Supports multiple RoPE variants:
 * - Standard (LLaMA, Mistral)
 * - NeoX (GPT-NeoX, Pythia)
 * - GPT-J
 *
 * @param input Input tensor [batch, seq_len, num_heads, head_dim]
 * @param output Output tensor [batch, seq_len, num_heads, head_dim]
 * @param positionOffset Starting position for cache continuation
 * @param freqBase Base frequency (default 10000)
 * @param freqScale Frequency scale factor
 * @param ropeType 0=standard, 1=neox, 2=gptj
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedRoPE(
    NDArray* input,
    NDArray* output,
    int positionOffset,
    float freqBase,
    float freqScale,
    int ropeType,
    LaunchContext* context,
    int rotaryDims = 0);

/**
 * Fused Layer Normalization
 * Computes: (x - mean) / sqrt(var + eps) * gain + bias in a single pass
 * using Welford's algorithm for numerical stability.
 *
 * @param input Input tensor
 * @param gain Scale parameter (gamma)
 * @param bias Shift parameter (beta), can be nullptr
 * @param output Output tensor
 * @param epsilon Numerical stability constant
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedLayerNorm(
    NDArray* input,
    NDArray* gain,
    NDArray* bias,
    NDArray* output,
    float epsilon,
    LaunchContext* context);

/**
 * Fused Bias + Dropout + Residual
 * Computes: dropout(input + bias) + residual in a single kernel
 *
 * @param input Input tensor
 * @param bias Bias tensor (broadcastable to input)
 * @param residual Residual connection tensor
 * @param output Output tensor
 * @param dropoutProb Dropout probability (0 = no dropout)
 * @param seed Random seed for dropout
 * @param training Whether in training mode
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedBiasDropoutResidual(
    NDArray* input,
    NDArray* bias,
    NDArray* residual,
    NDArray* output,
    float dropoutProb,
    LongType seed,
    bool training,
    LaunchContext* context);

/**
 * Fast GELU approximation
 * Computes: x * sigmoid(1.702 * x)
 * This is faster than the standard GELU while maintaining good accuracy.
 *
 * @param input Input tensor
 * @param output Output tensor
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedGELU(
    NDArray* input,
    NDArray* output,
    LaunchContext* context);

/**
 * Fused GELU backward
 * Computes gradient of fast GELU approximation.
 *
 * @param input Original input tensor
 * @param gradOut Gradient from upstream
 * @param gradIn Output gradient tensor
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedGELUBackward(
    NDArray* input,
    NDArray* gradOut,
    NDArray* gradIn,
    LaunchContext* context);

/**
 * Fused RMS Norm + SwiGLU backward
 * Computes gradients for the fused RMS norm + SwiGLU operation.
 *
 * @param input Original input tensor [batch, seq_len, hidden_dim]
 * @param gamma RMS norm scale [hidden_dim]
 * @param wGate Gate projection weights [hidden_dim, intermediate_dim]
 * @param wUp Up projection weights [hidden_dim, intermediate_dim]
 * @param gradOut Gradient from upstream [batch, seq_len, intermediate_dim]
 * @param gradInput Output gradient for input [batch, seq_len, hidden_dim]
 * @param gradGamma Output gradient for gamma [hidden_dim]
 * @param gradWGate Output gradient for wGate [hidden_dim, intermediate_dim]
 * @param gradWUp Output gradient for wUp [hidden_dim, intermediate_dim]
 * @param epsilon RMS norm epsilon
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedRmsNormSwiGLUBackward(
    NDArray* input,
    NDArray* gamma,
    NDArray* wGate,
    NDArray* wUp,
    NDArray* gradOut,
    NDArray* gradInput,
    NDArray* gradGamma,
    NDArray* gradWGate,
    NDArray* gradWUp,
    float epsilon,
    LaunchContext* context);

/**
 * Fused RoPE backward
 * Computes gradient of rotary position embedding.
 *
 * @param gradOut Gradient from upstream [batch, seq_len, num_heads, head_dim]
 * @param gradIn Output gradient [batch, seq_len, num_heads, head_dim]
 * @param positionOffset Starting position
 * @param freqBase Base frequency
 * @param freqScale Frequency scale
 * @param ropeType RoPE variant type
 * @param context Launch context
 */
/**
 * Fused RoPE with pre-computed cos/sin (cached variant)
 * Used when cos/sin are provided externally (e.g., from ONNX RotaryEmbedding).
 *
 * @param input Input tensor [batch, seq_len, num_heads, head_dim]
 * @param cosValues Pre-computed cosine [batch, seq_len, half_head_dim]
 * @param sinValues Pre-computed sine [batch, seq_len, half_head_dim]
 * @param output Output tensor [batch, seq_len, num_heads, head_dim]
 * @param ropeType 0=standard, 1=neox, 2=gptj
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedRoPECached(
    NDArray* input,
    NDArray* cosValues,
    NDArray* sinValues,
    NDArray* output,
    int ropeType,
    LaunchContext* context);

SD_LIB_HIDDEN void fusedRoPEBackward(
    NDArray* gradOut,
    NDArray* gradIn,
    int positionOffset,
    float freqBase,
    float freqScale,
    int ropeType,
    LaunchContext* context,
    int rotaryDims = 0);

/**
 * Fused Layer Norm backward
 * Computes gradients for layer normalization.
 *
 * @param input Original input tensor
 * @param gain Scale parameter (gamma)
 * @param gradOut Gradient from upstream
 * @param gradInput Output gradient for input
 * @param gradGain Output gradient for gain
 * @param gradBias Output gradient for bias (can be nullptr)
 * @param epsilon Numerical stability constant
 * @param context Launch context
 */
SD_LIB_HIDDEN void fusedLayerNormBackward(
    NDArray* input,
    NDArray* gain,
    NDArray* gradOut,
    NDArray* gradInput,
    NDArray* gradGain,
    NDArray* gradBias,
    float epsilon,
    LaunchContext* context);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_FUSED_LLM_OPS_H
