/* ******************************************************************************
 *
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
// Large Language Model (LLM) operations header
// These operations support modern transformer-based language models.
//

#ifndef LIBND4J_HEADERS_LLM_H
#define LIBND4J_HEADERS_LLM_H

#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

/**
 * rms_norm - Root Mean Square Layer Normalization
 *
 * Implements RMS normalization as used in LLaMA and other modern LLMs.
 * RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * gamma
 *
 * Input:
 *   0: input tensor [batch, ..., features]
 *   1: gamma weights [features] (optional)
 *
 * Output:
 *   0: normalized tensor [batch, ..., features]
 *
 * Float arguments:
 *   0: epsilon (default: 1e-5)
 */
#if NOT_EXCLUDED(OP_rms_norm)
DECLARE_CUSTOM_OP(rms_norm, 1, 1, false, 0, 0);
DECLARE_CUSTOM_OP(rms_norm_bp, 2, 1, false, 0, 0);
#endif

/**
 * rope - Rotary Position Embedding
 *
 * Applies rotary position embeddings to query and key tensors.
 * Used in LLaMA, Mistral, and other modern architectures.
 *
 * Input:
 *   0: input tensor [batch, seq_len, num_heads, head_dim]
 *   1: position indices [batch, seq_len] (optional)
 *
 * Output:
 *   0: tensor with rotary embeddings applied [batch, seq_len, num_heads, head_dim]
 *
 * Integer arguments:
 *   0: mode (0=standard, 1=neox, 2=gpt-j, default: 0)
 *   1: n_past (for KV cache, default: 0)
 *   2: n_dims (dimensions for rotation, default: head_dim)
 *   3: n_ctx (context length, default: 2048)
 *
 * Float arguments:
 *   0: freq_base (default: 10000.0)
 *   1: freq_scale (default: 1.0)
 */
#if NOT_EXCLUDED(OP_rope)
DECLARE_CUSTOM_OP(rope, 1, 1, false, 0, 0);
DECLARE_CUSTOM_OP(rope_bp, 2, 1, false, 0, 0);
#endif

/**
 * silu - SiLU/Swish Activation Function
 *
 * Implements SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x)
 * Also known as Swish activation. Used in LLaMA and other modern LLMs.
 *
 * Input:
 *   0: input tensor
 *
 * Output:
 *   0: output tensor with silu applied
 */
#if NOT_EXCLUDED(OP_silu)
DECLARE_CONFIGURABLE_OP(silu, 1, 1, true, 0, 0);
DECLARE_CONFIGURABLE_OP(silu_bp, 2, 1, true, 0, 0);
#endif

/**
 * quantized_matmul - Quantized Matrix Multiplication
 *
 * Performs matrix multiplication with quantized weights.
 * Supports various quantization formats (Q4_0, Q4_1, Q8_0, etc.)
 *
 * Input:
 *   0: input tensor (float) [batch, ..., in_features]
 *   1: quantized weight tensor [out_features, in_features]
 *
 * Output:
 *   0: output tensor [batch, ..., out_features]
 *
 * Integer arguments:
 *   0: quantization type (0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, default: 0)
 *   1: transpose_a (0=no, 1=yes, default: 0)
 *   2: transpose_b (0=no, 1=yes, default: 0)
 */
#if NOT_EXCLUDED(OP_quantized_matmul)
DECLARE_CUSTOM_OP(quantized_matmul, 2, 1, false, 0, 0);
#endif

/**
 * grouped_query_attention - Grouped Query Attention
 *
 * Implements GQA (Grouped Query Attention) as used in LLaMA 2, Mistral, etc.
 * Multiple query heads share the same key-value heads.
 *
 * Input:
 *   0: query tensor [batch, seq_len, num_heads, head_dim]
 *   1: key tensor [batch, kv_seq_len, num_kv_heads, head_dim]
 *   2: value tensor [batch, kv_seq_len, num_kv_heads, head_dim]
 *   3: attention mask (optional) [batch, 1, seq_len, kv_seq_len]
 *
 * Output:
 *   0: attention output [batch, seq_len, num_heads, head_dim]
 *
 * Integer arguments:
 *   0: number of query heads (default: 8)
 *   1: number of key-value heads (default: num_heads for MHA)
 *   2: causal mask (0=no, 1=yes, default: 1)
 *
 * Float arguments:
 *   0: attention scale (default: 1/sqrt(head_dim))
 */
#if NOT_EXCLUDED(OP_grouped_query_attention)
DECLARE_CUSTOM_OP(grouped_query_attention, 3, 1, false, 0, 0);
DECLARE_CUSTOM_OP(grouped_query_attention_bp, 4, 3, false, 0, 0);
#endif

/**
 * flash_attention - Flash Attention
 *
 * Memory-efficient attention implementation that processes attention
 * in blocks to reduce memory usage from O(N^2) to O(N).
 *
 * Input:
 *   0: query tensor [batch, seq_len, num_heads, head_dim]
 *   1: key tensor [batch, kv_seq_len, num_heads, head_dim]
 *   2: value tensor [batch, kv_seq_len, num_heads, head_dim]
 *   3: attention mask (optional)
 *
 * Output:
 *   0: attention output [batch, seq_len, num_heads, head_dim]
 *
 * Integer arguments:
 *   0: causal mask (0=no, 1=yes, default: 1)
 *
 * Float arguments:
 *   0: attention scale (default: 1/sqrt(head_dim))
 */
#if NOT_EXCLUDED(OP_flash_attention)
DECLARE_CUSTOM_OP(flash_attention, 3, 1, false, 0, 0);
DECLARE_CUSTOM_OP(flash_attention_bp, 4, 3, false, 0, 0);
#endif

/**
 * kv_cache_update - Key-Value Cache Update
 *
 * Updates the KV cache for autoregressive generation.
 *
 * Input:
 *   0: existing key cache [batch, max_seq_len, num_kv_heads, head_dim]
 *   1: existing value cache [batch, max_seq_len, num_kv_heads, head_dim]
 *   2: new keys [batch, new_seq_len, num_kv_heads, head_dim]
 *   3: new values [batch, new_seq_len, num_kv_heads, head_dim]
 *
 * Output:
 *   0: updated key cache [batch, max_seq_len, num_kv_heads, head_dim]
 *   1: updated value cache [batch, max_seq_len, num_kv_heads, head_dim]
 *
 * Integer arguments:
 *   0: start position in cache (default: 0)
 */
#if NOT_EXCLUDED(OP_kv_cache_update)
DECLARE_CUSTOM_OP(kv_cache_update, 4, 2, false, 0, 0);
#endif

/**
 * apply_alibi - ALiBi Position Encoding
 *
 * Applies Attention with Linear Biases position encoding.
 * Used in BLOOM and other models.
 *
 * Input:
 *   0: attention scores [batch, num_heads, seq_len, kv_seq_len]
 *
 * Output:
 *   0: attention scores with ALiBi applied
 *
 * Integer arguments:
 *   0: number of heads
 *
 * Float arguments:
 *   0: alibi slope base (default: calculated from num_heads)
 */
#if NOT_EXCLUDED(OP_apply_alibi)
DECLARE_CUSTOM_OP(apply_alibi, 1, 1, false, 0, 0);
#endif

/**
 * sliding_window_attention - Sliding Window Attention
 *
 * Implements sliding window attention as used in Mistral.
 * Each token only attends to a fixed window of previous tokens.
 *
 * Input:
 *   0: query tensor [batch, seq_len, num_heads, head_dim]
 *   1: key tensor [batch, seq_len, num_kv_heads, head_dim]
 *   2: value tensor [batch, seq_len, num_kv_heads, head_dim]
 *
 * Output:
 *   0: attention output [batch, seq_len, num_heads, head_dim]
 *
 * Integer arguments:
 *   0: window size (default: 4096)
 *   1: number of query heads
 *   2: number of kv heads
 *
 * Float arguments:
 *   0: attention scale
 */
#if NOT_EXCLUDED(OP_sliding_window_attention)
DECLARE_CUSTOM_OP(sliding_window_attention, 3, 1, false, 0, 0);
#endif

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HEADERS_LLM_H
