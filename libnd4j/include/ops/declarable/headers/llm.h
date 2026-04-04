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

/**
 * fused_gelu - Fast GELU Approximation
 *
 * Implements the fast GELU approximation: x * sigmoid(1.702 * x)
 * This is faster than standard GELU while maintaining good accuracy.
 * Used in many modern transformers including BERT variants.
 *
 * Input:
 *   0: input tensor
 *
 * Output:
 *   0: output tensor with GELU applied
 */
#if NOT_EXCLUDED(OP_fused_gelu)
DECLARE_CONFIGURABLE_OP(fused_gelu, 1, 1, true, 0, 0);
DECLARE_CONFIGURABLE_OP(fused_gelu_bp, 2, 1, true, 0, 0);
#endif

/**
 * fused_layer_norm - Fused Layer Normalization
 *
 * Computes layer normalization in a single fused kernel using Welford's
 * algorithm for numerical stability:
 * output = (input - mean) / sqrt(variance + epsilon) * gain + bias
 *
 * Input:
 *   0: input tensor [batch, ..., features]
 *   1: gain/gamma [features]
 *   2: bias/beta [features] (optional)
 *
 * Output:
 *   0: normalized tensor [batch, ..., features]
 *
 * Float arguments:
 *   0: epsilon (default: 1e-5)
 */
#if NOT_EXCLUDED(OP_fused_layer_norm)
DECLARE_CUSTOM_OP(fused_layer_norm, 2, 1, false, 0, 0);
DECLARE_CUSTOM_OP(fused_layer_norm_bp, 3, 2, false, 0, 0);
#endif

/**
 * fused_rope - Fused Rotary Position Embedding
 *
 * Applies rotary position embeddings with optimized sin/cos computation.
 * Supports multiple RoPE variants used in modern LLMs.
 *
 * Input:
 *   0: input tensor [batch, seq_len, num_heads, head_dim]
 *
 * Output:
 *   0: tensor with rotary embeddings applied [batch, seq_len, num_heads, head_dim]
 *
 * Integer arguments:
 *   0: rope_type (0=standard/LLaMA, 1=neox, 2=gpt-j, default: 0)
 *   1: position_offset (for KV cache continuation, default: 0)
 *
 * Float arguments:
 *   0: freq_base (default: 10000.0)
 *   1: freq_scale (default: 1.0)
 */
#if NOT_EXCLUDED(OP_fused_rope)
DECLARE_CUSTOM_OP(fused_rope, 1, 1, false, 0, 0);
DECLARE_CUSTOM_OP(fused_rope_bp, 2, 1, false, 0, 0);
#endif

/**
 * fused_bias_dropout_residual - Fused Bias + Dropout + Residual
 *
 * Computes: dropout(input + bias) + residual in a single kernel
 * to minimize memory bandwidth usage.
 *
 * Input:
 *   0: input tensor
 *   1: bias tensor (broadcastable to input)
 *   2: residual tensor
 *
 * Output:
 *   0: output tensor
 *
 * Integer arguments:
 *   0: seed for dropout RNG
 *
 * Float arguments:
 *   0: dropout probability (default: 0.0 = no dropout)
 *
 * Boolean arguments:
 *   0: training mode (default: false)
 */
#if NOT_EXCLUDED(OP_fused_bias_dropout_residual)
DECLARE_CUSTOM_OP(fused_bias_dropout_residual, 3, 1, false, 0, 0);
#endif

/**
 * fused_rms_norm_swiglu - Fused RMS Norm + SwiGLU
 *
 * Fuses the FFN computation in LLaMA-style models:
 * output = silu(rms_norm(input) @ W_gate) * (rms_norm(input) @ W_up)
 *
 * This mega-kernel avoids multiple memory passes and is optimized
 * for the feed-forward network pattern in modern LLMs.
 *
 * Input:
 *   0: input tensor [batch, seq_len, hidden_dim]
 *   1: gamma weights for RMS norm [hidden_dim]
 *   2: W_gate projection [hidden_dim, intermediate_dim]
 *   3: W_up projection [hidden_dim, intermediate_dim]
 *
 * Output:
 *   0: output tensor [batch, seq_len, intermediate_dim]
 *
 * Float arguments:
 *   0: epsilon for RMS norm (default: 1e-5)
 */
#if NOT_EXCLUDED(OP_fused_rms_norm_swiglu)
DECLARE_CUSTOM_OP(fused_rms_norm_swiglu, 4, 1, false, 0, 0);
DECLARE_CUSTOM_OP(fused_rms_norm_swiglu_bp, 5, 4, false, 0, 0);
#endif

/**
 * swish_mul - SwiGLU Component: swish(x) * y
 *
 * Computes the SwiGLU activation component: silu(x) * y
 * where silu(x) = x * sigmoid(x).
 *
 * This is used in LLaMA, Mistral, and other modern LLMs for their
 * feed-forward network's gated activation.
 *
 * Input:
 *   0: x tensor - the input to apply swish activation
 *   1: y tensor - the gate tensor to multiply with
 *
 * Output:
 *   0: output tensor = silu(x) * y
 */
#if NOT_EXCLUDED(OP_swish_mul)
DECLARE_CONFIGURABLE_OP(swish_mul, 2, 1, true, 0, 0);
DECLARE_CONFIGURABLE_OP(swish_mul_bp, 3, 2, true, 0, 0);
#endif

/**
 * mean_square - Mean of Squared Values
 *
 * Computes mean(x * x) along the last dimension.
 * This is a building block for RMSNorm and other normalization ops.
 *
 * Input:
 *   0: input tensor [batch, ..., features]
 *
 * Output:
 *   0: output tensor [batch, ..., 1] - mean of squared values along last dim
 *
 * Integer arguments:
 *   0: keepDims (0=no, 1=yes, default: 1)
 */
#if NOT_EXCLUDED(OP_mean_square)
DECLARE_CUSTOM_OP(mean_square, 1, 1, false, 0, 0);
DECLARE_CUSTOM_OP(mean_square_bp, 2, 1, false, 0, 0);
#endif

/**
 * Fused element-wise chain: executes a sequence of element-wise ops in a single kernel.
 *
 * Input 0: primary input tensor
 * Inputs 1..N: secondary inputs for binary ops in the chain (one per binary op)
 * iArgs: op codes from FusedElemOp enum (one per op in the chain)
 *   0=add, 1=sub, 2=mul, 3=div, 10=relu, 11=sigmoid, 12=tanh, 13=gelu,
 *   14=exp, 15=log, 16=abs, 17=neg, 18=square, 19=sqrt, 20=swish, 21=silu, 22=mish
 * tArgs[0]: clipMin (optional, for FUSED_CLIP=30)
 * tArgs[1]: clipMax (optional, for FUSED_CLIP=30)
 *
 * Output shape = input 0 shape.
 */
#if NOT_EXCLUDED(OP_fused_elementwise_chain)
DECLARE_CUSTOM_OP(fused_elementwise_chain, 1, 1, true, 0, 1);
#endif

/**
 * column_parallel_linear - Column-Parallel Linear Layer
 *
 * Tensor parallelism op: each GPU holds W[:, shard].
 * Output is gathered (AllGather) across TP ranks.
 *
 * Y_local = X @ W_shard  (each rank computes partial output)
 * Y = AllGather(Y_local)  (if gather_output=true)
 *
 * Input:
 *   0: X [batch, in_features]
 *   1: W_shard [in_features, out_features/tp_size] (this rank's column shard)
 *   2: bias_shard (optional) [out_features/tp_size]
 *
 * Output:
 *   0: Y [batch, out_features] (if gather) or [batch, out_features/tp_size] (if no gather)
 *
 * Integer arguments:
 *   0: tp_size (tensor parallel world size, default: 1)
 *   1: tp_rank (this rank, default: 0)
 *   2: gather_output (0=no, 1=yes, default: 1)
 */
#if NOT_EXCLUDED(OP_column_parallel_linear)
DECLARE_CUSTOM_OP(column_parallel_linear, 2, 1, false, 0, 0);
#endif

/**
 * row_parallel_linear - Row-Parallel Linear Layer
 *
 * Tensor parallelism op: each GPU holds W[shard, :].
 * Input is assumed to be split across ranks.
 * Output is reduced (AllReduce) across TP ranks.
 *
 * Y_partial = X_shard @ W_shard  (each rank computes partial sum)
 * Y = AllReduce(Y_partial)  (if reduce_output=true)
 *
 * Input:
 *   0: X_shard [batch, in_features/tp_size] (this rank's input shard)
 *   1: W_shard [in_features/tp_size, out_features] (this rank's row shard)
 *   2: bias (optional, only added on rank 0) [out_features]
 *
 * Output:
 *   0: Y [batch, out_features]
 *
 * Integer arguments:
 *   0: tp_size (tensor parallel world size, default: 1)
 *   1: tp_rank (this rank, default: 0)
 *   2: reduce_output (0=no, 1=yes, default: 1)
 */
#if NOT_EXCLUDED(OP_row_parallel_linear)
DECLARE_CUSTOM_OP(row_parallel_linear, 2, 1, false, 0, 0);
#endif

/**
 * decoder_masked_mha - Fused Decoder Masked Multi-Head Attention
 *
 * Single fused kernel for decode-phase attention:
 * QKV projection + RoPE + KV cache write + masked attention + output projection
 *
 * Optimized for single-token decode (memory-bandwidth bound).
 * Eliminates all intermediate global memory traffic.
 *
 * Input:
 *   0: hidden_states [batch, 1, hidden_dim]
 *   1: qkv_weight [3 * hidden_dim, hidden_dim] (fused QKV projection)
 *   2: o_weight [hidden_dim, hidden_dim] (output projection)
 *   3: key_cache [batch, num_kv_heads, max_seq_len, head_dim]
 *   4: value_cache [batch, num_kv_heads, max_seq_len, head_dim]
 *   5: attention_mask [batch, 1, 1, seq_len] (optional)
 *   6: cos_cache [max_seq_len, head_dim/2] (RoPE cos, optional)
 *   7: sin_cache [max_seq_len, head_dim/2] (RoPE sin, optional)
 *
 * Output:
 *   0: output [batch, 1, hidden_dim]
 *   1: updated key_cache
 *   2: updated value_cache
 *
 * Integer arguments:
 *   0: num_heads (query heads)
 *   1: num_kv_heads (key-value heads, for GQA)
 *   2: cache_position (current position in KV cache)
 *   3: rope_type (0=none, 1=standard, 2=neox, default: 1)
 *
 * Float arguments:
 *   0: attention_scale (default: 1/sqrt(head_dim))
 *   1: rope_freq_base (default: 10000.0)
 */
// TODO: Implementation moved to /tmp/new_ops_backup/ — re-enable when fixed
// #if NOT_EXCLUDED(OP_decoder_masked_mha)
// DECLARE_CUSTOM_OP(decoder_masked_mha, 5, 3, false, 0, 0);
// #endif

/**
 * fused_gemm_swiglu - Fused GEMM + SiLU + Element-wise Multiply
 *
 * Fuses the MLP computation in LLaMA-style models using CUTLASS EVT:
 * output = silu(X @ W_gate) * (X @ W_up)
 *
 * Input:
 *   0: X [batch, seq_len, hidden_dim]
 *   1: W_gate [hidden_dim, intermediate_dim]
 *   2: W_up [hidden_dim, intermediate_dim]
 *
 * Output:
 *   0: output [batch, seq_len, intermediate_dim]
 */
// TODO: Implementation moved to /tmp/new_ops_backup/ — re-enable when fixed
// #if NOT_EXCLUDED(OP_fused_gemm_swiglu)
// DECLARE_CUSTOM_OP(fused_gemm_swiglu, 3, 1, false, 0, 0);
// #endif

/**
 * moe_gate - Mixture of Experts Gating
 *
 * Top-K expert selection with load balancing auxiliary loss.
 *
 * Input:
 *   0: hidden_states [batch, seq_len, hidden_dim]
 *   1: gate_weights [hidden_dim, num_experts]
 *
 * Output:
 *   0: expert_indices [batch * seq_len, top_k] (INT64)
 *   1: expert_weights [batch * seq_len, top_k] (softmax scores)
 *   2: aux_loss (scalar) - load balancing loss
 *
 * Integer arguments:
 *   0: num_experts (default: 8)
 *   1: top_k (experts per token, default: 2)
 *
 * Float arguments:
 *   0: aux_loss_weight (default: 0.01)
 *   1: jitter_eps (training noise, default: 0.0)
 */
// TODO: Implementation moved to /tmp/new_ops_backup/ — re-enable when fixed
// #if NOT_EXCLUDED(OP_moe_gate)
// DECLARE_CUSTOM_OP(moe_gate, 2, 3, false, 0, 0);
// #endif

/**
 * selective_scan - Mamba/SSM Selective Scan
 *
 * Implements the selective scan operation for state space models (Mamba).
 * h_t = A_t * h_{t-1} + B_t * x_t
 * y_t = C_t * h_t + D * x_t
 *
 * Input:
 *   0: x [batch, seq_len, d_inner]
 *   1: delta [batch, seq_len, d_inner] (discretization step)
 *   2: A [d_inner, d_state] (state matrix)
 *   3: B [batch, seq_len, d_state] (input-dependent B)
 *   4: C [batch, seq_len, d_state] (input-dependent C)
 *   5: D [d_inner] (skip connection, optional)
 *
 * Output:
 *   0: y [batch, seq_len, d_inner]
 *   1: last_state [batch, d_inner, d_state] (for recurrent inference)
 *
 * Integer arguments:
 *   0: d_state (state dimension, default: 16)
 */
#if NOT_EXCLUDED(OP_selective_scan)
DECLARE_CUSTOM_OP(selective_scan, 5, 1, false, 0, 0);
#endif

/**
 * multi_lora_matmul - Batched LoRA GEMM with per-row adapter selection
 *
 * Computes Y = X @ W + X @ A[adapter_id] @ B[adapter_id] * scaling
 * where different rows of the batch can use different adapters.
 *
 * Input:
 *   0: X [batch, in_features]
 *   1: W [in_features, out_features] (base weights)
 *   2: adapter_ids [batch] (INT32, adapter index per row)
 *   3..3+N: LoRA A matrices [rank, in_features] (one per adapter)
 *   3+N..3+2N: LoRA B matrices [out_features, rank] (one per adapter)
 *
 * Output:
 *   0: Y [batch, out_features]
 *
 * Integer arguments:
 *   0: num_adapters
 *   1: lora_rank
 *
 * Float arguments:
 *   0: lora_scaling (alpha/rank)
 */
#if NOT_EXCLUDED(OP_multi_lora_matmul)
DECLARE_CUSTOM_OP(multi_lora_matmul, 5, 1, false, 0, 0);
#endif

/**
 * smooth_quant - SmoothQuant W8A8 quantized matmul
 *
 * Implements the SmoothQuant technique: Y = (X * diag(s)^-1) @ (diag(s) * W)
 * where s is a per-channel smoothing scale computed offline via calibration.
 *
 * Input arrays:
 *   0: input [M, K] - FP16/FP32 input activations
 *   1: weightQuant [K, N] - INT8 pre-quantized smoothed weights
 *   2: smoothScale [K] - per-channel smoothing scale
 *   3: weightScale scalar or [N] - dequant scale for weights
 *   4: (optional) actScale - per-channel activation scale
 *   5: (optional) bias [N]
 *
 * Output arrays:
 *   0: output [M, N]
 */
#if NOT_EXCLUDED(OP_smooth_quant)
DECLARE_CUSTOM_OP(smooth_quant, 4, 1, false, 0, 0);
#endif

/**
 * kv_cache_quantize - KV Cache Quantization
 *
 * Quantizes floating-point KV cache data using per-row symmetric quantization.
 * Each row (last dimension) gets its own scale factor: scale = max(abs(row)) / 127 (INT8).
 *
 * Input:
 *   0: input tensor [..., rowLen] (float KV data)
 *
 * Output:
 *   0: quantized tensor [..., rowLen] (INT8)
 *   1: scales tensor [...] (FLOAT32, one scale per row)
 *
 * Integer arguments:
 *   0: quantization format (0=INT8, 1=FP8_E4M3, 2=FP8_E5M2, 3=INT4)
 *   1: group size (optional, default: full row length)
 */
#if NOT_EXCLUDED(OP_kv_cache_quantize)
DECLARE_CUSTOM_OP(kv_cache_quantize, 1, 2, false, 0, 1);
#endif

/**
 * kv_cache_dequantize - KV Cache Dequantization
 *
 * Reverses kv_cache_quantize: reconstructs floating-point KV data
 * from quantized representation and per-row scales.
 * output = quantized * scale (per row)
 *
 * Input:
 *   0: quantized tensor [..., rowLen] (INT8)
 *   1: scales tensor [...] (FLOAT32, one scale per row)
 *
 * Output:
 *   0: dequantized tensor [..., rowLen] (FLOAT32)
 *
 * Integer arguments:
 *   0: quantization format (0=INT8, 1=FP8_E4M3, 2=FP8_E5M2, 3=INT4)
 */
#if NOT_EXCLUDED(OP_kv_cache_dequantize)
DECLARE_CUSTOM_OP(kv_cache_dequantize, 2, 1, false, 0, 1);
#endif

/**
 * gated_delta_rule - Gated Delta Network recurrent state update
 *
 * Implements the gated delta rule from arXiv:2412.06464 (ICLR 2025, NVIDIA Research):
 *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
 *   output_t = S_t^T * q_t
 *
 * Input:
 *   0: Q [B, L, H, D_k] - query
 *   1: K [B, L, H, D_k] - key (L2-normalized)
 *   2: V [B, L, H, D_v] - value
 *   3: beta [B, L, H] - per-step learning rate
 *   4: gate [B, L, H] - decay gate (pre-exp)
 *   5: state_in [B, H, D_k, D_v] - previous recurrent state (optional, zeros if absent)
 *
 * Output:
 *   0: output [B, L, H, D_v] - attention output
 *   1: state_out [B, H, D_k, D_v] - final recurrent state
 */
#if NOT_EXCLUDED(OP_gated_delta_rule)
DECLARE_CUSTOM_OP(gated_delta_rule, 5, 2, false, 0, 0);
#endif

/**
 * causal_conv1d - Depthwise causal 1D convolution with state
 *
 * Performs a causal (left-padded) depthwise 1D convolution, used in GDN and Mamba.
 *
 * Input:
 *   0: x [B, L, D] - input sequence
 *   1: weight [D, K] - depthwise conv weights (K = kernel size, typically 4)
 *   2: bias [D] (optional)
 *   3: state_in [B, D, K-1] (optional - conv state for autoregressive decode)
 *
 * Output:
 *   0: output [B, L, D] - convolved output
 *   1: state_out [B, D, K-1] - updated conv state
 *
 * Integer arguments:
 *   0: activation (0=none, 1=silu)
 *   1: wFormat (0=[D,K] PyTorch/ONNX default, 1=[K,D] TensorFlow)
 */
#if NOT_EXCLUDED(OP_causal_conv1d)
DECLARE_CUSTOM_OP(causal_conv1d, 2, 2, false, 0, 0);
#endif

/**
 * gated_delta_net_block - Full Gated Delta Network layer
 *
 * Fuses: linear projection -> causal_conv1d + SiLU -> gated_delta_rule -> RMSNorm + Swish gate -> output projection
 *
 * Input:
 *   0: x [B, L, D] - input
 *   1: Wqkv [D, (H*D_k*3 + H*D_v)] - QKV projection weights
 *   2: Wbeta [D, H] - beta projection weights
 *   3: Wgate [D, H] - gate projection weights
 *   4: Wout [H*D_v, D] - output projection weights
 *   5: conv_weight [D, K] - causal conv1d weights
 *   6: conv_bias [D] - causal conv1d bias
 *   7: state_in [B, H, D_k, D_v] - recurrent state (optional)
 *
 * Output:
 *   0: output [B, L, D]
 *   1: recurrent_state_out [B, H, D_k, D_v]
 *   2: conv_state_out [B, D, K-1]
 *
 * Integer arguments:
 *   0: num_heads (H)
 *   1: head_dim_k (D_k)
 *   2: head_dim_v (D_v)
 *
 * Float arguments:
 *   0: rms_norm_epsilon (default: 1e-5)
 */
#if NOT_EXCLUDED(OP_gated_delta_net_block)
DECLARE_CUSTOM_OP(gated_delta_net_block, 7, 3, false, 0, 3);
#endif

/**
 * ggml_dequantize - Dequantize raw GGML quantized bytes to float
 *
 * Uses GGML's own dequantization code to convert raw quantized tensor bytes
 * (Q4_0, Q4_K, Q5_K, Q6_K, etc.) directly to the target floating-point type.
 * Supports F32, F16, and BF16 output without intermediate allocations.
 *
 * Input:
 *   0: raw quantized bytes as INT8 NDArray (flat 1D)
 *
 * Output:
 *   0: dequantized NDArray with shape and type specified by iArgs
 *
 * Integer arguments:
 *   0: GGML quant type (0=Q4_0, 1=Q4_1, 2=Q5_0, 3=Q5_1, 4=Q8_0, 5=Q8_1,
 *      6=Q2_K, 7=Q3_K, 8=Q4_K, 9=Q5_K, 10=Q6_K)
 *   1: output data type (0=FLOAT32, 1=FLOAT16, 2=BFLOAT16)
 *   2..N: output shape dimensions
 */
#if NOT_EXCLUDED(OP_ggml_dequantize)
DECLARE_CUSTOM_OP(ggml_dequantize, 1, 1, false, 0, 1);
#endif

/**
 * per_layer_embedding - Per-Layer Embedding (Gemma 4)
 *
 * Adds a per-layer residual from a second embedding table to the hidden states.
 * Each decoder layer gets a small additive signal from a dedicated embedding table
 * indexed by the original token IDs. This is computed once before multimodal
 * features merge into the embedding sequence.
 *
 * output = hidden_states + ple_weight[token_ids] * scale
 *
 * Input:
 *   0: hidden_states [batch, seq_len, hidden_dim]
 *   1: ple_weight [vocab_size, hidden_dim] - per-layer embedding table
 *   2: token_ids [batch, seq_len] (INT64 or INT32)
 *
 * Output:
 *   0: output [batch, seq_len, hidden_dim]
 *
 * Float arguments:
 *   0: scale (default: 1.0)
 */
#if NOT_EXCLUDED(OP_per_layer_embedding)
DECLARE_CUSTOM_OP(per_layer_embedding, 3, 1, false, 0, 0);
#endif

/**
 * shared_kv_attention - Attention with Shared KV from Earlier Layer (Gemma 4)
 *
 * The last N layers do not compute their own K and V projections. Instead they
 * reuse K/V tensors produced by an earlier layer (the last non-shared layer of
 * the same attention type). This reduces memory and compute cost with minimal
 * quality impact.
 *
 * Computes standard scaled dot-product attention but takes pre-computed K/V
 * from a donor layer rather than projecting from the current hidden state.
 *
 * Input:
 *   0: query [batch, seq_len, num_heads, head_dim] - projected from current layer
 *   1: shared_key [batch, kv_seq_len, num_kv_heads, head_dim] - from donor layer
 *   2: shared_value [batch, kv_seq_len, num_kv_heads, head_dim] - from donor layer
 *   3: attention_mask [batch, 1, seq_len, kv_seq_len] (optional)
 *
 * Output:
 *   0: attention_output [batch, seq_len, num_heads, head_dim]
 *
 * Integer arguments:
 *   0: num_heads (query heads)
 *   1: num_kv_heads
 *   2: causal (0=bidirectional, 1=causal, default: 1)
 *   3: sliding_window_size (0=full context, >0=local window, default: 0)
 *
 * Float arguments:
 *   0: attention_scale (default: 1/sqrt(head_dim))
 */
#if NOT_EXCLUDED(OP_shared_kv_attention)
DECLARE_CUSTOM_OP(shared_kv_attention, 3, 1, false, 0, 0);
#endif

/**
 * squared_relu - Squared ReLU Activation Function
 *
 * Implements Squared ReLU: squared_relu(x) = max(0, x)^2
 * Used in Nemotron and other NVIDIA model architectures as an alternative
 * to SiLU/GELU in FFN layers.
 *
 * Input:
 *   0: input tensor
 *
 * Output:
 *   0: output tensor with squared ReLU applied
 */
#if NOT_EXCLUDED(OP_squared_relu)
DECLARE_CUSTOM_OP(squared_relu, 1, 1, true, 0, 0);
DECLARE_CUSTOM_OP(squared_relu_bp, 2, 1, true, 0, 0);
#endif

/**
 * mamba2_ssm - Mamba-2 State Space Model (SSD)
 *
 * Implements the Mamba-2 SSD (State Space Duality) recurrence.
 * Unlike Mamba-1 selective_scan which uses per-element diagonal state,
 * Mamba-2 uses head-structured state with scalar per-head decay:
 *
 *   A_bar = exp(A * dt)           (per-head scalar decay)
 *   h_t = A_bar * h_{t-1} + (B_t * dt) outer x_t
 *   y_t = C_t * h_t + D * x_t    (skip connection)
 *
 * Input:
 *   0: x [B, L, D] - input after conv+activation
 *   1: A [H] - per-head scalar decay (log-space)
 *   2: B [B, L, N] - input-dependent state expansion
 *   3: C [B, L, N] - input-dependent state contraction
 *   4: dt [B, L, H] - discretization timestep (post-softplus)
 *   5: D [H] - skip connection (optional)
 *   6: state_in [B, H, N, P] - previous state (optional)
 *
 * Output:
 *   0: y [B, L, D] - output
 *   1: state_out [B, H, N, P] - final state
 *
 * Integer arguments:
 *   0: num_heads (H)
 *   1: head_dim (P = D/H)
 *   2: state_dim (N)
 */
#if NOT_EXCLUDED(OP_mamba2_ssm)
DECLARE_CUSTOM_OP(mamba2_ssm, 5, 2, false, 0, 3);
#endif

/**
 * dual_rope - Dual Rotary Position Embedding (Gemma 4)
 *
 * Applies two different RoPE configurations depending on attention type:
 * - Standard RoPE for sliding-window (local) attention layers
 * - Proportional RoPE with a different freq_base for global full-context layers
 *
 * This enables longer context windows by using different position encoding
 * frequencies for local vs global attention.
 *
 * Input:
 *   0: input tensor [batch, seq_len, num_heads, head_dim]
 *
 * Output:
 *   0: tensor with rotary embeddings applied [batch, seq_len, num_heads, head_dim]
 *
 * Integer arguments:
 *   0: attention_type (0=sliding/local, 1=global/full-context)
 *   1: position_offset (for KV cache continuation, default: 0)
 *
 * Float arguments:
 *   0: local_freq_base (for sliding window layers, default: 10000.0)
 *   1: global_freq_base (for full-context layers, default: 1000000.0)
 *   2: local_freq_scale (default: 1.0)
 *   3: global_freq_scale (default: 1.0)
 */
#if NOT_EXCLUDED(OP_dual_rope)
DECLARE_CUSTOM_OP(dual_rope, 1, 1, false, 0, 0);
#endif

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HEADERS_LLM_H
