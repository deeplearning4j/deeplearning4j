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
//  @author raver119@gmail.com
//

#ifndef LIBND4J_HEADERS_NN_H
#define LIBND4J_HEADERS_NN_H
#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

#if NOT_EXCLUDED(OP_softmax)
DECLARE_CONFIGURABLE_OP(softmax, 1, 1, true, 0, 0);
DECLARE_CONFIGURABLE_OP(softmax_bp, 3, 1, true, 0, 0);
#endif

/**
 * Local response normalization implementation as TF.
 * input: 4D array
 *
 * T args:
 *
 * 0: bias
 * 1: alpha
 * 2: beta
 *
 * Int arg: depth - optional local radius
 *
 * output - 4D array
 */
#if NOT_EXCLUDED(OP_lrn)
DECLARE_CONFIGURABLE_OP(lrn, 1, 1, true, 3, 0);
#endif

/**
 * Local response normalization - backprop variant.
 * input:
 *  0 - 4D array of data
 *  1 - epsilon - 4D array of approximation
 *
 * T args:
 *
 * 0: bias
 * 1: alpha
 * 2: beta
 *
 * Int arg: depth - optional local radius
 *
 * output - next approximation as 4D array
 */
#if NOT_EXCLUDED(OP_lrn)
DECLARE_CONFIGURABLE_OP(lrn_bp, 2, 1, true, 3, 0);
#endif

/**
 * Batch normalization implementation.
 * Reference: https://arxiv.org/abs/1502.03167v3
 *
 * Expected arguments:
 * input: input array (any number of dimensions)
 * mean:
 * variance:
 * gamma:
 * beta:
 *
 * Int args:
 * 0: apply scale
 * 1: apply offset
 *
 *
 * T args:
 * 0: epsilon
 */
#if NOT_EXCLUDED(OP_batchnorm)
DECLARE_CUSTOM_OP(batchnorm, 3, 1, false, 1, 2);
/**
 * back prop in batch normalization
 *
 * Expected arguments:
 * input: input array (any number of dimensions)
 * mean:
 * variance:
 * gamma: optional
 * beta: optional
 * dLdOut: next epsilon
 *
 * Int args:
 * 0: apply scale
 * 1: apply offset
 *
 * T args:
 * 0: epsilon
 *
 * output arrays:
 * dL/dInput
 * dL/dMean
 * dL/dVariance
 * dL/dGamma, optional
 * dL/dBeta, optional
 */
DECLARE_CUSTOM_OP(batchnorm_bp, 4, 3, false, 1, 2);
#endif

/**
 * This operation updates parameters with provided gradients, wrt learning rate
 * Expected arguments:
 * x: parameters, any shape
 * y: gradients. same shape as x
 * lr: optional, learning rate
 *
 * T args:
 * 0: optional, learning rate
 */
#if NOT_EXCLUDED(OP_apply_sgd)
DECLARE_CONFIGURABLE_OP(apply_sgd, 2, 1, true, -2, 0);
#endif

/**
 * This operation performs batch normalization of layer, it is based on following article
 * https://arxiv.org/abs/1502.03167. Expected arguments: x: input 4D array of shape [bS,iH,iW,iD] (data format = NHWC)
 * or [bS,iD,iH,iW] (data format = NCHW), where bS - batch size iH - input height iW - input width iD - input depth (or
 * number of channels) scale:  1D input array of scale factors, shape [iD] offset: 1D input array of offsets (shifts),
 * shape [iD] mean: 1D input array of population mean used for inference, shape [iD], this array is required only if
 * isTraining = false variance: 1D input array of population mean used for inference, shape [iD], this array is required
 * only if isTraining = false
 *
 * T input arguments:
 * 0: epsilon, it is optional argument, default value is 0.001, this is small number to be added to the variance of x
 *
 * integer input arguments:
 * 0: dataFormat, may have two values: zero -> NHWC, unity -> NCHW
 * 1: isTraining, may have two values: zero -> inference, unity -> training
 */
#if NOT_EXCLUDED(OP_fused_batch_norm)
DECLARE_CUSTOM_OP(fused_batch_norm, 3, 1, false, 0, 2);
#endif

#if NOT_EXCLUDED(OP_log_softmax)
DECLARE_CONFIGURABLE_OP(log_softmax, 1, 1, true, 0, 0);
DECLARE_CONFIGURABLE_OP(log_softmax_bp, 2, 1, true, 0, 0);
#endif

/**
 * relu_layer = relu(x*w + b)
 */
#if NOT_EXCLUDED(OP_relu_layer)
DECLARE_CUSTOM_OP(relu_layer, 3, 1, false, 0, 0);
#endif
/**
 * applies layer normalization to input
 * y = g * standardize(x) + b
 *
 * see sd::ops::standardize
 *
 */
#if NOT_EXCLUDED(OP_layer_norm)
DECLARE_CONFIGURABLE_OP(layer_norm, 3, 1, true, 0, -2);
DECLARE_CUSTOM_OP(layer_norm_bp, 4, 1, false, 0, -2);
#endif

/**
 * This operation performs dot product attention on the given timeseries input with the given queries
 * out = sum(similarity(k_i, q) * v_i)
 *
 * similarity(k, q) = softmax(k * q) where x * q is the dot product of x and q
 *
 * Optionally with normalization step:
 * similarity(k, q) = softmax(k * q / sqrt(size(q))
 *
 * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, p. 4, eq. 1)
 *
 * Note: This supports multiple queries at once, if only one query is available the queries vector still has to
 * be 3D but can have queryCount = 1
 *
 * Note: keys and values usually is the same array. If you want to use it as the same array, simply pass it for
 * both.
 *
 * Expected arguments:
 * q: input 3D array "queries" of shape [batchSize, featureKeys, queryCount] or 4D array of shape [batchSize, numHeads,
 * featureKeys, queryCount] k: input 3D array "keys" of shape [batchSize, featureKeys, timesteps] or 4D array of shape
 * [batchSize, numHeads, featureKeys, timesteps] v: input 3D array "values" of shape [batchSize, featureValues,
 * timesteps] or 4D array of shape [batchSize, numHeads, featureValues, timesteps] mask: OPTIONAL; array that defines
 * which values should be skipped of shape [batchSize, timesteps]
 *
 * integer input arguments:
 * 0: normalization, may have two values: zero -> do not apply normalization, one -> apply normalization
 * 1: withWeights, may have two values: zero -> do not return weights, one -> return weights
 *
 * Output Arrays:
 * 0: Attention result arrays of shape [batchSize, featureValues, queryCount] or [batchSize, numHeads, featureValues,
 * queryCount] 1: OPTIONAL; Attention weights of shape [batchSize, timesteps, queryCount] or [batchSize, numHeads,
 * timesteps, queryCount]
 */
#if NOT_EXCLUDED(OP_dot_product_attention)
DECLARE_CUSTOM_OP(dot_product_attention, 3, -1, false, 0, 2);
DECLARE_CUSTOM_OP(dot_product_attention_bp, 4, 3, false, 0, 1);
#endif


/**
 * This operation performs dot product attention on the given timeseries input with the given queries
 * out = sum(similarity(k_i, q) * v_i)
 *
 * similarity(k, q) = softmax(k * q) where x * q is the dot product of x and q
 *
 * Optionally with normalization step:
 * similarity(k, q) = softmax(k * q / sqrt(size(q))
 *
 * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, p. 4, eq. 1)
 *
 * Note: This supports multiple queries at once, if only one query is available the queries vector still has to
 * be 3D but can have queryCount = 1
 *
 * Note: keys and values usually is the same array. If you want to use it as the same array, simply pass it for
 * both.
 *
 * Expected arguments:
 * q: input 3D array "queries" of shape [batchSize, featureKeys, queryCount] or 4D array of shape [batchSize, numHeads,
 * featureKeys, queryCount] k: input 3D array "keys" of shape [batchSize, featureKeys, timesteps] or 4D array of shape
 * [batchSize, numHeads, featureKeys, timesteps] v: input 3D array "values" of shape [batchSize, featureValues,
 * timesteps] or 4D array of shape [batchSize, numHeads, featureValues, timesteps] mask: OPTIONAL; array that defines
 * which values should be skipped of shape [batchSize, timesteps]
 *
 * integer input arguments:
 * 0: normalization, may have two values: zero -> do not apply normalization, one -> apply normalization
 * 1: withWeights, may have two values: zero -> do not return weights, one -> return weights
 *
 * Output Arrays:
 * 0: Attention result arrays of shape [batchSize, featureValues, queryCount] or [batchSize, numHeads, featureValues,
 * queryCount] 1: OPTIONAL; Attention weights of shape [batchSize, timesteps, queryCount] or [batchSize, numHeads,
 * timesteps, queryCount]
 */
#if NOT_EXCLUDED(OP_dot_product_attention_v2)
DECLARE_CUSTOM_OP(dot_product_attention_v2, -2, -1, false, -2, 2);
DECLARE_CUSTOM_OP(dot_product_attention_v2_bp, -2, -3, false, -2, 1);
#endif


/**
 * This performs multi-headed dot product attention on the given timeseries input
 * out = concat(head_1, head_2, ..., head_n) * Wo
 * head_i = dot_product_attention(Wq_i*q, Wk_i*k, Wv_i*v)
 *
 * Optionally with normalization when calculating the attention for each head.
 *
 * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, pp. 4,5, "3.2.2 Multi-Head Attention")
 *
 * This makes use of dot_product_attention OP support for rank 4 inputs.
 *
 * Expected arguments:
 * q: input 3D array "queries" of shape [batchSize, featureKeys, queryCount]
 * k: input 3D array "keys" of shape [batchSize, featureKeys, timesteps]
 * v: input 3D array "values" of shape [batchSize, featureValues, timesteps]
 * Wq: input query projection weights of shape [numHeads, projectedKeys, featureKeys]
 * Wk: input key projection weights of shape [numHeads, projectedKeys, featureKeys]
 * Wv: input value projection weights of shape [numHeads, projectedValues, featureValues]
 * Wo: output projection weights of shape [numHeads * projectedValues, outSize]
 * mask: OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps]
 *
 * integer input arguments:
 * 0: normalization, may have two values: zero -> do not apply normalization, one -> apply normalization
 * 1: withWeights, may have two values: zero -> do not return weights, one -> return weights
 *
 * Output Arrays:
 * 0: Attention result arrays of shape [batchSize, outSize, queryCount]
 * 1: OPTIONAL; Attention weights of shape [batchSize, numHeads, timesteps, queryCount]
 */
#if NOT_EXCLUDED(OP_multi_head_dot_product_attention)
DECLARE_CUSTOM_OP(multi_head_dot_product_attention, 7, -1, false, 0, 2);
DECLARE_CUSTOM_OP(multi_head_dot_product_attention_bp, 8, 7, false, 0, 1);
#endif

/**
 * ONNX MultiHeadAttention - for pre-projected queries, keys, values
 *
 * Compatible with Microsoft's ONNX MultiHeadAttention operator.
 * Takes already-projected Q, K, V tensors (unlike multi_head_dot_product_attention
 * which takes unprojected inputs and weight matrices).
 *
 * Inputs:
 *   0: query      [batch, seqQ, hidden] - already projected
 *   1: key        [batch, seqKV, hidden] - already projected  
 *   2: value      [batch, seqKV, hidden] - already projected
 *   3: attn_bias  [batch, numHeads, seqQ, seqKV] or broadcastable (optional)
 *   4: past_key   [batch, numHeads, pastSeq, headDim] (optional)
 *   5: past_value [batch, numHeads, pastSeq, headDim] (optional)
 *
 * Int args:
 *   0: numHeads
 *   1: useCausalMask (0 or 1)
 *
 * Float args:
 *   0: scale (0 = auto compute 1/sqrt(headDim))
 *
 * Outputs:
 *   0: output        [batch, seqQ, hidden]
 *   1: present_key   [batch, numHeads, totalSeq, headDim] (optional)
 *   2: present_value [batch, numHeads, totalSeq, headDim] (optional)
 */
#if NOT_EXCLUDED(OP_onnx_multi_head_attention)
DECLARE_CUSTOM_OP(onnx_multi_head_attention, 3, -1, false, -2, 2);
#endif

/**
 * CTC Greedy Decoder - Connectionist Temporal Classification decoding
 *
 * Used in OCR and speech recognition to decode CTC output to text sequences.
 *
 * Inputs:
 *   0: logits [batch, time, num_classes] - log probabilities from CTC output
 *   1: sequence_length [batch] (optional) - actual sequence lengths
 *
 * Integer args:
 *   0: merge_repeated - whether to merge repeated characters (default 1)
 *   1: blank_index - index of blank label (default 0)
 *
 * Outputs:
 *   0: decoded [batch, time] - decoded sequences (padded with blank)
 *   1: log_probability [batch] - log probability of decoded sequences
 */
#if NOT_EXCLUDED(OP_ctc_greedy_decoder)
DECLARE_CUSTOM_OP(ctc_greedy_decoder, 1, 2, false, 0, 2);
#endif

/**
 * Adaptive Average Pooling 2D
 *
 * Automatically computes kernel size and stride to produce output
 * of the specified spatial dimensions. Common in vision models.
 *
 * Inputs:
 *   0: input [batch, channels, height, width] (NCHW) or [batch, height, width, channels] (NHWC)
 *
 * Integer args:
 *   0: output_height - target output height
 *   1: output_width - target output width
 *   2: isNCHW - 1 for NCHW format, 0 for NHWC (default 1)
 *
 * Outputs:
 *   0: output [batch, channels, output_height, output_width] or corresponding NHWC
 */
#if NOT_EXCLUDED(OP_adaptive_avgpool2d)
DECLARE_CUSTOM_OP(adaptive_avgpool2d, 1, 1, false, 0, 3);
DECLARE_CUSTOM_OP(adaptive_avgpool2d_bp, 2, 1, false, 0, 3);
DECLARE_CUSTOM_OP(adaptive_maxpool2d, 1, 1, false, 0, 3);
DECLARE_CUSTOM_OP(adaptive_maxpool2d_bp, 2, 1, false, 0, 3);
DECLARE_CUSTOM_OP(adaptive_avgpool3d, 1, 1, false, 0, 4);
#endif

/**
 * Mixture of Experts (MoE) layer
 *
 * Implements sparse MoE routing where each token is processed by top-k
 * selected experts. Used in models like DeepSeek, Mixtral, etc.
 *
 * Inputs:
 *   0: input [batch, seq_len, hidden_size] - input embeddings
 *   1: router_weights [hidden_size, num_experts] - router projection
 *   2: expert_weights [num_experts, hidden_size, expert_hidden] - expert weight matrices
 *   3: expert_bias [num_experts, expert_hidden] (optional) - expert biases
 *
 * Integer args:
 *   0: num_experts - total number of experts
 *   1: top_k - number of experts to route to per token (default 2)
 *   2: normalize_probs - whether to normalize router probs (default 1)
 *
 * T args:
 *   0: capacity_factor - expert capacity factor (default 1.0)
 *
 * Outputs:
 *   0: output [batch, seq_len, expert_hidden] - combined expert outputs
 *   1: router_probs [batch, seq_len, num_experts] - router probabilities
 *   2: expert_indices [batch, seq_len, top_k] - selected expert indices
 */
#if NOT_EXCLUDED(OP_mixture_of_experts)
DECLARE_CUSTOM_OP(mixture_of_experts, 3, 3, false, -2, 3);
#endif

/**
 * Deformable Convolution 2D
 *
 * Implements deformable convolution where learned offsets are added to
 * the regular sampling grid, allowing the convolution to adapt to
 * geometric transformations in the input.
 *
 * Reference: "Deformable Convolutional Networks" (Dai et al., 2017)
 *            "Deformable ConvNets v2" (Zhu et al., 2019)
 *
 * Inputs:
 *   0: input [batch, channels, height, width] (NCHW) or NHWC
 *   1: weights [out_channels, in_channels/groups, kernel_h, kernel_w]
 *   2: offset [batch, 2*kernel_h*kernel_w*offset_groups, out_h, out_w]
 *   3: bias [out_channels] (optional)
 *   4: mask [batch, kernel_h*kernel_w*offset_groups, out_h, out_w] (optional, for v2)
 *
 * Integer args:
 *   0: kernel_h
 *   1: kernel_w
 *   2: stride_h
 *   3: stride_w
 *   4: pad_h
 *   5: pad_w
 *   6: dilation_h
 *   7: dilation_w
 *   8: groups
 *   9: offset_groups (deformable groups)
 *   10: isNCHW (1 for NCHW, 0 for NHWC)
 *
 * Outputs:
 *   0: output [batch, out_channels, out_h, out_w] or corresponding NHWC
 */
#if NOT_EXCLUDED(OP_deformable_conv2d)
DECLARE_CUSTOM_OP(deformable_conv2d, 3, 1, false, 0, 11);
#endif

/**
 * Windowed Attention - Local/Sliding Window Attention
 *
 * Implements windowed attention mechanisms used in efficient transformers
 * like Longformer, BigBird, Swin Transformer, and SAM.
 *
 * Supports both 1D (sequence) and 2D (spatial) windowed attention.
 *
 * Inputs:
 *   0: query [batch, seq_len, num_heads, head_dim] or [batch, height, width, num_heads, head_dim]
 *   1: key [batch, seq_len, num_heads, head_dim] or corresponding 2D
 *   2: value [batch, seq_len, num_heads, head_dim] or corresponding 2D
 *   3: relative_position_bias [num_heads, window_size, window_size] (optional)
 *   4: attention_mask [batch, num_heads, window_size, window_size] (optional)
 *
 * Integer args:
 *   0: window_size - size of attention window
 *   1: num_heads - number of attention heads
 *   2: shift_size - shift for shifted window attention (default 0)
 *
 * T args:
 *   0: scale - attention scale factor (default 1/sqrt(head_dim))
 *
 * Outputs:
 *   0: output - same shape as query
 *   1: attention_weights (optional) [batch, num_windows, num_heads, window_size, window_size]
 */
#if NOT_EXCLUDED(OP_windowed_attention)
DECLARE_CUSTOM_OP(windowed_attention, 3, -1, false, -2, 2);
#endif

/**
 * Relative Position Bias - Compute relative position bias for attention
 *
 * Supports two modes:
 * 1. Learned bias (Swin/SAM style): looks up bias values from a learned table
 * 2. ALiBi: computes linear position-based bias
 *
 * Inputs (for learned bias):
 *   0: bias_table [num_relative_positions, num_heads] - learned bias values
 *   1: relative_position_index [window_size^2, window_size^2] (optional)
 *
 * Inputs (for ALiBi):
 *   0: sequence_length (scalar) - length of sequence
 *
 * Integer args:
 *   0: num_heads - number of attention heads
 *   1: window_size - window size for 2D position encoding
 *   2: use_alibi - 1 for ALiBi mode, 0 for learned bias mode
 *
 * Outputs:
 *   0: bias [num_heads, window_size^2, window_size^2] or [num_heads, seq_len, seq_len]
 */
#if NOT_EXCLUDED(OP_relative_position_bias)
DECLARE_CUSTOM_OP(relative_position_bias, 1, 1, false, 0, 3);
#endif

/**
 * token_sample - Token sampling for LLM inference
 *
 * Full sampling pipeline in a single native call:
 *   temperature scaling -> top-K -> softmax -> top-P -> sample/argmax
 *
 * Input:
 *   0: logits [batch, vocabSize], [vocabSize], or [batch, seqLen, vocabSize]
 *      For rank 3, samples from the last sequence position (seqLen-1).
 *
 * Output:
 *   0: token indices INT64 [batch] or scalar
 *
 * Float args:
 *   0: temperature (default: 0.0 = greedy/argmax)
 *   1: topP nucleus threshold (default: 0.0 = disabled)
 *
 * Int args:
 *   0: topK (default: 0 = disabled)
 *   1: seed (default: 0 = random)
 */
#if NOT_EXCLUDED(OP_token_sample)
DECLARE_CUSTOM_OP(token_sample, 1, 1, false, 0, 0);
#endif

//////////////////////////////////////////////////////////////////////////
/**
 * ema_update - Exponential Moving Average parameter update
 *
 * Used in DINOv2 for updating teacher network parameters:
 *   output = decay * shadow + (1 - decay) * model
 *
 * Input:
 *   0: model parameters (student) - any shape
 *   1: shadow parameters (teacher/EMA) - same shape as model
 *
 * Float args:
 *   0: decay factor (default: 0.999, typically 0.996-0.9999)
 *
 * Output:
 *   0: updated shadow parameters - same shape as inputs
 */
#if NOT_EXCLUDED(OP_ema_update)
DECLARE_CUSTOM_OP(ema_update, 2, 1, false, 0, 0);
DECLARE_CUSTOM_OP(ema_update_bp, 3, 2, false, 0, 0);
#endif

//////////////////////////////////////////////////////////////////////////
/**
 * center_and_sharpen - DINOv2 centering and sharpening
 *
 * Prevents mode collapse in self-supervised learning:
 *   output = softmax((input - center) / temperature)
 *
 * Input:
 *   0: input logits [batch, features] - teacher output
 *   1: center vector [features] - running mean of teacher outputs
 *
 * Float args:
 *   0: temperature (default: 0.07, typically 0.04-0.07)
 *
 * Output:
 *   0: sharpened probabilities [batch, features]
 */
#if NOT_EXCLUDED(OP_center_and_sharpen)
DECLARE_CUSTOM_OP(center_and_sharpen, 2, 1, false, 0, 0);
DECLARE_CUSTOM_OP(center_and_sharpen_bp, 3, 2, false, 0, 0);
#endif

//////////////////////////////////////////////////////////////////////////
/**
 * two_way_cross_attention - SAM-style bidirectional cross-attention
 *
 * Performs two cross-attention operations simultaneously:
 *   1. tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV
 *   2. imageOutput = softmax(imageQ @ tokenK^T * scale) @ tokenV
 *
 * Input:
 *   0: tokenQuery  [batch, tokenSeqLen, embedDim]
 *   1: tokenKey    [batch, tokenSeqLen, embedDim]
 *   2: tokenValue  [batch, tokenSeqLen, embedDim]
 *   3: imageQuery  [batch, imageSeqLen, embedDim]
 *   4: imageKey    [batch, imageSeqLen, embedDim]
 *   5: imageValue  [batch, imageSeqLen, embedDim]
 *
 * Float args:
 *   0: scale factor (default: 1/sqrt(embedDim))
 *
 * Output:
 *   0: tokenOutput [batch, tokenSeqLen, embedDim]
 *   1: imageOutput [batch, imageSeqLen, embedDim]
 */
#if NOT_EXCLUDED(OP_two_way_cross_attention)
DECLARE_CUSTOM_OP(two_way_cross_attention, 6, 2, false, 0, 0);
DECLARE_CUSTOM_OP(two_way_cross_attention_bp, 8, 6, false, 0, 0);
#endif

/**
 * kv_scatter - Batch KV cache scatter update
 *
 * Copies a single time-step slice from each present KV tensor into the
 * corresponding static KV buffer at a given cache position.
 * Replaces N individual Java view+assign calls with one native kernel launch.
 *
 * Inputs (2*N tensors):
 *   [0..N-1]:   present_kv tensors [batch, heads, seqLen, dim] (source; last position extracted)
 *   [N..2N-1]:  static_kv  tensors [batch, heads, maxKvLen, dim] (destination; written in-place)
 *
 * Int args:
 *   0: cachePos  — position in static buffer to write to
 *   1: numPairs  — N (number of present/static pairs)
 *
 * Output:
 *   0: scalar INT64 (returns 0 on success; framework requires at least 1 output)
 */
#if NOT_EXCLUDED(OP_kv_scatter)
DECLARE_CUSTOM_OP(kv_scatter, 2, 1, false, 0, 1);
#endif

/**
 * turbo_quant_attention - Asymmetric attention with TurboQuant compressed keys.
 *
 * Computes scaled dot-product attention using compressed key representations
 * from TurboQuant's two-stage quantization (ICLR 2026). The asymmetric inner
 * product estimator combines MSE reconstruction with QJL correction:
 *
 *   score(q, k) ≈ <q, k_mse> + ||r|| * sqrt(π/2)/m * <S@q, signs>
 *
 * Input 0: Q              [B, H, Sq, D] query
 * Input 1: K_mse          [B, H, Sk, D] MSE-reconstructed keys (FLOAT16)
 * Input 2: QJL_signs      [B, H, Sk, D] sign bits (INT8, +1/-1)
 * Input 3: residual_norms [B, H, Sk]    residual L2 norms (FLOAT16)
 * Input 4: QJL_matrix     [D, D]        random Gaussian projection matrix
 * Input 5: V              [B, H, Sk, D] dequantized values
 * Input 6: attention_mask [B, 1, 1, Sk] or compatible broadcastable shape
 *
 * IArgs: 0=numHeads, 1=headDim
 * TArgs: 0=scale (0=auto: 1/sqrt(headDim))
 *
 * Output 0: [B, H, Sq, D]
 */
#if NOT_EXCLUDED(OP_turbo_quant_attention)
DECLARE_CUSTOM_OP(turbo_quant_attention, 7, 1, false, 1, 2);
#endif

}  // namespace ops
}  // namespace sd

#endif
