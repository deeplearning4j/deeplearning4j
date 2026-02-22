/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.common.base.Preconditions;
import org.nd4j.enums.PadMode;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDNN {
  public NDNN() {
  }

  /**
   * Concatenates a ReLU which selects only the positive part of the activation with a ReLU which selects only the negative part of the activation. Note that as a result this non-linearity doubles the depth of the activations.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray cReLU(INDArray x) {
    NDValidation.validateNumerical("CReLU", "x", x);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CReLU(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Neural network batch normalization operation.<br>
   * For details, see <a href="https://arxiv.org/abs/1502.03167">https://arxiv.org/abs/1502.03167</a><br>
   *
   * @param input Input variable. (NUMERIC type)
   * @param mean Mean value. For 1d axis, this should match input.size(axis) (NUMERIC type)
   * @param variance Variance value. For 1d axis, this should match input.size(axis) (NUMERIC type)
   * @param gamma Gamma value. For 1d axis, this should match input.size(axis) (NUMERIC type)
   * @param beta Beta value. For 1d axis, this should match input.size(axis) (NUMERIC type)
   * @param epsilon Epsilon constant for numerical stability (to avoid division by 0)
   * @param axis For 2d CNN activations: 1 for NCHW format activations, or 3 for NHWC format activations.
   * For 3d CNN activations: 1 for NCDHW format, 4 for NDHWC
   * For 1d/RNN activations: 1 for NCW format, 2 for NWC (Size: AtLeast(min=1))
   * @return output variable for batch normalization (NUMERIC type)
   */
  public INDArray batchNorm(INDArray input, INDArray mean, INDArray variance, INDArray gamma,
      INDArray beta, double epsilon, int... axis) {
    NDValidation.validateNumerical("batchNorm", "input", input);
    NDValidation.validateNumerical("batchNorm", "mean", mean);
    NDValidation.validateNumerical("batchNorm", "variance", variance);
    NDValidation.validateNumerical("batchNorm", "gamma", gamma);
    NDValidation.validateNumerical("batchNorm", "beta", beta);
    Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm(input, mean, variance, gamma, beta, epsilon, axis));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Bias addition operation: a special case of addition, typically used with CNN 4D activations and a 1D bias vector<br>
   *
   * @param input 4d input variable (NUMERIC type)
   * @param bias 1d bias (NUMERIC type)
   * @param nchw The format - nchw=true means [minibatch, channels, height, width] format; nchw=false - [minibatch, height, width, channels].
   * Unused for 2d inputs
   * @return output Output variable, after applying bias add operation (NUMERIC type)
   */
  public INDArray biasAdd(INDArray input, INDArray bias, boolean nchw) {
    NDValidation.validateNumerical("biasAdd", "input", input);
    NDValidation.validateNumerical("biasAdd", "bias", bias);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd(input, bias, nchw));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * DINOv2 centering and sharpening operation.<br>
   * Prevents mode collapse in self-supervised learning by centering the teacher output<br>
   * and applying temperature-based sharpening:<br>
   *   output = softmax((input - center) / temperature)<br>
   *
   * @param input Teacher output logits [batch, features] (NUMERIC type)
   * @param center Running center vector [features] (NUMERIC type)
   * @param temperature Sharpening temperature (typically 0.04-0.07)
   * @return output Sharpened probabilities [batch, features] (NUMERIC type)
   */
  public INDArray centerAndSharpen(INDArray input, INDArray center, double temperature) {
    NDValidation.validateNumerical("centerAndSharpen", "input", input);
    NDValidation.validateNumerical("centerAndSharpen", "center", center);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CenterAndSharpen(input, center, temperature));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * DINOv2 centering and sharpening operation.<br>
   * Prevents mode collapse in self-supervised learning by centering the teacher output<br>
   * and applying temperature-based sharpening:<br>
   *   output = softmax((input - center) / temperature)<br>
   *
   * @param input Teacher output logits [batch, features] (NUMERIC type)
   * @param center Running center vector [features] (NUMERIC type)
   * @return output Sharpened probabilities [batch, features] (NUMERIC type)
   */
  public INDArray centerAndSharpen(INDArray input, INDArray center) {
    NDValidation.validateNumerical("centerAndSharpen", "input", input);
    NDValidation.validateNumerical("centerAndSharpen", "center", center);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CenterAndSharpen(input, center, 0.07));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * CTC Greedy Decoder - Connectionist Temporal Classification decoding.<br>
   * <br>
   * Performs greedy (best path) decoding on CTC output. Used in:<br>
   * - OCR (Optical Character Recognition) - PaddleOCR, CRNN<br>
   * - Speech recognition - DeepSpeech, Wav2Vec<br>
   * - Handwriting recognition<br>
   * <br>
   * Algorithm:<br>
   * 1. At each timestep, select the class with highest probability<br>
   * 2. Optionally merge consecutive repeated characters<br>
   * 3. Remove blank labels from the output<br>
   * <br>
   * For example, with mergeRepeated=true and blankIndex=0:<br>
   * Input:  [0, 1, 1, 0, 2, 2, 2, 0] (0=blank, 1='a', 2='b')<br>
   * Output: [1, 2] -> "ab"<br>
   * <br>
   * Note: This is greedy decoding. For better accuracy with language models,<br>
   * use beam search decoding instead.<br>
   *
   * @param logits Log probabilities from CTC output. Shape: [batch, timeSteps, numClasses] (NUMERIC type)
   * @param mergeRepeated Whether to merge repeated characters in output
   * @param blankIndex Index of the blank label in the vocabulary
   */
  public INDArray[] ctcGreedyDecoder(INDArray logits, boolean mergeRepeated, int blankIndex) {
    NDValidation.validateNumerical("ctcGreedyDecoder", "logits", logits);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CTCGreedyDecoder(logits, null, mergeRepeated, blankIndex));
  }

  /**
   * CTC Greedy Decoder - Connectionist Temporal Classification decoding.<br>
   * <br>
   * Performs greedy (best path) decoding on CTC output. Used in:<br>
   * - OCR (Optical Character Recognition) - PaddleOCR, CRNN<br>
   * - Speech recognition - DeepSpeech, Wav2Vec<br>
   * - Handwriting recognition<br>
   * <br>
   * Algorithm:<br>
   * 1. At each timestep, select the class with highest probability<br>
   * 2. Optionally merge consecutive repeated characters<br>
   * 3. Remove blank labels from the output<br>
   * <br>
   * For example, with mergeRepeated=true and blankIndex=0:<br>
   * Input:  [0, 1, 1, 0, 2, 2, 2, 0] (0=blank, 1='a', 2='b')<br>
   * Output: [1, 2] -> "ab"<br>
   * <br>
   * Note: This is greedy decoding. For better accuracy with language models,<br>
   * use beam search decoding instead.<br>
   *
   * @param logits Log probabilities from CTC output. Shape: [batch, timeSteps, numClasses] (NUMERIC type)
   * @param sequenceLength Optional actual sequence lengths. Shape: [batch] (NUMERIC type)
   * @param mergeRepeated Whether to merge repeated characters in output
   * @param blankIndex Index of the blank label in the vocabulary
   */
  public INDArray[] ctcGreedyDecoder(INDArray logits, INDArray sequenceLength,
      boolean mergeRepeated, int blankIndex) {
    NDValidation.validateNumerical("ctcGreedyDecoder", "logits", logits);
    NDValidation.validateNumerical("ctcGreedyDecoder", "sequenceLength", sequenceLength);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CTCGreedyDecoder(logits, sequenceLength, mergeRepeated, blankIndex));
  }

  /**
   * This operation performs dot product attention on the given timeseries input with the given queries<br>
   * out = sum(similarity(k_i, q) * v_i)<br>
   * <br>
   * similarity(k, q) = softmax(k * q) where x * q is the dot product of x and q<br>
   * <br>
   * Optionally with normalization step:<br>
   * similarity(k, q) = softmax(k * q / sqrt(size(q))<br>
   * <br>
   * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, p. 4, eq. 1)<br>
   * <br>
   * Note: This supports multiple queries at once, if only one query is available the queries vector still has to<br>
   * be 3D but can have queryCount = 1<br>
   * <br>
   * Note: keys and values usually is the same array. If you want to use it as the same array, simply pass it for<br>
   * both.<br>
   * <br>
   * Note: Queries, keys and values must either be all rank 3 or all rank 4 arrays. Mixing them doesn't work. The<br>
   * output rank will depend on the input rank.<br>
   *
   * @param queries input 3D array "queries" of shape [batchSize, featureKeys, queryCount]
   * or 4D array of shape [batchSize, numHeads, featureKeys, queryCount] (NUMERIC type)
   * @param keys input 3D array "keys" of shape [batchSize, featureKeys, timesteps]
   * or 4D array of shape [batchSize, numHeads, featureKeys, timesteps] (NUMERIC type)
   * @param values input 3D array "values" of shape [batchSize, featureValues, timesteps]
   * or 4D array of shape [batchSize, numHeads, featureValues, timesteps] (NUMERIC type)
   * @param mask OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps] (NUMERIC type)
   * @param scaled normalization, false -> do not apply normalization, true -> apply normalization
   * @return output  Attention result arrays of shape [batchSize, featureValues, queryCount] or [batchSize, numHeads, featureValues, queryCount],
   * (optionally) Attention Weights of shape [batchSize, timesteps, queryCount] or [batchSize, numHeads, timesteps, queryCount] (NUMERIC type)
   */
  public INDArray dotProductAttention(INDArray queries, INDArray keys, INDArray values,
      INDArray mask, boolean scaled) {
    NDValidation.validateNumerical("dotProductAttention", "queries", queries);
    NDValidation.validateNumerical("dotProductAttention", "keys", keys);
    NDValidation.validateNumerical("dotProductAttention", "values", values);
    NDValidation.validateNumerical("dotProductAttention", "mask", mask);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttention(queries, keys, values, mask, scaled, false));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Dot product attention operation with flash attention and KV cache support.<br>
   * <br>
   * out = softmax(Q * K^T / scale + attentionBias) * V<br>
   * <br>
   * For 4D inputs [batch, seq, heads, dim], uses memory-efficient flash attention algorithm.<br>
   * For 2D/3D inputs, uses standard attention computation.<br>
   * <br>
   * Flash attention features:<br>
   * - O(N) memory complexity instead of O(N^2)<br>
   * - Tiled computation with online softmax<br>
   * - Supports grouped query attention (GQA) where numHeads > numKvHeads<br>
   * - Supports attention bias (relative position bias, ALiBi, etc.)<br>
   * <br>
   * KV Cache support for autoregressive generation:<br>
   * - Pass keyCache and valueCache tensors<br>
   * - Set kvCachePosition to current generation position<br>
   * - Cached keys/values are updated in-place<br>
   * <br>
   * See "Attention is all you need" (https://arxiv.org/abs/1706.03762)<br>
   * See "FlashAttention: Fast and Memory-Efficient Exact Attention" (https://arxiv.org/abs/2205.14135)<br>
   *
   * @param queries Query tensor. Shape: [batchSize, numQueries, queryDim] or [batchSize, numQueries, numHeads, headDim] for flash attention (NUMERIC type)
   * @param values Value tensor. Shape: [batchSize, numValues, valueDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param keys Key tensor. Shape: [batchSize, numValues, keyDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param queryMask Query mask tensor (optional). Shape: [batchSize, numQueries] (NUMERIC type)
   * @param valueMask Value mask tensor (optional). Shape: [batchSize, numValues] (NUMERIC type)
   * @param scaleFactor Scaling factor applied to attention scores. 0 = auto (1/sqrt(headDim))
   * @param dropoutProbability Dropout probability applied to attention weights
   * @param useCausalMask Whether to apply causal mask for autoregressive tasks
   * @param training Whether in training mode (affects dropout)
   * @return output Output tensor. Shape: [batchSize, numQueries, valueDim] or [batchSize, numQueries, numHeads, headDim] (NUMERIC type)
   */
  public INDArray dotProductAttentionV2(INDArray queries, INDArray values, INDArray keys,
      INDArray queryMask, INDArray valueMask, double scaleFactor, double dropoutProbability,
      boolean useCausalMask, boolean training) {
    NDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    NDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    NDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    NDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    NDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(queries, values, keys, queryMask, valueMask, null, scaleFactor, dropoutProbability, useCausalMask, training));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Dot product attention operation with flash attention and KV cache support.<br>
   * <br>
   * out = softmax(Q * K^T / scale + attentionBias) * V<br>
   * <br>
   * For 4D inputs [batch, seq, heads, dim], uses memory-efficient flash attention algorithm.<br>
   * For 2D/3D inputs, uses standard attention computation.<br>
   * <br>
   * Flash attention features:<br>
   * - O(N) memory complexity instead of O(N^2)<br>
   * - Tiled computation with online softmax<br>
   * - Supports grouped query attention (GQA) where numHeads > numKvHeads<br>
   * - Supports attention bias (relative position bias, ALiBi, etc.)<br>
   * <br>
   * KV Cache support for autoregressive generation:<br>
   * - Pass keyCache and valueCache tensors<br>
   * - Set kvCachePosition to current generation position<br>
   * - Cached keys/values are updated in-place<br>
   * <br>
   * See "Attention is all you need" (https://arxiv.org/abs/1706.03762)<br>
   * See "FlashAttention: Fast and Memory-Efficient Exact Attention" (https://arxiv.org/abs/2205.14135)<br>
   *
   * @param queries Query tensor. Shape: [batchSize, numQueries, queryDim] or [batchSize, numQueries, numHeads, headDim] for flash attention (NUMERIC type)
   * @param values Value tensor. Shape: [batchSize, numValues, valueDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param keys Key tensor. Shape: [batchSize, numValues, keyDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param queryMask Query mask tensor (optional). Shape: [batchSize, numQueries] (NUMERIC type)
   * @param valueMask Value mask tensor (optional). Shape: [batchSize, numValues] (NUMERIC type)
   * @param attentionBias Attention bias tensor (optional). Shape: [batchSize, numHeads, numQueries, numKeys] or broadcastable. Added to attention scores before softmax. (NUMERIC type)
   * @param scaleFactor Scaling factor applied to attention scores. 0 = auto (1/sqrt(headDim))
   * @param dropoutProbability Dropout probability applied to attention weights
   * @param useCausalMask Whether to apply causal mask for autoregressive tasks
   * @param training Whether in training mode (affects dropout)
   * @return output Output tensor. Shape: [batchSize, numQueries, valueDim] or [batchSize, numQueries, numHeads, headDim] (NUMERIC type)
   */
  public INDArray dotProductAttentionV2(INDArray queries, INDArray values, INDArray keys,
      INDArray queryMask, INDArray valueMask, INDArray attentionBias, double scaleFactor,
      double dropoutProbability, boolean useCausalMask, boolean training) {
    NDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    NDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    NDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    NDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    NDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    NDValidation.validateNumerical("dotProductAttentionV2", "attentionBias", attentionBias);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(queries, values, keys, queryMask, valueMask, attentionBias, scaleFactor, dropoutProbability, useCausalMask, training));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Dropout operation<br>
   *
   * @param input Input array (NUMERIC type)
   * @param inverted Whether dropout should be inverted or not.
   * @param seed the seed for dropout
   * @param probabilityValue the chance of dropping a value to 0. Maybe interpreted as 1 - p if inverted is true.
   * @return output Output (NUMERIC type)
   */
  public INDArray dropout(INDArray input, boolean inverted, int seed, double probabilityValue) {
    NDValidation.validateNumerical("dropout", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.CustomDropOut(input, inverted, seed, probabilityValue));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Dropout operation<br>
   *
   * @param input Input array (NUMERIC type)
   * @param inverted Whether dropout should be inverted or not.
   * @param probabilityValue the chance of dropping a value to 0. Maybe interpreted as 1 - p if inverted is true.
   * @return output Output (NUMERIC type)
   */
  public INDArray dropout(INDArray input, boolean inverted, double probabilityValue) {
    NDValidation.validateNumerical("dropout", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.CustomDropOut(input, inverted, 0, probabilityValue));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise exponential linear unit (ELU) function:<br>
   * out = x if x > 0<br>
   * out = a * (exp(x) - 1) if x <= 0<br>
   * with constant a = 1.0<br>
   * <p><br>
   * See: <a href="https://arxiv.org/abs/1511.07289">https://arxiv.org/abs/1511.07289</a><br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray elu(INDArray x) {
    NDValidation.validateNumerical("elu", "x", x);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ELU(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Exponential Moving Average parameter update for DINOv2 teacher networks.<br>
   * Computes: output = decay * shadow + (1 - decay) * model<br>
   * Used in self-supervised learning to maintain a slowly-updated teacher model.<br>
   *
   * @param model Current model parameters (student) (NUMERIC type)
   * @param shadow EMA shadow parameters (teacher) (NUMERIC type)
   * @param decay EMA decay factor (typically 0.996-0.9999)
   * @return output Updated shadow parameters (NUMERIC type)
   */
  public INDArray emaUpdate(INDArray model, INDArray shadow, double decay) {
    NDValidation.validateNumerical("emaUpdate", "model", model);
    NDValidation.validateNumerical("emaUpdate", "shadow", shadow);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.EmaUpdate(model, shadow, decay));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Exponential Moving Average parameter update for DINOv2 teacher networks.<br>
   * Computes: output = decay * shadow + (1 - decay) * model<br>
   * Used in self-supervised learning to maintain a slowly-updated teacher model.<br>
   *
   * @param model Current model parameters (student) (NUMERIC type)
   * @param shadow EMA shadow parameters (teacher) (NUMERIC type)
   * @return output Updated shadow parameters (NUMERIC type)
   */
  public INDArray emaUpdate(INDArray model, INDArray shadow) {
    NDValidation.validateNumerical("emaUpdate", "model", model);
    NDValidation.validateNumerical("emaUpdate", "shadow", shadow);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.EmaUpdate(model, shadow, 0.999));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Flash Attention - Memory-efficient attention computation.<br>
   * <br>
   * Uses tiled computation with online softmax to achieve O(N) memory complexity<br>
   * instead of O(N^2) for standard attention.<br>
   * <br>
   * Supports Grouped Query Attention (GQA) where numHeads > numKvHeads,<br>
   * allowing multiple query heads to share the same KV heads.<br>
   * <br>
   * out = softmax(Q * K^T / scale) * V<br>
   * <br>
   * See "FlashAttention: Fast and Memory-Efficient Exact Attention" (https://arxiv.org/abs/2205.14135)<br>
   *
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param key Key tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param value Value tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param scale Scaling factor. 0 = auto (1/sqrt(headDim))
   * @param isCausal Whether to apply causal masking
   * @param numHeads Number of query attention heads
   * @param numKvHeads Number of KV heads (0 = same as numHeads, for GQA use smaller value)
   * @return output Attention output. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public INDArray flashAttention(INDArray query, INDArray key, INDArray value, double scale,
      boolean isCausal, int numHeads, int numKvHeads) {
    NDValidation.validateNumerical("flashAttention", "query", query);
    NDValidation.validateNumerical("flashAttention", "key", key);
    NDValidation.validateNumerical("flashAttention", "value", value);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FlashAttention(query, key, value, scale, isCausal, numHeads, numKvHeads));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * GELU activation function - Gaussian Error Linear Units<br>
   * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a><br>
   * This method uses the sigmoid approximation<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray gelu(INDArray x) {
    NDValidation.validateNumerical("gelu", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.GELU(x));
  }

  /**
   * Grouped Query Attention (GQA) - Efficient attention with shared KV heads.<br>
   * <br>
   * Multiple query heads share the same key-value heads, reducing memory and<br>
   * computation while maintaining model quality. Used in LLaMA 2, Mistral, etc.<br>
   * <br>
   * numHeads must be divisible by numKvHeads. Each KV head is repeated<br>
   * (numHeads / numKvHeads) times to match query heads.<br>
   * <br>
   * Special cases:<br>
   * - numKvHeads == numHeads: Standard Multi-Head Attention (MHA)<br>
   * - numKvHeads == 1: Multi-Query Attention (MQA)<br>
   * <br>
   * See "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"<br>
   *
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param key Key tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param value Value tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param scale Scaling factor. 0 = auto (1/sqrt(headDim))
   * @param isCausal Whether to apply causal masking
   * @param numHeads Number of query attention heads
   * @param numKvHeads Number of KV heads (must divide numHeads evenly)
   * @return output Attention output. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public INDArray groupedQueryAttention(INDArray query, INDArray key, INDArray value, double scale,
      boolean isCausal, int numHeads, int numKvHeads) {
    NDValidation.validateNumerical("groupedQueryAttention", "query", query);
    NDValidation.validateNumerical("groupedQueryAttention", "key", key);
    NDValidation.validateNumerical("groupedQueryAttention", "value", value);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GroupedQueryAttention(query, key, value, scale, isCausal, numHeads, numKvHeads));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise hard sigmoid function:<br>
   * out[i] = 0 if in[i] <= -2.5<br>
   * out[1] = 0.2*in[i]+0.5 if -2.5 < in[i] < 2.5<br>
   * out[i] = 1 if in[i] >= 2.5<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray hardSigmoid(INDArray x) {
    NDValidation.validateNumerical("hardSigmoid", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid(x));
  }

  /**
   * Element-wise hard tanh function:<br>
   * out[i] = -1 if in[i] <= -1<br>
   * out[1] = in[i] if -1 < in[i] < 1<br>
   * out[i] = 1 if in[i] >= 1<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray hardTanh(INDArray x) {
    NDValidation.validateNumerical("hardTanh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.HardTanh(x));
  }

  /**
   * Derivative (dOut/dIn) of the element-wise hard Tanh function - hardTanh(INDArray)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray hardTanhDerivative(INDArray x) {
    NDValidation.validateNumerical("hardTanhDerivative", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative(x));
  }

  /**
   * KV Cache Update - Updates key-value cache for autoregressive generation.<br>
   * <br>
   * During LLM inference, past key-value pairs are cached to avoid redundant<br>
   * computation during token-by-token generation. This operation efficiently<br>
   * inserts new keys/values at the specified position.<br>
   * <br>
   * Usage pattern:<br>
   * 1. Initialize cache with zeros: [batch, maxSeqLen, numKvHeads, headDim]<br>
   * 2. For each new token, compute new K/V and update cache<br>
   * 3. Use full cached K/V for attention computation<br>
   * <br>
   * Returns updated keyCache and valueCache tensors.<br>
   *
   * @param keyCache Existing key cache. Shape: [batch, maxSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param valueCache Existing value cache. Shape: [batch, maxSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param newKeys New keys to insert. Shape: [batch, newSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param newValues New values to insert. Shape: [batch, newSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param startPosition Position in cache where new keys/values should be inserted
   */
  public INDArray[] kvCacheUpdate(INDArray keyCache, INDArray valueCache, INDArray newKeys,
      INDArray newValues, int startPosition) {
    NDValidation.validateNumerical("kvCacheUpdate", "keyCache", keyCache);
    NDValidation.validateNumerical("kvCacheUpdate", "valueCache", valueCache);
    NDValidation.validateNumerical("kvCacheUpdate", "newKeys", newKeys);
    NDValidation.validateNumerical("kvCacheUpdate", "newValues", newValues);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheUpdate(keyCache, valueCache, newKeys, newValues, startPosition));
  }

  /**
   * Batch KV cache scatter update for LLM autoregressive decoding.<br>
   * <br>
   * Copies a single time-step slice from each present KV tensor into the<br>
   * corresponding static KV buffer at a given cache position. Replaces N<br>
   * individual Java view+assign calls with a single native kernel launch.<br>
   * <br>
   * The present tensor has shape [batch, heads, seqLen, dim] where the new<br>
   * token's KV entry is at the last sequence position. This entry is extracted<br>
   * and written into the static buffer at cachePos.<br>
   * <br>
   * For multiple pairs, inputs are ordered as:<br>
   * [present_0, ..., present_{N-1}, static_0, ..., static_{N-1}]<br>
   *
   * @param present Present KV tensor from decoder output. Shape: [batch, heads, seqLen, dim] (NUMERIC type)
   * @param staticBuffer Static KV cache buffer. Shape: [batch, heads, maxKvLen, dim]. Updated in-place. (NUMERIC type)
   * @param cachePos Position in static buffer to write the new entry
   * @return output Scalar 0 on success (LONG type)
   */
  public INDArray kvScatter(INDArray present, INDArray staticBuffer, long cachePos) {
    NDValidation.validateNumerical("kvScatter", "present", present);
    NDValidation.validateNumerical("kvScatter", "staticBuffer", staticBuffer);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(present, staticBuffer, cachePos, 1));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Batch KV cache scatter update for LLM autoregressive decoding.<br>
   * <br>
   * Copies a single time-step slice from each present KV tensor into the<br>
   * corresponding static KV buffer at a given cache position. Replaces N<br>
   * individual Java view+assign calls with a single native kernel launch.<br>
   * <br>
   * The present tensor has shape [batch, heads, seqLen, dim] where the new<br>
   * token's KV entry is at the last sequence position. This entry is extracted<br>
   * and written into the static buffer at cachePos.<br>
   * <br>
   * For multiple pairs, inputs are ordered as:<br>
   * [present_0, ..., present_{N-1}, static_0, ..., static_{N-1}]<br>
   *
   * @param present Present KV tensor from decoder output. Shape: [batch, heads, seqLen, dim] (NUMERIC type)
   * @param staticBuffer Static KV cache buffer. Shape: [batch, heads, maxKvLen, dim]. Updated in-place. (NUMERIC type)
   * @param cachePos Position in static buffer to write the new entry
   * @param numPairs Number of present/static KV pairs. When > 1, inputs are [present_0..N-1, static_0..N-1]
   * @return output Scalar 0 on success (LONG type)
   */
  public INDArray kvScatter(INDArray present, INDArray staticBuffer, long cachePos, int numPairs) {
    NDValidation.validateNumerical("kvScatter", "present", present);
    NDValidation.validateNumerical("kvScatter", "staticBuffer", staticBuffer);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(present, staticBuffer, cachePos, numPairs));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Apply Layer Normalization<br>
   * <br>
   * y = gain * standardize(x) + bias<br>
   *
   * @param input Input variable (NUMERIC type)
   * @param gain Gain (NUMERIC type)
   * @param bias Bias (NUMERIC type)
   * @param channelsFirst For 2D input - unused. True for NCHW (minibatch, channels, height, width), false for NHWC data
   * @param dimensions Dimensions to perform layer norm over - dimension=1 for 2d/MLP data, dimension=1,2,3 for CNNs (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray layerNorm(INDArray input, INDArray gain, INDArray bias, boolean channelsFirst,
      long... dimensions) {
    NDValidation.validateNumerical("layerNorm", "input", input);
    NDValidation.validateNumerical("layerNorm", "gain", gain);
    NDValidation.validateNumerical("layerNorm", "bias", bias);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(input, gain, bias, channelsFirst, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Apply Layer Normalization<br>
   * <br>
   * y = gain * standardize(x) + bias<br>
   *
   * @param input Input variable (NUMERIC type)
   * @param gain Gain (NUMERIC type)
   * @param channelsFirst For 2D input - unused. True for NCHW (minibatch, channels, height, width), false for NHWC data
   * @param dimensions Dimensions to perform layer norm over - dimension=1 for 2d/MLP data, dimension=1,2,3 for CNNs (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray layerNorm(INDArray input, INDArray gain, boolean channelsFirst,
      long... dimensions) {
    NDValidation.validateNumerical("layerNorm", "input", input);
    NDValidation.validateNumerical("layerNorm", "gain", gain);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(input, gain, null, channelsFirst, dimensions));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise leaky ReLU function:<br>
   * out = x if x >= 0.0<br>
   * out = alpha * x if x < cutoff<br>
   * Alpha value is most commonly set to 0.01<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param alpha Cutoff - commonly 0.01
   * @return output Output variable (NUMERIC type)
   */
  public INDArray leakyRelu(INDArray x, double alpha) {
    NDValidation.validateNumerical("leakyRelu", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU(x, alpha));
  }

  /**
   * Leaky ReLU derivative: dOut/dIn given input.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param alpha Cutoff - commonly 0.01
   * @return output Output variable (NUMERIC type)
   */
  public INDArray leakyReluDerivative(INDArray x, double alpha) {
    NDValidation.validateNumerical("leakyReluDerivative", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative(x, alpha));
  }

  /**
   * Linear layer operation: out = mmul(in,w) + bias<br>
   * Note that bias array is optional<br>
   *
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable, shape [nIn, nOut] (NUMERIC type)
   * @param bias Optional bias variable (may be null) (NUMERIC type)
   * @param transposeA Whether to transpose input or not
   * @param transposeB Whether to transpose second input or not
   * @param transposeC Whether to transpose result or not
   * @return output Output variable (NUMERIC type)
   */
  public INDArray linear(INDArray input, INDArray weights, INDArray bias, boolean transposeA,
      boolean transposeB, boolean transposeC) {
    NDValidation.validateNumerical("linear", "input", input);
    NDValidation.validateNumerical("linear", "weights", weights);
    NDValidation.validateNumerical("linear", "bias", bias);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(input, weights, bias, transposeA, transposeB, transposeC));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Linear layer operation: out = mmul(in,w) + bias<br>
   * Note that bias array is optional<br>
   *
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable, shape [nIn, nOut] (NUMERIC type)
   * @param bias Optional bias variable (may be null) (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray linear(INDArray input, INDArray weights, INDArray bias) {
    NDValidation.validateNumerical("linear", "input", input);
    NDValidation.validateNumerical("linear", "weights", weights);
    NDValidation.validateNumerical("linear", "bias", bias);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(input, weights, bias, false, false, false));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray logSigmoid(INDArray x) {
    NDValidation.validateNumerical("logSigmoid", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.LogSigmoid(x));
  }

  /**
   * Log softmax activation<br>
   *
   * @param x  (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public INDArray logSoftmax(INDArray x) {
    NDValidation.validateNumerical("logSoftmax", "x", x);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(x));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Log softmax activation<br>
   *
   * @param x Input (NUMERIC type)
   * @param dimension Dimension along which to apply log softmax
   * @return output Output - log(softmax(input)) (NUMERIC type)
   */
  public INDArray logSoftmax(INDArray x, int dimension) {
    NDValidation.validateNumerical("logSoftmax", "x", x);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(x, dimension));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Mixture of Experts (MoE) Layer.<br>
   * <br>
   * Implements sparse MoE routing where each token is processed by only the top-k<br>
   * selected experts out of a larger pool. This enables scaling model capacity<br>
   * without proportionally increasing computation.<br>
   * <br>
   * Used in large language models like:<br>
   * - DeepSeek (DeepSeekMoE)<br>
   * - Mixtral (Mistral AI)<br>
   * - Switch Transformer (Google)<br>
   * - GShard (Google)<br>
   * <br>
   * The router computes expert selection probabilities:<br>
   * router_probs = softmax(input @ routerWeights)<br>
   * <br>
   * Top-k experts are selected and their outputs are weighted by normalized probs:<br>
   * output = sum(normalized_prob[i] * expert[i](input) for i in top_k)<br>
   * <br>
   * Benefits:<br>
   * - Scales model capacity with sublinear compute increase<br>
   * - Enables very large models with efficient inference<br>
   * - Supports expert parallelism across devices<br>
   *
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numExperts] (NUMERIC type)
   * @param expertWeights Expert weight matrices. Shape: [numExperts, hiddenSize, expertHiddenSize] (NUMERIC type)
   * @param numExperts Total number of experts
   * @param topK Number of experts to route to per token
   */
  public INDArray[] mixtureOfExperts(INDArray input, INDArray routerWeights, INDArray expertWeights,
      int numExperts, int topK) {
    NDValidation.validateNumerical("mixtureOfExperts", "input", input);
    NDValidation.validateNumerical("mixtureOfExperts", "routerWeights", routerWeights);
    NDValidation.validateNumerical("mixtureOfExperts", "expertWeights", expertWeights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MixtureOfExperts(input, routerWeights, expertWeights, null, numExperts, topK, true, 1.0));
  }

  /**
   * Mixture of Experts (MoE) Layer.<br>
   * <br>
   * Implements sparse MoE routing where each token is processed by only the top-k<br>
   * selected experts out of a larger pool. This enables scaling model capacity<br>
   * without proportionally increasing computation.<br>
   * <br>
   * Used in large language models like:<br>
   * - DeepSeek (DeepSeekMoE)<br>
   * - Mixtral (Mistral AI)<br>
   * - Switch Transformer (Google)<br>
   * - GShard (Google)<br>
   * <br>
   * The router computes expert selection probabilities:<br>
   * router_probs = softmax(input @ routerWeights)<br>
   * <br>
   * Top-k experts are selected and their outputs are weighted by normalized probs:<br>
   * output = sum(normalized_prob[i] * expert[i](input) for i in top_k)<br>
   * <br>
   * Benefits:<br>
   * - Scales model capacity with sublinear compute increase<br>
   * - Enables very large models with efficient inference<br>
   * - Supports expert parallelism across devices<br>
   *
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numExperts] (NUMERIC type)
   * @param expertWeights Expert weight matrices. Shape: [numExperts, hiddenSize, expertHiddenSize] (NUMERIC type)
   * @param expertBias Optional expert biases. Shape: [numExperts, expertHiddenSize] (NUMERIC type)
   * @param numExperts Total number of experts
   * @param topK Number of experts to route to per token
   * @param normalizeProbs Whether to normalize router probabilities for selected experts
   * @param capacityFactor Expert capacity factor for load balancing
   */
  public INDArray[] mixtureOfExperts(INDArray input, INDArray routerWeights, INDArray expertWeights,
      INDArray expertBias, int numExperts, int topK, boolean normalizeProbs,
      double capacityFactor) {
    NDValidation.validateNumerical("mixtureOfExperts", "input", input);
    NDValidation.validateNumerical("mixtureOfExperts", "routerWeights", routerWeights);
    NDValidation.validateNumerical("mixtureOfExperts", "expertWeights", expertWeights);
    NDValidation.validateNumerical("mixtureOfExperts", "expertBias", expertBias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MixtureOfExperts(input, routerWeights, expertWeights, expertBias, numExperts, topK, normalizeProbs, capacityFactor));
  }

  /**
   * This performs multi-headed dot product attention on the given timeseries input<br>
   * out = concat(head_1, head_2, ..., head_n) * Wo<br>
   * head_i = dot_product_attention(Wq_i*q, Wk_i*k, Wv_i*v)<br>
   * <br>
   * Optionally with normalization when calculating the attention for each head.<br>
   * <br>
   * See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, pp. 4,5, "3.2.2 Multi-Head Attention")<br>
   * <br>
   * This makes use of dot_product_attention OP support for rank 4 inputs.<br>
   * see dotProductAttention(INDArray, INDArray, INDArray, INDArray, boolean, boolean)<br>
   *
   * @param queries input 3D array "queries" of shape [batchSize, featureKeys, queryCount] (NUMERIC type)
   * @param keys input 3D array "keys" of shape [batchSize, featureKeys, timesteps] (NUMERIC type)
   * @param values input 3D array "values" of shape [batchSize, featureValues, timesteps] (NUMERIC type)
   * @param Wq input query projection weights of shape [numHeads, projectedKeys, featureKeys] (NUMERIC type)
   * @param Wk input key projection weights of shape [numHeads, projectedKeys, featureKeys] (NUMERIC type)
   * @param Wv input value projection weights of shape [numHeads, projectedValues, featureValues] (NUMERIC type)
   * @param Wo output projection weights of shape [numHeads * projectedValues, outSize] (NUMERIC type)
   * @param mask OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps] (NUMERIC type)
   * @param scaled normalization, false -> do not apply normalization, true -> apply normalization
   * @return output Attention result arrays of shape [batchSize, outSize, queryCount]
   * (optionally) Attention Weights of shape [batchSize, numHeads, timesteps, queryCount] (NUMERIC type)
   */
  public INDArray multiHeadDotProductAttention(INDArray queries, INDArray keys, INDArray values,
      INDArray Wq, INDArray Wk, INDArray Wv, INDArray Wo, INDArray mask, boolean scaled) {
    NDValidation.validateNumerical("multiHeadDotProductAttention", "queries", queries);
    NDValidation.validateNumerical("multiHeadDotProductAttention", "keys", keys);
    NDValidation.validateNumerical("multiHeadDotProductAttention", "values", values);
    NDValidation.validateNumerical("multiHeadDotProductAttention", "Wq", Wq);
    NDValidation.validateNumerical("multiHeadDotProductAttention", "Wk", Wk);
    NDValidation.validateNumerical("multiHeadDotProductAttention", "Wv", Wv);
    NDValidation.validateNumerical("multiHeadDotProductAttention", "Wo", Wo);
    NDValidation.validateNumerical("multiHeadDotProductAttention", "mask", mask);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttention(queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, false));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Padding operation<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param padding Padding value (NUMERIC type)
   * @param PadMode Padding format
   * @param constant Padding constant
   * @return output Padded input (NUMERIC type)
   */
  public INDArray pad(INDArray input, INDArray padding, PadMode PadMode, double constant) {
    NDValidation.validateNumerical("pad", "input", input);
    NDValidation.validateNumerical("pad", "padding", padding);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.Pad(input, padding, PadMode, constant));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Padding operation<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param padding Padding value (NUMERIC type)
   * @param constant Padding constant
   * @return output Padded input (NUMERIC type)
   */
  public INDArray pad(INDArray input, INDArray padding, double constant) {
    NDValidation.validateNumerical("pad", "input", input);
    NDValidation.validateNumerical("pad", "padding", padding);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.Pad(input, padding, PadMode.CONSTANT, constant));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * GELU activation function - Gaussian Error Linear Units<br>
   * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a><br>
   * This method uses the precise method<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray preciseGelu(INDArray x) {
    NDValidation.validateNumerical("preciseGelu", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELU(x));
  }

  /**
   * PReLU (Parameterized Rectified Linear Unit) operation.  Like LeakyReLU with a learnable alpha:<br>
   * out[i] = in[i] if in[i] >= 0<br>
   * out[i] = in[i] * alpha[i] otherwise<br>
   * <br>
   * sharedAxes allows you to share learnable parameters along axes.<br>
   * For example, if the input has shape [batchSize, channels, height, width]<br>
   * and you want each channel to have its own cutoff, use sharedAxes = [2, 3] and an<br>
   * alpha with shape [channels].<br>
   *
   * @param input Input data (NUMERIC type)
   * @param alpha The cutoff variable.  Note that the batch dimension (the 0th, whether it is batch or not) should not be part of alpha. (NUMERIC type)
   * @param sharedAxes Which axes to share cutoff parameters along. (Size: AtLeast(min=1))
   * @return output Output (NUMERIC type)
   */
  public INDArray prelu(INDArray input, INDArray alpha, int... sharedAxes) {
    NDValidation.validateNumerical("prelu", "input", input);
    NDValidation.validateNumerical("prelu", "alpha", alpha);
    Preconditions.checkArgument(sharedAxes.length >= 1, "sharedAxes has incorrect size/length. Expected: sharedAxes.length >= 1, got %s", sharedAxes.length);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.PRelu(input, alpha, sharedAxes));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Relative Position Bias - Compute relative position bias for attention.<br>
   * <br>
   * Supports two modes:<br>
   * 1. Learned bias (Swin/SAM style): Looks up bias values from a learned table<br>
   *    based on relative positions between query and key positions.<br>
   * <br>
   * 2. ALiBi (Attention with Linear Biases): Computes linear position-based bias<br>
   *    without learned parameters. More efficient for very long sequences.<br>
   * <br>
   * For learned bias mode:<br>
   * - biasTable shape: [(2*windowSize-1)^2, numHeads] for 2D<br>
   * - Output is gathered based on relative position indices<br>
   * <br>
   * For ALiBi mode:<br>
   * - biasTable can be sequence length (scalar) or input tensor<br>
   * - Computes m_h * |i - j| where m_h = 2^(-8*h/H)<br>
   * <br>
   * Reference: "Swin Transformer" (Liu et al., 2021)<br>
   *            "Train Short, Test Long" (Press et al., 2021) for ALiBi<br>
   *
   * @param biasTable Learned bias table. Shape: [numRelativePositions, numHeads] for learned mode, or scalar/tensor for ALiBi mode (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param windowSize Window size for 2D position encoding (used if generating index)
   * @return output Position bias. Shape: [numHeads, windowSize^2, windowSize^2] or [numHeads, seqLen, seqLen] (NUMERIC type)
   */
  public INDArray relativePositionBias(INDArray biasTable, int numHeads, int windowSize) {
    NDValidation.validateNumerical("relativePositionBias", "biasTable", biasTable);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RelativePositionBias(biasTable, null, numHeads, windowSize, false));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Relative Position Bias - Compute relative position bias for attention.<br>
   * <br>
   * Supports two modes:<br>
   * 1. Learned bias (Swin/SAM style): Looks up bias values from a learned table<br>
   *    based on relative positions between query and key positions.<br>
   * <br>
   * 2. ALiBi (Attention with Linear Biases): Computes linear position-based bias<br>
   *    without learned parameters. More efficient for very long sequences.<br>
   * <br>
   * For learned bias mode:<br>
   * - biasTable shape: [(2*windowSize-1)^2, numHeads] for 2D<br>
   * - Output is gathered based on relative position indices<br>
   * <br>
   * For ALiBi mode:<br>
   * - biasTable can be sequence length (scalar) or input tensor<br>
   * - Computes m_h * |i - j| where m_h = 2^(-8*h/H)<br>
   * <br>
   * Reference: "Swin Transformer" (Liu et al., 2021)<br>
   *            "Train Short, Test Long" (Press et al., 2021) for ALiBi<br>
   *
   * @param biasTable Learned bias table. Shape: [numRelativePositions, numHeads] for learned mode, or scalar/tensor for ALiBi mode (NUMERIC type)
   * @param relativePositionIndex Optional precomputed relative position index. Shape: [windowSize^2, windowSize^2] (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param windowSize Window size for 2D position encoding (used if generating index)
   * @return output Position bias. Shape: [numHeads, windowSize^2, windowSize^2] or [numHeads, seqLen, seqLen] (NUMERIC type)
   */
  public INDArray relativePositionBias(INDArray biasTable, INDArray relativePositionIndex,
      int numHeads, int windowSize) {
    NDValidation.validateNumerical("relativePositionBias", "biasTable", biasTable);
    NDValidation.validateNumerical("relativePositionBias", "relativePositionIndex", relativePositionIndex);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RelativePositionBias(biasTable, relativePositionIndex, numHeads, windowSize, false));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise rectified linear function with specified cutoff:<br>
   * out[i] = in[i] if in[i] >= cutoff<br>
   * out[i] = 0 otherwise<br>
   *
   * @param x Input (NUMERIC type)
   * @param cutoff Cutoff value for ReLU operation - x > cutoff ? x : 0. Usually 0
   * @return output Output (NUMERIC type)
   */
  public INDArray relu(INDArray x, double cutoff) {
    NDValidation.validateNumerical("relu", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear(x, cutoff));
  }

  /**
   * Element-wise "rectified linear 6" function with specified cutoff:<br>
   * out[i] = min(max(in, cutoff), 6)<br>
   *
   * @param x Input (NUMERIC type)
   * @param cutoff Cutoff value for ReLU operation. Usually 0
   * @return output Output (NUMERIC type)
   */
  public INDArray relu6(INDArray x, double cutoff) {
    NDValidation.validateNumerical("relu6", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.Relu6(x, cutoff));
  }

  /**
   * ReLU (Rectified Linear Unit) layer operation: out = relu(mmul(in,w) + bias)<br>
   *
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable (NUMERIC type)
   * @param bias  Bias variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray reluLayer(INDArray input, INDArray weights, INDArray bias) {
    NDValidation.validateNumerical("reluLayer", "input", input);
    NDValidation.validateNumerical("reluLayer", "weights", weights);
    NDValidation.validateNumerical("reluLayer", "bias", bias);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.ReluLayer(input, weights, bias));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param input Input variable (NUMERIC type)
   * @param gamma Scale/gain vector (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output RMS normalized output (NUMERIC type)
   */
  public INDArray rmsNorm(INDArray input, INDArray gamma, double epsilon) {
    NDValidation.validateNumerical("rmsNorm", "input", input);
    NDValidation.validateNumerical("rmsNorm", "gamma", gamma);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(input, gamma, epsilon));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param input Input variable (NUMERIC type)
   * @param gamma Scale/gain vector (NUMERIC type)
   * @return output RMS normalized output (NUMERIC type)
   */
  public INDArray rmsNorm(INDArray input, INDArray gamma) {
    NDValidation.validateNumerical("rmsNorm", "input", input);
    NDValidation.validateNumerical("rmsNorm", "gamma", gamma);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(input, gamma, 1.0E-5));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param input Input variable (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output RMS normalized output (NUMERIC type)
   */
  public INDArray rmsNorm(INDArray input, double epsilon) {
    NDValidation.validateNumerical("rmsNorm", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(input, null, epsilon));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param input Input variable (NUMERIC type)
   * @return output RMS normalized output (NUMERIC type)
   */
  public INDArray rmsNorm(INDArray input) {
    NDValidation.validateNumerical("rmsNorm", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(input, null, 1.0E-5));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise SeLU function - Scaled exponential Lineal Unit: see <a href="https://arxiv.org/abs/1706.02515">Self-Normalizing Neural Networks</a><br>
   * <br>
   * out[i] = scale * alpha * (exp(in[i])-1) if in[i]>0, or 0 if in[i] <= 0<br>
   * Uses default scale and alpha values.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray selu(INDArray x) {
    NDValidation.validateNumerical("selu", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.SELU(x));
  }

  /**
   * Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray sigmoid(INDArray x) {
    NDValidation.validateNumerical("sigmoid", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid(x));
  }

  /**
   * Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut<br>
   *
   * @param x Input Variable (NUMERIC type)
   * @param wrt Gradient at the output - dL/dOut. Must have same shape as the input (NUMERIC type)
   * @return output Output (gradient at input of sigmoid) (NUMERIC type)
   */
  public INDArray sigmoidDerivative(INDArray x, INDArray wrt) {
    NDValidation.validateNumerical("sigmoidDerivative", "x", x);
    NDValidation.validateNumerical("sigmoidDerivative", "wrt", wrt);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative(x, wrt));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Sliding Window Attention - Efficient attention for long sequences.<br>
   * <br>
   * Each token only attends to a fixed window of previous tokens, enabling<br>
   * efficient processing of very long sequences. Used in Mistral and other<br>
   * modern LLMs for handling long contexts.<br>
   * <br>
   * Benefits:<br>
   * - O(N * windowSize) complexity instead of O(N^2)<br>
   * - Memory efficient for long sequences<br>
   * - Supports very long context lengths (e.g., 32K with 4K window)<br>
   * <br>
   * The attention mask is automatically applied to restrict each position<br>
   * to only attend to positions within [pos - windowSize, pos].<br>
   *
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param key Key tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param value Value tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param windowSize Sliding window size - tokens can only attend to this many previous positions
   * @param numHeads Number of query attention heads
   * @param numKvHeads Number of KV heads (0 = same as numHeads)
   * @param scale Scaling factor. 0 = auto (1/sqrt(headDim))
   * @return output Attention output. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public INDArray slidingWindowAttention(INDArray query, INDArray key, INDArray value,
      int windowSize, int numHeads, int numKvHeads, double scale) {
    NDValidation.validateNumerical("slidingWindowAttention", "query", query);
    NDValidation.validateNumerical("slidingWindowAttention", "key", key);
    NDValidation.validateNumerical("slidingWindowAttention", "value", value);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SlidingWindowAttention(query, key, value, windowSize, numHeads, numKvHeads, scale));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Softmax activation, along the specified dimension<br>
   *
   * @param x Input (NUMERIC type)
   * @param dimension Dimension along which to apply softmax
   * @return output Output variable (NUMERIC type)
   */
  public INDArray softmax(INDArray x, int dimension) {
    NDValidation.validateNumerical("softmax", "x", x);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(x, dimension));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Softmax activation, along the specified dimension<br>
   *
   * @param x Input (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray softmax(INDArray x) {
    NDValidation.validateNumerical("softmax", "x", x);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(x, -1));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Element-wise softplus function: out = log(exp(x) + 1)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray softplus(INDArray x) {
    NDValidation.validateNumerical("softplus", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus(x));
  }

  /**
   * Element-wise softsign function: out = x / (abs(x) + 1)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray softsign(INDArray x) {
    NDValidation.validateNumerical("softsign", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftSign(x));
  }

  /**
   * Element-wise derivative (dOut/dIn) of the softsign function softsign(INDArray)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output (NUMERIC type)
   */
  public INDArray softsignDerivative(INDArray x) {
    NDValidation.validateNumerical("softsignDerivative", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative(x));
  }

  /**
   * Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0<br>
   * See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a><br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray swish(INDArray x) {
    NDValidation.validateNumerical("swish", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Swish(x));
  }

  /**
   * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray tanh(INDArray x) {
    NDValidation.validateNumerical("tanh", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh(x));
  }

  /**
   * Token sampling for LLM inference.<br>
   * <br>
   * Full sampling pipeline in a single native GPU call:<br>
   *   temperature scaling -> top-K filtering -> softmax -> top-P filtering -> sample/argmax<br>
   * <br>
   * For greedy decoding (temperature=0 or no top-k/top-p), performs GPU-side argmax<br>
   * with shared-memory reduction — avoids transferring the full logits tensor to host.<br>
   * <br>
   * Supports rank 1 [vocabSize], rank 2 [batch, vocabSize], and rank 3<br>
   * [batch, seqLen, vocabSize] inputs. For rank 3, the last sequence position<br>
   * is automatically extracted for sampling.<br>
   *
   * @param logits Logits tensor. Shape: [vocabSize], [batch, vocabSize], or [batch, seqLen, vocabSize]. For rank-3, samples from the last sequence position. (NUMERIC type)
   * @return output Sampled token indices. Shape: [batch] or scalar (LONG type)
   */
  public INDArray tokenSample(INDArray logits) {
    NDValidation.validateNumerical("tokenSample", "logits", logits);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample(logits, 0.0, 0, 0.0, 0));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Token sampling for LLM inference.<br>
   * <br>
   * Full sampling pipeline in a single native GPU call:<br>
   *   temperature scaling -> top-K filtering -> softmax -> top-P filtering -> sample/argmax<br>
   * <br>
   * For greedy decoding (temperature=0 or no top-k/top-p), performs GPU-side argmax<br>
   * with shared-memory reduction — avoids transferring the full logits tensor to host.<br>
   * <br>
   * Supports rank 1 [vocabSize], rank 2 [batch, vocabSize], and rank 3<br>
   * [batch, seqLen, vocabSize] inputs. For rank 3, the last sequence position<br>
   * is automatically extracted for sampling.<br>
   *
   * @param logits Logits tensor. Shape: [vocabSize], [batch, vocabSize], or [batch, seqLen, vocabSize]. For rank-3, samples from the last sequence position. (NUMERIC type)
   * @param temperature Temperature for sampling. 0 = greedy (argmax)
   * @param topK Top-K filtering: keep only top K logits. 0 = disabled
   * @param topP Top-P (nucleus) filtering threshold. 0 = disabled
   * @param seed Random seed for sampling. 0 = random
   * @return output Sampled token indices. Shape: [batch] or scalar (LONG type)
   */
  public INDArray tokenSample(INDArray logits, double temperature, int topK, double topP,
      long seed) {
    NDValidation.validateNumerical("tokenSample", "logits", logits);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample(logits, temperature, topK, topP, seed));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Find values and indices for the largest k entries along the last dimension.<br>
   *
   * @param input Input data (NUMERIC type)
   * @param k The number of values to return
   * @param sorted Whether to return the values sorted or not
   */
  public INDArray[] topK(INDArray input, double k, boolean sorted) {
    NDValidation.validateNumerical("topK", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TopK(input, k, sorted));
  }

  /**
   * SAM-style Two-Way Cross Attention.<br>
   * Bidirectional cross-attention where tokens attend to image features and<br>
   * image features attend to tokens simultaneously:<br>
   *   tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV<br>
   *   imageOutput = softmax(imageQ @ tokenK^T * scale) @ tokenV<br>
   *
   * @param tokenQuery Token queries [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenKey Token keys [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenValue Token values [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param imageQuery Image queries [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageKey Image keys [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageValue Image values [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param scale Attention scale factor (default: 1/sqrt(embedDim))
   */
  public INDArray[] twoWayCrossAttention(INDArray tokenQuery, INDArray tokenKey,
      INDArray tokenValue, INDArray imageQuery, INDArray imageKey, INDArray imageValue,
      double scale) {
    NDValidation.validateNumerical("twoWayCrossAttention", "tokenQuery", tokenQuery);
    NDValidation.validateNumerical("twoWayCrossAttention", "tokenKey", tokenKey);
    NDValidation.validateNumerical("twoWayCrossAttention", "tokenValue", tokenValue);
    NDValidation.validateNumerical("twoWayCrossAttention", "imageQuery", imageQuery);
    NDValidation.validateNumerical("twoWayCrossAttention", "imageKey", imageKey);
    NDValidation.validateNumerical("twoWayCrossAttention", "imageValue", imageValue);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TwoWayCrossAttention(tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue, scale));
  }

  /**
   * SAM-style Two-Way Cross Attention.<br>
   * Bidirectional cross-attention where tokens attend to image features and<br>
   * image features attend to tokens simultaneously:<br>
   *   tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV<br>
   *   imageOutput = softmax(imageQ @ tokenK^T * scale) @ tokenV<br>
   *
   * @param tokenQuery Token queries [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenKey Token keys [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenValue Token values [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param imageQuery Image queries [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageKey Image keys [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageValue Image values [batch, imageSeqLen, embedDim] (NUMERIC type)
   */
  public INDArray[] twoWayCrossAttention(INDArray tokenQuery, INDArray tokenKey,
      INDArray tokenValue, INDArray imageQuery, INDArray imageKey, INDArray imageValue) {
    NDValidation.validateNumerical("twoWayCrossAttention", "tokenQuery", tokenQuery);
    NDValidation.validateNumerical("twoWayCrossAttention", "tokenKey", tokenKey);
    NDValidation.validateNumerical("twoWayCrossAttention", "tokenValue", tokenValue);
    NDValidation.validateNumerical("twoWayCrossAttention", "imageQuery", imageQuery);
    NDValidation.validateNumerical("twoWayCrossAttention", "imageKey", imageKey);
    NDValidation.validateNumerical("twoWayCrossAttention", "imageValue", imageValue);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TwoWayCrossAttention(tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue, 0.0));
  }

  /**
   * Windowed Attention - Local/Sliding Window Attention.<br>
   * <br>
   * Implements windowed attention mechanisms used in efficient transformers like<br>
   * Longformer, BigBird, Swin Transformer, and SAM (Segment Anything Model).<br>
   * <br>
   * Supports both:<br>
   * - 1D windowed attention: for sequences [batch, seqLen, heads, dim]<br>
   * - 2D windowed attention: for images [batch, height, width, heads, dim]<br>
   * <br>
   * Shifted window attention (shiftSize > 0) enables cross-window connections<br>
   * as used in Swin Transformer.<br>
   * <br>
   * Benefits:<br>
   * - O(N * windowSize) complexity instead of O(N^2)<br>
   * - Efficient for long sequences and high-resolution images<br>
   * - Supports relative position bias for position-aware attention<br>
   *
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] for 1D or [batch, height, width, numHeads, headDim] for 2D (NUMERIC type)
   * @param key Key tensor. Same shape as query (NUMERIC type)
   * @param value Value tensor. Same shape as query (NUMERIC type)
   * @param windowSize Size of attention window
   * @param numHeads Number of attention heads
   * @return output Attention output. Same shape as query (NUMERIC type)
   */
  public INDArray windowedAttention(INDArray query, INDArray key, INDArray value, int windowSize,
      int numHeads) {
    NDValidation.validateNumerical("windowedAttention", "query", query);
    NDValidation.validateNumerical("windowedAttention", "key", key);
    NDValidation.validateNumerical("windowedAttention", "value", value);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.WindowedAttention(query, key, value, null, null, windowSize, numHeads, 0, 0.0, false));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Windowed Attention - Local/Sliding Window Attention.<br>
   * <br>
   * Implements windowed attention mechanisms used in efficient transformers like<br>
   * Longformer, BigBird, Swin Transformer, and SAM (Segment Anything Model).<br>
   * <br>
   * Supports both:<br>
   * - 1D windowed attention: for sequences [batch, seqLen, heads, dim]<br>
   * - 2D windowed attention: for images [batch, height, width, heads, dim]<br>
   * <br>
   * Shifted window attention (shiftSize > 0) enables cross-window connections<br>
   * as used in Swin Transformer.<br>
   * <br>
   * Benefits:<br>
   * - O(N * windowSize) complexity instead of O(N^2)<br>
   * - Efficient for long sequences and high-resolution images<br>
   * - Supports relative position bias for position-aware attention<br>
   *
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] for 1D or [batch, height, width, numHeads, headDim] for 2D (NUMERIC type)
   * @param key Key tensor. Same shape as query (NUMERIC type)
   * @param value Value tensor. Same shape as query (NUMERIC type)
   * @param relativePositionBias Optional relative position bias. Shape: [numHeads, windowSize, windowSize] (NUMERIC type)
   * @param attentionMask Optional attention mask (NUMERIC type)
   * @param windowSize Size of attention window
   * @param numHeads Number of attention heads
   * @param shiftSize Shift size for shifted window attention (Swin style). 0 = no shift
   * @param scale Attention scale factor. 0 = auto (1/sqrt(headDim))
   * @param returnWeights Whether to return attention weights
   * @return output Attention output. Same shape as query (NUMERIC type)
   */
  public INDArray windowedAttention(INDArray query, INDArray key, INDArray value,
      INDArray relativePositionBias, INDArray attentionMask, int windowSize, int numHeads,
      int shiftSize, double scale, boolean returnWeights) {
    NDValidation.validateNumerical("windowedAttention", "query", query);
    NDValidation.validateNumerical("windowedAttention", "key", key);
    NDValidation.validateNumerical("windowedAttention", "value", value);
    NDValidation.validateNumerical("windowedAttention", "relativePositionBias", relativePositionBias);
    NDValidation.validateNumerical("windowedAttention", "attentionMask", attentionMask);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.WindowedAttention(query, key, value, relativePositionBias, attentionMask, windowSize, numHeads, shiftSize, scale, returnWeights));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  public INDArray fusedElementwiseChain(INDArray input, int... opCodes) {
    return fusedElementwiseChain(input, null, opCodes);
  }

  public INDArray fusedElementwiseChain(INDArray input, INDArray[] secondaryInputs, int[] opCodes) {
    INDArray output = Nd4j.createUninitialized(input.dataType(), input.shape());
    org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain.ChainBuilder builder =
            org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain.builder().input(input);
    int secIdx = 0;
    for (int code : opCodes) {
      if (code < 10) {
        // binary op — needs secondary input
        if (secondaryInputs != null && secIdx < secondaryInputs.length) {
          builder.addOp(code, secondaryInputs[secIdx++]);
        } else {
          throw new IllegalArgumentException("Binary op code " + code + " requires a secondary input");
        }
      } else {
        builder.addOp(code);
      }
    }
    builder.output(output);
    Nd4j.exec(builder.build());
    return output;
  }
}
