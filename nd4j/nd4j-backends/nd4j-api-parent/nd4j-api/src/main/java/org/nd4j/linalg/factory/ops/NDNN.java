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
   * Applies Attention with Linear Biases (ALiBi) position encoding to attention scores.<br>
   *
   * @param scores Attention scores [batch, num_heads, seq_len, kv_len] (NUMERIC type)
   * @param numHeads Number of attention heads
   * @return output Scores with ALiBi position bias applied (NUMERIC type)
   */
  public INDArray applyAlibi(INDArray scores, int numHeads) {
    NDValidation.validateNumerical("applyAlibi", "scores", scores);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ApplyAlibi(scores, numHeads));
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
   * Activation-aware Weight Quantization (AWQ) matrix multiplication.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param weightPacked AWQ-packed weight (NUMERIC type)
   * @param weightScale Weight quantization scales (NUMERIC type)
   * @param groupSize Quantization group size
   * @return output Dequantized matmul result (NUMERIC type)
   */
  public INDArray awqMatmul(INDArray input, INDArray weightPacked, INDArray weightScale,
      int groupSize) {
    NDValidation.validateNumerical("awqMatmul", "input", input);
    NDValidation.validateNumerical("awqMatmul", "weightPacked", weightPacked);
    NDValidation.validateNumerical("awqMatmul", "weightScale", weightScale);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.AwqMatmul(input, weightPacked, weightScale, groupSize));
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
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   * @param convStateIn Conv state for autoregressive decode [batch, dim, kernelSize-1] (NUMERIC type)
   * @param activation Activation function (0=none, 1=silu)
   * @param wFormat Weight format (0=[D,K] PyTorch/ONNX default, 1=[K,D] TensorFlow)
   */
  public INDArray[] causalConv1d(INDArray x, INDArray weight, INDArray bias, INDArray convStateIn,
      int activation, int wFormat) {
    NDValidation.validateNumerical("causalConv1d", "x", x);
    NDValidation.validateNumerical("causalConv1d", "weight", weight);
    NDValidation.validateNumerical("causalConv1d", "bias", bias);
    NDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(x, weight, bias, convStateIn, activation, wFormat));
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   */
  public INDArray[] causalConv1d(INDArray x, INDArray weight) {
    NDValidation.validateNumerical("causalConv1d", "x", x);
    NDValidation.validateNumerical("causalConv1d", "weight", weight);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(x, weight, null, null, 0, 0));
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   */
  public INDArray[] causalConv1d(INDArray x, INDArray weight, INDArray bias) {
    NDValidation.validateNumerical("causalConv1d", "x", x);
    NDValidation.validateNumerical("causalConv1d", "weight", weight);
    NDValidation.validateNumerical("causalConv1d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(x, weight, bias, null, 0, 0));
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   * @param convStateIn Conv state for autoregressive decode [batch, dim, kernelSize-1] (NUMERIC type)
   */
  public INDArray[] causalConv1d(INDArray x, INDArray weight, INDArray bias, INDArray convStateIn) {
    NDValidation.validateNumerical("causalConv1d", "x", x);
    NDValidation.validateNumerical("causalConv1d", "weight", weight);
    NDValidation.validateNumerical("causalConv1d", "bias", bias);
    NDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(x, weight, bias, convStateIn, 0, 0));
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   * @param convStateIn Conv state for autoregressive decode [batch, dim, kernelSize-1] (NUMERIC type)
   * @param activation Activation function (0=none, 1=silu)
   */
  public INDArray[] causalConv1d(INDArray x, INDArray weight, INDArray bias, INDArray convStateIn,
      int activation) {
    NDValidation.validateNumerical("causalConv1d", "x", x);
    NDValidation.validateNumerical("causalConv1d", "weight", weight);
    NDValidation.validateNumerical("causalConv1d", "bias", bias);
    NDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(x, weight, bias, convStateIn, activation, 0));
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
   * Column-parallel linear layer for tensor parallelism.<br>
   * Splits weight columns across tensor parallel ranks.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param weight Weight matrix (NUMERIC type)
   * @param tpRank Tensor parallel rank
   * @param tpSize Tensor parallel world size
   * @param gatherOutput Whether to all-gather output
   * @return output Column-parallel linear output (NUMERIC type)
   */
  public INDArray columnParallelLinear(INDArray input, INDArray weight, int tpRank, int tpSize,
      boolean gatherOutput) {
    NDValidation.validateNumerical("columnParallelLinear", "input", input);
    NDValidation.validateNumerical("columnParallelLinear", "weight", weight);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.ColumnParallelLinear(input, weight, tpRank, tpSize, gatherOutput));
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
   * Decoder-optimized masked multi-head attention.<br>
   * Optimized for autoregressive decoding with incremental KV cache.<br>
   *
   * @param query Query tensor (NUMERIC type)
   * @param key Key tensor (NUMERIC type)
   * @param value Value tensor (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param isCausal Whether to apply causal mask
   * @return output Attention output (NUMERIC type)
   */
  public INDArray decoderMaskedMha(INDArray query, INDArray key, INDArray value, int numHeads,
      boolean isCausal) {
    NDValidation.validateNumerical("decoderMaskedMha", "query", query);
    NDValidation.validateNumerical("decoderMaskedMha", "key", key);
    NDValidation.validateNumerical("decoderMaskedMha", "value", value);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DecoderMaskedMha(query, key, value, numHeads, isCausal));
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
   * Weight-Decomposed Low-Rank Adaptation (DoRA) fused matrix multiplication.<br>
   * Decomposes weight into magnitude and direction, applies LoRA to direction only.<br>
   *
   * @param input Input [batch, in_features] (NUMERIC type)
   * @param weight Base weight [out_features, in_features] (NUMERIC type)
   * @param loraA LoRA down-projection [r, in_features] (NUMERIC type)
   * @param loraB LoRA up-projection [out_features, r] (NUMERIC type)
   * @param magnitude Per-output magnitude [out_features] (NUMERIC type)
   * @param scaling LoRA scaling factor (default 1.0)
   * @return output DoRA result with weight-decomposed adaptation (NUMERIC type)
   */
  public INDArray doraMatMul(INDArray input, INDArray weight, INDArray loraA, INDArray loraB,
      INDArray magnitude, double scaling) {
    NDValidation.validateNumerical("doraMatMul", "input", input);
    NDValidation.validateNumerical("doraMatMul", "weight", weight);
    NDValidation.validateNumerical("doraMatMul", "loraA", loraA);
    NDValidation.validateNumerical("doraMatMul", "loraB", loraB);
    NDValidation.validateNumerical("doraMatMul", "magnitude", magnitude);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DoraMatMul(input, weight, loraA, loraB, magnitude, scaling));
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
   * FP8 matrix multiplication with per-tensor scaling.<br>
   *
   * @param a First matrix (NUMERIC type)
   * @param b Second matrix (NUMERIC type)
   * @param scaleA Scale for matrix A (NUMERIC type)
   * @param scaleB Scale for matrix B (NUMERIC type)
   * @return output Scaled FP8 matmul result (NUMERIC type)
   */
  public INDArray fp8Matmul(INDArray a, INDArray b, INDArray scaleA, INDArray scaleB) {
    NDValidation.validateNumerical("fp8Matmul", "a", a);
    NDValidation.validateNumerical("fp8Matmul", "b", b);
    NDValidation.validateNumerical("fp8Matmul", "scaleA", scaleA);
    NDValidation.validateNumerical("fp8Matmul", "scaleB", scaleB);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Fp8Matmul(a, b, scaleA, scaleB));
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
   * Fused bias addition, dropout, and residual connection in a single kernel.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param bias Bias tensor (NUMERIC type)
   * @param residual Residual connection tensor (NUMERIC type)
   * @param dropoutProb Dropout probability
   * @param training Whether in training mode
   * @return output dropout(input + bias) + residual (NUMERIC type)
   */
  public INDArray fusedBiasDropoutResidual(INDArray input, INDArray bias, INDArray residual,
      double dropoutProb, boolean training) {
    NDValidation.validateNumerical("fusedBiasDropoutResidual", "input", input);
    NDValidation.validateNumerical("fusedBiasDropoutResidual", "bias", bias);
    NDValidation.validateNumerical("fusedBiasDropoutResidual", "residual", residual);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedBiasDropoutResidual(input, bias, residual, dropoutProb, training));
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
   * Executes a fused chain of element-wise operations in a single kernel pass.<br>
   * Intermediate values stay in registers instead of global memory. Replaces N separate kernel launches with 1.<br>
   *
   * @param input Primary input array (NUMERIC type)
   * @param secondaryInputs Optional secondary input arrays for binary ops (add, sub, mul, div) (NUMERIC type)
   * @param opCodes Op codes: 0=add, 1=sub, 2=mul, 3=div, 10=relu, 11=sigmoid, 12=tanh, 13=gelu, 14=exp, 15=log, 16=abs, 17=neg, 18=square, 19=sqrt, 20=swish, 21=silu, 22=mish, 30=clip, 31=leaky_relu (Size: AtLeast(min=1))
   * @return output Result of applying the fused element-wise chain (NUMERIC type)
   */
  public INDArray fusedElementwiseChain(INDArray input, INDArray[] secondaryInputs, int[] opCodes) {
    NDValidation.validateNumerical("fusedElementwiseChain", "input", input);
    NDValidation.validateNumerical("fusedElementwiseChain", "secondaryInputs", secondaryInputs);
    Preconditions.checkArgument(secondaryInputs.length >= 0, "secondaryInputs has incorrect size/length. Expected: secondaryInputs.length >= 0, got %s", secondaryInputs.length);
    Preconditions.checkArgument(opCodes.length >= 1, "opCodes has incorrect size/length. Expected: opCodes.length >= 1, got %s", opCodes.length);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain(input, secondaryInputs, opCodes));
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
   * Fused Gaussian Error Linear Unit (GELU) activation function.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output GELU(x) (NUMERIC type)
   */
  public INDArray fusedGelu(INDArray input) {
    NDValidation.validateNumerical("fusedGelu", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGELU(input));
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
   * Fused GEMM + SwiGLU: combines two matrix multiplications with gated activation.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param wGate Gate projection weight (NUMERIC type)
   * @param wUp Up projection weight (NUMERIC type)
   * @return output SwiGLU(input @ wGate, input @ wUp) (NUMERIC type)
   */
  public INDArray fusedGemmSwiglu(INDArray input, INDArray wGate, INDArray wUp) {
    NDValidation.validateNumerical("fusedGemmSwiglu", "input", input);
    NDValidation.validateNumerical("fusedGemmSwiglu", "wGate", wGate);
    NDValidation.validateNumerical("fusedGemmSwiglu", "wUp", wUp);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGemmSwiglu(input, wGate, wUp));
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
   * Fused layer normalization. Computes mean, variance, normalize, scale and shift in one pass.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param gamma Scale parameter (NUMERIC type)
   * @param beta Bias parameter (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output Layer-normalized output (NUMERIC type)
   */
  public INDArray fusedLayerNorm(INDArray input, INDArray gamma, INDArray beta, double epsilon) {
    NDValidation.validateNumerical("fusedLayerNorm", "input", input);
    NDValidation.validateNumerical("fusedLayerNorm", "gamma", gamma);
    NDValidation.validateNumerical("fusedLayerNorm", "beta", beta);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedLayerNorm(input, gamma, beta, epsilon));
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
   * Fused normalization + quantization in a single kernel.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param gamma Norm scale parameter (NUMERIC type)
   * @param epsilon Epsilon for normalization
   * @param quantType Quantization type
   * @return output Normalized and quantized output (NUMERIC type)
   */
  public INDArray fusedNormQuantize(INDArray input, INDArray gamma, double epsilon, int quantType) {
    NDValidation.validateNumerical("fusedNormQuantize", "input", input);
    NDValidation.validateNumerical("fusedNormQuantize", "gamma", gamma);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedNormQuantize(input, gamma, epsilon, quantType));
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
   * Fused RMSNorm + SwiGLU activation. Combines normalization and gated activation<br>
   * into a single kernel for better memory efficiency.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param gamma RMS norm scale (NUMERIC type)
   * @param wGate Gate projection weight (NUMERIC type)
   * @param wUp Up projection weight (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output Result of fused RMSNorm + SwiGLU (NUMERIC type)
   */
  public INDArray fusedRmsNormSwiglu(INDArray input, INDArray gamma, INDArray wGate, INDArray wUp,
      double epsilon) {
    NDValidation.validateNumerical("fusedRmsNormSwiglu", "input", input);
    NDValidation.validateNumerical("fusedRmsNormSwiglu", "gamma", gamma);
    NDValidation.validateNumerical("fusedRmsNormSwiglu", "wGate", wGate);
    NDValidation.validateNumerical("fusedRmsNormSwiglu", "wUp", wUp);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRmsNormSwiGLU(input, gamma, wGate, wUp, epsilon));
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
   * Fused Rotary Position Embedding using precomputed cache.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param ropeCache Precomputed RoPE cache (cos/sin) (NUMERIC type)
   * @param startPosition Start position for RoPE application
   * @return output Input with RoPE applied (NUMERIC type)
   */
  public INDArray fusedRoPE(INDArray input, INDArray ropeCache, int startPosition) {
    NDValidation.validateNumerical("fusedRoPE", "input", input);
    NDValidation.validateNumerical("fusedRoPE", "ropeCache", ropeCache);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(input, ropeCache, startPosition));
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
   * Full Gated Delta Network (GDN) layer block.<br>
   * <br>
   * Fuses the complete GDN layer pipeline:<br>
   *   1. Linear projection (QKV + beta + gate)<br>
   *   2. Causal depthwise conv1d with SiLU activation<br>
   *   3. Gated delta rule recurrent state update<br>
   *   4. RMSNorm + Swish gate<br>
   *   5. Output linear projection<br>
   * <br>
   * This is the building block for Gated Delta Network architectures<br>
   * (arXiv:2412.06464, ICLR 2025).<br>
   *
   * @param x Input tensor [batch, seqLen, modelDim] (NUMERIC type)
   * @param wqkv QKV projection weights [modelDim, qkvDim] (NUMERIC type)
   * @param wbeta Beta projection weights [modelDim, numHeads] (NUMERIC type)
   * @param wgate Gate projection weights [modelDim, numHeads] (NUMERIC type)
   * @param wout Output projection weights [numHeads*headDimV, modelDim] (NUMERIC type)
   * @param convWeight Causal conv1d weights [modelDim, kernelSize] (NUMERIC type)
   * @param convBias Causal conv1d bias [modelDim] (NUMERIC type)
   * @param recurrentStateIn Previous recurrent state [batch, numHeads, headDimK, headDimV] (NUMERIC type)
   * @param numHeads Number of attention heads (H)
   * @param headDimK Key head dimension (D_k)
   * @param headDimV Value head dimension (D_v)
   * @param rmsNormEpsilon RMSNorm epsilon
   */
  public INDArray[] gatedDeltaNetBlock(INDArray x, INDArray wqkv, INDArray wbeta, INDArray wgate,
      INDArray wout, INDArray convWeight, INDArray convBias, INDArray recurrentStateIn,
      int numHeads, int headDimK, int headDimV, double rmsNormEpsilon) {
    NDValidation.validateNumerical("gatedDeltaNetBlock", "x", x);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wqkv", wqkv);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wbeta", wbeta);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wgate", wgate);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wout", wout);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "convWeight", convWeight);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "convBias", convBias);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "recurrentStateIn", recurrentStateIn);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaNetBlock(x, wqkv, wbeta, wgate, wout, convWeight, convBias, recurrentStateIn, numHeads, headDimK, headDimV, rmsNormEpsilon));
  }

  /**
   * Full Gated Delta Network (GDN) layer block.<br>
   * <br>
   * Fuses the complete GDN layer pipeline:<br>
   *   1. Linear projection (QKV + beta + gate)<br>
   *   2. Causal depthwise conv1d with SiLU activation<br>
   *   3. Gated delta rule recurrent state update<br>
   *   4. RMSNorm + Swish gate<br>
   *   5. Output linear projection<br>
   * <br>
   * This is the building block for Gated Delta Network architectures<br>
   * (arXiv:2412.06464, ICLR 2025).<br>
   *
   * @param x Input tensor [batch, seqLen, modelDim] (NUMERIC type)
   * @param wqkv QKV projection weights [modelDim, qkvDim] (NUMERIC type)
   * @param wbeta Beta projection weights [modelDim, numHeads] (NUMERIC type)
   * @param wgate Gate projection weights [modelDim, numHeads] (NUMERIC type)
   * @param wout Output projection weights [numHeads*headDimV, modelDim] (NUMERIC type)
   * @param convWeight Causal conv1d weights [modelDim, kernelSize] (NUMERIC type)
   * @param convBias Causal conv1d bias [modelDim] (NUMERIC type)
   * @param numHeads Number of attention heads (H)
   * @param headDimK Key head dimension (D_k)
   * @param headDimV Value head dimension (D_v)
   */
  public INDArray[] gatedDeltaNetBlock(INDArray x, INDArray wqkv, INDArray wbeta, INDArray wgate,
      INDArray wout, INDArray convWeight, INDArray convBias, int numHeads, int headDimK,
      int headDimV) {
    NDValidation.validateNumerical("gatedDeltaNetBlock", "x", x);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wqkv", wqkv);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wbeta", wbeta);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wgate", wgate);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "wout", wout);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "convWeight", convWeight);
    NDValidation.validateNumerical("gatedDeltaNetBlock", "convBias", convBias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaNetBlock(x, wqkv, wbeta, wgate, wout, convWeight, convBias, null, numHeads, headDimK, headDimV, 1.0E-5));
  }

  /**
   * Gated Delta Rule (arXiv:2412.06464, ICLR 2025, NVIDIA Research).<br>
   * <br>
   * Recurrent linear attention with gated exponential decay and delta update rule:<br>
   *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)<br>
   *   output_t = S_t^T * q_t<br>
   * <br>
   * State shape: [batch, numHeads, headDimK, headDimV].<br>
   * Used in Gated Delta Networks (Qwen3.5 and other production models).<br>
   *
   * @param q Query tensor [batch, seqLen, numHeads, headDimK] (NUMERIC type)
   * @param k Key tensor [batch, seqLen, numHeads, headDimK] (L2-normalized) (NUMERIC type)
   * @param v Value tensor [batch, seqLen, numHeads, headDimV] (NUMERIC type)
   * @param beta Per-step learning rate [batch, seqLen, numHeads] (NUMERIC type)
   * @param gate Decay gate (pre-exp) [batch, seqLen, numHeads] (NUMERIC type)
   * @param stateIn Previous recurrent state [batch, numHeads, headDimK, headDimV] (NUMERIC type)
   */
  public INDArray[] gatedDeltaRule(INDArray q, INDArray k, INDArray v, INDArray beta, INDArray gate,
      INDArray stateIn) {
    NDValidation.validateNumerical("gatedDeltaRule", "q", q);
    NDValidation.validateNumerical("gatedDeltaRule", "k", k);
    NDValidation.validateNumerical("gatedDeltaRule", "v", v);
    NDValidation.validateNumerical("gatedDeltaRule", "beta", beta);
    NDValidation.validateNumerical("gatedDeltaRule", "gate", gate);
    NDValidation.validateNumerical("gatedDeltaRule", "stateIn", stateIn);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(q, k, v, beta, gate, stateIn));
  }

  /**
   * Gated Delta Rule (arXiv:2412.06464, ICLR 2025, NVIDIA Research).<br>
   * <br>
   * Recurrent linear attention with gated exponential decay and delta update rule:<br>
   *   S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)<br>
   *   output_t = S_t^T * q_t<br>
   * <br>
   * State shape: [batch, numHeads, headDimK, headDimV].<br>
   * Used in Gated Delta Networks (Qwen3.5 and other production models).<br>
   *
   * @param q Query tensor [batch, seqLen, numHeads, headDimK] (NUMERIC type)
   * @param k Key tensor [batch, seqLen, numHeads, headDimK] (L2-normalized) (NUMERIC type)
   * @param v Value tensor [batch, seqLen, numHeads, headDimV] (NUMERIC type)
   * @param beta Per-step learning rate [batch, seqLen, numHeads] (NUMERIC type)
   * @param gate Decay gate (pre-exp) [batch, seqLen, numHeads] (NUMERIC type)
   */
  public INDArray[] gatedDeltaRule(INDArray q, INDArray k, INDArray v, INDArray beta,
      INDArray gate) {
    NDValidation.validateNumerical("gatedDeltaRule", "q", q);
    NDValidation.validateNumerical("gatedDeltaRule", "k", k);
    NDValidation.validateNumerical("gatedDeltaRule", "v", v);
    NDValidation.validateNumerical("gatedDeltaRule", "beta", beta);
    NDValidation.validateNumerical("gatedDeltaRule", "gate", gate);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(q, k, v, beta, gate, null));
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
   * GPU-accelerated top-K sampling for autoregressive text generation.<br>
   *
   * @param logits Logit scores (NUMERIC type)
   * @param k Number of top candidates
   * @param temperature Sampling temperature
   * @return output Sampled token indices (NUMERIC type)
   */
  public INDArray gpuTopKSample(INDArray logits, int k, double temperature) {
    NDValidation.validateNumerical("gpuTopKSample", "logits", logits);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopKSample(logits, k, temperature));
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
   * GPU-accelerated nucleus (top-P) sampling for autoregressive text generation.<br>
   *
   * @param logits Logit scores (NUMERIC type)
   * @param p Cumulative probability threshold (nucleus)
   * @param temperature Sampling temperature
   * @return output Sampled token indices (NUMERIC type)
   */
  public INDArray gpuTopPSample(INDArray logits, double p, double temperature) {
    NDValidation.validateNumerical("gpuTopPSample", "logits", logits);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopPSample(logits, p, temperature));
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
   * Dequantizes quantized KV cache tensors back to floating point.<br>
   *
   * @param input Quantized key or value tensor (NUMERIC type)
   * @param scale Quantization scales (NUMERIC type)
   * @param quantType Quantization type
   * @return output Dequantized tensor (NUMERIC type)
   */
  public INDArray kvCacheDequantize(INDArray input, INDArray scale, int quantType) {
    NDValidation.validateNumerical("kvCacheDequantize", "input", input);
    NDValidation.validateNumerical("kvCacheDequantize", "scale", scale);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize(input, scale, quantType));
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
   * Quantizes KV cache tensors for memory-efficient inference.<br>
   *
   * @param input Key or value tensor to quantize (NUMERIC type)
   * @param quantType Quantization type
   * @param groupSize Group size for quantization
   * @return output Quantized tensor (NUMERIC type)
   */
  public INDArray kvCacheQuantize(INDArray input, int quantType, int groupSize) {
    NDValidation.validateNumerical("kvCacheQuantize", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize(input, quantType, groupSize));
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
   * Creates a contiguous copy of the input tensor with linear (row-major) memory layout.<br>
   *
   * @param input Source tensor (NUMERIC type)
   * @return output Contiguous copy of input (NUMERIC type)
   */
  public INDArray linearCopy(INDArray input) {
    NDValidation.validateNumerical("linearCopy", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.custom.LinearCopy(input));
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
   * Low-Rank Hadamard Product (LoHa) fused matrix multiplication.<br>
   * Uses Hadamard product of two low-rank matrices as the adapter.<br>
   *
   * @param input Input [batch, in_features] (NUMERIC type)
   * @param weight Base weight [out_features, in_features] (NUMERIC type)
   * @param lohaA1 First Hadamard factor A [dim, in_features] (NUMERIC type)
   * @param lohaB1 First Hadamard factor B [out_features, dim] (NUMERIC type)
   * @param lohaA2 Second Hadamard factor A [dim, in_features] (NUMERIC type)
   * @param lohaB2 Second Hadamard factor B [out_features, dim] (NUMERIC type)
   * @param scaling Scaling factor (default 1.0)
   * @param transposeWeight Whether to transpose weight (default true)
   * @return output LoHa result (NUMERIC type)
   */
  public INDArray lohaMatMul(INDArray input, INDArray weight, INDArray lohaA1, INDArray lohaB1,
      INDArray lohaA2, INDArray lohaB2, double scaling, boolean transposeWeight) {
    NDValidation.validateNumerical("lohaMatMul", "input", input);
    NDValidation.validateNumerical("lohaMatMul", "weight", weight);
    NDValidation.validateNumerical("lohaMatMul", "lohaA1", lohaA1);
    NDValidation.validateNumerical("lohaMatMul", "lohaB1", lohaB1);
    NDValidation.validateNumerical("lohaMatMul", "lohaA2", lohaA2);
    NDValidation.validateNumerical("lohaMatMul", "lohaB2", lohaB2);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LohaMatMul(input, weight, lohaA1, lohaB1, lohaA2, lohaB2, scaling, transposeWeight));
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
   * Low-Rank Kronecker Product (LoKr) fused matrix multiplication.<br>
   * Uses Kronecker product of matrices as the adapter.<br>
   *
   * @param input Input [batch, in_features] (NUMERIC type)
   * @param weight Base weight [out_features, in_features] (NUMERIC type)
   * @param lokrC Kronecker factor C [f1, f2] (NUMERIC type)
   * @param lokrA Kronecker factor A [dim, d2] (NUMERIC type)
   * @param lokrB Kronecker factor B [d1, dim] (NUMERIC type)
   * @param factor1 First Kronecker factor dimension
   * @param factor2 Second Kronecker factor dimension
   * @param scaling Scaling factor (default 1.0)
   * @param transposeWeight Whether to transpose weight (default true)
   * @return output LoKr result (NUMERIC type)
   */
  public INDArray lokrMatMul(INDArray input, INDArray weight, INDArray lokrC, INDArray lokrA,
      INDArray lokrB, int factor1, int factor2, double scaling, boolean transposeWeight) {
    NDValidation.validateNumerical("lokrMatMul", "input", input);
    NDValidation.validateNumerical("lokrMatMul", "weight", weight);
    NDValidation.validateNumerical("lokrMatMul", "lokrC", lokrC);
    NDValidation.validateNumerical("lokrMatMul", "lokrA", lokrA);
    NDValidation.validateNumerical("lokrMatMul", "lokrB", lokrB);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LokrMatMul(input, weight, lokrC, lokrA, lokrB, factor1, factor2, scaling, transposeWeight));
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
   * Low-Rank Adaptation (LoRA) fused matrix multiplication.<br>
   * Computes base weight matmul + low-rank adapter in a single operation.<br>
   *
   * @param input Input [batch, in_features] (NUMERIC type)
   * @param weight Base weight [out_features, in_features] (NUMERIC type)
   * @param loraA LoRA down-projection [r, in_features] (NUMERIC type)
   * @param loraB LoRA up-projection [out_features, r] (NUMERIC type)
   * @param scaling LoRA scaling factor (default 1.0)
   * @param transposeWeight Whether to transpose weight (default true)
   * @return output Result: input @ weight^T + scaling * input @ loraA^T @ loraB^T (NUMERIC type)
   */
  public INDArray loraMatMul(INDArray input, INDArray weight, INDArray loraA, INDArray loraB,
      double scaling, boolean transposeWeight) {
    NDValidation.validateNumerical("loraMatMul", "input", input);
    NDValidation.validateNumerical("loraMatMul", "weight", weight);
    NDValidation.validateNumerical("loraMatMul", "loraA", loraA);
    NDValidation.validateNumerical("loraMatMul", "loraB", loraB);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LoraMatMul(input, weight, loraA, loraB, scaling, transposeWeight));
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
   * Computes the mean of squared values. Used in RMSNorm and similar operations.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output Mean of squared values (NUMERIC type)
   */
  public INDArray meanSquare(INDArray input) {
    NDValidation.validateNumerical("meanSquare", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MeanSquare(input));
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
   * Multi-head Latent Attention (MLA) from DeepSeek-V2.<br>
   * Uses low-rank KV compression for efficient long-context inference.<br>
   *
   * @param input Input hidden states (NUMERIC type)
   * @param kvDownProj KV down-projection weight (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param latentDim Latent dimension for compressed KV
   * @return output Attention output (NUMERIC type)
   */
  public INDArray mlaAttention(INDArray input, INDArray kvDownProj, int numHeads, int latentDim) {
    NDValidation.validateNumerical("mlaAttention", "input", input);
    NDValidation.validateNumerical("mlaAttention", "kvDownProj", kvDownProj);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MLAAttention(input, kvDownProj, numHeads, latentDim));
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
   * Mixture of Experts (MoE) gating/routing function.<br>
   * Selects top-K experts and computes routing weights.<br>
   *
   * @param input Input hidden states (NUMERIC type)
   * @param gateWeights Router gate weights (NUMERIC type)
   * @param numExperts Number of experts
   * @param topK Top-K experts to select
   * @return output Gating weights and expert indices (NUMERIC type)
   */
  public INDArray moeGate(INDArray input, INDArray gateWeights, int numExperts, int topK) {
    NDValidation.validateNumerical("moeGate", "input", input);
    NDValidation.validateNumerical("moeGate", "gateWeights", gateWeights);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MoeGate(input, gateWeights, numExperts, topK));
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
   * Multi-adapter LoRA matrix multiplication. Selects different LoRA adapters per batch element.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param baseWeight Base weight matrix (NUMERIC type)
   * @param loraAWeights Stacked LoRA A weights (NUMERIC type)
   * @param loraBWeights Stacked LoRA B weights (NUMERIC type)
   * @param adapterIds Adapter selection indices (NUMERIC type)
   * @param scaling LoRA scaling factor
   * @return output Result with per-sample adapter selection (NUMERIC type)
   */
  public INDArray multiLoraMatmul(INDArray input, INDArray baseWeight, INDArray loraAWeights,
      INDArray loraBWeights, INDArray adapterIds, double scaling) {
    NDValidation.validateNumerical("multiLoraMatmul", "input", input);
    NDValidation.validateNumerical("multiLoraMatmul", "baseWeight", baseWeight);
    NDValidation.validateNumerical("multiLoraMatmul", "loraAWeights", loraAWeights);
    NDValidation.validateNumerical("multiLoraMatmul", "loraBWeights", loraBWeights);
    NDValidation.validateNumerical("multiLoraMatmul", "adapterIds", adapterIds);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MultiLoraMatmul(input, baseWeight, loraAWeights, loraBWeights, adapterIds, scaling));
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
   * Quantized matrix multiplication. Supports mixed precision (float/int) inputs.<br>
   *
   * @param a First matrix (NUMERIC type)
   * @param b Second matrix (NUMERIC type)
   * @return output Matrix product (NUMERIC type)
   */
  public INDArray quantizedMatmul(INDArray a, INDArray b) {
    NDValidation.validateNumerical("quantizedMatmul", "a", a);
    NDValidation.validateNumerical("quantizedMatmul", "b", b);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.QuantizedMatmul(a, b));
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
   * Reshapes a tensor without copying data. Returns a view if possible.<br>
   * If the reshape cannot be done without copying, this op will fail.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param shape Target shape (NUMERIC type)
   * @return output Reshaped view (no data copy) (NUMERIC type)
   */
  public INDArray reshapeNoCopy(INDArray input, INDArray shape) {
    NDValidation.validateNumerical("reshapeNoCopy", "input", input);
    NDValidation.validateNumerical("reshapeNoCopy", "shape", shape);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(input, shape));
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
   * Applies Rotary Position Embedding (RoPE) to the input tensor.<br>
   * Encodes position information by rotating pairs of dimensions in the input.<br>
   *
   * @param input Input tensor [batch, seq_len, num_heads, head_dim] (NUMERIC type)
   * @param mode RoPE mode (default 0)
   * @param nPast Number of past tokens (default 0)
   * @param nDims Dimension subset for rotation (default last dim)
   * @param freqBase Frequency base (default 10000.0)
   * @param freqScale Frequency scale (default 1.0)
   * @return output Output with rotary position embeddings applied (NUMERIC type)
   */
  public INDArray rope(INDArray input, int mode, int nPast, int nDims, double freqBase,
      double freqScale) {
    NDValidation.validateNumerical("rope", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RoPE(input, mode, nPast, nDims, freqBase, freqScale));
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
   * Row-parallel linear layer for tensor parallelism.<br>
   * Splits weight rows across tensor parallel ranks.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param weight Weight matrix (NUMERIC type)
   * @param tpRank Tensor parallel rank
   * @param tpSize Tensor parallel world size
   * @param reduceOutput Whether to all-reduce output
   * @return output Row-parallel linear output (NUMERIC type)
   */
  public INDArray rowParallelLinear(INDArray input, INDArray weight, int tpRank, int tpSize,
      boolean reduceOutput) {
    NDValidation.validateNumerical("rowParallelLinear", "input", input);
    NDValidation.validateNumerical("rowParallelLinear", "weight", weight);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.RowParallelLinear(input, weight, tpRank, tpSize, reduceOutput));
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
   * Selective scan operation for state space models (Mamba architecture).<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output Selective scan output (NUMERIC type)
   */
  public INDArray selectiveScan(INDArray input) {
    NDValidation.validateNumerical("selectiveScan", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SelectiveScan(input));
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
   * SiLU (Sigmoid Linear Unit) activation function, also known as Swish.<br>
   * Computes f(x) = x * sigmoid(x).<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output SiLU(x) = x * sigmoid(x) (NUMERIC type)
   */
  public INDArray silu(INDArray input) {
    NDValidation.validateNumerical("silu", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SiLU(input));
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
   * SmoothQuant: migrates quantization difficulty from activations to weights.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param smoothScale Smooth quantization scale (NUMERIC type)
   * @return output Smoothly quantized output (NUMERIC type)
   */
  public INDArray smoothQuant(INDArray input, INDArray smoothScale) {
    NDValidation.validateNumerical("smoothQuant", "input", input);
    NDValidation.validateNumerical("smoothQuant", "smoothScale", smoothScale);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SmoothQuant(input, smoothScale));
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
   * Fused Swish-Mul: computes swish(input) * gate in a single kernel.<br>
   * Used in SwiGLU and similar gated architectures.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param gate Gate tensor (NUMERIC type)
   * @return output swish(input) * gate (NUMERIC type)
   */
  public INDArray swishMul(INDArray input, INDArray gate) {
    NDValidation.validateNumerical("swishMul", "input", input);
    NDValidation.validateNumerical("swishMul", "gate", gate);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul(input, gate));
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
   * TurboQuant asymmetric attention with compressed keys.<br>
   * <br>
   * Computes scaled dot-product attention using compressed key representations<br>
   * from TurboQuant's two-stage quantization (ICLR 2026). The asymmetric inner<br>
   * product estimator combines MSE reconstruction with QJL correction:<br>
   * <br>
   *   score(q, k) ≈ <q, k_mse> + ||r|| * sqrt(π/2)/m * <S@q, signs><br>
   * <br>
   * Keys use full two-stage compression (MSE + QJL) for asymmetric attention.<br>
   * Values use MSE-only decompression (error averages out in softmax-weighted sum).<br>
   *
   * @param query Query tensor [B, H, Sq, D] (NUMERIC type)
   * @param kMse MSE-reconstructed keys [B, H, Sk, D] (NUMERIC type)
   * @param qjlSigns QJL sign bits [B, H, Sk, D] INT8 (INT type)
   * @param residualNorms Residual L2 norms [B, H, Sk] (NUMERIC type)
   * @param qjlMatrix QJL projection matrix [D, D] (NUMERIC type)
   * @param values Dequantized values [B, H, Sk, D] (NUMERIC type)
   * @param attentionMask Attention mask [B, 1, 1, Sk] (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param headDim Dimension per head
   * @param scale Attention scale (0 = auto: 1/sqrt(headDim))
   * @return output Attention output [B, H, Sq, D] (NUMERIC type)
   */
  public INDArray turboQuantAttention(INDArray query, INDArray kMse, INDArray qjlSigns,
      INDArray residualNorms, INDArray qjlMatrix, INDArray values, INDArray attentionMask,
      int numHeads, int headDim, double scale) {
    NDValidation.validateNumerical("turboQuantAttention", "query", query);
    NDValidation.validateNumerical("turboQuantAttention", "kMse", kMse);
    NDValidation.validateInteger("turboQuantAttention", "qjlSigns", qjlSigns);
    NDValidation.validateNumerical("turboQuantAttention", "residualNorms", residualNorms);
    NDValidation.validateNumerical("turboQuantAttention", "qjlMatrix", qjlMatrix);
    NDValidation.validateNumerical("turboQuantAttention", "values", values);
    NDValidation.validateNumerical("turboQuantAttention", "attentionMask", attentionMask);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TurboQuantAttention(query, kMse, qjlSigns, residualNorms, qjlMatrix, values, attentionMask, numHeads, headDim, scale));
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
   * TurboQuant asymmetric attention with compressed keys.<br>
   * <br>
   * Computes scaled dot-product attention using compressed key representations<br>
   * from TurboQuant's two-stage quantization (ICLR 2026). The asymmetric inner<br>
   * product estimator combines MSE reconstruction with QJL correction:<br>
   * <br>
   *   score(q, k) ≈ <q, k_mse> + ||r|| * sqrt(π/2)/m * <S@q, signs><br>
   * <br>
   * Keys use full two-stage compression (MSE + QJL) for asymmetric attention.<br>
   * Values use MSE-only decompression (error averages out in softmax-weighted sum).<br>
   *
   * @param query Query tensor [B, H, Sq, D] (NUMERIC type)
   * @param kMse MSE-reconstructed keys [B, H, Sk, D] (NUMERIC type)
   * @param qjlSigns QJL sign bits [B, H, Sk, D] INT8 (INT type)
   * @param residualNorms Residual L2 norms [B, H, Sk] (NUMERIC type)
   * @param qjlMatrix QJL projection matrix [D, D] (NUMERIC type)
   * @param values Dequantized values [B, H, Sk, D] (NUMERIC type)
   * @param attentionMask Attention mask [B, 1, 1, Sk] (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param headDim Dimension per head
   * @return output Attention output [B, H, Sq, D] (NUMERIC type)
   */
  public INDArray turboQuantAttention(INDArray query, INDArray kMse, INDArray qjlSigns,
      INDArray residualNorms, INDArray qjlMatrix, INDArray values, INDArray attentionMask,
      int numHeads, int headDim) {
    NDValidation.validateNumerical("turboQuantAttention", "query", query);
    NDValidation.validateNumerical("turboQuantAttention", "kMse", kMse);
    NDValidation.validateInteger("turboQuantAttention", "qjlSigns", qjlSigns);
    NDValidation.validateNumerical("turboQuantAttention", "residualNorms", residualNorms);
    NDValidation.validateNumerical("turboQuantAttention", "qjlMatrix", qjlMatrix);
    NDValidation.validateNumerical("turboQuantAttention", "values", values);
    NDValidation.validateNumerical("turboQuantAttention", "attentionMask", attentionMask);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.TurboQuantAttention(query, kMse, qjlSigns, residualNorms, qjlMatrix, values, attentionMask, numHeads, headDim, 0.0));
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
}
