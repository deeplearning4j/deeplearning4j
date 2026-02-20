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

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.base.Preconditions;
import org.nd4j.enums.PadMode;

public class SDNN extends SDOps {
  public SDNN(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * Concatenates a ReLU which selects only the positive part of the activation with a ReLU which selects only the negative part of the activation. Note that as a result this non-linearity doubles the depth of the activations.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cReLU(SDVariable x) {
    SDValidation.validateNumerical("CReLU", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CReLU(sd,x).outputVariable();
  }

  /**
   * Concatenates a ReLU which selects only the positive part of the activation with a ReLU which selects only the negative part of the activation. Note that as a result this non-linearity doubles the depth of the activations.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable cReLU(String name, SDVariable x) {
    SDValidation.validateNumerical("CReLU", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CReLU(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable batchNorm(SDVariable input, SDVariable mean, SDVariable variance,
      SDVariable gamma, SDVariable beta, double epsilon, int... axis) {
    SDValidation.validateNumerical("batchNorm", "input", input);
    SDValidation.validateNumerical("batchNorm", "mean", mean);
    SDValidation.validateNumerical("batchNorm", "variance", variance);
    SDValidation.validateNumerical("batchNorm", "gamma", gamma);
    SDValidation.validateNumerical("batchNorm", "beta", beta);
    Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm(sd,input, mean, variance, gamma, beta, epsilon, axis).outputVariable();
  }

  /**
   * Neural network batch normalization operation.<br>
   * For details, see <a href="https://arxiv.org/abs/1502.03167">https://arxiv.org/abs/1502.03167</a><br>
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable batchNorm(String name, SDVariable input, SDVariable mean, SDVariable variance,
      SDVariable gamma, SDVariable beta, double epsilon, int... axis) {
    SDValidation.validateNumerical("batchNorm", "input", input);
    SDValidation.validateNumerical("batchNorm", "mean", mean);
    SDValidation.validateNumerical("batchNorm", "variance", variance);
    SDValidation.validateNumerical("batchNorm", "gamma", gamma);
    SDValidation.validateNumerical("batchNorm", "beta", beta);
    Preconditions.checkArgument(axis.length >= 1, "axis has incorrect size/length. Expected: axis.length >= 1, got %s", axis.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm(sd,input, mean, variance, gamma, beta, epsilon, axis).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable biasAdd(SDVariable input, SDVariable bias, boolean nchw) {
    SDValidation.validateNumerical("biasAdd", "input", input);
    SDValidation.validateNumerical("biasAdd", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd(sd,input, bias, nchw).outputVariable();
  }

  /**
   * Bias addition operation: a special case of addition, typically used with CNN 4D activations and a 1D bias vector<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input 4d input variable (NUMERIC type)
   * @param bias 1d bias (NUMERIC type)
   * @param nchw The format - nchw=true means [minibatch, channels, height, width] format; nchw=false - [minibatch, height, width, channels].
   * Unused for 2d inputs
   * @return output Output variable, after applying bias add operation (NUMERIC type)
   */
  public SDVariable biasAdd(String name, SDVariable input, SDVariable bias, boolean nchw) {
    SDValidation.validateNumerical("biasAdd", "input", input);
    SDValidation.validateNumerical("biasAdd", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd(sd,input, bias, nchw).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable centerAndSharpen(SDVariable input, SDVariable center, double temperature) {
    SDValidation.validateNumerical("centerAndSharpen", "input", input);
    SDValidation.validateNumerical("centerAndSharpen", "center", center);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CenterAndSharpen(sd,input, center, temperature).outputVariable();
  }

  /**
   * DINOv2 centering and sharpening operation.<br>
   * Prevents mode collapse in self-supervised learning by centering the teacher output<br>
   * and applying temperature-based sharpening:<br>
   *   output = softmax((input - center) / temperature)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Teacher output logits [batch, features] (NUMERIC type)
   * @param center Running center vector [features] (NUMERIC type)
   * @param temperature Sharpening temperature (typically 0.04-0.07)
   * @return output Sharpened probabilities [batch, features] (NUMERIC type)
   */
  public SDVariable centerAndSharpen(String name, SDVariable input, SDVariable center,
      double temperature) {
    SDValidation.validateNumerical("centerAndSharpen", "input", input);
    SDValidation.validateNumerical("centerAndSharpen", "center", center);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CenterAndSharpen(sd,input, center, temperature).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable centerAndSharpen(SDVariable input, SDVariable center) {
    SDValidation.validateNumerical("centerAndSharpen", "input", input);
    SDValidation.validateNumerical("centerAndSharpen", "center", center);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CenterAndSharpen(sd,input, center, 0.07).outputVariable();
  }

  /**
   * DINOv2 centering and sharpening operation.<br>
   * Prevents mode collapse in self-supervised learning by centering the teacher output<br>
   * and applying temperature-based sharpening:<br>
   *   output = softmax((input - center) / temperature)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Teacher output logits [batch, features] (NUMERIC type)
   * @param center Running center vector [features] (NUMERIC type)
   * @return output Sharpened probabilities [batch, features] (NUMERIC type)
   */
  public SDVariable centerAndSharpen(String name, SDVariable input, SDVariable center) {
    SDValidation.validateNumerical("centerAndSharpen", "input", input);
    SDValidation.validateNumerical("centerAndSharpen", "center", center);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CenterAndSharpen(sd,input, center, 0.07).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] ctcGreedyDecoder(SDVariable logits, boolean mergeRepeated, int blankIndex) {
    SDValidation.validateNumerical("ctcGreedyDecoder", "logits", logits);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CTCGreedyDecoder(sd,logits, null, mergeRepeated, blankIndex).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param logits Log probabilities from CTC output. Shape: [batch, timeSteps, numClasses] (NUMERIC type)
   * @param mergeRepeated Whether to merge repeated characters in output
   * @param blankIndex Index of the blank label in the vocabulary
   */
  public SDVariable[] ctcGreedyDecoder(String[] names, SDVariable logits, boolean mergeRepeated,
      int blankIndex) {
    SDValidation.validateNumerical("ctcGreedyDecoder", "logits", logits);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CTCGreedyDecoder(sd,logits, null, mergeRepeated, blankIndex).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] ctcGreedyDecoder(SDVariable logits, SDVariable sequenceLength,
      boolean mergeRepeated, int blankIndex) {
    SDValidation.validateNumerical("ctcGreedyDecoder", "logits", logits);
    SDValidation.validateNumerical("ctcGreedyDecoder", "sequenceLength", sequenceLength);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CTCGreedyDecoder(sd,logits, sequenceLength, mergeRepeated, blankIndex).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param logits Log probabilities from CTC output. Shape: [batch, timeSteps, numClasses] (NUMERIC type)
   * @param sequenceLength Optional actual sequence lengths. Shape: [batch] (NUMERIC type)
   * @param mergeRepeated Whether to merge repeated characters in output
   * @param blankIndex Index of the blank label in the vocabulary
   */
  public SDVariable[] ctcGreedyDecoder(String[] names, SDVariable logits, SDVariable sequenceLength,
      boolean mergeRepeated, int blankIndex) {
    SDValidation.validateNumerical("ctcGreedyDecoder", "logits", logits);
    SDValidation.validateNumerical("ctcGreedyDecoder", "sequenceLength", sequenceLength);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CTCGreedyDecoder(sd,logits, sequenceLength, mergeRepeated, blankIndex).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable dotProductAttention(SDVariable queries, SDVariable keys, SDVariable values,
      SDVariable mask, boolean scaled) {
    SDValidation.validateNumerical("dotProductAttention", "queries", queries);
    SDValidation.validateNumerical("dotProductAttention", "keys", keys);
    SDValidation.validateNumerical("dotProductAttention", "values", values);
    SDValidation.validateNumerical("dotProductAttention", "mask", mask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttention(sd,queries, keys, values, mask, scaled, false).outputVariable();
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable dotProductAttention(String name, SDVariable queries, SDVariable keys,
      SDVariable values, SDVariable mask, boolean scaled) {
    SDValidation.validateNumerical("dotProductAttention", "queries", queries);
    SDValidation.validateNumerical("dotProductAttention", "keys", keys);
    SDValidation.validateNumerical("dotProductAttention", "values", values);
    SDValidation.validateNumerical("dotProductAttention", "mask", mask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttention(sd,queries, keys, values, mask, scaled, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable dotProductAttentionV2(SDVariable queries, SDVariable values, SDVariable keys,
      SDVariable queryMask, SDVariable valueMask, double scaleFactor, double dropoutProbability,
      boolean useCausalMask, boolean training) {
    SDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    SDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    SDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    SDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, null, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable dotProductAttentionV2(String name, SDVariable queries, SDVariable values,
      SDVariable keys, SDVariable queryMask, SDVariable valueMask, double scaleFactor,
      double dropoutProbability, boolean useCausalMask, boolean training) {
    SDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    SDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    SDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    SDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, null, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable dotProductAttentionV2(SDVariable queries, SDVariable values, SDVariable keys,
      SDVariable queryMask, SDVariable valueMask, SDVariable attentionBias, double scaleFactor,
      double dropoutProbability, boolean useCausalMask, boolean training) {
    SDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    SDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    SDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    SDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "attentionBias", attentionBias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, attentionBias, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable dotProductAttentionV2(String name, SDVariable queries, SDVariable values,
      SDVariable keys, SDVariable queryMask, SDVariable valueMask, SDVariable attentionBias,
      double scaleFactor, double dropoutProbability, boolean useCausalMask, boolean training) {
    SDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    SDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    SDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    SDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "attentionBias", attentionBias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, attentionBias, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable dropout(SDVariable input, boolean inverted, int seed, double probabilityValue) {
    SDValidation.validateNumerical("dropout", "input", input);
    return new org.nd4j.linalg.api.ops.random.impl.CustomDropOut(sd,input, inverted, seed, probabilityValue).outputVariable();
  }

  /**
   * Dropout operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input array (NUMERIC type)
   * @param inverted Whether dropout should be inverted or not.
   * @param seed the seed for dropout
   * @param probabilityValue the chance of dropping a value to 0. Maybe interpreted as 1 - p if inverted is true.
   * @return output Output (NUMERIC type)
   */
  public SDVariable dropout(String name, SDVariable input, boolean inverted, int seed,
      double probabilityValue) {
    SDValidation.validateNumerical("dropout", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.CustomDropOut(sd,input, inverted, seed, probabilityValue).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Dropout operation<br>
   *
   * @param input Input array (NUMERIC type)
   * @param inverted Whether dropout should be inverted or not.
   * @param probabilityValue the chance of dropping a value to 0. Maybe interpreted as 1 - p if inverted is true.
   * @return output Output (NUMERIC type)
   */
  public SDVariable dropout(SDVariable input, boolean inverted, double probabilityValue) {
    SDValidation.validateNumerical("dropout", "input", input);
    return new org.nd4j.linalg.api.ops.random.impl.CustomDropOut(sd,input, inverted, 0, probabilityValue).outputVariable();
  }

  /**
   * Dropout operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input array (NUMERIC type)
   * @param inverted Whether dropout should be inverted or not.
   * @param probabilityValue the chance of dropping a value to 0. Maybe interpreted as 1 - p if inverted is true.
   * @return output Output (NUMERIC type)
   */
  public SDVariable dropout(String name, SDVariable input, boolean inverted,
      double probabilityValue) {
    SDValidation.validateNumerical("dropout", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.CustomDropOut(sd,input, inverted, 0, probabilityValue).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable elu(SDVariable x) {
    SDValidation.validateNumerical("elu", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.ELU(sd,x).outputVariable();
  }

  /**
   * Element-wise exponential linear unit (ELU) function:<br>
   * out = x if x > 0<br>
   * out = a * (exp(x) - 1) if x <= 0<br>
   * with constant a = 1.0<br>
   * <p><br>
   * See: <a href="https://arxiv.org/abs/1511.07289">https://arxiv.org/abs/1511.07289</a><br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable elu(String name, SDVariable x) {
    SDValidation.validateNumerical("elu", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.ELU(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable emaUpdate(SDVariable model, SDVariable shadow, double decay) {
    SDValidation.validateNumerical("emaUpdate", "model", model);
    SDValidation.validateNumerical("emaUpdate", "shadow", shadow);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.EmaUpdate(sd,model, shadow, decay).outputVariable();
  }

  /**
   * Exponential Moving Average parameter update for DINOv2 teacher networks.<br>
   * Computes: output = decay * shadow + (1 - decay) * model<br>
   * Used in self-supervised learning to maintain a slowly-updated teacher model.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param model Current model parameters (student) (NUMERIC type)
   * @param shadow EMA shadow parameters (teacher) (NUMERIC type)
   * @param decay EMA decay factor (typically 0.996-0.9999)
   * @return output Updated shadow parameters (NUMERIC type)
   */
  public SDVariable emaUpdate(String name, SDVariable model, SDVariable shadow, double decay) {
    SDValidation.validateNumerical("emaUpdate", "model", model);
    SDValidation.validateNumerical("emaUpdate", "shadow", shadow);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.EmaUpdate(sd,model, shadow, decay).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable emaUpdate(SDVariable model, SDVariable shadow) {
    SDValidation.validateNumerical("emaUpdate", "model", model);
    SDValidation.validateNumerical("emaUpdate", "shadow", shadow);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.EmaUpdate(sd,model, shadow, 0.999).outputVariable();
  }

  /**
   * Exponential Moving Average parameter update for DINOv2 teacher networks.<br>
   * Computes: output = decay * shadow + (1 - decay) * model<br>
   * Used in self-supervised learning to maintain a slowly-updated teacher model.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param model Current model parameters (student) (NUMERIC type)
   * @param shadow EMA shadow parameters (teacher) (NUMERIC type)
   * @return output Updated shadow parameters (NUMERIC type)
   */
  public SDVariable emaUpdate(String name, SDVariable model, SDVariable shadow) {
    SDValidation.validateNumerical("emaUpdate", "model", model);
    SDValidation.validateNumerical("emaUpdate", "shadow", shadow);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.EmaUpdate(sd,model, shadow, 0.999).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable flashAttention(SDVariable query, SDVariable key, SDVariable value, double scale,
      boolean isCausal, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("flashAttention", "query", query);
    SDValidation.validateNumerical("flashAttention", "key", key);
    SDValidation.validateNumerical("flashAttention", "value", value);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FlashAttention(sd,query, key, value, scale, isCausal, numHeads, numKvHeads).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param key Key tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param value Value tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param scale Scaling factor. 0 = auto (1/sqrt(headDim))
   * @param isCausal Whether to apply causal masking
   * @param numHeads Number of query attention heads
   * @param numKvHeads Number of KV heads (0 = same as numHeads, for GQA use smaller value)
   * @return output Attention output. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable flashAttention(String name, SDVariable query, SDVariable key, SDVariable value,
      double scale, boolean isCausal, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("flashAttention", "query", query);
    SDValidation.validateNumerical("flashAttention", "key", key);
    SDValidation.validateNumerical("flashAttention", "value", value);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FlashAttention(sd,query, key, value, scale, isCausal, numHeads, numKvHeads).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * GELU activation function - Gaussian Error Linear Units<br>
   * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a><br>
   * This method uses the sigmoid approximation<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable gelu(SDVariable x) {
    SDValidation.validateNumerical("gelu", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.GELU(sd,x).outputVariable();
  }

  /**
   * GELU activation function - Gaussian Error Linear Units<br>
   * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a><br>
   * This method uses the sigmoid approximation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable gelu(String name, SDVariable x) {
    SDValidation.validateNumerical("gelu", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.GELU(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable groupedQueryAttention(SDVariable query, SDVariable key, SDVariable value,
      double scale, boolean isCausal, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("groupedQueryAttention", "query", query);
    SDValidation.validateNumerical("groupedQueryAttention", "key", key);
    SDValidation.validateNumerical("groupedQueryAttention", "value", value);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.GroupedQueryAttention(sd,query, key, value, scale, isCausal, numHeads, numKvHeads).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param key Key tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param value Value tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param scale Scaling factor. 0 = auto (1/sqrt(headDim))
   * @param isCausal Whether to apply causal masking
   * @param numHeads Number of query attention heads
   * @param numKvHeads Number of KV heads (must divide numHeads evenly)
   * @return output Attention output. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable groupedQueryAttention(String name, SDVariable query, SDVariable key,
      SDVariable value, double scale, boolean isCausal, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("groupedQueryAttention", "query", query);
    SDValidation.validateNumerical("groupedQueryAttention", "key", key);
    SDValidation.validateNumerical("groupedQueryAttention", "value", value);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GroupedQueryAttention(sd,query, key, value, scale, isCausal, numHeads, numKvHeads).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable hardSigmoid(SDVariable x) {
    SDValidation.validateNumerical("hardSigmoid", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid(sd,x).outputVariable();
  }

  /**
   * Element-wise hard sigmoid function:<br>
   * out[i] = 0 if in[i] <= -2.5<br>
   * out[1] = 0.2*in[i]+0.5 if -2.5 < in[i] < 2.5<br>
   * out[i] = 1 if in[i] >= 2.5<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable hardSigmoid(String name, SDVariable x) {
    SDValidation.validateNumerical("hardSigmoid", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.HardSigmoid(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable hardTanh(SDVariable x) {
    SDValidation.validateNumerical("hardTanh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.HardTanh(sd,x).outputVariable();
  }

  /**
   * Element-wise hard tanh function:<br>
   * out[i] = -1 if in[i] <= -1<br>
   * out[1] = in[i] if -1 < in[i] < 1<br>
   * out[i] = 1 if in[i] >= 1<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable hardTanh(String name, SDVariable x) {
    SDValidation.validateNumerical("hardTanh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.HardTanh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Derivative (dOut/dIn) of the element-wise hard Tanh function - hardTanh(INDArray)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable hardTanhDerivative(SDVariable x) {
    SDValidation.validateNumerical("hardTanhDerivative", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative(sd,x).outputVariable();
  }

  /**
   * Derivative (dOut/dIn) of the element-wise hard Tanh function - hardTanh(INDArray)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable hardTanhDerivative(String name, SDVariable x) {
    SDValidation.validateNumerical("hardTanhDerivative", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.gradient.HardTanhDerivative(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] kvCacheUpdate(SDVariable keyCache, SDVariable valueCache, SDVariable newKeys,
      SDVariable newValues, int startPosition) {
    SDValidation.validateNumerical("kvCacheUpdate", "keyCache", keyCache);
    SDValidation.validateNumerical("kvCacheUpdate", "valueCache", valueCache);
    SDValidation.validateNumerical("kvCacheUpdate", "newKeys", newKeys);
    SDValidation.validateNumerical("kvCacheUpdate", "newValues", newValues);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheUpdate(sd,keyCache, valueCache, newKeys, newValues, startPosition).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param keyCache Existing key cache. Shape: [batch, maxSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param valueCache Existing value cache. Shape: [batch, maxSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param newKeys New keys to insert. Shape: [batch, newSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param newValues New values to insert. Shape: [batch, newSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param startPosition Position in cache where new keys/values should be inserted
   */
  public SDVariable[] kvCacheUpdate(String[] names, SDVariable keyCache, SDVariable valueCache,
      SDVariable newKeys, SDVariable newValues, int startPosition) {
    SDValidation.validateNumerical("kvCacheUpdate", "keyCache", keyCache);
    SDValidation.validateNumerical("kvCacheUpdate", "valueCache", valueCache);
    SDValidation.validateNumerical("kvCacheUpdate", "newKeys", newKeys);
    SDValidation.validateNumerical("kvCacheUpdate", "newValues", newValues);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheUpdate(sd,keyCache, valueCache, newKeys, newValues, startPosition).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable kvScatter(SDVariable present, SDVariable staticBuffer, long cachePos) {
    SDValidation.validateNumerical("kvScatter", "present", present);
    SDValidation.validateNumerical("kvScatter", "staticBuffer", staticBuffer);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(sd,present, staticBuffer, cachePos, 1).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param present Present KV tensor from decoder output. Shape: [batch, heads, seqLen, dim] (NUMERIC type)
   * @param staticBuffer Static KV cache buffer. Shape: [batch, heads, maxKvLen, dim]. Updated in-place. (NUMERIC type)
   * @param cachePos Position in static buffer to write the new entry
   * @return output Scalar 0 on success (LONG type)
   */
  public SDVariable kvScatter(String name, SDVariable present, SDVariable staticBuffer,
      long cachePos) {
    SDValidation.validateNumerical("kvScatter", "present", present);
    SDValidation.validateNumerical("kvScatter", "staticBuffer", staticBuffer);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(sd,present, staticBuffer, cachePos, 1).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable kvScatter(SDVariable present, SDVariable staticBuffer, long cachePos,
      int numPairs) {
    SDValidation.validateNumerical("kvScatter", "present", present);
    SDValidation.validateNumerical("kvScatter", "staticBuffer", staticBuffer);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(sd,present, staticBuffer, cachePos, numPairs).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param present Present KV tensor from decoder output. Shape: [batch, heads, seqLen, dim] (NUMERIC type)
   * @param staticBuffer Static KV cache buffer. Shape: [batch, heads, maxKvLen, dim]. Updated in-place. (NUMERIC type)
   * @param cachePos Position in static buffer to write the new entry
   * @param numPairs Number of present/static KV pairs. When > 1, inputs are [present_0..N-1, static_0..N-1]
   * @return output Scalar 0 on success (LONG type)
   */
  public SDVariable kvScatter(String name, SDVariable present, SDVariable staticBuffer,
      long cachePos, int numPairs) {
    SDValidation.validateNumerical("kvScatter", "present", present);
    SDValidation.validateNumerical("kvScatter", "staticBuffer", staticBuffer);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.KvScatter(sd,present, staticBuffer, cachePos, numPairs).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable layerNorm(SDVariable input, SDVariable gain, SDVariable bias,
      boolean channelsFirst, long... dimensions) {
    SDValidation.validateNumerical("layerNorm", "input", input);
    SDValidation.validateNumerical("layerNorm", "gain", gain);
    SDValidation.validateNumerical("layerNorm", "bias", bias);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(sd,input, gain, bias, channelsFirst, dimensions).outputVariable();
  }

  /**
   * Apply Layer Normalization<br>
   * <br>
   * y = gain * standardize(x) + bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @param gain Gain (NUMERIC type)
   * @param bias Bias (NUMERIC type)
   * @param channelsFirst For 2D input - unused. True for NCHW (minibatch, channels, height, width), false for NHWC data
   * @param dimensions Dimensions to perform layer norm over - dimension=1 for 2d/MLP data, dimension=1,2,3 for CNNs (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable layerNorm(String name, SDVariable input, SDVariable gain, SDVariable bias,
      boolean channelsFirst, long... dimensions) {
    SDValidation.validateNumerical("layerNorm", "input", input);
    SDValidation.validateNumerical("layerNorm", "gain", gain);
    SDValidation.validateNumerical("layerNorm", "bias", bias);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(sd,input, gain, bias, channelsFirst, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable layerNorm(SDVariable input, SDVariable gain, boolean channelsFirst,
      long... dimensions) {
    SDValidation.validateNumerical("layerNorm", "input", input);
    SDValidation.validateNumerical("layerNorm", "gain", gain);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(sd,input, gain, null, channelsFirst, dimensions).outputVariable();
  }

  /**
   * Apply Layer Normalization<br>
   * <br>
   * y = gain * standardize(x) + bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @param gain Gain (NUMERIC type)
   * @param channelsFirst For 2D input - unused. True for NCHW (minibatch, channels, height, width), false for NHWC data
   * @param dimensions Dimensions to perform layer norm over - dimension=1 for 2d/MLP data, dimension=1,2,3 for CNNs (Size: AtLeast(min=1))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable layerNorm(String name, SDVariable input, SDVariable gain, boolean channelsFirst,
      long... dimensions) {
    SDValidation.validateNumerical("layerNorm", "input", input);
    SDValidation.validateNumerical("layerNorm", "gain", gain);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(sd,input, gain, null, channelsFirst, dimensions).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable leakyRelu(SDVariable x, double alpha) {
    SDValidation.validateNumerical("leakyRelu", "x", x);
    return new org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU(sd,x, alpha).outputVariable();
  }

  /**
   * Element-wise leaky ReLU function:<br>
   * out = x if x >= 0.0<br>
   * out = alpha * x if x < cutoff<br>
   * Alpha value is most commonly set to 0.01<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param alpha Cutoff - commonly 0.01
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable leakyRelu(String name, SDVariable x, double alpha) {
    SDValidation.validateNumerical("leakyRelu", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.LeakyReLU(sd,x, alpha).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Leaky ReLU derivative: dOut/dIn given input.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @param alpha Cutoff - commonly 0.01
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable leakyReluDerivative(SDVariable x, double alpha) {
    SDValidation.validateNumerical("leakyReluDerivative", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative(sd,x, alpha).outputVariable();
  }

  /**
   * Leaky ReLU derivative: dOut/dIn given input.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @param alpha Cutoff - commonly 0.01
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable leakyReluDerivative(String name, SDVariable x, double alpha) {
    SDValidation.validateNumerical("leakyReluDerivative", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.gradient.LeakyReLUDerivative(sd,x, alpha).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable linear(SDVariable input, SDVariable weights, SDVariable bias,
      boolean transposeA, boolean transposeB, boolean transposeC) {
    SDValidation.validateNumerical("linear", "input", input);
    SDValidation.validateNumerical("linear", "weights", weights);
    SDValidation.validateNumerical("linear", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(sd,input, weights, bias, transposeA, transposeB, transposeC).outputVariable();
  }

  /**
   * Linear layer operation: out = mmul(in,w) + bias<br>
   * Note that bias array is optional<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable, shape [nIn, nOut] (NUMERIC type)
   * @param bias Optional bias variable (may be null) (NUMERIC type)
   * @param transposeA Whether to transpose input or not
   * @param transposeB Whether to transpose second input or not
   * @param transposeC Whether to transpose result or not
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable linear(String name, SDVariable input, SDVariable weights, SDVariable bias,
      boolean transposeA, boolean transposeB, boolean transposeC) {
    SDValidation.validateNumerical("linear", "input", input);
    SDValidation.validateNumerical("linear", "weights", weights);
    SDValidation.validateNumerical("linear", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(sd,input, weights, bias, transposeA, transposeB, transposeC).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable linear(SDVariable input, SDVariable weights, SDVariable bias) {
    SDValidation.validateNumerical("linear", "input", input);
    SDValidation.validateNumerical("linear", "weights", weights);
    SDValidation.validateNumerical("linear", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(sd,input, weights, bias, false, false, false).outputVariable();
  }

  /**
   * Linear layer operation: out = mmul(in,w) + bias<br>
   * Note that bias array is optional<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable, shape [nIn, nOut] (NUMERIC type)
   * @param bias Optional bias variable (may be null) (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable linear(String name, SDVariable input, SDVariable weights, SDVariable bias) {
    SDValidation.validateNumerical("linear", "input", input);
    SDValidation.validateNumerical("linear", "weights", weights);
    SDValidation.validateNumerical("linear", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(sd,input, weights, bias, false, false, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable logSigmoid(SDVariable x) {
    SDValidation.validateNumerical("logSigmoid", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.LogSigmoid(sd,x).outputVariable();
  }

  /**
   * Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable logSigmoid(String name, SDVariable x) {
    SDValidation.validateNumerical("logSigmoid", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.LogSigmoid(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log softmax activation<br>
   *
   * @param x  (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public SDVariable logSoftmax(SDVariable x) {
    SDValidation.validateNumerical("logSoftmax", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(sd,x).outputVariable();
  }

  /**
   * Log softmax activation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x  (NUMERIC type)
   * @return output  (NUMERIC type)
   */
  public SDVariable logSoftmax(String name, SDVariable x) {
    SDValidation.validateNumerical("logSoftmax", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Log softmax activation<br>
   *
   * @param x Input (NUMERIC type)
   * @param dimension Dimension along which to apply log softmax
   * @return output Output - log(softmax(input)) (NUMERIC type)
   */
  public SDVariable logSoftmax(SDVariable x, int dimension) {
    SDValidation.validateNumerical("logSoftmax", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(sd,x, dimension).outputVariable();
  }

  /**
   * Log softmax activation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input (NUMERIC type)
   * @param dimension Dimension along which to apply log softmax
   * @return output Output - log(softmax(input)) (NUMERIC type)
   */
  public SDVariable logSoftmax(String name, SDVariable x, int dimension) {
    SDValidation.validateNumerical("logSoftmax", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(sd,x, dimension).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] mixtureOfExperts(SDVariable input, SDVariable routerWeights,
      SDVariable expertWeights, int numExperts, int topK) {
    SDValidation.validateNumerical("mixtureOfExperts", "input", input);
    SDValidation.validateNumerical("mixtureOfExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("mixtureOfExperts", "expertWeights", expertWeights);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MixtureOfExperts(sd,input, routerWeights, expertWeights, null, numExperts, topK, true, 1.0).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numExperts] (NUMERIC type)
   * @param expertWeights Expert weight matrices. Shape: [numExperts, hiddenSize, expertHiddenSize] (NUMERIC type)
   * @param numExperts Total number of experts
   * @param topK Number of experts to route to per token
   */
  public SDVariable[] mixtureOfExperts(String[] names, SDVariable input, SDVariable routerWeights,
      SDVariable expertWeights, int numExperts, int topK) {
    SDValidation.validateNumerical("mixtureOfExperts", "input", input);
    SDValidation.validateNumerical("mixtureOfExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("mixtureOfExperts", "expertWeights", expertWeights);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MixtureOfExperts(sd,input, routerWeights, expertWeights, null, numExperts, topK, true, 1.0).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] mixtureOfExperts(SDVariable input, SDVariable routerWeights,
      SDVariable expertWeights, SDVariable expertBias, int numExperts, int topK,
      boolean normalizeProbs, double capacityFactor) {
    SDValidation.validateNumerical("mixtureOfExperts", "input", input);
    SDValidation.validateNumerical("mixtureOfExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("mixtureOfExperts", "expertWeights", expertWeights);
    SDValidation.validateNumerical("mixtureOfExperts", "expertBias", expertBias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MixtureOfExperts(sd,input, routerWeights, expertWeights, expertBias, numExperts, topK, normalizeProbs, capacityFactor).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numExperts] (NUMERIC type)
   * @param expertWeights Expert weight matrices. Shape: [numExperts, hiddenSize, expertHiddenSize] (NUMERIC type)
   * @param expertBias Optional expert biases. Shape: [numExperts, expertHiddenSize] (NUMERIC type)
   * @param numExperts Total number of experts
   * @param topK Number of experts to route to per token
   * @param normalizeProbs Whether to normalize router probabilities for selected experts
   * @param capacityFactor Expert capacity factor for load balancing
   */
  public SDVariable[] mixtureOfExperts(String[] names, SDVariable input, SDVariable routerWeights,
      SDVariable expertWeights, SDVariable expertBias, int numExperts, int topK,
      boolean normalizeProbs, double capacityFactor) {
    SDValidation.validateNumerical("mixtureOfExperts", "input", input);
    SDValidation.validateNumerical("mixtureOfExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("mixtureOfExperts", "expertWeights", expertWeights);
    SDValidation.validateNumerical("mixtureOfExperts", "expertBias", expertBias);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MixtureOfExperts(sd,input, routerWeights, expertWeights, expertBias, numExperts, topK, normalizeProbs, capacityFactor).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable multiHeadDotProductAttention(SDVariable queries, SDVariable keys,
      SDVariable values, SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo,
      SDVariable mask, boolean scaled) {
    SDValidation.validateNumerical("multiHeadDotProductAttention", "queries", queries);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "keys", keys);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "values", values);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wq", Wq);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wk", Wk);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wv", Wv);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wo", Wo);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "mask", mask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttention(sd,queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, false).outputVariable();
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable multiHeadDotProductAttention(String name, SDVariable queries, SDVariable keys,
      SDVariable values, SDVariable Wq, SDVariable Wk, SDVariable Wv, SDVariable Wo,
      SDVariable mask, boolean scaled) {
    SDValidation.validateNumerical("multiHeadDotProductAttention", "queries", queries);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "keys", keys);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "values", values);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wq", Wq);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wk", Wk);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wv", Wv);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "Wo", Wo);
    SDValidation.validateNumerical("multiHeadDotProductAttention", "mask", mask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttention(sd,queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable pad(SDVariable input, SDVariable padding, PadMode PadMode, double constant) {
    SDValidation.validateNumerical("pad", "input", input);
    SDValidation.validateNumerical("pad", "padding", padding);
    return new org.nd4j.linalg.api.ops.impl.transforms.Pad(sd,input, padding, PadMode, constant).outputVariable();
  }

  /**
   * Padding operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param padding Padding value (NUMERIC type)
   * @param PadMode Padding format
   * @param constant Padding constant
   * @return output Padded input (NUMERIC type)
   */
  public SDVariable pad(String name, SDVariable input, SDVariable padding, PadMode PadMode,
      double constant) {
    SDValidation.validateNumerical("pad", "input", input);
    SDValidation.validateNumerical("pad", "padding", padding);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.Pad(sd,input, padding, PadMode, constant).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Padding operation<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param padding Padding value (NUMERIC type)
   * @param constant Padding constant
   * @return output Padded input (NUMERIC type)
   */
  public SDVariable pad(SDVariable input, SDVariable padding, double constant) {
    SDValidation.validateNumerical("pad", "input", input);
    SDValidation.validateNumerical("pad", "padding", padding);
    return new org.nd4j.linalg.api.ops.impl.transforms.Pad(sd,input, padding, PadMode.CONSTANT, constant).outputVariable();
  }

  /**
   * Padding operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param padding Padding value (NUMERIC type)
   * @param constant Padding constant
   * @return output Padded input (NUMERIC type)
   */
  public SDVariable pad(String name, SDVariable input, SDVariable padding, double constant) {
    SDValidation.validateNumerical("pad", "input", input);
    SDValidation.validateNumerical("pad", "padding", padding);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.Pad(sd,input, padding, PadMode.CONSTANT, constant).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * GELU activation function - Gaussian Error Linear Units<br>
   * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a><br>
   * This method uses the precise method<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable preciseGelu(SDVariable x) {
    SDValidation.validateNumerical("preciseGelu", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELU(sd,x).outputVariable();
  }

  /**
   * GELU activation function - Gaussian Error Linear Units<br>
   * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a><br>
   * This method uses the precise method<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable preciseGelu(String name, SDVariable x) {
    SDValidation.validateNumerical("preciseGelu", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.PreciseGELU(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable prelu(SDVariable input, SDVariable alpha, int... sharedAxes) {
    SDValidation.validateNumerical("prelu", "input", input);
    SDValidation.validateNumerical("prelu", "alpha", alpha);
    Preconditions.checkArgument(sharedAxes.length >= 1, "sharedAxes has incorrect size/length. Expected: sharedAxes.length >= 1, got %s", sharedAxes.length);
    return new org.nd4j.linalg.api.ops.impl.scalar.PRelu(sd,input, alpha, sharedAxes).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param input Input data (NUMERIC type)
   * @param alpha The cutoff variable.  Note that the batch dimension (the 0th, whether it is batch or not) should not be part of alpha. (NUMERIC type)
   * @param sharedAxes Which axes to share cutoff parameters along. (Size: AtLeast(min=1))
   * @return output Output (NUMERIC type)
   */
  public SDVariable prelu(String name, SDVariable input, SDVariable alpha, int... sharedAxes) {
    SDValidation.validateNumerical("prelu", "input", input);
    SDValidation.validateNumerical("prelu", "alpha", alpha);
    Preconditions.checkArgument(sharedAxes.length >= 1, "sharedAxes has incorrect size/length. Expected: sharedAxes.length >= 1, got %s", sharedAxes.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.PRelu(sd,input, alpha, sharedAxes).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable relativePositionBias(SDVariable biasTable, int numHeads, int windowSize) {
    SDValidation.validateNumerical("relativePositionBias", "biasTable", biasTable);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RelativePositionBias(sd,biasTable, null, numHeads, windowSize, false).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param biasTable Learned bias table. Shape: [numRelativePositions, numHeads] for learned mode, or scalar/tensor for ALiBi mode (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param windowSize Window size for 2D position encoding (used if generating index)
   * @return output Position bias. Shape: [numHeads, windowSize^2, windowSize^2] or [numHeads, seqLen, seqLen] (NUMERIC type)
   */
  public SDVariable relativePositionBias(String name, SDVariable biasTable, int numHeads,
      int windowSize) {
    SDValidation.validateNumerical("relativePositionBias", "biasTable", biasTable);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RelativePositionBias(sd,biasTable, null, numHeads, windowSize, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable relativePositionBias(SDVariable biasTable, SDVariable relativePositionIndex,
      int numHeads, int windowSize) {
    SDValidation.validateNumerical("relativePositionBias", "biasTable", biasTable);
    SDValidation.validateNumerical("relativePositionBias", "relativePositionIndex", relativePositionIndex);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RelativePositionBias(sd,biasTable, relativePositionIndex, numHeads, windowSize, false).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param biasTable Learned bias table. Shape: [numRelativePositions, numHeads] for learned mode, or scalar/tensor for ALiBi mode (NUMERIC type)
   * @param relativePositionIndex Optional precomputed relative position index. Shape: [windowSize^2, windowSize^2] (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param windowSize Window size for 2D position encoding (used if generating index)
   * @return output Position bias. Shape: [numHeads, windowSize^2, windowSize^2] or [numHeads, seqLen, seqLen] (NUMERIC type)
   */
  public SDVariable relativePositionBias(String name, SDVariable biasTable,
      SDVariable relativePositionIndex, int numHeads, int windowSize) {
    SDValidation.validateNumerical("relativePositionBias", "biasTable", biasTable);
    SDValidation.validateNumerical("relativePositionBias", "relativePositionIndex", relativePositionIndex);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RelativePositionBias(sd,biasTable, relativePositionIndex, numHeads, windowSize, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable relu(SDVariable x, double cutoff) {
    SDValidation.validateNumerical("relu", "x", x);
    return new org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear(sd,x, cutoff).outputVariable();
  }

  /**
   * Element-wise rectified linear function with specified cutoff:<br>
   * out[i] = in[i] if in[i] >= cutoff<br>
   * out[i] = 0 otherwise<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input (NUMERIC type)
   * @param cutoff Cutoff value for ReLU operation - x > cutoff ? x : 0. Usually 0
   * @return output Output (NUMERIC type)
   */
  public SDVariable relu(String name, SDVariable x, double cutoff) {
    SDValidation.validateNumerical("relu", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.RectifiedLinear(sd,x, cutoff).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise "rectified linear 6" function with specified cutoff:<br>
   * out[i] = min(max(in, cutoff), 6)<br>
   *
   * @param x Input (NUMERIC type)
   * @param cutoff Cutoff value for ReLU operation. Usually 0
   * @return output Output (NUMERIC type)
   */
  public SDVariable relu6(SDVariable x, double cutoff) {
    SDValidation.validateNumerical("relu6", "x", x);
    return new org.nd4j.linalg.api.ops.impl.scalar.Relu6(sd,x, cutoff).outputVariable();
  }

  /**
   * Element-wise "rectified linear 6" function with specified cutoff:<br>
   * out[i] = min(max(in, cutoff), 6)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input (NUMERIC type)
   * @param cutoff Cutoff value for ReLU operation. Usually 0
   * @return output Output (NUMERIC type)
   */
  public SDVariable relu6(String name, SDVariable x, double cutoff) {
    SDValidation.validateNumerical("relu6", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.scalar.Relu6(sd,x, cutoff).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * ReLU (Rectified Linear Unit) layer operation: out = relu(mmul(in,w) + bias)<br>
   *
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable (NUMERIC type)
   * @param bias  Bias variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable reluLayer(SDVariable input, SDVariable weights, SDVariable bias) {
    SDValidation.validateNumerical("reluLayer", "input", input);
    SDValidation.validateNumerical("reluLayer", "weights", weights);
    SDValidation.validateNumerical("reluLayer", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.transforms.ReluLayer(sd,input, weights, bias).outputVariable();
  }

  /**
   * ReLU (Rectified Linear Unit) layer operation: out = relu(mmul(in,w) + bias)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable (NUMERIC type)
   * @param bias  Bias variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable reluLayer(String name, SDVariable input, SDVariable weights, SDVariable bias) {
    SDValidation.validateNumerical("reluLayer", "input", input);
    SDValidation.validateNumerical("reluLayer", "weights", weights);
    SDValidation.validateNumerical("reluLayer", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.ReluLayer(sd,input, weights, bias).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rmsNorm(SDVariable input, SDVariable gamma, double epsilon) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    SDValidation.validateNumerical("rmsNorm", "gamma", gamma);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, gamma, epsilon).outputVariable();
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @param gamma Scale/gain vector (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output RMS normalized output (NUMERIC type)
   */
  public SDVariable rmsNorm(String name, SDVariable input, SDVariable gamma, double epsilon) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    SDValidation.validateNumerical("rmsNorm", "gamma", gamma);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, gamma, epsilon).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rmsNorm(SDVariable input, SDVariable gamma) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    SDValidation.validateNumerical("rmsNorm", "gamma", gamma);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, gamma, 1.0E-5).outputVariable();
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @param gamma Scale/gain vector (NUMERIC type)
   * @return output RMS normalized output (NUMERIC type)
   */
  public SDVariable rmsNorm(String name, SDVariable input, SDVariable gamma) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    SDValidation.validateNumerical("rmsNorm", "gamma", gamma);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, gamma, 1.0E-5).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rmsNorm(SDVariable input, double epsilon) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, null, epsilon).outputVariable();
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output RMS normalized output (NUMERIC type)
   */
  public SDVariable rmsNorm(String name, SDVariable input, double epsilon) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, null, epsilon).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rmsNorm(SDVariable input) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, null, 1.0E-5).outputVariable();
  }

  /**
   * Root Mean Square Layer Normalization (RMSNorm):<br>
   * <br>
   * output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma<br>
   * <br>
   * If gamma is not provided, only RMS normalization is applied.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable (NUMERIC type)
   * @return output RMS normalized output (NUMERIC type)
   */
  public SDVariable rmsNorm(String name, SDVariable input) {
    SDValidation.validateNumerical("rmsNorm", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm(sd,input, null, 1.0E-5).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable selu(SDVariable x) {
    SDValidation.validateNumerical("selu", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.SELU(sd,x).outputVariable();
  }

  /**
   * Element-wise SeLU function - Scaled exponential Lineal Unit: see <a href="https://arxiv.org/abs/1706.02515">Self-Normalizing Neural Networks</a><br>
   * <br>
   * out[i] = scale * alpha * (exp(in[i])-1) if in[i]>0, or 0 if in[i] <= 0<br>
   * Uses default scale and alpha values.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable selu(String name, SDVariable x) {
    SDValidation.validateNumerical("selu", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.SELU(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sigmoid(SDVariable x) {
    SDValidation.validateNumerical("sigmoid", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid(sd,x).outputVariable();
  }

  /**
   * Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable sigmoid(String name, SDVariable x) {
    SDValidation.validateNumerical("sigmoid", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut<br>
   *
   * @param x Input Variable (NUMERIC type)
   * @param wrt Gradient at the output - dL/dOut. Must have same shape as the input (NUMERIC type)
   * @return output Output (gradient at input of sigmoid) (NUMERIC type)
   */
  public SDVariable sigmoidDerivative(SDVariable x, SDVariable wrt) {
    SDValidation.validateNumerical("sigmoidDerivative", "x", x);
    SDValidation.validateNumerical("sigmoidDerivative", "wrt", wrt);
    return new org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative(sd,x, wrt).outputVariable();
  }

  /**
   * Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input Variable (NUMERIC type)
   * @param wrt Gradient at the output - dL/dOut. Must have same shape as the input (NUMERIC type)
   * @return output Output (gradient at input of sigmoid) (NUMERIC type)
   */
  public SDVariable sigmoidDerivative(String name, SDVariable x, SDVariable wrt) {
    SDValidation.validateNumerical("sigmoidDerivative", "x", x);
    SDValidation.validateNumerical("sigmoidDerivative", "wrt", wrt);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative(sd,x, wrt).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable slidingWindowAttention(SDVariable query, SDVariable key, SDVariable value,
      int windowSize, int numHeads, int numKvHeads, double scale) {
    SDValidation.validateNumerical("slidingWindowAttention", "query", query);
    SDValidation.validateNumerical("slidingWindowAttention", "key", key);
    SDValidation.validateNumerical("slidingWindowAttention", "value", value);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SlidingWindowAttention(sd,query, key, value, windowSize, numHeads, numKvHeads, scale).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param key Key tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param value Value tensor. Shape: [batch, seqLen, numKvHeads, headDim] (NUMERIC type)
   * @param windowSize Sliding window size - tokens can only attend to this many previous positions
   * @param numHeads Number of query attention heads
   * @param numKvHeads Number of KV heads (0 = same as numHeads)
   * @param scale Scaling factor. 0 = auto (1/sqrt(headDim))
   * @return output Attention output. Shape: [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable slidingWindowAttention(String name, SDVariable query, SDVariable key,
      SDVariable value, int windowSize, int numHeads, int numKvHeads, double scale) {
    SDValidation.validateNumerical("slidingWindowAttention", "query", query);
    SDValidation.validateNumerical("slidingWindowAttention", "key", key);
    SDValidation.validateNumerical("slidingWindowAttention", "value", value);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SlidingWindowAttention(sd,query, key, value, windowSize, numHeads, numKvHeads, scale).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Softmax activation, along the specified dimension<br>
   *
   * @param x Input (NUMERIC type)
   * @param dimension Dimension along which to apply softmax
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softmax(SDVariable x, int dimension) {
    SDValidation.validateNumerical("softmax", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(sd,x, dimension).outputVariable();
  }

  /**
   * Softmax activation, along the specified dimension<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input (NUMERIC type)
   * @param dimension Dimension along which to apply softmax
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softmax(String name, SDVariable x, int dimension) {
    SDValidation.validateNumerical("softmax", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(sd,x, dimension).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Softmax activation, along the specified dimension<br>
   *
   * @param x Input (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softmax(SDVariable x) {
    SDValidation.validateNumerical("softmax", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(sd,x, -1).outputVariable();
  }

  /**
   * Softmax activation, along the specified dimension<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softmax(String name, SDVariable x) {
    SDValidation.validateNumerical("softmax", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(sd,x, -1).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise softplus function: out = log(exp(x) + 1)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softplus(SDVariable x) {
    SDValidation.validateNumerical("softplus", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus(sd,x).outputVariable();
  }

  /**
   * Element-wise softplus function: out = log(exp(x) + 1)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softplus(String name, SDVariable x) {
    SDValidation.validateNumerical("softplus", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise softsign function: out = x / (abs(x) + 1)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softsign(SDVariable x) {
    SDValidation.validateNumerical("softsign", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftSign(sd,x).outputVariable();
  }

  /**
   * Element-wise softsign function: out = x / (abs(x) + 1)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable softsign(String name, SDVariable x) {
    SDValidation.validateNumerical("softsign", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.SoftSign(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise derivative (dOut/dIn) of the softsign function softsign(INDArray)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output (NUMERIC type)
   */
  public SDVariable softsignDerivative(SDVariable x) {
    SDValidation.validateNumerical("softsignDerivative", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative(sd,x).outputVariable();
  }

  /**
   * Element-wise derivative (dOut/dIn) of the softsign function softsign(INDArray)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output (NUMERIC type)
   */
  public SDVariable softsignDerivative(String name, SDVariable x) {
    SDValidation.validateNumerical("softsignDerivative", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftSignDerivative(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0<br>
   * See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a><br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable swish(SDVariable x) {
    SDValidation.validateNumerical("swish", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Swish(sd,x).outputVariable();
  }

  /**
   * Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0<br>
   * See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a><br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable swish(String name, SDVariable x) {
    SDValidation.validateNumerical("swish", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Swish(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable tanh(SDVariable x) {
    SDValidation.validateNumerical("tanh", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh(sd,x).outputVariable();
  }

  /**
   * Elementwise tanh (hyperbolic tangent) operation: out = tanh(x)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable tanh(String name, SDVariable x) {
    SDValidation.validateNumerical("tanh", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.strict.Tanh(sd,x).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable tokenSample(SDVariable logits) {
    SDValidation.validateNumerical("tokenSample", "logits", logits);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample(sd,logits, 0.0, 0, 0.0, 0).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param logits Logits tensor. Shape: [vocabSize], [batch, vocabSize], or [batch, seqLen, vocabSize]. For rank-3, samples from the last sequence position. (NUMERIC type)
   * @return output Sampled token indices. Shape: [batch] or scalar (LONG type)
   */
  public SDVariable tokenSample(String name, SDVariable logits) {
    SDValidation.validateNumerical("tokenSample", "logits", logits);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample(sd,logits, 0.0, 0, 0.0, 0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable tokenSample(SDVariable logits, double temperature, int topK, double topP,
      long seed) {
    SDValidation.validateNumerical("tokenSample", "logits", logits);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample(sd,logits, temperature, topK, topP, seed).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param logits Logits tensor. Shape: [vocabSize], [batch, vocabSize], or [batch, seqLen, vocabSize]. For rank-3, samples from the last sequence position. (NUMERIC type)
   * @param temperature Temperature for sampling. 0 = greedy (argmax)
   * @param topK Top-K filtering: keep only top K logits. 0 = disabled
   * @param topP Top-P (nucleus) filtering threshold. 0 = disabled
   * @param seed Random seed for sampling. 0 = random
   * @return output Sampled token indices. Shape: [batch] or scalar (LONG type)
   */
  public SDVariable tokenSample(String name, SDVariable logits, double temperature, int topK,
      double topP, long seed) {
    SDValidation.validateNumerical("tokenSample", "logits", logits);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.TokenSample(sd,logits, temperature, topK, topP, seed).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Find values and indices for the largest k entries along the last dimension.<br>
   *
   * @param input Input data (NUMERIC type)
   * @param k The number of values to return
   * @param sorted Whether to return the values sorted or not
   */
  public SDVariable[] topK(SDVariable input, double k, boolean sorted) {
    SDValidation.validateNumerical("topK", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.TopK(sd,input, k, sorted).outputVariables();
  }

  /**
   * Find values and indices for the largest k entries along the last dimension.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input Input data (NUMERIC type)
   * @param k The number of values to return
   * @param sorted Whether to return the values sorted or not
   */
  public SDVariable[] topK(String[] names, SDVariable input, double k, boolean sorted) {
    SDValidation.validateNumerical("topK", "input", input);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.TopK(sd,input, k, sorted).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] twoWayCrossAttention(SDVariable tokenQuery, SDVariable tokenKey,
      SDVariable tokenValue, SDVariable imageQuery, SDVariable imageKey, SDVariable imageValue,
      double scale) {
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenQuery", tokenQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenKey", tokenKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenValue", tokenValue);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageQuery", imageQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageKey", imageKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageValue", imageValue);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.TwoWayCrossAttention(sd,tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue, scale).outputVariables();
  }

  /**
   * SAM-style Two-Way Cross Attention.<br>
   * Bidirectional cross-attention where tokens attend to image features and<br>
   * image features attend to tokens simultaneously:<br>
   *   tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV<br>
   *   imageOutput = softmax(imageQ @ tokenK^T * scale) @ tokenV<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param tokenQuery Token queries [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenKey Token keys [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenValue Token values [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param imageQuery Image queries [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageKey Image keys [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageValue Image values [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param scale Attention scale factor (default: 1/sqrt(embedDim))
   */
  public SDVariable[] twoWayCrossAttention(String[] names, SDVariable tokenQuery,
      SDVariable tokenKey, SDVariable tokenValue, SDVariable imageQuery, SDVariable imageKey,
      SDVariable imageValue, double scale) {
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenQuery", tokenQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenKey", tokenKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenValue", tokenValue);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageQuery", imageQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageKey", imageKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageValue", imageValue);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.TwoWayCrossAttention(sd,tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue, scale).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] twoWayCrossAttention(SDVariable tokenQuery, SDVariable tokenKey,
      SDVariable tokenValue, SDVariable imageQuery, SDVariable imageKey, SDVariable imageValue) {
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenQuery", tokenQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenKey", tokenKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenValue", tokenValue);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageQuery", imageQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageKey", imageKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageValue", imageValue);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.TwoWayCrossAttention(sd,tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue, 0.0).outputVariables();
  }

  /**
   * SAM-style Two-Way Cross Attention.<br>
   * Bidirectional cross-attention where tokens attend to image features and<br>
   * image features attend to tokens simultaneously:<br>
   *   tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV<br>
   *   imageOutput = softmax(imageQ @ tokenK^T * scale) @ tokenV<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param tokenQuery Token queries [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenKey Token keys [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param tokenValue Token values [batch, tokenSeqLen, embedDim] (NUMERIC type)
   * @param imageQuery Image queries [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageKey Image keys [batch, imageSeqLen, embedDim] (NUMERIC type)
   * @param imageValue Image values [batch, imageSeqLen, embedDim] (NUMERIC type)
   */
  public SDVariable[] twoWayCrossAttention(String[] names, SDVariable tokenQuery,
      SDVariable tokenKey, SDVariable tokenValue, SDVariable imageQuery, SDVariable imageKey,
      SDVariable imageValue) {
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenQuery", tokenQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenKey", tokenKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "tokenValue", tokenValue);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageQuery", imageQuery);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageKey", imageKey);
    SDValidation.validateNumerical("twoWayCrossAttention", "imageValue", imageValue);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.TwoWayCrossAttention(sd,tokenQuery, tokenKey, tokenValue, imageQuery, imageKey, imageValue, 0.0).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable windowedAttention(SDVariable query, SDVariable key, SDVariable value,
      int windowSize, int numHeads) {
    SDValidation.validateNumerical("windowedAttention", "query", query);
    SDValidation.validateNumerical("windowedAttention", "key", key);
    SDValidation.validateNumerical("windowedAttention", "value", value);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.WindowedAttention(sd,query, key, value, null, null, windowSize, numHeads, 0, 0.0, false).outputVariable();
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
   * @param name name May be null. Name for the output variable
   * @param query Query tensor. Shape: [batch, seqLen, numHeads, headDim] for 1D or [batch, height, width, numHeads, headDim] for 2D (NUMERIC type)
   * @param key Key tensor. Same shape as query (NUMERIC type)
   * @param value Value tensor. Same shape as query (NUMERIC type)
   * @param windowSize Size of attention window
   * @param numHeads Number of attention heads
   * @return output Attention output. Same shape as query (NUMERIC type)
   */
  public SDVariable windowedAttention(String name, SDVariable query, SDVariable key,
      SDVariable value, int windowSize, int numHeads) {
    SDValidation.validateNumerical("windowedAttention", "query", query);
    SDValidation.validateNumerical("windowedAttention", "key", key);
    SDValidation.validateNumerical("windowedAttention", "value", value);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.WindowedAttention(sd,query, key, value, null, null, windowSize, numHeads, 0, 0.0, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable windowedAttention(SDVariable query, SDVariable key, SDVariable value,
      SDVariable relativePositionBias, SDVariable attentionMask, int windowSize, int numHeads,
      int shiftSize, double scale, boolean returnWeights) {
    SDValidation.validateNumerical("windowedAttention", "query", query);
    SDValidation.validateNumerical("windowedAttention", "key", key);
    SDValidation.validateNumerical("windowedAttention", "value", value);
    SDValidation.validateNumerical("windowedAttention", "relativePositionBias", relativePositionBias);
    SDValidation.validateNumerical("windowedAttention", "attentionMask", attentionMask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.WindowedAttention(sd,query, key, value, relativePositionBias, attentionMask, windowSize, numHeads, shiftSize, scale, returnWeights).outputVariable();
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable windowedAttention(String name, SDVariable query, SDVariable key,
      SDVariable value, SDVariable relativePositionBias, SDVariable attentionMask, int windowSize,
      int numHeads, int shiftSize, double scale, boolean returnWeights) {
    SDValidation.validateNumerical("windowedAttention", "query", query);
    SDValidation.validateNumerical("windowedAttention", "key", key);
    SDValidation.validateNumerical("windowedAttention", "value", value);
    SDValidation.validateNumerical("windowedAttention", "relativePositionBias", relativePositionBias);
    SDValidation.validateNumerical("windowedAttention", "attentionMask", attentionMask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.WindowedAttention(sd,query, key, value, relativePositionBias, attentionMask, windowSize, numHeads, shiftSize, scale, returnWeights).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
