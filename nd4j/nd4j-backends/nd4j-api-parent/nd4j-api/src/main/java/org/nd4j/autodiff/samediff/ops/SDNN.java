/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.autodiff.samediff.ops;

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
   * Dropout operation<br>
   *
   * @param input Input array (NUMERIC type)
   * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
   * @return output Output (NUMERIC type)
   */
  public SDVariable dropout(SDVariable input, double inputRetainProbability) {
    SDValidation.validateNumerical("dropout", "input", input);
    return new org.nd4j.linalg.api.ops.random.impl.DropOut(sd,input, inputRetainProbability).outputVariable();
  }

  /**
   * Dropout operation<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input array (NUMERIC type)
   * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
   * @return output Output (NUMERIC type)
   */
  public SDVariable dropout(String name, SDVariable input, double inputRetainProbability) {
    SDValidation.validateNumerical("dropout", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.random.impl.DropOut(sd,input, inputRetainProbability).outputVariable();
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
      boolean channelsFirst, int... dimensions) {
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
      boolean channelsFirst, int... dimensions) {
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
      int... dimensions) {
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
      int... dimensions) {
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
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable linear(SDVariable input, SDVariable weights, SDVariable bias) {
    SDValidation.validateNumerical("linear", "input", input);
    SDValidation.validateNumerical("linear", "weights", weights);
    SDValidation.validateNumerical("linear", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(sd,input, weights, bias).outputVariable();
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
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(sd,input, weights, bias).outputVariable();
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
   * Padding operation <br>
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
   * Padding operation <br>
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
   * Padding operation <br>
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
   * Padding operation <br>
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
   * Note that bias array is optional<br>
   *
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable (NUMERIC type)
   * @param bias Optional bias variable (may be null) (NUMERIC type)
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
   * Note that bias array is optional<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable (NUMERIC type)
   * @param bias Optional bias variable (may be null) (NUMERIC type)
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
   * Softmax derivative function<br>
   *
   * @param x Softmax input (NUMERIC type)
   * @param wrt Gradient at output, dL/dx (NUMERIC type)
   * @param dimension Softmax dimension
   * @return output  (NUMERIC type)
   */
  public SDVariable softmaxDerivative(SDVariable x, SDVariable wrt, int dimension) {
    SDValidation.validateNumerical("softmaxDerivative", "x", x);
    SDValidation.validateNumerical("softmaxDerivative", "wrt", wrt);
    return new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp(sd,x, wrt, dimension).outputVariable();
  }

  /**
   * Softmax derivative function<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Softmax input (NUMERIC type)
   * @param wrt Gradient at output, dL/dx (NUMERIC type)
   * @param dimension Softmax dimension
   * @return output  (NUMERIC type)
   */
  public SDVariable softmaxDerivative(String name, SDVariable x, SDVariable wrt, int dimension) {
    SDValidation.validateNumerical("softmaxDerivative", "x", x);
    SDValidation.validateNumerical("softmaxDerivative", "wrt", wrt);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp(sd,x, wrt, dimension).outputVariable();
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
}
