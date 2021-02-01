/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.CReLU(x))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.BatchNorm(input, mean, variance, gamma, beta, epsilon, axis))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.broadcast.BiasAdd(input, bias, nchw))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttention(queries, keys, values, mask, scaled, false))[0];
  }

  /**
   * Dropout operation<br>
   *
   * @param input Input array (NUMERIC type)
   * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
   * @return output Output (NUMERIC type)
   */
  public INDArray dropout(INDArray input, double inputRetainProbability) {
    NDValidation.validateNumerical("dropout", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.random.impl.DropOut(input, inputRetainProbability));
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.strict.ELU(x))[0];
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
      int... dimensions) {
    NDValidation.validateNumerical("layerNorm", "input", input);
    NDValidation.validateNumerical("layerNorm", "gain", gain);
    NDValidation.validateNumerical("layerNorm", "bias", bias);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(input, gain, bias, channelsFirst, dimensions))[0];
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
      int... dimensions) {
    NDValidation.validateNumerical("layerNorm", "input", input);
    NDValidation.validateNumerical("layerNorm", "gain", gain);
    Preconditions.checkArgument(dimensions.length >= 1, "dimensions has incorrect size/length. Expected: dimensions.length >= 1, got %s", dimensions.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm(input, gain, null, channelsFirst, dimensions))[0];
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
   * @return output Output variable (NUMERIC type)
   */
  public INDArray linear(INDArray input, INDArray weights, INDArray bias) {
    NDValidation.validateNumerical("linear", "input", input);
    NDValidation.validateNumerical("linear", "weights", weights);
    NDValidation.validateNumerical("linear", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.XwPlusB(input, weights, bias))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(x))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.LogSoftMax(x, dimension))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.MultiHeadDotProductAttention(queries, keys, values, Wq, Wk, Wv, Wo, mask, scaled, false))[0];
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
  public INDArray pad(INDArray input, INDArray padding, PadMode PadMode, double constant) {
    NDValidation.validateNumerical("pad", "input", input);
    NDValidation.validateNumerical("pad", "padding", padding);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.Pad(input, padding, PadMode, constant))[0];
  }

  /**
   * Padding operation <br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param padding Padding value (NUMERIC type)
   * @param constant Padding constant
   * @return output Padded input (NUMERIC type)
   */
  public INDArray pad(INDArray input, INDArray padding, double constant) {
    NDValidation.validateNumerical("pad", "input", input);
    NDValidation.validateNumerical("pad", "padding", padding);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.Pad(input, padding, PadMode.CONSTANT, constant))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.scalar.PRelu(input, alpha, sharedAxes))[0];
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
   * Note that bias array is optional<br>
   *
   * @param input Input data (NUMERIC type)
   * @param weights Weights variable (NUMERIC type)
   * @param bias Optional bias variable (may be null) (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray reluLayer(INDArray input, INDArray weights, INDArray bias) {
    NDValidation.validateNumerical("reluLayer", "input", input);
    NDValidation.validateNumerical("reluLayer", "weights", weights);
    NDValidation.validateNumerical("reluLayer", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.ReluLayer(input, weights, bias))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.SigmoidDerivative(x, wrt))[0];
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
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(x, dimension))[0];
  }

  /**
   * Softmax activation, along the specified dimension<br>
   *
   * @param x Input (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public INDArray softmax(INDArray x) {
    NDValidation.validateNumerical("softmax", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SoftMax(x, -1))[0];
  }

  /**
   * Softmax derivative function<br>
   *
   * @param x Softmax input (NUMERIC type)
   * @param wrt Gradient at output, dL/dx (NUMERIC type)
   * @param dimension Softmax dimension
   * @return output  (NUMERIC type)
   */
  public INDArray softmaxDerivative(INDArray x, INDArray wrt, int dimension) {
    NDValidation.validateNumerical("softmaxDerivative", "x", x);
    NDValidation.validateNumerical("softmaxDerivative", "wrt", wrt);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftmaxBp(x, wrt, dimension))[0];
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
}
