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
   * Applies Attention with Linear Biases (ALiBi) position encoding to attention scores.<br>
   *
   * @param scores Attention scores [batch, num_heads, seq_len, kv_len] (NUMERIC type)
   * @param numHeads Number of attention heads
   * @return output Scores with ALiBi position bias applied (NUMERIC type)
   */
  public SDVariable applyAlibi(SDVariable scores, int numHeads) {
    SDValidation.validateNumerical("applyAlibi", "scores", scores);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.ApplyAlibi(sd,scores, numHeads).outputVariable();
  }

  /**
   * Applies Attention with Linear Biases (ALiBi) position encoding to attention scores.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param scores Attention scores [batch, num_heads, seq_len, kv_len] (NUMERIC type)
   * @param numHeads Number of attention heads
   * @return output Scores with ALiBi position bias applied (NUMERIC type)
   */
  public SDVariable applyAlibi(String name, SDVariable scores, int numHeads) {
    SDValidation.validateNumerical("applyAlibi", "scores", scores);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ApplyAlibi(sd,scores, numHeads).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable awqMatmul(SDVariable input, SDVariable weightPacked, SDVariable weightScale,
      int groupSize) {
    SDValidation.validateNumerical("awqMatmul", "input", input);
    SDValidation.validateNumerical("awqMatmul", "weightPacked", weightPacked);
    SDValidation.validateNumerical("awqMatmul", "weightScale", weightScale);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.AwqMatmul(sd,input, weightPacked, weightScale, groupSize).outputVariable();
  }

  /**
   * Activation-aware Weight Quantization (AWQ) matrix multiplication.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param weightPacked AWQ-packed weight (NUMERIC type)
   * @param weightScale Weight quantization scales (NUMERIC type)
   * @param groupSize Quantization group size
   * @return output Dequantized matmul result (NUMERIC type)
   */
  public SDVariable awqMatmul(String name, SDVariable input, SDVariable weightPacked,
      SDVariable weightScale, int groupSize) {
    SDValidation.validateNumerical("awqMatmul", "input", input);
    SDValidation.validateNumerical("awqMatmul", "weightPacked", weightPacked);
    SDValidation.validateNumerical("awqMatmul", "weightScale", weightScale);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.AwqMatmul(sd,input, weightPacked, weightScale, groupSize).outputVariable();
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
  public SDVariable[] causalConv1d(SDVariable x, SDVariable weight, SDVariable bias,
      SDVariable convStateIn, int activation, int wFormat) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    SDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, convStateIn, activation, wFormat).outputVariables();
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   * @param convStateIn Conv state for autoregressive decode [batch, dim, kernelSize-1] (NUMERIC type)
   * @param activation Activation function (0=none, 1=silu)
   * @param wFormat Weight format (0=[D,K] PyTorch/ONNX default, 1=[K,D] TensorFlow)
   */
  public SDVariable[] causalConv1d(String[] names, SDVariable x, SDVariable weight, SDVariable bias,
      SDVariable convStateIn, int activation, int wFormat) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    SDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, convStateIn, activation, wFormat).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] causalConv1d(SDVariable x, SDVariable weight) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, null, null, 0, 0).outputVariables();
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   */
  public SDVariable[] causalConv1d(String[] names, SDVariable x, SDVariable weight) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, null, null, 0, 0).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] causalConv1d(SDVariable x, SDVariable weight, SDVariable bias) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, null, 0, 0).outputVariables();
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   */
  public SDVariable[] causalConv1d(String[] names, SDVariable x, SDVariable weight,
      SDVariable bias) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, null, 0, 0).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] causalConv1d(SDVariable x, SDVariable weight, SDVariable bias,
      SDVariable convStateIn) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    SDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, convStateIn, 0, 0).outputVariables();
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   * @param convStateIn Conv state for autoregressive decode [batch, dim, kernelSize-1] (NUMERIC type)
   */
  public SDVariable[] causalConv1d(String[] names, SDVariable x, SDVariable weight, SDVariable bias,
      SDVariable convStateIn) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    SDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, convStateIn, 0, 0).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] causalConv1d(SDVariable x, SDVariable weight, SDVariable bias,
      SDVariable convStateIn, int activation) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    SDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, convStateIn, activation, 0).outputVariables();
  }

  /**
   * Causal depthwise 1D convolution with state for autoregressive decoding.<br>
   * <br>
   * Performs a causal (left-padded) depthwise 1D convolution.<br>
   * Used in Gated Delta Networks (GDN) and Mamba architectures.<br>
   * The state output preserves the last (kernelSize-1) input elements<br>
   * for use as initial state in the next autoregressive step.<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input sequence [batch, seqLen, dim] (NUMERIC type)
   * @param weight Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1) (NUMERIC type)
   * @param bias Bias [dim] (NUMERIC type)
   * @param convStateIn Conv state for autoregressive decode [batch, dim, kernelSize-1] (NUMERIC type)
   * @param activation Activation function (0=none, 1=silu)
   */
  public SDVariable[] causalConv1d(String[] names, SDVariable x, SDVariable weight, SDVariable bias,
      SDVariable convStateIn, int activation) {
    SDValidation.validateNumerical("causalConv1d", "x", x);
    SDValidation.validateNumerical("causalConv1d", "weight", weight);
    SDValidation.validateNumerical("causalConv1d", "bias", bias);
    SDValidation.validateNumerical("causalConv1d", "convStateIn", convStateIn);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(sd,x, weight, bias, convStateIn, activation, 0).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable columnParallelLinear(SDVariable input, SDVariable weight, int tpRank,
      int tpSize, boolean gatherOutput) {
    SDValidation.validateNumerical("columnParallelLinear", "input", input);
    SDValidation.validateNumerical("columnParallelLinear", "weight", weight);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.ColumnParallelLinear(sd,input, weight, tpRank, tpSize, gatherOutput).outputVariable();
  }

  /**
   * Column-parallel linear layer for tensor parallelism.<br>
   * Splits weight columns across tensor parallel ranks.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param weight Weight matrix (NUMERIC type)
   * @param tpRank Tensor parallel rank
   * @param tpSize Tensor parallel world size
   * @param gatherOutput Whether to all-gather output
   * @return output Column-parallel linear output (NUMERIC type)
   */
  public SDVariable columnParallelLinear(String name, SDVariable input, SDVariable weight,
      int tpRank, int tpSize, boolean gatherOutput) {
    SDValidation.validateNumerical("columnParallelLinear", "input", input);
    SDValidation.validateNumerical("columnParallelLinear", "weight", weight);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.ColumnParallelLinear(sd,input, weight, tpRank, tpSize, gatherOutput).outputVariable();
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
  public SDVariable decoderMaskedMha(SDVariable query, SDVariable key, SDVariable value,
      int numHeads, boolean isCausal) {
    SDValidation.validateNumerical("decoderMaskedMha", "query", query);
    SDValidation.validateNumerical("decoderMaskedMha", "key", key);
    SDValidation.validateNumerical("decoderMaskedMha", "value", value);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DecoderMaskedMha(sd,query, key, value, numHeads, isCausal).outputVariable();
  }

  /**
   * Decoder-optimized masked multi-head attention.<br>
   * Optimized for autoregressive decoding with incremental KV cache.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param query Query tensor (NUMERIC type)
   * @param key Key tensor (NUMERIC type)
   * @param value Value tensor (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param isCausal Whether to apply causal mask
   * @return output Attention output (NUMERIC type)
   */
  public SDVariable decoderMaskedMha(String name, SDVariable query, SDVariable key,
      SDVariable value, int numHeads, boolean isCausal) {
    SDValidation.validateNumerical("decoderMaskedMha", "query", query);
    SDValidation.validateNumerical("decoderMaskedMha", "key", key);
    SDValidation.validateNumerical("decoderMaskedMha", "value", value);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DecoderMaskedMha(sd,query, key, value, numHeads, isCausal).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable doraMatMul(SDVariable input, SDVariable weight, SDVariable loraA,
      SDVariable loraB, SDVariable magnitude, double scaling) {
    SDValidation.validateNumerical("doraMatMul", "input", input);
    SDValidation.validateNumerical("doraMatMul", "weight", weight);
    SDValidation.validateNumerical("doraMatMul", "loraA", loraA);
    SDValidation.validateNumerical("doraMatMul", "loraB", loraB);
    SDValidation.validateNumerical("doraMatMul", "magnitude", magnitude);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DoraMatMul(sd,input, weight, loraA, loraB, magnitude, scaling).outputVariable();
  }

  /**
   * Weight-Decomposed Low-Rank Adaptation (DoRA) fused matrix multiplication.<br>
   * Decomposes weight into magnitude and direction, applies LoRA to direction only.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input [batch, in_features] (NUMERIC type)
   * @param weight Base weight [out_features, in_features] (NUMERIC type)
   * @param loraA LoRA down-projection [r, in_features] (NUMERIC type)
   * @param loraB LoRA up-projection [out_features, r] (NUMERIC type)
   * @param magnitude Per-output magnitude [out_features] (NUMERIC type)
   * @param scaling LoRA scaling factor (default 1.0)
   * @return output DoRA result with weight-decomposed adaptation (NUMERIC type)
   */
  public SDVariable doraMatMul(String name, SDVariable input, SDVariable weight, SDVariable loraA,
      SDVariable loraB, SDVariable magnitude, double scaling) {
    SDValidation.validateNumerical("doraMatMul", "input", input);
    SDValidation.validateNumerical("doraMatMul", "weight", weight);
    SDValidation.validateNumerical("doraMatMul", "loraA", loraA);
    SDValidation.validateNumerical("doraMatMul", "loraB", loraB);
    SDValidation.validateNumerical("doraMatMul", "magnitude", magnitude);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DoraMatMul(sd,input, weight, loraA, loraB, magnitude, scaling).outputVariable();
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
   * - Pass keyCache and valueCache tensors with cachePosition<br>
   * - Current K/V are written at cachePosition in-place, then full cache used for attention<br>
   * - attentionBias masks zero-padded cache positions (set -1e9 beyond cachePosition)<br>
   * - All tensor shapes are fixed after first decode step, enabling DSP replay<br>
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
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, null, null, null, null, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * - Pass keyCache and valueCache tensors with cachePosition<br>
   * - Current K/V are written at cachePosition in-place, then full cache used for attention<br>
   * - attentionBias masks zero-padded cache positions (set -1e9 beyond cachePosition)<br>
   * - All tensor shapes are fixed after first decode step, enabling DSP replay<br>
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
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, null, null, null, null, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * - Pass keyCache and valueCache tensors with cachePosition<br>
   * - Current K/V are written at cachePosition in-place, then full cache used for attention<br>
   * - attentionBias masks zero-padded cache positions (set -1e9 beyond cachePosition)<br>
   * - All tensor shapes are fixed after first decode step, enabling DSP replay<br>
   * <br>
   * See "Attention is all you need" (https://arxiv.org/abs/1706.03762)<br>
   * See "FlashAttention: Fast and Memory-Efficient Exact Attention" (https://arxiv.org/abs/2205.14135)<br>
   *
   * @param queries Query tensor. Shape: [batchSize, numQueries, queryDim] or [batchSize, numQueries, numHeads, headDim] for flash attention (NUMERIC type)
   * @param values Value tensor. Shape: [batchSize, numValues, valueDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param keys Key tensor. Shape: [batchSize, numValues, keyDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param queryMask Query mask tensor (optional). Shape: [batchSize, numQueries] (NUMERIC type)
   * @param valueMask Value mask tensor (optional). Shape: [batchSize, numValues] (NUMERIC type)
   * @param attentionBias Attention bias tensor (optional). Shape: [batchSize, numHeads, numQueries, numKeys] or broadcastable. Added to attention scores before softmax. When KV cache is active, placed at input[8] to mask zero-padded cache positions. (NUMERIC type)
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
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, null, null, null, attentionBias, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * - Pass keyCache and valueCache tensors with cachePosition<br>
   * - Current K/V are written at cachePosition in-place, then full cache used for attention<br>
   * - attentionBias masks zero-padded cache positions (set -1e9 beyond cachePosition)<br>
   * - All tensor shapes are fixed after first decode step, enabling DSP replay<br>
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
   * @param attentionBias Attention bias tensor (optional). Shape: [batchSize, numHeads, numQueries, numKeys] or broadcastable. Added to attention scores before softmax. When KV cache is active, placed at input[8] to mask zero-padded cache positions. (NUMERIC type)
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
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, null, null, null, attentionBias, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * - Pass keyCache and valueCache tensors with cachePosition<br>
   * - Current K/V are written at cachePosition in-place, then full cache used for attention<br>
   * - attentionBias masks zero-padded cache positions (set -1e9 beyond cachePosition)<br>
   * - All tensor shapes are fixed after first decode step, enabling DSP replay<br>
   * <br>
   * See "Attention is all you need" (https://arxiv.org/abs/1706.03762)<br>
   * See "FlashAttention: Fast and Memory-Efficient Exact Attention" (https://arxiv.org/abs/2205.14135)<br>
   *
   * @param queries Query tensor. Shape: [batchSize, numQueries, queryDim] or [batchSize, numQueries, numHeads, headDim] for flash attention (NUMERIC type)
   * @param values Value tensor. Shape: [batchSize, numValues, valueDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param keys Key tensor. Shape: [batchSize, numValues, keyDim] or [batchSize, numValues, numHeads, headDim] (NUMERIC type)
   * @param queryMask Query mask tensor (optional). Shape: [batchSize, numQueries] (NUMERIC type)
   * @param valueMask Value mask tensor (optional). Shape: [batchSize, numValues] (NUMERIC type)
   * @param keyCache Key cache tensor (optional). Shape: [batchSize, maxSeqLen, numKvHeads, headDim]. For in-place KV cache during autoregressive decoding. (NUMERIC type)
   * @param valueCache Value cache tensor (optional). Shape: [batchSize, maxSeqLen, numKvHeads, headDim]. For in-place KV cache during autoregressive decoding. (NUMERIC type)
   * @param cachePosition Cache write position (optional). Scalar INT64 tensor indicating where to write current K/V in the cache. Enables DSP replay with fixed graph shapes. (NUMERIC type)
   * @param attentionBias Attention bias tensor (optional). Shape: [batchSize, numHeads, numQueries, numKeys] or broadcastable. Added to attention scores before softmax. When KV cache is active, placed at input[8] to mask zero-padded cache positions. (NUMERIC type)
   * @param scaleFactor Scaling factor applied to attention scores. 0 = auto (1/sqrt(headDim))
   * @param dropoutProbability Dropout probability applied to attention weights
   * @param useCausalMask Whether to apply causal mask for autoregressive tasks
   * @param training Whether in training mode (affects dropout)
   * @return output Output tensor. Shape: [batchSize, numQueries, valueDim] or [batchSize, numQueries, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dotProductAttentionV2(SDVariable queries, SDVariable values, SDVariable keys,
      SDVariable queryMask, SDVariable valueMask, SDVariable keyCache, SDVariable valueCache,
      SDVariable cachePosition, SDVariable attentionBias, double scaleFactor,
      double dropoutProbability, boolean useCausalMask, boolean training) {
    SDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    SDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    SDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    SDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "keyCache", keyCache);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueCache", valueCache);
    SDValidation.validateNumerical("dotProductAttentionV2", "cachePosition", cachePosition);
    SDValidation.validateNumerical("dotProductAttentionV2", "attentionBias", attentionBias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, keyCache, valueCache, cachePosition, attentionBias, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * - Pass keyCache and valueCache tensors with cachePosition<br>
   * - Current K/V are written at cachePosition in-place, then full cache used for attention<br>
   * - attentionBias masks zero-padded cache positions (set -1e9 beyond cachePosition)<br>
   * - All tensor shapes are fixed after first decode step, enabling DSP replay<br>
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
   * @param keyCache Key cache tensor (optional). Shape: [batchSize, maxSeqLen, numKvHeads, headDim]. For in-place KV cache during autoregressive decoding. (NUMERIC type)
   * @param valueCache Value cache tensor (optional). Shape: [batchSize, maxSeqLen, numKvHeads, headDim]. For in-place KV cache during autoregressive decoding. (NUMERIC type)
   * @param cachePosition Cache write position (optional). Scalar INT64 tensor indicating where to write current K/V in the cache. Enables DSP replay with fixed graph shapes. (NUMERIC type)
   * @param attentionBias Attention bias tensor (optional). Shape: [batchSize, numHeads, numQueries, numKeys] or broadcastable. Added to attention scores before softmax. When KV cache is active, placed at input[8] to mask zero-padded cache positions. (NUMERIC type)
   * @param scaleFactor Scaling factor applied to attention scores. 0 = auto (1/sqrt(headDim))
   * @param dropoutProbability Dropout probability applied to attention weights
   * @param useCausalMask Whether to apply causal mask for autoregressive tasks
   * @param training Whether in training mode (affects dropout)
   * @return output Output tensor. Shape: [batchSize, numQueries, valueDim] or [batchSize, numQueries, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dotProductAttentionV2(String name, SDVariable queries, SDVariable values,
      SDVariable keys, SDVariable queryMask, SDVariable valueMask, SDVariable keyCache,
      SDVariable valueCache, SDVariable cachePosition, SDVariable attentionBias, double scaleFactor,
      double dropoutProbability, boolean useCausalMask, boolean training) {
    SDValidation.validateNumerical("dotProductAttentionV2", "queries", queries);
    SDValidation.validateNumerical("dotProductAttentionV2", "values", values);
    SDValidation.validateNumerical("dotProductAttentionV2", "keys", keys);
    SDValidation.validateNumerical("dotProductAttentionV2", "queryMask", queryMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueMask", valueMask);
    SDValidation.validateNumerical("dotProductAttentionV2", "keyCache", keyCache);
    SDValidation.validateNumerical("dotProductAttentionV2", "valueCache", valueCache);
    SDValidation.validateNumerical("dotProductAttentionV2", "cachePosition", cachePosition);
    SDValidation.validateNumerical("dotProductAttentionV2", "attentionBias", attentionBias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DotProductAttentionV2(sd,queries, values, keys, queryMask, valueMask, keyCache, valueCache, cachePosition, attentionBias, scaleFactor, dropoutProbability, useCausalMask, training).outputVariable();
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
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @param attentionType Attention type (0=local/sliding-window, 1=global/full-context)
   * @param positionOffset Position offset for KV cache continuation
   * @param localFreqBase RoPE frequency base for local/sliding-window layers
   * @param globalFreqBase RoPE frequency base for global/full-context layers
   * @param localFreqScale RoPE frequency scale for local layers
   * @param globalFreqScale RoPE frequency scale for global layers
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(SDVariable input, int attentionType, int positionOffset,
      double localFreqBase, double globalFreqBase, double localFreqScale, double globalFreqScale) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, attentionType, positionOffset, localFreqBase, globalFreqBase, localFreqScale, globalFreqScale).outputVariable();
  }

  /**
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @param attentionType Attention type (0=local/sliding-window, 1=global/full-context)
   * @param positionOffset Position offset for KV cache continuation
   * @param localFreqBase RoPE frequency base for local/sliding-window layers
   * @param globalFreqBase RoPE frequency base for global/full-context layers
   * @param localFreqScale RoPE frequency scale for local layers
   * @param globalFreqScale RoPE frequency scale for global layers
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(String name, SDVariable input, int attentionType, int positionOffset,
      double localFreqBase, double globalFreqBase, double localFreqScale, double globalFreqScale) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, attentionType, positionOffset, localFreqBase, globalFreqBase, localFreqScale, globalFreqScale).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(SDVariable input) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, 0, 0, 10000.0, 1000000.0, 1.0, 1.0).outputVariable();
  }

  /**
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(String name, SDVariable input) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, 0, 0, 10000.0, 1000000.0, 1.0, 1.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @param attentionType Attention type (0=local/sliding-window, 1=global/full-context)
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(SDVariable input, int attentionType) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, attentionType, 0, 10000.0, 1000000.0, 1.0, 1.0).outputVariable();
  }

  /**
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @param attentionType Attention type (0=local/sliding-window, 1=global/full-context)
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(String name, SDVariable input, int attentionType) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, attentionType, 0, 10000.0, 1000000.0, 1.0, 1.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @param attentionType Attention type (0=local/sliding-window, 1=global/full-context)
   * @param positionOffset Position offset for KV cache continuation
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(SDVariable input, int attentionType, int positionOffset) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, attentionType, positionOffset, 10000.0, 1000000.0, 1.0, 1.0).outputVariable();
  }

  /**
   * Dual Rotary Position Embedding (Gemma 4).<br>
   * <br>
   * Applies two different RoPE configurations depending on attention type:<br>
   * - Standard RoPE (localFreqBase) for sliding-window (local) attention layers<br>
   * - Proportional RoPE (globalFreqBase) for global full-context attention layers<br>
   * <br>
   * This enables longer context windows by using different position encoding<br>
   * frequencies for local vs global attention. For each dimension pair (2i, 2i+1):<br>
   *   theta_i = freqBase ^ (-2i / headDim) * freqScale<br>
   *   output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)<br>
   *   output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even (NUMERIC type)
   * @param attentionType Attention type (0=local/sliding-window, 1=global/full-context)
   * @param positionOffset Position offset for KV cache continuation
   * @return output Output with rotary embeddings applied [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable dualRoPE(String name, SDVariable input, int attentionType, int positionOffset) {
    SDValidation.validateNumerical("dualRoPE", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.DualRoPE(sd,input, attentionType, positionOffset, 10000.0, 1000000.0, 1.0, 1.0).outputVariable();
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
   * FP8 matrix multiplication with per-tensor scaling.<br>
   *
   * @param a First matrix (NUMERIC type)
   * @param b Second matrix (NUMERIC type)
   * @param scaleA Scale for matrix A (NUMERIC type)
   * @param scaleB Scale for matrix B (NUMERIC type)
   * @return output Scaled FP8 matmul result (NUMERIC type)
   */
  public SDVariable fp8Matmul(SDVariable a, SDVariable b, SDVariable scaleA, SDVariable scaleB) {
    SDValidation.validateNumerical("fp8Matmul", "a", a);
    SDValidation.validateNumerical("fp8Matmul", "b", b);
    SDValidation.validateNumerical("fp8Matmul", "scaleA", scaleA);
    SDValidation.validateNumerical("fp8Matmul", "scaleB", scaleB);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Fp8Matmul(sd,a, b, scaleA, scaleB).outputVariable();
  }

  /**
   * FP8 matrix multiplication with per-tensor scaling.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param a First matrix (NUMERIC type)
   * @param b Second matrix (NUMERIC type)
   * @param scaleA Scale for matrix A (NUMERIC type)
   * @param scaleB Scale for matrix B (NUMERIC type)
   * @return output Scaled FP8 matmul result (NUMERIC type)
   */
  public SDVariable fp8Matmul(String name, SDVariable a, SDVariable b, SDVariable scaleA,
      SDVariable scaleB) {
    SDValidation.validateNumerical("fp8Matmul", "a", a);
    SDValidation.validateNumerical("fp8Matmul", "b", b);
    SDValidation.validateNumerical("fp8Matmul", "scaleA", scaleA);
    SDValidation.validateNumerical("fp8Matmul", "scaleB", scaleB);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Fp8Matmul(sd,a, b, scaleA, scaleB).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable fusedBiasDropoutResidual(SDVariable input, SDVariable bias, SDVariable residual,
      double dropoutProb, boolean training) {
    SDValidation.validateNumerical("fusedBiasDropoutResidual", "input", input);
    SDValidation.validateNumerical("fusedBiasDropoutResidual", "bias", bias);
    SDValidation.validateNumerical("fusedBiasDropoutResidual", "residual", residual);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedBiasDropoutResidual(sd,input, bias, residual, dropoutProb, training).outputVariable();
  }

  /**
   * Fused bias addition, dropout, and residual connection in a single kernel.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param bias Bias tensor (NUMERIC type)
   * @param residual Residual connection tensor (NUMERIC type)
   * @param dropoutProb Dropout probability
   * @param training Whether in training mode
   * @return output dropout(input + bias) + residual (NUMERIC type)
   */
  public SDVariable fusedBiasDropoutResidual(String name, SDVariable input, SDVariable bias,
      SDVariable residual, double dropoutProb, boolean training) {
    SDValidation.validateNumerical("fusedBiasDropoutResidual", "input", input);
    SDValidation.validateNumerical("fusedBiasDropoutResidual", "bias", bias);
    SDValidation.validateNumerical("fusedBiasDropoutResidual", "residual", residual);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedBiasDropoutResidual(sd,input, bias, residual, dropoutProb, training).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable fusedElementwiseChain(SDVariable input, SDVariable[] secondaryInputs,
      int[] opCodes) {
    SDValidation.validateNumerical("fusedElementwiseChain", "input", input);
    SDValidation.validateNumerical("fusedElementwiseChain", "secondaryInputs", secondaryInputs);
    Preconditions.checkArgument(secondaryInputs.length >= 0, "secondaryInputs has incorrect size/length. Expected: secondaryInputs.length >= 0, got %s", secondaryInputs.length);
    Preconditions.checkArgument(opCodes.length >= 1, "opCodes has incorrect size/length. Expected: opCodes.length >= 1, got %s", opCodes.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain(sd,input, secondaryInputs, opCodes).outputVariable();
  }

  /**
   * Executes a fused chain of element-wise operations in a single kernel pass.<br>
   * Intermediate values stay in registers instead of global memory. Replaces N separate kernel launches with 1.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Primary input array (NUMERIC type)
   * @param secondaryInputs Optional secondary input arrays for binary ops (add, sub, mul, div) (NUMERIC type)
   * @param opCodes Op codes: 0=add, 1=sub, 2=mul, 3=div, 10=relu, 11=sigmoid, 12=tanh, 13=gelu, 14=exp, 15=log, 16=abs, 17=neg, 18=square, 19=sqrt, 20=swish, 21=silu, 22=mish, 30=clip, 31=leaky_relu (Size: AtLeast(min=1))
   * @return output Result of applying the fused element-wise chain (NUMERIC type)
   */
  public SDVariable fusedElementwiseChain(String name, SDVariable input,
      SDVariable[] secondaryInputs, int[] opCodes) {
    SDValidation.validateNumerical("fusedElementwiseChain", "input", input);
    SDValidation.validateNumerical("fusedElementwiseChain", "secondaryInputs", secondaryInputs);
    Preconditions.checkArgument(secondaryInputs.length >= 0, "secondaryInputs has incorrect size/length. Expected: secondaryInputs.length >= 0, got %s", secondaryInputs.length);
    Preconditions.checkArgument(opCodes.length >= 1, "opCodes has incorrect size/length. Expected: opCodes.length >= 1, got %s", opCodes.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain(sd,input, secondaryInputs, opCodes).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Fused Gaussian Error Linear Unit (GELU) activation function.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output GELU(x) (NUMERIC type)
   */
  public SDVariable fusedGelu(SDVariable input) {
    SDValidation.validateNumerical("fusedGelu", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGELU(sd,input).outputVariable();
  }

  /**
   * Fused Gaussian Error Linear Unit (GELU) activation function.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @return output GELU(x) (NUMERIC type)
   */
  public SDVariable fusedGelu(String name, SDVariable input) {
    SDValidation.validateNumerical("fusedGelu", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGELU(sd,input).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Fused GEMM + SwiGLU: combines two matrix multiplications with gated activation.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param wGate Gate projection weight (NUMERIC type)
   * @param wUp Up projection weight (NUMERIC type)
   * @return output SwiGLU(input @ wGate, input @ wUp) (NUMERIC type)
   */
  public SDVariable fusedGemmSwiglu(SDVariable input, SDVariable wGate, SDVariable wUp) {
    SDValidation.validateNumerical("fusedGemmSwiglu", "input", input);
    SDValidation.validateNumerical("fusedGemmSwiglu", "wGate", wGate);
    SDValidation.validateNumerical("fusedGemmSwiglu", "wUp", wUp);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGemmSwiglu(sd,input, wGate, wUp).outputVariable();
  }

  /**
   * Fused GEMM + SwiGLU: combines two matrix multiplications with gated activation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param wGate Gate projection weight (NUMERIC type)
   * @param wUp Up projection weight (NUMERIC type)
   * @return output SwiGLU(input @ wGate, input @ wUp) (NUMERIC type)
   */
  public SDVariable fusedGemmSwiglu(String name, SDVariable input, SDVariable wGate,
      SDVariable wUp) {
    SDValidation.validateNumerical("fusedGemmSwiglu", "input", input);
    SDValidation.validateNumerical("fusedGemmSwiglu", "wGate", wGate);
    SDValidation.validateNumerical("fusedGemmSwiglu", "wUp", wUp);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedGemmSwiglu(sd,input, wGate, wUp).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable fusedLayerNorm(SDVariable input, SDVariable gamma, SDVariable beta,
      double epsilon) {
    SDValidation.validateNumerical("fusedLayerNorm", "input", input);
    SDValidation.validateNumerical("fusedLayerNorm", "gamma", gamma);
    SDValidation.validateNumerical("fusedLayerNorm", "beta", beta);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedLayerNorm(sd,input, gamma, beta, epsilon).outputVariable();
  }

  /**
   * Fused layer normalization. Computes mean, variance, normalize, scale and shift in one pass.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param gamma Scale parameter (NUMERIC type)
   * @param beta Bias parameter (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output Layer-normalized output (NUMERIC type)
   */
  public SDVariable fusedLayerNorm(String name, SDVariable input, SDVariable gamma, SDVariable beta,
      double epsilon) {
    SDValidation.validateNumerical("fusedLayerNorm", "input", input);
    SDValidation.validateNumerical("fusedLayerNorm", "gamma", gamma);
    SDValidation.validateNumerical("fusedLayerNorm", "beta", beta);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedLayerNorm(sd,input, gamma, beta, epsilon).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable fusedNormQuantize(SDVariable input, SDVariable gamma, double epsilon,
      int quantType) {
    SDValidation.validateNumerical("fusedNormQuantize", "input", input);
    SDValidation.validateNumerical("fusedNormQuantize", "gamma", gamma);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedNormQuantize(sd,input, gamma, epsilon, quantType).outputVariable();
  }

  /**
   * Fused normalization + quantization in a single kernel.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param gamma Norm scale parameter (NUMERIC type)
   * @param epsilon Epsilon for normalization
   * @param quantType Quantization type
   * @return output Normalized and quantized output (NUMERIC type)
   */
  public SDVariable fusedNormQuantize(String name, SDVariable input, SDVariable gamma,
      double epsilon, int quantType) {
    SDValidation.validateNumerical("fusedNormQuantize", "input", input);
    SDValidation.validateNumerical("fusedNormQuantize", "gamma", gamma);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedNormQuantize(sd,input, gamma, epsilon, quantType).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable fusedRmsNormSwiglu(SDVariable input, SDVariable gamma, SDVariable wGate,
      SDVariable wUp, double epsilon) {
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "input", input);
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "gamma", gamma);
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "wGate", wGate);
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "wUp", wUp);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRmsNormSwiGLU(sd,input, gamma, wGate, wUp, epsilon).outputVariable();
  }

  /**
   * Fused RMSNorm + SwiGLU activation. Combines normalization and gated activation<br>
   * into a single kernel for better memory efficiency.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param gamma RMS norm scale (NUMERIC type)
   * @param wGate Gate projection weight (NUMERIC type)
   * @param wUp Up projection weight (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output Result of fused RMSNorm + SwiGLU (NUMERIC type)
   */
  public SDVariable fusedRmsNormSwiglu(String name, SDVariable input, SDVariable gamma,
      SDVariable wGate, SDVariable wUp, double epsilon) {
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "input", input);
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "gamma", gamma);
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "wGate", wGate);
    SDValidation.validateNumerical("fusedRmsNormSwiglu", "wUp", wUp);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRmsNormSwiGLU(sd,input, gamma, wGate, wUp, epsilon).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Fused Rotary Position Embedding (RoPE).<br>
   * <br>
   * Two modes:<br>
   * 1. Precomputed cache: provide ropeCache with cos/sin values and startPosition<br>
   * 2. Dynamic position: provide scalar positionOffset tensor for KV cache decode<br>
   *    (enables DSP replay with fixed graph shapes)<br>
   * <br>
   * Supports RoPE variants: standard (LLaMA/Mistral), NeoX, GPT-J.<br>
   *
   * @param input Input tensor [batch, seq_len, num_heads, head_dim] (NUMERIC type)
   * @param ropeCache Precomputed RoPE cache (cos/sin) (NUMERIC type)
   * @param startPosition Start position for RoPE application
   * @return output Input with RoPE applied (NUMERIC type)
   */
  public SDVariable fusedRoPE(SDVariable input, SDVariable ropeCache, int startPosition) {
    SDValidation.validateNumerical("fusedRoPE", "input", input);
    SDValidation.validateNumerical("fusedRoPE", "ropeCache", ropeCache);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(sd,input, ropeCache, null, startPosition, 0, 10000.0, 1.0, 0).outputVariable();
  }

  /**
   * Fused Rotary Position Embedding (RoPE).<br>
   * <br>
   * Two modes:<br>
   * 1. Precomputed cache: provide ropeCache with cos/sin values and startPosition<br>
   * 2. Dynamic position: provide scalar positionOffset tensor for KV cache decode<br>
   *    (enables DSP replay with fixed graph shapes)<br>
   * <br>
   * Supports RoPE variants: standard (LLaMA/Mistral), NeoX, GPT-J.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor [batch, seq_len, num_heads, head_dim] (NUMERIC type)
   * @param ropeCache Precomputed RoPE cache (cos/sin) (NUMERIC type)
   * @param startPosition Start position for RoPE application
   * @return output Input with RoPE applied (NUMERIC type)
   */
  public SDVariable fusedRoPE(String name, SDVariable input, SDVariable ropeCache,
      int startPosition) {
    SDValidation.validateNumerical("fusedRoPE", "input", input);
    SDValidation.validateNumerical("fusedRoPE", "ropeCache", ropeCache);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(sd,input, ropeCache, null, startPosition, 0, 10000.0, 1.0, 0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Fused Rotary Position Embedding (RoPE).<br>
   * <br>
   * Two modes:<br>
   * 1. Precomputed cache: provide ropeCache with cos/sin values and startPosition<br>
   * 2. Dynamic position: provide scalar positionOffset tensor for KV cache decode<br>
   *    (enables DSP replay with fixed graph shapes)<br>
   * <br>
   * Supports RoPE variants: standard (LLaMA/Mistral), NeoX, GPT-J.<br>
   *
   * @param input Input tensor [batch, seq_len, num_heads, head_dim] (NUMERIC type)
   * @param positionOffset Scalar INT64 tensor with dynamic position offset for KV cache decode (NUMERIC type)
   * @param ropeType RoPE variant: 0=standard (LLaMA), 1=NeoX, 2=GPT-J
   * @param freqBase Base frequency for RoPE computation
   * @param freqScale Frequency scale factor
   * @param rotaryDims Number of dimensions to rotate (0 = all head dims)
   * @return output Input with RoPE applied (NUMERIC type)
   */
  public SDVariable fusedRoPE(SDVariable input, SDVariable positionOffset, int ropeType,
      double freqBase, double freqScale, int rotaryDims) {
    SDValidation.validateNumerical("fusedRoPE", "input", input);
    SDValidation.validateNumerical("fusedRoPE", "positionOffset", positionOffset);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(sd,input, null, positionOffset, 0, ropeType, freqBase, freqScale, rotaryDims).outputVariable();
  }

  /**
   * Fused Rotary Position Embedding (RoPE).<br>
   * <br>
   * Two modes:<br>
   * 1. Precomputed cache: provide ropeCache with cos/sin values and startPosition<br>
   * 2. Dynamic position: provide scalar positionOffset tensor for KV cache decode<br>
   *    (enables DSP replay with fixed graph shapes)<br>
   * <br>
   * Supports RoPE variants: standard (LLaMA/Mistral), NeoX, GPT-J.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor [batch, seq_len, num_heads, head_dim] (NUMERIC type)
   * @param positionOffset Scalar INT64 tensor with dynamic position offset for KV cache decode (NUMERIC type)
   * @param ropeType RoPE variant: 0=standard (LLaMA), 1=NeoX, 2=GPT-J
   * @param freqBase Base frequency for RoPE computation
   * @param freqScale Frequency scale factor
   * @param rotaryDims Number of dimensions to rotate (0 = all head dims)
   * @return output Input with RoPE applied (NUMERIC type)
   */
  public SDVariable fusedRoPE(String name, SDVariable input, SDVariable positionOffset,
      int ropeType, double freqBase, double freqScale, int rotaryDims) {
    SDValidation.validateNumerical("fusedRoPE", "input", input);
    SDValidation.validateNumerical("fusedRoPE", "positionOffset", positionOffset);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE(sd,input, null, positionOffset, 0, ropeType, freqBase, freqScale, rotaryDims).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable[] gatedDeltaNetBlock(SDVariable x, SDVariable wqkv, SDVariable wbeta,
      SDVariable wgate, SDVariable wout, SDVariable convWeight, SDVariable convBias,
      SDVariable recurrentStateIn, int numHeads, int headDimK, int headDimV,
      double rmsNormEpsilon) {
    SDValidation.validateNumerical("gatedDeltaNetBlock", "x", x);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wqkv", wqkv);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wbeta", wbeta);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wgate", wgate);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wout", wout);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convWeight", convWeight);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convBias", convBias);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "recurrentStateIn", recurrentStateIn);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaNetBlock(sd,x, wqkv, wbeta, wgate, wout, convWeight, convBias, recurrentStateIn, numHeads, headDimK, headDimV, rmsNormEpsilon).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
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
  public SDVariable[] gatedDeltaNetBlock(String[] names, SDVariable x, SDVariable wqkv,
      SDVariable wbeta, SDVariable wgate, SDVariable wout, SDVariable convWeight,
      SDVariable convBias, SDVariable recurrentStateIn, int numHeads, int headDimK, int headDimV,
      double rmsNormEpsilon) {
    SDValidation.validateNumerical("gatedDeltaNetBlock", "x", x);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wqkv", wqkv);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wbeta", wbeta);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wgate", wgate);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wout", wout);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convWeight", convWeight);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convBias", convBias);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "recurrentStateIn", recurrentStateIn);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaNetBlock(sd,x, wqkv, wbeta, wgate, wout, convWeight, convBias, recurrentStateIn, numHeads, headDimK, headDimV, rmsNormEpsilon).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] gatedDeltaNetBlock(SDVariable x, SDVariable wqkv, SDVariable wbeta,
      SDVariable wgate, SDVariable wout, SDVariable convWeight, SDVariable convBias, int numHeads,
      int headDimK, int headDimV) {
    SDValidation.validateNumerical("gatedDeltaNetBlock", "x", x);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wqkv", wqkv);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wbeta", wbeta);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wgate", wgate);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wout", wout);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convWeight", convWeight);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convBias", convBias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaNetBlock(sd,x, wqkv, wbeta, wgate, wout, convWeight, convBias, null, numHeads, headDimK, headDimV, 1.0E-5).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
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
  public SDVariable[] gatedDeltaNetBlock(String[] names, SDVariable x, SDVariable wqkv,
      SDVariable wbeta, SDVariable wgate, SDVariable wout, SDVariable convWeight,
      SDVariable convBias, int numHeads, int headDimK, int headDimV) {
    SDValidation.validateNumerical("gatedDeltaNetBlock", "x", x);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wqkv", wqkv);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wbeta", wbeta);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wgate", wgate);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "wout", wout);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convWeight", convWeight);
    SDValidation.validateNumerical("gatedDeltaNetBlock", "convBias", convBias);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaNetBlock(sd,x, wqkv, wbeta, wgate, wout, convWeight, convBias, null, numHeads, headDimK, headDimV, 1.0E-5).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] gatedDeltaRule(SDVariable q, SDVariable k, SDVariable v, SDVariable beta,
      SDVariable gate, SDVariable stateIn) {
    SDValidation.validateNumerical("gatedDeltaRule", "q", q);
    SDValidation.validateNumerical("gatedDeltaRule", "k", k);
    SDValidation.validateNumerical("gatedDeltaRule", "v", v);
    SDValidation.validateNumerical("gatedDeltaRule", "beta", beta);
    SDValidation.validateNumerical("gatedDeltaRule", "gate", gate);
    SDValidation.validateNumerical("gatedDeltaRule", "stateIn", stateIn);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(sd,q, k, v, beta, gate, stateIn).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param q Query tensor [batch, seqLen, numHeads, headDimK] (NUMERIC type)
   * @param k Key tensor [batch, seqLen, numHeads, headDimK] (L2-normalized) (NUMERIC type)
   * @param v Value tensor [batch, seqLen, numHeads, headDimV] (NUMERIC type)
   * @param beta Per-step learning rate [batch, seqLen, numHeads] (NUMERIC type)
   * @param gate Decay gate (pre-exp) [batch, seqLen, numHeads] (NUMERIC type)
   * @param stateIn Previous recurrent state [batch, numHeads, headDimK, headDimV] (NUMERIC type)
   */
  public SDVariable[] gatedDeltaRule(String[] names, SDVariable q, SDVariable k, SDVariable v,
      SDVariable beta, SDVariable gate, SDVariable stateIn) {
    SDValidation.validateNumerical("gatedDeltaRule", "q", q);
    SDValidation.validateNumerical("gatedDeltaRule", "k", k);
    SDValidation.validateNumerical("gatedDeltaRule", "v", v);
    SDValidation.validateNumerical("gatedDeltaRule", "beta", beta);
    SDValidation.validateNumerical("gatedDeltaRule", "gate", gate);
    SDValidation.validateNumerical("gatedDeltaRule", "stateIn", stateIn);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(sd,q, k, v, beta, gate, stateIn).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
  public SDVariable[] gatedDeltaRule(SDVariable q, SDVariable k, SDVariable v, SDVariable beta,
      SDVariable gate) {
    SDValidation.validateNumerical("gatedDeltaRule", "q", q);
    SDValidation.validateNumerical("gatedDeltaRule", "k", k);
    SDValidation.validateNumerical("gatedDeltaRule", "v", v);
    SDValidation.validateNumerical("gatedDeltaRule", "beta", beta);
    SDValidation.validateNumerical("gatedDeltaRule", "gate", gate);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(sd,q, k, v, beta, gate, null).outputVariables();
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
   * @param names names May be null. Arrays of names for the output variables.
   * @param q Query tensor [batch, seqLen, numHeads, headDimK] (NUMERIC type)
   * @param k Key tensor [batch, seqLen, numHeads, headDimK] (L2-normalized) (NUMERIC type)
   * @param v Value tensor [batch, seqLen, numHeads, headDimV] (NUMERIC type)
   * @param beta Per-step learning rate [batch, seqLen, numHeads] (NUMERIC type)
   * @param gate Decay gate (pre-exp) [batch, seqLen, numHeads] (NUMERIC type)
   */
  public SDVariable[] gatedDeltaRule(String[] names, SDVariable q, SDVariable k, SDVariable v,
      SDVariable beta, SDVariable gate) {
    SDValidation.validateNumerical("gatedDeltaRule", "q", q);
    SDValidation.validateNumerical("gatedDeltaRule", "k", k);
    SDValidation.validateNumerical("gatedDeltaRule", "v", v);
    SDValidation.validateNumerical("gatedDeltaRule", "beta", beta);
    SDValidation.validateNumerical("gatedDeltaRule", "gate", gate);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(sd,q, k, v, beta, gate, null).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
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
   * GPU-accelerated top-K sampling for autoregressive text generation.<br>
   *
   * @param logits Logit scores (NUMERIC type)
   * @param k Number of top candidates
   * @param temperature Sampling temperature
   * @return output Sampled token indices (NUMERIC type)
   */
  public SDVariable gpuTopKSample(SDVariable logits, int k, double temperature) {
    SDValidation.validateNumerical("gpuTopKSample", "logits", logits);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopKSample(sd,logits, k, temperature).outputVariable();
  }

  /**
   * GPU-accelerated top-K sampling for autoregressive text generation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param logits Logit scores (NUMERIC type)
   * @param k Number of top candidates
   * @param temperature Sampling temperature
   * @return output Sampled token indices (NUMERIC type)
   */
  public SDVariable gpuTopKSample(String name, SDVariable logits, int k, double temperature) {
    SDValidation.validateNumerical("gpuTopKSample", "logits", logits);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopKSample(sd,logits, k, temperature).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * GPU-accelerated nucleus (top-P) sampling for autoregressive text generation.<br>
   *
   * @param logits Logit scores (NUMERIC type)
   * @param p Cumulative probability threshold (nucleus)
   * @param temperature Sampling temperature
   * @return output Sampled token indices (NUMERIC type)
   */
  public SDVariable gpuTopPSample(SDVariable logits, double p, double temperature) {
    SDValidation.validateNumerical("gpuTopPSample", "logits", logits);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopPSample(sd,logits, p, temperature).outputVariable();
  }

  /**
   * GPU-accelerated nucleus (top-P) sampling for autoregressive text generation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param logits Logit scores (NUMERIC type)
   * @param p Cumulative probability threshold (nucleus)
   * @param temperature Sampling temperature
   * @return output Sampled token indices (NUMERIC type)
   */
  public SDVariable gpuTopPSample(String name, SDVariable logits, double p, double temperature) {
    SDValidation.validateNumerical("gpuTopPSample", "logits", logits);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.GpuTopPSample(sd,logits, p, temperature).outputVariable();
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
   * Dequantizes quantized KV cache tensors back to floating point.<br>
   *
   * @param input Quantized key or value tensor (NUMERIC type)
   * @param scale Quantization scales (NUMERIC type)
   * @param quantType Quantization type
   * @return output Dequantized tensor (NUMERIC type)
   */
  public SDVariable kvCacheDequantize(SDVariable input, SDVariable scale, int quantType) {
    SDValidation.validateNumerical("kvCacheDequantize", "input", input);
    SDValidation.validateNumerical("kvCacheDequantize", "scale", scale);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize(sd,input, scale, quantType).outputVariable();
  }

  /**
   * Dequantizes quantized KV cache tensors back to floating point.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Quantized key or value tensor (NUMERIC type)
   * @param scale Quantization scales (NUMERIC type)
   * @param quantType Quantization type
   * @return output Dequantized tensor (NUMERIC type)
   */
  public SDVariable kvCacheDequantize(String name, SDVariable input, SDVariable scale,
      int quantType) {
    SDValidation.validateNumerical("kvCacheDequantize", "input", input);
    SDValidation.validateNumerical("kvCacheDequantize", "scale", scale);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheDequantize(sd,input, scale, quantType).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Quantizes KV cache tensors for memory-efficient inference.<br>
   *
   * @param input Key or value tensor to quantize (NUMERIC type)
   * @param quantType Quantization type
   * @param groupSize Group size for quantization
   * @return output Quantized tensor (NUMERIC type)
   */
  public SDVariable kvCacheQuantize(SDVariable input, int quantType, int groupSize) {
    SDValidation.validateNumerical("kvCacheQuantize", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize(sd,input, quantType, groupSize).outputVariable();
  }

  /**
   * Quantizes KV cache tensors for memory-efficient inference.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Key or value tensor to quantize (NUMERIC type)
   * @param quantType Quantization type
   * @param groupSize Group size for quantization
   * @return output Quantized tensor (NUMERIC type)
   */
  public SDVariable kvCacheQuantize(String name, SDVariable input, int quantType, int groupSize) {
    SDValidation.validateNumerical("kvCacheQuantize", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.KVCacheQuantize(sd,input, quantType, groupSize).outputVariable();
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
   * Creates a contiguous copy of the input tensor with linear (row-major) memory layout.<br>
   *
   * @param input Source tensor (NUMERIC type)
   * @return output Contiguous copy of input (NUMERIC type)
   */
  public SDVariable linearCopy(SDVariable input) {
    SDValidation.validateNumerical("linearCopy", "input", input);
    return new org.nd4j.linalg.api.ops.custom.LinearCopy(sd,input).outputVariable();
  }

  /**
   * Creates a contiguous copy of the input tensor with linear (row-major) memory layout.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Source tensor (NUMERIC type)
   * @return output Contiguous copy of input (NUMERIC type)
   */
  public SDVariable linearCopy(String name, SDVariable input) {
    SDValidation.validateNumerical("linearCopy", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.custom.LinearCopy(sd,input).outputVariable();
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
  public SDVariable lohaMatMul(SDVariable input, SDVariable weight, SDVariable lohaA1,
      SDVariable lohaB1, SDVariable lohaA2, SDVariable lohaB2, double scaling,
      boolean transposeWeight) {
    SDValidation.validateNumerical("lohaMatMul", "input", input);
    SDValidation.validateNumerical("lohaMatMul", "weight", weight);
    SDValidation.validateNumerical("lohaMatMul", "lohaA1", lohaA1);
    SDValidation.validateNumerical("lohaMatMul", "lohaB1", lohaB1);
    SDValidation.validateNumerical("lohaMatMul", "lohaA2", lohaA2);
    SDValidation.validateNumerical("lohaMatMul", "lohaB2", lohaB2);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.LohaMatMul(sd,input, weight, lohaA1, lohaB1, lohaA2, lohaB2, scaling, transposeWeight).outputVariable();
  }

  /**
   * Low-Rank Hadamard Product (LoHa) fused matrix multiplication.<br>
   * Uses Hadamard product of two low-rank matrices as the adapter.<br>
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable lohaMatMul(String name, SDVariable input, SDVariable weight, SDVariable lohaA1,
      SDVariable lohaB1, SDVariable lohaA2, SDVariable lohaB2, double scaling,
      boolean transposeWeight) {
    SDValidation.validateNumerical("lohaMatMul", "input", input);
    SDValidation.validateNumerical("lohaMatMul", "weight", weight);
    SDValidation.validateNumerical("lohaMatMul", "lohaA1", lohaA1);
    SDValidation.validateNumerical("lohaMatMul", "lohaB1", lohaB1);
    SDValidation.validateNumerical("lohaMatMul", "lohaA2", lohaA2);
    SDValidation.validateNumerical("lohaMatMul", "lohaB2", lohaB2);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LohaMatMul(sd,input, weight, lohaA1, lohaB1, lohaA2, lohaB2, scaling, transposeWeight).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable lokrMatMul(SDVariable input, SDVariable weight, SDVariable lokrC,
      SDVariable lokrA, SDVariable lokrB, int factor1, int factor2, double scaling,
      boolean transposeWeight) {
    SDValidation.validateNumerical("lokrMatMul", "input", input);
    SDValidation.validateNumerical("lokrMatMul", "weight", weight);
    SDValidation.validateNumerical("lokrMatMul", "lokrC", lokrC);
    SDValidation.validateNumerical("lokrMatMul", "lokrA", lokrA);
    SDValidation.validateNumerical("lokrMatMul", "lokrB", lokrB);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.LokrMatMul(sd,input, weight, lokrC, lokrA, lokrB, factor1, factor2, scaling, transposeWeight).outputVariable();
  }

  /**
   * Low-Rank Kronecker Product (LoKr) fused matrix multiplication.<br>
   * Uses Kronecker product of matrices as the adapter.<br>
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable lokrMatMul(String name, SDVariable input, SDVariable weight, SDVariable lokrC,
      SDVariable lokrA, SDVariable lokrB, int factor1, int factor2, double scaling,
      boolean transposeWeight) {
    SDValidation.validateNumerical("lokrMatMul", "input", input);
    SDValidation.validateNumerical("lokrMatMul", "weight", weight);
    SDValidation.validateNumerical("lokrMatMul", "lokrC", lokrC);
    SDValidation.validateNumerical("lokrMatMul", "lokrA", lokrA);
    SDValidation.validateNumerical("lokrMatMul", "lokrB", lokrB);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LokrMatMul(sd,input, weight, lokrC, lokrA, lokrB, factor1, factor2, scaling, transposeWeight).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable loraMatMul(SDVariable input, SDVariable weight, SDVariable loraA,
      SDVariable loraB, double scaling, boolean transposeWeight) {
    SDValidation.validateNumerical("loraMatMul", "input", input);
    SDValidation.validateNumerical("loraMatMul", "weight", weight);
    SDValidation.validateNumerical("loraMatMul", "loraA", loraA);
    SDValidation.validateNumerical("loraMatMul", "loraB", loraB);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.LoraMatMul(sd,input, weight, loraA, loraB, scaling, transposeWeight).outputVariable();
  }

  /**
   * Low-Rank Adaptation (LoRA) fused matrix multiplication.<br>
   * Computes base weight matmul + low-rank adapter in a single operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input [batch, in_features] (NUMERIC type)
   * @param weight Base weight [out_features, in_features] (NUMERIC type)
   * @param loraA LoRA down-projection [r, in_features] (NUMERIC type)
   * @param loraB LoRA up-projection [out_features, r] (NUMERIC type)
   * @param scaling LoRA scaling factor (default 1.0)
   * @param transposeWeight Whether to transpose weight (default true)
   * @return output Result: input @ weight^T + scaling * input @ loraA^T @ loraB^T (NUMERIC type)
   */
  public SDVariable loraMatMul(String name, SDVariable input, SDVariable weight, SDVariable loraA,
      SDVariable loraB, double scaling, boolean transposeWeight) {
    SDValidation.validateNumerical("loraMatMul", "input", input);
    SDValidation.validateNumerical("loraMatMul", "weight", weight);
    SDValidation.validateNumerical("loraMatMul", "loraA", loraA);
    SDValidation.validateNumerical("loraMatMul", "loraB", loraB);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.LoraMatMul(sd,input, weight, loraA, loraB, scaling, transposeWeight).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Mamba-2 State Space Model (SSD) - head-structured recurrence.<br>
   * <br>
   * Implements the Mamba-2 SSD recurrence with scalar per-head decay:<br>
   *   h_t = exp(A * dt) * h_{t-1} + (B * dt) outer x_t<br>
   *   y_t = C * h_t + D * x_t<br>
   * <br>
   * Unlike Mamba-1 (selective_scan) which uses per-element diagonal state,<br>
   * Mamba-2 uses head-structured state for improved hardware utilization.<br>
   * <br>
   * See "Transformers are SSMs: Generalized Models and Efficient Algorithms Through<br>
   * Structured State Space Duality" (https://arxiv.org/abs/2405.21060)<br>
   *
   * @param x Input tensor [batch, seqLen, D] where D = numHeads * headDim (NUMERIC type)
   * @param A Per-head scalar decay in log-space [numHeads] (NUMERIC type)
   * @param B Input-dependent state expansion [batch, seqLen, stateDim] (NUMERIC type)
   * @param C Input-dependent state contraction [batch, seqLen, stateDim] (NUMERIC type)
   * @param dt Discretization timestep (post-softplus) [batch, seqLen, numHeads] (NUMERIC type)
   * @param numHeads Number of SSM heads (H)
   * @param headDim Dimension per head (P = D/H)
   * @param stateDim State dimension (N)
   */
  public SDVariable[] mamba2Ssm(SDVariable x, SDVariable A, SDVariable B, SDVariable C,
      SDVariable dt, int numHeads, int headDim, int stateDim) {
    SDValidation.validateNumerical("mamba2Ssm", "x", x);
    SDValidation.validateNumerical("mamba2Ssm", "A", A);
    SDValidation.validateNumerical("mamba2Ssm", "B", B);
    SDValidation.validateNumerical("mamba2Ssm", "C", C);
    SDValidation.validateNumerical("mamba2Ssm", "dt", dt);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Mamba2SSM(sd,x, A, B, C, dt, numHeads, headDim, stateDim).outputVariables();
  }

  /**
   * Mamba-2 State Space Model (SSD) - head-structured recurrence.<br>
   * <br>
   * Implements the Mamba-2 SSD recurrence with scalar per-head decay:<br>
   *   h_t = exp(A * dt) * h_{t-1} + (B * dt) outer x_t<br>
   *   y_t = C * h_t + D * x_t<br>
   * <br>
   * Unlike Mamba-1 (selective_scan) which uses per-element diagonal state,<br>
   * Mamba-2 uses head-structured state for improved hardware utilization.<br>
   * <br>
   * See "Transformers are SSMs: Generalized Models and Efficient Algorithms Through<br>
   * Structured State Space Duality" (https://arxiv.org/abs/2405.21060)<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param x Input tensor [batch, seqLen, D] where D = numHeads * headDim (NUMERIC type)
   * @param A Per-head scalar decay in log-space [numHeads] (NUMERIC type)
   * @param B Input-dependent state expansion [batch, seqLen, stateDim] (NUMERIC type)
   * @param C Input-dependent state contraction [batch, seqLen, stateDim] (NUMERIC type)
   * @param dt Discretization timestep (post-softplus) [batch, seqLen, numHeads] (NUMERIC type)
   * @param numHeads Number of SSM heads (H)
   * @param headDim Dimension per head (P = D/H)
   * @param stateDim State dimension (N)
   */
  public SDVariable[] mamba2Ssm(String[] names, SDVariable x, SDVariable A, SDVariable B,
      SDVariable C, SDVariable dt, int numHeads, int headDim, int stateDim) {
    SDValidation.validateNumerical("mamba2Ssm", "x", x);
    SDValidation.validateNumerical("mamba2Ssm", "A", A);
    SDValidation.validateNumerical("mamba2Ssm", "B", B);
    SDValidation.validateNumerical("mamba2Ssm", "C", C);
    SDValidation.validateNumerical("mamba2Ssm", "dt", dt);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Mamba2SSM(sd,x, A, B, C, dt, numHeads, headDim, stateDim).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Computes the mean of squared values. Used in RMSNorm and similar operations.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output Mean of squared values (NUMERIC type)
   */
  public SDVariable meanSquare(SDVariable input) {
    SDValidation.validateNumerical("meanSquare", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MeanSquare(sd,input).outputVariable();
  }

  /**
   * Computes the mean of squared values. Used in RMSNorm and similar operations.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @return output Mean of squared values (NUMERIC type)
   */
  public SDVariable meanSquare(String name, SDVariable input) {
    SDValidation.validateNumerical("meanSquare", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MeanSquare(sd,input).outputVariable();
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
   * Multi-head Latent Attention (MLA) from DeepSeek-V2.<br>
   * Uses low-rank KV compression for efficient long-context inference.<br>
   *
   * @param input Input hidden states (NUMERIC type)
   * @param kvDownProj KV down-projection weight (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param latentDim Latent dimension for compressed KV
   * @return output Attention output (NUMERIC type)
   */
  public SDVariable mlaAttention(SDVariable input, SDVariable kvDownProj, int numHeads,
      int latentDim) {
    SDValidation.validateNumerical("mlaAttention", "input", input);
    SDValidation.validateNumerical("mlaAttention", "kvDownProj", kvDownProj);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MLAAttention(sd,input, kvDownProj, numHeads, latentDim).outputVariable();
  }

  /**
   * Multi-head Latent Attention (MLA) from DeepSeek-V2.<br>
   * Uses low-rank KV compression for efficient long-context inference.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input hidden states (NUMERIC type)
   * @param kvDownProj KV down-projection weight (NUMERIC type)
   * @param numHeads Number of attention heads
   * @param latentDim Latent dimension for compressed KV
   * @return output Attention output (NUMERIC type)
   */
  public SDVariable mlaAttention(String name, SDVariable input, SDVariable kvDownProj, int numHeads,
      int latentDim) {
    SDValidation.validateNumerical("mlaAttention", "input", input);
    SDValidation.validateNumerical("mlaAttention", "kvDownProj", kvDownProj);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MLAAttention(sd,input, kvDownProj, numHeads, latentDim).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable moeGate(SDVariable input, SDVariable gateWeights, int numExperts, int topK) {
    SDValidation.validateNumerical("moeGate", "input", input);
    SDValidation.validateNumerical("moeGate", "gateWeights", gateWeights);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MoeGate(sd,input, gateWeights, numExperts, topK).outputVariable();
  }

  /**
   * Mixture of Experts (MoE) gating/routing function.<br>
   * Selects top-K experts and computes routing weights.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input hidden states (NUMERIC type)
   * @param gateWeights Router gate weights (NUMERIC type)
   * @param numExperts Number of experts
   * @param topK Top-K experts to select
   * @return output Gating weights and expert indices (NUMERIC type)
   */
  public SDVariable moeGate(String name, SDVariable input, SDVariable gateWeights, int numExperts,
      int topK) {
    SDValidation.validateNumerical("moeGate", "input", input);
    SDValidation.validateNumerical("moeGate", "gateWeights", gateWeights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MoeGate(sd,input, gateWeights, numExperts, topK).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Mixture of Experts with Shared Experts (IBM Granite 4.0 pattern).<br>
   * <br>
   * Extends MoE with an always-on shared expert pathway. The shared expert<br>
   * processes every token unconditionally using SwiGLU activation, while<br>
   * routed experts are selected via top-K gating.<br>
   * <br>
   * output = shared_expert(input) + weighted_sum(routed_experts(input))<br>
   * <br>
   * where shared_expert uses SwiGLU:<br>
   * shared_out = down_proj(silu(gate_proj(x)) * up_proj(x))<br>
   * <br>
   * Used in:<br>
   * - IBM Granite 4.0 (granitemoeshared architecture)<br>
   * - DeepSeek V2/V3 (shared expert variant)<br>
   *
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numRoutedExperts] (NUMERIC type)
   * @param routedExpertWeights Routed expert weight matrices. Shape: [numRoutedExperts, hiddenSize, expertHidden] (NUMERIC type)
   * @param sharedGateProj Shared expert gate projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedUpProj Shared expert up projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedDownProj Shared expert down projection. Shape: [sharedIntermediateSize, hiddenSize] (NUMERIC type)
   * @param numRoutedExperts Number of routed experts
   * @param topK Number of experts to route to per token
   */
  public SDVariable[] moeSharedExperts(SDVariable input, SDVariable routerWeights,
      SDVariable routedExpertWeights, SDVariable sharedGateProj, SDVariable sharedUpProj,
      SDVariable sharedDownProj, int numRoutedExperts, int topK) {
    SDValidation.validateNumerical("moeSharedExperts", "input", input);
    SDValidation.validateNumerical("moeSharedExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("moeSharedExperts", "routedExpertWeights", routedExpertWeights);
    SDValidation.validateNumerical("moeSharedExperts", "sharedGateProj", sharedGateProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedUpProj", sharedUpProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedDownProj", sharedDownProj);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MoeSharedExperts(sd,input, routerWeights, routedExpertWeights, sharedGateProj, sharedUpProj, sharedDownProj, null, numRoutedExperts, topK, true, 1.0).outputVariables();
  }

  /**
   * Mixture of Experts with Shared Experts (IBM Granite 4.0 pattern).<br>
   * <br>
   * Extends MoE with an always-on shared expert pathway. The shared expert<br>
   * processes every token unconditionally using SwiGLU activation, while<br>
   * routed experts are selected via top-K gating.<br>
   * <br>
   * output = shared_expert(input) + weighted_sum(routed_experts(input))<br>
   * <br>
   * where shared_expert uses SwiGLU:<br>
   * shared_out = down_proj(silu(gate_proj(x)) * up_proj(x))<br>
   * <br>
   * Used in:<br>
   * - IBM Granite 4.0 (granitemoeshared architecture)<br>
   * - DeepSeek V2/V3 (shared expert variant)<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numRoutedExperts] (NUMERIC type)
   * @param routedExpertWeights Routed expert weight matrices. Shape: [numRoutedExperts, hiddenSize, expertHidden] (NUMERIC type)
   * @param sharedGateProj Shared expert gate projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedUpProj Shared expert up projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedDownProj Shared expert down projection. Shape: [sharedIntermediateSize, hiddenSize] (NUMERIC type)
   * @param numRoutedExperts Number of routed experts
   * @param topK Number of experts to route to per token
   */
  public SDVariable[] moeSharedExperts(String[] names, SDVariable input, SDVariable routerWeights,
      SDVariable routedExpertWeights, SDVariable sharedGateProj, SDVariable sharedUpProj,
      SDVariable sharedDownProj, int numRoutedExperts, int topK) {
    SDValidation.validateNumerical("moeSharedExperts", "input", input);
    SDValidation.validateNumerical("moeSharedExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("moeSharedExperts", "routedExpertWeights", routedExpertWeights);
    SDValidation.validateNumerical("moeSharedExperts", "sharedGateProj", sharedGateProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedUpProj", sharedUpProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedDownProj", sharedDownProj);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MoeSharedExperts(sd,input, routerWeights, routedExpertWeights, sharedGateProj, sharedUpProj, sharedDownProj, null, numRoutedExperts, topK, true, 1.0).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * Mixture of Experts with Shared Experts (IBM Granite 4.0 pattern).<br>
   * <br>
   * Extends MoE with an always-on shared expert pathway. The shared expert<br>
   * processes every token unconditionally using SwiGLU activation, while<br>
   * routed experts are selected via top-K gating.<br>
   * <br>
   * output = shared_expert(input) + weighted_sum(routed_experts(input))<br>
   * <br>
   * where shared_expert uses SwiGLU:<br>
   * shared_out = down_proj(silu(gate_proj(x)) * up_proj(x))<br>
   * <br>
   * Used in:<br>
   * - IBM Granite 4.0 (granitemoeshared architecture)<br>
   * - DeepSeek V2/V3 (shared expert variant)<br>
   *
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numRoutedExperts] (NUMERIC type)
   * @param routedExpertWeights Routed expert weight matrices. Shape: [numRoutedExperts, hiddenSize, expertHidden] (NUMERIC type)
   * @param sharedGateProj Shared expert gate projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedUpProj Shared expert up projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedDownProj Shared expert down projection. Shape: [sharedIntermediateSize, hiddenSize] (NUMERIC type)
   * @param routedExpertBias Optional routed expert biases (NUMERIC type)
   * @param numRoutedExperts Number of routed experts
   * @param topK Number of experts to route to per token
   * @param normalizeProbs Whether to normalize router probabilities for selected experts
   * @param capacityFactor Expert capacity factor for load balancing
   */
  public SDVariable[] moeSharedExperts(SDVariable input, SDVariable routerWeights,
      SDVariable routedExpertWeights, SDVariable sharedGateProj, SDVariable sharedUpProj,
      SDVariable sharedDownProj, SDVariable routedExpertBias, int numRoutedExperts, int topK,
      boolean normalizeProbs, double capacityFactor) {
    SDValidation.validateNumerical("moeSharedExperts", "input", input);
    SDValidation.validateNumerical("moeSharedExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("moeSharedExperts", "routedExpertWeights", routedExpertWeights);
    SDValidation.validateNumerical("moeSharedExperts", "sharedGateProj", sharedGateProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedUpProj", sharedUpProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedDownProj", sharedDownProj);
    SDValidation.validateNumerical("moeSharedExperts", "routedExpertBias", routedExpertBias);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MoeSharedExperts(sd,input, routerWeights, routedExpertWeights, sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias, numRoutedExperts, topK, normalizeProbs, capacityFactor).outputVariables();
  }

  /**
   * Mixture of Experts with Shared Experts (IBM Granite 4.0 pattern).<br>
   * <br>
   * Extends MoE with an always-on shared expert pathway. The shared expert<br>
   * processes every token unconditionally using SwiGLU activation, while<br>
   * routed experts are selected via top-K gating.<br>
   * <br>
   * output = shared_expert(input) + weighted_sum(routed_experts(input))<br>
   * <br>
   * where shared_expert uses SwiGLU:<br>
   * shared_out = down_proj(silu(gate_proj(x)) * up_proj(x))<br>
   * <br>
   * Used in:<br>
   * - IBM Granite 4.0 (granitemoeshared architecture)<br>
   * - DeepSeek V2/V3 (shared expert variant)<br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input Input embeddings. Shape: [batch, seqLen, hiddenSize] (NUMERIC type)
   * @param routerWeights Router projection weights. Shape: [hiddenSize, numRoutedExperts] (NUMERIC type)
   * @param routedExpertWeights Routed expert weight matrices. Shape: [numRoutedExperts, hiddenSize, expertHidden] (NUMERIC type)
   * @param sharedGateProj Shared expert gate projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedUpProj Shared expert up projection. Shape: [hiddenSize, sharedIntermediateSize] (NUMERIC type)
   * @param sharedDownProj Shared expert down projection. Shape: [sharedIntermediateSize, hiddenSize] (NUMERIC type)
   * @param routedExpertBias Optional routed expert biases (NUMERIC type)
   * @param numRoutedExperts Number of routed experts
   * @param topK Number of experts to route to per token
   * @param normalizeProbs Whether to normalize router probabilities for selected experts
   * @param capacityFactor Expert capacity factor for load balancing
   */
  public SDVariable[] moeSharedExperts(String[] names, SDVariable input, SDVariable routerWeights,
      SDVariable routedExpertWeights, SDVariable sharedGateProj, SDVariable sharedUpProj,
      SDVariable sharedDownProj, SDVariable routedExpertBias, int numRoutedExperts, int topK,
      boolean normalizeProbs, double capacityFactor) {
    SDValidation.validateNumerical("moeSharedExperts", "input", input);
    SDValidation.validateNumerical("moeSharedExperts", "routerWeights", routerWeights);
    SDValidation.validateNumerical("moeSharedExperts", "routedExpertWeights", routedExpertWeights);
    SDValidation.validateNumerical("moeSharedExperts", "sharedGateProj", sharedGateProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedUpProj", sharedUpProj);
    SDValidation.validateNumerical("moeSharedExperts", "sharedDownProj", sharedDownProj);
    SDValidation.validateNumerical("moeSharedExperts", "routedExpertBias", routedExpertBias);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MoeSharedExperts(sd,input, routerWeights, routedExpertWeights, sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias, numRoutedExperts, topK, normalizeProbs, capacityFactor).outputVariables();
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
  public SDVariable multiLoraMatmul(SDVariable input, SDVariable baseWeight,
      SDVariable loraAWeights, SDVariable loraBWeights, SDVariable adapterIds, double scaling) {
    SDValidation.validateNumerical("multiLoraMatmul", "input", input);
    SDValidation.validateNumerical("multiLoraMatmul", "baseWeight", baseWeight);
    SDValidation.validateNumerical("multiLoraMatmul", "loraAWeights", loraAWeights);
    SDValidation.validateNumerical("multiLoraMatmul", "loraBWeights", loraBWeights);
    SDValidation.validateNumerical("multiLoraMatmul", "adapterIds", adapterIds);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.MultiLoraMatmul(sd,input, baseWeight, loraAWeights, loraBWeights, adapterIds, scaling).outputVariable();
  }

  /**
   * Multi-adapter LoRA matrix multiplication. Selects different LoRA adapters per batch element.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param baseWeight Base weight matrix (NUMERIC type)
   * @param loraAWeights Stacked LoRA A weights (NUMERIC type)
   * @param loraBWeights Stacked LoRA B weights (NUMERIC type)
   * @param adapterIds Adapter selection indices (NUMERIC type)
   * @param scaling LoRA scaling factor
   * @return output Result with per-sample adapter selection (NUMERIC type)
   */
  public SDVariable multiLoraMatmul(String name, SDVariable input, SDVariable baseWeight,
      SDVariable loraAWeights, SDVariable loraBWeights, SDVariable adapterIds, double scaling) {
    SDValidation.validateNumerical("multiLoraMatmul", "input", input);
    SDValidation.validateNumerical("multiLoraMatmul", "baseWeight", baseWeight);
    SDValidation.validateNumerical("multiLoraMatmul", "loraAWeights", loraAWeights);
    SDValidation.validateNumerical("multiLoraMatmul", "loraBWeights", loraBWeights);
    SDValidation.validateNumerical("multiLoraMatmul", "adapterIds", adapterIds);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.MultiLoraMatmul(sd,input, baseWeight, loraAWeights, loraBWeights, adapterIds, scaling).outputVariable();
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
   * Per-Layer Embedding (Gemma 4).<br>
   * <br>
   * Adds a per-layer residual from a second embedding table to the hidden states:<br>
   *   output = hiddenStates + pleWeight[tokenIds] * scale<br>
   * <br>
   * Each decoder layer receives a small additive signal from a dedicated<br>
   * embedding table indexed by the original token IDs. This is computed once<br>
   * before multimodal features merge into the embedding sequence, since PLE<br>
   * relies on token IDs that are lost once multimodal features replace placeholders.<br>
   *
   * @param hiddenStates Hidden states [batch, seqLen, hiddenDim] (NUMERIC type)
   * @param pleWeight Per-layer embedding table [vocabSize, hiddenDim] (NUMERIC type)
   * @param tokenIds Token IDs [batch, seqLen] (INT type)
   * @param scale Scale factor for the embedding addition
   * @return output Output [batch, seqLen, hiddenDim] (NUMERIC type)
   */
  public SDVariable perLayerEmbedding(SDVariable hiddenStates, SDVariable pleWeight,
      SDVariable tokenIds, double scale) {
    SDValidation.validateNumerical("perLayerEmbedding", "hiddenStates", hiddenStates);
    SDValidation.validateNumerical("perLayerEmbedding", "pleWeight", pleWeight);
    SDValidation.validateInteger("perLayerEmbedding", "tokenIds", tokenIds);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.PerLayerEmbedding(sd,hiddenStates, pleWeight, tokenIds, scale).outputVariable();
  }

  /**
   * Per-Layer Embedding (Gemma 4).<br>
   * <br>
   * Adds a per-layer residual from a second embedding table to the hidden states:<br>
   *   output = hiddenStates + pleWeight[tokenIds] * scale<br>
   * <br>
   * Each decoder layer receives a small additive signal from a dedicated<br>
   * embedding table indexed by the original token IDs. This is computed once<br>
   * before multimodal features merge into the embedding sequence, since PLE<br>
   * relies on token IDs that are lost once multimodal features replace placeholders.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param hiddenStates Hidden states [batch, seqLen, hiddenDim] (NUMERIC type)
   * @param pleWeight Per-layer embedding table [vocabSize, hiddenDim] (NUMERIC type)
   * @param tokenIds Token IDs [batch, seqLen] (INT type)
   * @param scale Scale factor for the embedding addition
   * @return output Output [batch, seqLen, hiddenDim] (NUMERIC type)
   */
  public SDVariable perLayerEmbedding(String name, SDVariable hiddenStates, SDVariable pleWeight,
      SDVariable tokenIds, double scale) {
    SDValidation.validateNumerical("perLayerEmbedding", "hiddenStates", hiddenStates);
    SDValidation.validateNumerical("perLayerEmbedding", "pleWeight", pleWeight);
    SDValidation.validateInteger("perLayerEmbedding", "tokenIds", tokenIds);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.PerLayerEmbedding(sd,hiddenStates, pleWeight, tokenIds, scale).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Per-Layer Embedding (Gemma 4).<br>
   * <br>
   * Adds a per-layer residual from a second embedding table to the hidden states:<br>
   *   output = hiddenStates + pleWeight[tokenIds] * scale<br>
   * <br>
   * Each decoder layer receives a small additive signal from a dedicated<br>
   * embedding table indexed by the original token IDs. This is computed once<br>
   * before multimodal features merge into the embedding sequence, since PLE<br>
   * relies on token IDs that are lost once multimodal features replace placeholders.<br>
   *
   * @param hiddenStates Hidden states [batch, seqLen, hiddenDim] (NUMERIC type)
   * @param pleWeight Per-layer embedding table [vocabSize, hiddenDim] (NUMERIC type)
   * @param tokenIds Token IDs [batch, seqLen] (INT type)
   * @return output Output [batch, seqLen, hiddenDim] (NUMERIC type)
   */
  public SDVariable perLayerEmbedding(SDVariable hiddenStates, SDVariable pleWeight,
      SDVariable tokenIds) {
    SDValidation.validateNumerical("perLayerEmbedding", "hiddenStates", hiddenStates);
    SDValidation.validateNumerical("perLayerEmbedding", "pleWeight", pleWeight);
    SDValidation.validateInteger("perLayerEmbedding", "tokenIds", tokenIds);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.PerLayerEmbedding(sd,hiddenStates, pleWeight, tokenIds, 1.0).outputVariable();
  }

  /**
   * Per-Layer Embedding (Gemma 4).<br>
   * <br>
   * Adds a per-layer residual from a second embedding table to the hidden states:<br>
   *   output = hiddenStates + pleWeight[tokenIds] * scale<br>
   * <br>
   * Each decoder layer receives a small additive signal from a dedicated<br>
   * embedding table indexed by the original token IDs. This is computed once<br>
   * before multimodal features merge into the embedding sequence, since PLE<br>
   * relies on token IDs that are lost once multimodal features replace placeholders.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param hiddenStates Hidden states [batch, seqLen, hiddenDim] (NUMERIC type)
   * @param pleWeight Per-layer embedding table [vocabSize, hiddenDim] (NUMERIC type)
   * @param tokenIds Token IDs [batch, seqLen] (INT type)
   * @return output Output [batch, seqLen, hiddenDim] (NUMERIC type)
   */
  public SDVariable perLayerEmbedding(String name, SDVariable hiddenStates, SDVariable pleWeight,
      SDVariable tokenIds) {
    SDValidation.validateNumerical("perLayerEmbedding", "hiddenStates", hiddenStates);
    SDValidation.validateNumerical("perLayerEmbedding", "pleWeight", pleWeight);
    SDValidation.validateInteger("perLayerEmbedding", "tokenIds", tokenIds);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.PerLayerEmbedding(sd,hiddenStates, pleWeight, tokenIds, 1.0).outputVariable();
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
   * Quantized matrix multiplication. Supports mixed precision (float/int) inputs.<br>
   *
   * @param a First matrix (NUMERIC type)
   * @param b Second matrix (NUMERIC type)
   * @return output Matrix product (NUMERIC type)
   */
  public SDVariable quantizedMatmul(SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("quantizedMatmul", "a", a);
    SDValidation.validateNumerical("quantizedMatmul", "b", b);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.QuantizedMatmul(sd,a, b).outputVariable();
  }

  /**
   * Quantized matrix multiplication. Supports mixed precision (float/int) inputs.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param a First matrix (NUMERIC type)
   * @param b Second matrix (NUMERIC type)
   * @return output Matrix product (NUMERIC type)
   */
  public SDVariable quantizedMatmul(String name, SDVariable a, SDVariable b) {
    SDValidation.validateNumerical("quantizedMatmul", "a", a);
    SDValidation.validateNumerical("quantizedMatmul", "b", b);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.QuantizedMatmul(sd,a, b).outputVariable();
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
   * Reshapes a tensor without copying data. Returns a view if possible.<br>
   * If the reshape cannot be done without copying, this op will fail.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param shape Target shape (NUMERIC type)
   * @return output Reshaped view (no data copy) (NUMERIC type)
   */
  public SDVariable reshapeNoCopy(SDVariable input, SDVariable shape) {
    SDValidation.validateNumerical("reshapeNoCopy", "input", input);
    SDValidation.validateNumerical("reshapeNoCopy", "shape", shape);
    return new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(sd,input, shape).outputVariable();
  }

  /**
   * Reshapes a tensor without copying data. Returns a view if possible.<br>
   * If the reshape cannot be done without copying, this op will fail.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param shape Target shape (NUMERIC type)
   * @return output Reshaped view (no data copy) (NUMERIC type)
   */
  public SDVariable reshapeNoCopy(String name, SDVariable input, SDVariable shape) {
    SDValidation.validateNumerical("reshapeNoCopy", "input", input);
    SDValidation.validateNumerical("reshapeNoCopy", "shape", shape);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.shape.ReshapeNoCopy(sd,input, shape).outputVariable();
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
   * Fused RMSNorm + Linear (MatMul) operation:<br>
   * Computes matmul(rms_norm(x, gamma, eps), W) without materializing the intermediate<br>
   * normalized tensor. Common in transformer models where RMSNorm feeds directly into<br>
   * Q/K/V projections or FFN layers.<br>
   *
   * @param input Input variable [batch, ..., features] (NUMERIC type)
   * @param gamma RMSNorm scale weights [features] (NUMERIC type)
   * @param weights Weight matrix [features, outFeatures] (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output Result of rms_norm(input, gamma, eps) @ weights (NUMERIC type)
   */
  public SDVariable rmsNormLinear(SDVariable input, SDVariable gamma, SDVariable weights,
      double epsilon) {
    SDValidation.validateNumerical("rmsNormLinear", "input", input);
    SDValidation.validateNumerical("rmsNormLinear", "gamma", gamma);
    SDValidation.validateNumerical("rmsNormLinear", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormLinear(sd,input, gamma, weights, epsilon).outputVariable();
  }

  /**
   * Fused RMSNorm + Linear (MatMul) operation:<br>
   * Computes matmul(rms_norm(x, gamma, eps), W) without materializing the intermediate<br>
   * normalized tensor. Common in transformer models where RMSNorm feeds directly into<br>
   * Q/K/V projections or FFN layers.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable [batch, ..., features] (NUMERIC type)
   * @param gamma RMSNorm scale weights [features] (NUMERIC type)
   * @param weights Weight matrix [features, outFeatures] (NUMERIC type)
   * @param epsilon Epsilon for numerical stability
   * @return output Result of rms_norm(input, gamma, eps) @ weights (NUMERIC type)
   */
  public SDVariable rmsNormLinear(String name, SDVariable input, SDVariable gamma,
      SDVariable weights, double epsilon) {
    SDValidation.validateNumerical("rmsNormLinear", "input", input);
    SDValidation.validateNumerical("rmsNormLinear", "gamma", gamma);
    SDValidation.validateNumerical("rmsNormLinear", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormLinear(sd,input, gamma, weights, epsilon).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Fused RMSNorm + Linear (MatMul) operation:<br>
   * Computes matmul(rms_norm(x, gamma, eps), W) without materializing the intermediate<br>
   * normalized tensor. Common in transformer models where RMSNorm feeds directly into<br>
   * Q/K/V projections or FFN layers.<br>
   *
   * @param input Input variable [batch, ..., features] (NUMERIC type)
   * @param gamma RMSNorm scale weights [features] (NUMERIC type)
   * @param weights Weight matrix [features, outFeatures] (NUMERIC type)
   * @return output Result of rms_norm(input, gamma, eps) @ weights (NUMERIC type)
   */
  public SDVariable rmsNormLinear(SDVariable input, SDVariable gamma, SDVariable weights) {
    SDValidation.validateNumerical("rmsNormLinear", "input", input);
    SDValidation.validateNumerical("rmsNormLinear", "gamma", gamma);
    SDValidation.validateNumerical("rmsNormLinear", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormLinear(sd,input, gamma, weights, 1.0E-6).outputVariable();
  }

  /**
   * Fused RMSNorm + Linear (MatMul) operation:<br>
   * Computes matmul(rms_norm(x, gamma, eps), W) without materializing the intermediate<br>
   * normalized tensor. Common in transformer models where RMSNorm feeds directly into<br>
   * Q/K/V projections or FFN layers.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input variable [batch, ..., features] (NUMERIC type)
   * @param gamma RMSNorm scale weights [features] (NUMERIC type)
   * @param weights Weight matrix [features, outFeatures] (NUMERIC type)
   * @return output Result of rms_norm(input, gamma, eps) @ weights (NUMERIC type)
   */
  public SDVariable rmsNormLinear(String name, SDVariable input, SDVariable gamma,
      SDVariable weights) {
    SDValidation.validateNumerical("rmsNormLinear", "input", input);
    SDValidation.validateNumerical("rmsNormLinear", "gamma", gamma);
    SDValidation.validateNumerical("rmsNormLinear", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormLinear(sd,input, gamma, weights, 1.0E-6).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rope(SDVariable input, int mode, int nPast, int nDims, double freqBase,
      double freqScale) {
    SDValidation.validateNumerical("rope", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RoPE(sd,input, mode, nPast, nDims, freqBase, freqScale).outputVariable();
  }

  /**
   * Applies Rotary Position Embedding (RoPE) to the input tensor.<br>
   * Encodes position information by rotating pairs of dimensions in the input.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor [batch, seq_len, num_heads, head_dim] (NUMERIC type)
   * @param mode RoPE mode (default 0)
   * @param nPast Number of past tokens (default 0)
   * @param nDims Dimension subset for rotation (default last dim)
   * @param freqBase Frequency base (default 10000.0)
   * @param freqScale Frequency scale (default 1.0)
   * @return output Output with rotary position embeddings applied (NUMERIC type)
   */
  public SDVariable rope(String name, SDVariable input, int mode, int nPast, int nDims,
      double freqBase, double freqScale) {
    SDValidation.validateNumerical("rope", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RoPE(sd,input, mode, nPast, nDims, freqBase, freqScale).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable rowParallelLinear(SDVariable input, SDVariable weight, int tpRank, int tpSize,
      boolean reduceOutput) {
    SDValidation.validateNumerical("rowParallelLinear", "input", input);
    SDValidation.validateNumerical("rowParallelLinear", "weight", weight);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.RowParallelLinear(sd,input, weight, tpRank, tpSize, reduceOutput).outputVariable();
  }

  /**
   * Row-parallel linear layer for tensor parallelism.<br>
   * Splits weight rows across tensor parallel ranks.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param weight Weight matrix (NUMERIC type)
   * @param tpRank Tensor parallel rank
   * @param tpSize Tensor parallel world size
   * @param reduceOutput Whether to all-reduce output
   * @return output Row-parallel linear output (NUMERIC type)
   */
  public SDVariable rowParallelLinear(String name, SDVariable input, SDVariable weight, int tpRank,
      int tpSize, boolean reduceOutput) {
    SDValidation.validateNumerical("rowParallelLinear", "input", input);
    SDValidation.validateNumerical("rowParallelLinear", "weight", weight);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.RowParallelLinear(sd,input, weight, tpRank, tpSize, reduceOutput).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Selective scan operation for state space models (Mamba architecture).<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output Selective scan output (NUMERIC type)
   */
  public SDVariable selectiveScan(SDVariable input) {
    SDValidation.validateNumerical("selectiveScan", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SelectiveScan(sd,input).outputVariable();
  }

  /**
   * Selective scan operation for state space models (Mamba architecture).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @return output Selective scan output (NUMERIC type)
   */
  public SDVariable selectiveScan(String name, SDVariable input) {
    SDValidation.validateNumerical("selectiveScan", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SelectiveScan(sd,input).outputVariable();
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
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param mask Attention mask [batch, 1, seqLen, kvSeqLen] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @param causal Causal masking (0=bidirectional, 1=causal)
   * @param slidingWindowSize Sliding window size (0=disabled)
   * @param scale Attention scale (0=auto: 1/sqrt(headDim))
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, SDVariable mask, int numHeads, int numKvHeads, int causal,
      int slidingWindowSize, double scale) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    SDValidation.validateNumerical("sharedKvAttention", "mask", mask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, mask, numHeads, numKvHeads, causal, slidingWindowSize, scale).outputVariable();
  }

  /**
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param mask Attention mask [batch, 1, seqLen, kvSeqLen] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @param causal Causal masking (0=bidirectional, 1=causal)
   * @param slidingWindowSize Sliding window size (0=disabled)
   * @param scale Attention scale (0=auto: 1/sqrt(headDim))
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(String name, SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, SDVariable mask, int numHeads, int numKvHeads, int causal,
      int slidingWindowSize, double scale) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    SDValidation.validateNumerical("sharedKvAttention", "mask", mask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, mask, numHeads, numKvHeads, causal, slidingWindowSize, scale).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, null, numHeads, numKvHeads, 1, 0, 0.0).outputVariable();
  }

  /**
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(String name, SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, null, numHeads, numKvHeads, 1, 0, 0.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param mask Attention mask [batch, 1, seqLen, kvSeqLen] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, SDVariable mask, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    SDValidation.validateNumerical("sharedKvAttention", "mask", mask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, mask, numHeads, numKvHeads, 1, 0, 0.0).outputVariable();
  }

  /**
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param mask Attention mask [batch, 1, seqLen, kvSeqLen] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(String name, SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, SDVariable mask, int numHeads, int numKvHeads) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    SDValidation.validateNumerical("sharedKvAttention", "mask", mask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, mask, numHeads, numKvHeads, 1, 0, 0.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param mask Attention mask [batch, 1, seqLen, kvSeqLen] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @param causal Causal masking (0=bidirectional, 1=causal)
   * @param slidingWindowSize Sliding window size (0=disabled)
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, SDVariable mask, int numHeads, int numKvHeads, int causal,
      int slidingWindowSize) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    SDValidation.validateNumerical("sharedKvAttention", "mask", mask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, mask, numHeads, numKvHeads, causal, slidingWindowSize, 0.0).outputVariable();
  }

  /**
   * Shared KV Attention (Gemma 4).<br>
   * <br>
   * Grouped-query attention where K/V come from a donor layer rather than<br>
   * being projected from the current hidden state. The last N layers reuse<br>
   * K/V tensors produced by an earlier layer (the last non-shared layer of<br>
   * the same attention type: sliding or full). This reduces memory and<br>
   * compute with minimal quality impact.<br>
   * <br>
   * Supports causal masking and optional sliding window for local attention.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param query Query [batch, seqLen, numHeads, headDim] (NUMERIC type)
   * @param sharedKey Key from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param sharedValue Value from donor layer [batch, kvSeqLen, numKvHeads, headDim] (NUMERIC type)
   * @param mask Attention mask [batch, 1, seqLen, kvSeqLen] (NUMERIC type)
   * @param numHeads Number of query heads
   * @param numKvHeads Number of key-value heads (for GQA)
   * @param causal Causal masking (0=bidirectional, 1=causal)
   * @param slidingWindowSize Sliding window size (0=disabled)
   * @return output Attention output [batch, seqLen, numHeads, headDim] (NUMERIC type)
   */
  public SDVariable sharedKvAttention(String name, SDVariable query, SDVariable sharedKey,
      SDVariable sharedValue, SDVariable mask, int numHeads, int numKvHeads, int causal,
      int slidingWindowSize) {
    SDValidation.validateNumerical("sharedKvAttention", "query", query);
    SDValidation.validateNumerical("sharedKvAttention", "sharedKey", sharedKey);
    SDValidation.validateNumerical("sharedKvAttention", "sharedValue", sharedValue);
    SDValidation.validateNumerical("sharedKvAttention", "mask", mask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SharedKvAttention(sd,query, sharedKey, sharedValue, mask, numHeads, numKvHeads, causal, slidingWindowSize, 0.0).outputVariable();
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
   * SiLU (Sigmoid Linear Unit) activation function, also known as Swish.<br>
   * Computes f(x) = x * sigmoid(x).<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @return output SiLU(x) = x * sigmoid(x) (NUMERIC type)
   */
  public SDVariable silu(SDVariable input) {
    SDValidation.validateNumerical("silu", "input", input);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SiLU(sd,input).outputVariable();
  }

  /**
   * SiLU (Sigmoid Linear Unit) activation function, also known as Swish.<br>
   * Computes f(x) = x * sigmoid(x).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @return output SiLU(x) = x * sigmoid(x) (NUMERIC type)
   */
  public SDVariable silu(String name, SDVariable input) {
    SDValidation.validateNumerical("silu", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SiLU(sd,input).outputVariable();
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
   * SmoothQuant: migrates quantization difficulty from activations to weights.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param smoothScale Smooth quantization scale (NUMERIC type)
   * @return output Smoothly quantized output (NUMERIC type)
   */
  public SDVariable smoothQuant(SDVariable input, SDVariable smoothScale) {
    SDValidation.validateNumerical("smoothQuant", "input", input);
    SDValidation.validateNumerical("smoothQuant", "smoothScale", smoothScale);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SmoothQuant(sd,input, smoothScale).outputVariable();
  }

  /**
   * SmoothQuant: migrates quantization difficulty from activations to weights.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param smoothScale Smooth quantization scale (NUMERIC type)
   * @return output Smoothly quantized output (NUMERIC type)
   */
  public SDVariable smoothQuant(String name, SDVariable input, SDVariable smoothScale) {
    SDValidation.validateNumerical("smoothQuant", "input", input);
    SDValidation.validateNumerical("smoothQuant", "smoothScale", smoothScale);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SmoothQuant(sd,input, smoothScale).outputVariable();
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
   * Squared ReLU activation function: out = max(0, x)^2.<br>
   * Used in Nemotron and other NVIDIA model architectures.<br>
   *
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable squaredRelu(SDVariable x) {
    SDValidation.validateNumerical("squaredRelu", "x", x);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SquaredReLU(sd,x).outputVariable();
  }

  /**
   * Squared ReLU activation function: out = max(0, x)^2.<br>
   * Used in Nemotron and other NVIDIA model architectures.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable (NUMERIC type)
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable squaredRelu(String name, SDVariable x) {
    SDValidation.validateNumerical("squaredRelu", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SquaredReLU(sd,x).outputVariable();
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
   * Fused Swish-Mul: computes swish(input) * gate in a single kernel.<br>
   * Used in SwiGLU and similar gated architectures.<br>
   *
   * @param input Input tensor (NUMERIC type)
   * @param gate Gate tensor (NUMERIC type)
   * @return output swish(input) * gate (NUMERIC type)
   */
  public SDVariable swishMul(SDVariable input, SDVariable gate) {
    SDValidation.validateNumerical("swishMul", "input", input);
    SDValidation.validateNumerical("swishMul", "gate", gate);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul(sd,input, gate).outputVariable();
  }

  /**
   * Fused Swish-Mul: computes swish(input) * gate in a single kernel.<br>
   * Used in SwiGLU and similar gated architectures.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input tensor (NUMERIC type)
   * @param gate Gate tensor (NUMERIC type)
   * @return output swish(input) * gate (NUMERIC type)
   */
  public SDVariable swishMul(String name, SDVariable input, SDVariable gate) {
    SDValidation.validateNumerical("swishMul", "input", input);
    SDValidation.validateNumerical("swishMul", "gate", gate);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SwishMul(sd,input, gate).outputVariable();
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
  public SDVariable turboQuantAttention(SDVariable query, SDVariable kMse, SDVariable qjlSigns,
      SDVariable residualNorms, SDVariable qjlMatrix, SDVariable values, SDVariable attentionMask,
      int numHeads, int headDim, double scale) {
    SDValidation.validateNumerical("turboQuantAttention", "query", query);
    SDValidation.validateNumerical("turboQuantAttention", "kMse", kMse);
    SDValidation.validateInteger("turboQuantAttention", "qjlSigns", qjlSigns);
    SDValidation.validateNumerical("turboQuantAttention", "residualNorms", residualNorms);
    SDValidation.validateNumerical("turboQuantAttention", "qjlMatrix", qjlMatrix);
    SDValidation.validateNumerical("turboQuantAttention", "values", values);
    SDValidation.validateNumerical("turboQuantAttention", "attentionMask", attentionMask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.TurboQuantAttention(sd,query, kMse, qjlSigns, residualNorms, qjlMatrix, values, attentionMask, numHeads, headDim, scale).outputVariable();
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable turboQuantAttention(String name, SDVariable query, SDVariable kMse,
      SDVariable qjlSigns, SDVariable residualNorms, SDVariable qjlMatrix, SDVariable values,
      SDVariable attentionMask, int numHeads, int headDim, double scale) {
    SDValidation.validateNumerical("turboQuantAttention", "query", query);
    SDValidation.validateNumerical("turboQuantAttention", "kMse", kMse);
    SDValidation.validateInteger("turboQuantAttention", "qjlSigns", qjlSigns);
    SDValidation.validateNumerical("turboQuantAttention", "residualNorms", residualNorms);
    SDValidation.validateNumerical("turboQuantAttention", "qjlMatrix", qjlMatrix);
    SDValidation.validateNumerical("turboQuantAttention", "values", values);
    SDValidation.validateNumerical("turboQuantAttention", "attentionMask", attentionMask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.TurboQuantAttention(sd,query, kMse, qjlSigns, residualNorms, qjlMatrix, values, attentionMask, numHeads, headDim, scale).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable turboQuantAttention(SDVariable query, SDVariable kMse, SDVariable qjlSigns,
      SDVariable residualNorms, SDVariable qjlMatrix, SDVariable values, SDVariable attentionMask,
      int numHeads, int headDim) {
    SDValidation.validateNumerical("turboQuantAttention", "query", query);
    SDValidation.validateNumerical("turboQuantAttention", "kMse", kMse);
    SDValidation.validateInteger("turboQuantAttention", "qjlSigns", qjlSigns);
    SDValidation.validateNumerical("turboQuantAttention", "residualNorms", residualNorms);
    SDValidation.validateNumerical("turboQuantAttention", "qjlMatrix", qjlMatrix);
    SDValidation.validateNumerical("turboQuantAttention", "values", values);
    SDValidation.validateNumerical("turboQuantAttention", "attentionMask", attentionMask);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.TurboQuantAttention(sd,query, kMse, qjlSigns, residualNorms, qjlMatrix, values, attentionMask, numHeads, headDim, 0.0).outputVariable();
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
   * @param name name May be null. Name for the output variable
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
  public SDVariable turboQuantAttention(String name, SDVariable query, SDVariable kMse,
      SDVariable qjlSigns, SDVariable residualNorms, SDVariable qjlMatrix, SDVariable values,
      SDVariable attentionMask, int numHeads, int headDim) {
    SDValidation.validateNumerical("turboQuantAttention", "query", query);
    SDValidation.validateNumerical("turboQuantAttention", "kMse", kMse);
    SDValidation.validateInteger("turboQuantAttention", "qjlSigns", qjlSigns);
    SDValidation.validateNumerical("turboQuantAttention", "residualNorms", residualNorms);
    SDValidation.validateNumerical("turboQuantAttention", "qjlMatrix", qjlMatrix);
    SDValidation.validateNumerical("turboQuantAttention", "values", values);
    SDValidation.validateNumerical("turboQuantAttention", "attentionMask", attentionMask);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.TurboQuantAttention(sd,query, kMse, qjlSigns, residualNorms, qjlMatrix, values, attentionMask, numHeads, headDim, 0.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
