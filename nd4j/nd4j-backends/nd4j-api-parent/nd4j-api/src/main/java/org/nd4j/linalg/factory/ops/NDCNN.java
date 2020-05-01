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

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.common.base.Preconditions;
import org.nd4j.enums.DataFormat;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDCNN {
  public NDCNN() {
  }

  /**
   * 2D Convolution layer operation - average pooling 2d<br>
   *
   * @param input the input to average pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   * @return output Result after applying average pooling on the input (NUMERIC type)
   */
  public INDArray avgPooling2d(INDArray input, Pooling2DConfig Pooling2DConfig) {
    NDValidation.validateNumerical("avgPooling2d", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D(input, Pooling2DConfig))[0];
  }

  /**
   * 3D convolution layer operation - average pooling 3d <br>
   *
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   *                         (shape [minibatch, channels, depth, height, width]) or NDHWC format
   *                         (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param Pooling3DConfig Configuration Object
   * @return output after applying average pooling on the input (NUMERIC type)
   */
  public INDArray avgPooling3d(INDArray input, Pooling3DConfig Pooling3DConfig) {
    NDValidation.validateNumerical("avgPooling3d", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling3D(input, Pooling3DConfig))[0];
  }

  /**
   * Convolution 2d layer batch to space operation on 4d input.<br>
   * Reduces input batch dimension by rearranging data into a larger spatial dimensions<br>
   *
   * @param x Input variable. 4d input (NUMERIC type)
   * @param blocks Block size, in the height/width dimension (Size: Exactly(count=2))
   * @param croppingTop  (Size: Exactly(count=2))
   * @param croppingBottom  (Size: Exactly(count=2))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray batchToSpace(INDArray x, int[] blocks, int[] croppingTop, int... croppingBottom) {
    NDValidation.validateNumerical("batchToSpace", "x", x);
    Preconditions.checkArgument(blocks.length == 2, "blocks has incorrect size/length. Expected: blocks.length == 2, got %s", blocks.length);
    Preconditions.checkArgument(croppingTop.length == 2, "croppingTop has incorrect size/length. Expected: croppingTop.length == 2, got %s", croppingTop.length);
    Preconditions.checkArgument(croppingBottom.length == 2, "croppingBottom has incorrect size/length. Expected: croppingBottom.length == 2, got %s", croppingBottom.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpace(x, blocks, croppingTop, croppingBottom))[0];
  }

  /**
   * col2im operation for use in 2D convolution operations. Outputs a 4d array with shape<br>
   * [minibatch, inputChannels, height, width]<br>
   *
   * @param in Input - rank 6 input with shape [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output Col2Im output variable (NUMERIC type)
   */
  public INDArray col2Im(INDArray in, Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("col2Im", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Col2Im(in, Conv2DConfig))[0];
  }

  /**
   * Conv1d operation.<br>
   *
   * @param input the inputs to conv1d (NUMERIC type)
   * @param weights weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels] (NUMERIC type)
   * @param bias bias for conv1d op - rank 1 array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv1DConfig Configuration Object
   * @return output result of conv1d op (NUMERIC type)
   */
  public INDArray conv1d(INDArray input, INDArray weights, INDArray bias,
      Conv1DConfig Conv1DConfig) {
    NDValidation.validateNumerical("conv1d", "input", input);
    NDValidation.validateNumerical("conv1d", "weights", weights);
    NDValidation.validateNumerical("conv1d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D(input, weights, bias, Conv1DConfig))[0];
  }

  /**
   * Conv1d operation.<br>
   *
   * @param input the inputs to conv1d (NUMERIC type)
   * @param weights weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels] (NUMERIC type)
   * @param Conv1DConfig Configuration Object
   * @return output result of conv1d op (NUMERIC type)
   */
  public INDArray conv1d(INDArray input, INDArray weights, Conv1DConfig Conv1DConfig) {
    NDValidation.validateNumerical("conv1d", "input", input);
    NDValidation.validateNumerical("conv1d", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D(input, weights, null, Conv1DConfig))[0];
  }

  /**
   * 2D Convolution operation with optional bias<br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param weights Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels] (NUMERIC type)
   * @param bias Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of conv2d op (NUMERIC type)
   */
  public INDArray conv2d(INDArray layerInput, INDArray weights, INDArray bias,
      Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("conv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("conv2d", "weights", weights);
    NDValidation.validateNumerical("conv2d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(layerInput, weights, bias, Conv2DConfig))[0];
  }

  /**
   * 2D Convolution operation with optional bias<br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param weights Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of conv2d op (NUMERIC type)
   */
  public INDArray conv2d(INDArray layerInput, INDArray weights, Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("conv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("conv2d", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(layerInput, weights, null, Conv2DConfig))[0];
  }

  /**
   * Convolution 3D operation with optional bias <br>
   *
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   * (shape [minibatch, channels, depth, height, width]) or NDHWC format
   * (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param weights  Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]. (NUMERIC type)
   * @param bias  Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv3DConfig Configuration Object
   * @return output Conv3d output variable (NUMERIC type)
   */
  public INDArray conv3d(INDArray input, INDArray weights, INDArray bias,
      Conv3DConfig Conv3DConfig) {
    NDValidation.validateNumerical("conv3d", "input", input);
    NDValidation.validateNumerical("conv3d", "weights", weights);
    NDValidation.validateNumerical("conv3d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D(input, weights, bias, Conv3DConfig))[0];
  }

  /**
   * Convolution 3D operation with optional bias <br>
   *
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   * (shape [minibatch, channels, depth, height, width]) or NDHWC format
   * (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param weights  Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]. (NUMERIC type)
   * @param Conv3DConfig Configuration Object
   * @return output Conv3d output variable (NUMERIC type)
   */
  public INDArray conv3d(INDArray input, INDArray weights, Conv3DConfig Conv3DConfig) {
    NDValidation.validateNumerical("conv3d", "input", input);
    NDValidation.validateNumerical("conv3d", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D(input, weights, null, Conv3DConfig))[0];
  }

  /**
   * 2D deconvolution operation with optional bias<br>
   *
   * @param layerInput the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format
   * (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param weights Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth] (NUMERIC type)
   * @param bias Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param DeConv2DConfig Configuration Object
   * @return output result of deconv2d op (NUMERIC type)
   */
  public INDArray deconv2d(INDArray layerInput, INDArray weights, INDArray bias,
      DeConv2DConfig DeConv2DConfig) {
    NDValidation.validateNumerical("deconv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("deconv2d", "weights", weights);
    NDValidation.validateNumerical("deconv2d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D(layerInput, weights, bias, DeConv2DConfig))[0];
  }

  /**
   * 2D deconvolution operation with optional bias<br>
   *
   * @param layerInput the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format
   * (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param weights Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth] (NUMERIC type)
   * @param DeConv2DConfig Configuration Object
   * @return output result of deconv2d op (NUMERIC type)
   */
  public INDArray deconv2d(INDArray layerInput, INDArray weights, DeConv2DConfig DeConv2DConfig) {
    NDValidation.validateNumerical("deconv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("deconv2d", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D(layerInput, weights, null, DeConv2DConfig))[0];
  }

  /**
   * 3D CNN deconvolution operation with or without optional bias<br>
   *
   * @param input Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW) (NUMERIC type)
   * @param weights Weights array - shape [kD, kH, kW, oC, iC] (NUMERIC type)
   * @param bias Bias array - optional, may be null. If non-null, must have shape [outputChannels] (NUMERIC type)
   * @param DeConv3DConfig Configuration Object
   * @return output result of 3D CNN deconvolution operation (NUMERIC type)
   */
  public INDArray deconv3d(INDArray input, INDArray weights, INDArray bias,
      DeConv3DConfig DeConv3DConfig) {
    NDValidation.validateNumerical("deconv3d", "input", input);
    NDValidation.validateNumerical("deconv3d", "weights", weights);
    NDValidation.validateNumerical("deconv3d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3D(input, weights, bias, DeConv3DConfig))[0];
  }

  /**
   * 3D CNN deconvolution operation with or without optional bias<br>
   *
   * @param input Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW) (NUMERIC type)
   * @param weights Weights array - shape [kD, kH, kW, oC, iC] (NUMERIC type)
   * @param DeConv3DConfig Configuration Object
   * @return output result of 3D CNN deconvolution operation (NUMERIC type)
   */
  public INDArray deconv3d(INDArray input, INDArray weights, DeConv3DConfig DeConv3DConfig) {
    NDValidation.validateNumerical("deconv3d", "input", input);
    NDValidation.validateNumerical("deconv3d", "weights", weights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3D(input, weights, null, DeConv3DConfig))[0];
  }

  /**
   * Convolution 2d layer batch to space operation on 4d input.<br>
   * Reduces input channels dimension by rearranging data into a larger spatial dimensions<br>
   * Example: if input has shape [mb, 8, 2, 2] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]<br>
   * = [mb, 2, 4, 4]<br>
   *
   * @param x the input to depth to space pooling 2d operation - 4d activations in NCHW format
   *                    (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param blockSize Block size, in the height/width dimension
   * @param dataFormat Data format: "NCHW" or "NHWC"
   * @return output Output variable (NUMERIC type)
   */
  public INDArray depthToSpace(INDArray x, int blockSize, DataFormat dataFormat) {
    NDValidation.validateNumerical("depthToSpace", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthToSpace(x, blockSize, dataFormat))[0];
  }

  /**
   * Depth-wise 2D convolution operation with optional bias <br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param depthWeights Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param bias Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of depthwise conv2d op (NUMERIC type)
   */
  public INDArray depthWiseConv2d(INDArray layerInput, INDArray depthWeights, INDArray bias,
      Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("depthWiseConv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("depthWiseConv2d", "depthWeights", depthWeights);
    NDValidation.validateNumerical("depthWiseConv2d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D(layerInput, depthWeights, bias, Conv2DConfig))[0];
  }

  /**
   * Depth-wise 2D convolution operation with optional bias <br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param depthWeights Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of depthwise conv2d op (NUMERIC type)
   */
  public INDArray depthWiseConv2d(INDArray layerInput, INDArray depthWeights,
      Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("depthWiseConv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("depthWiseConv2d", "depthWeights", depthWeights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D(layerInput, depthWeights, null, Conv2DConfig))[0];
  }

  /**
   * TODO doc string<br>
   *
   * @param df  (NUMERIC type)
   * @param weights df (NUMERIC type)
   * @param strides weights (Size: Exactly(count=2))
   * @param rates strides (Size: Exactly(count=2))
   * @param isSameMode isSameMode
   * @return output Computed the grayscale dilation of 4-D input and 3-D filters tensors. (NUMERIC type)
   */
  public INDArray dilation2D(INDArray df, INDArray weights, int[] strides, int[] rates,
      boolean isSameMode) {
    NDValidation.validateNumerical("dilation2D", "df", df);
    NDValidation.validateNumerical("dilation2D", "weights", weights);
    Preconditions.checkArgument(strides.length == 2, "strides has incorrect size/length. Expected: strides.length == 2, got %s", strides.length);
    Preconditions.checkArgument(rates.length == 2, "rates has incorrect size/length. Expected: rates.length == 2, got %s", rates.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.Dilation2D(df, weights, strides, rates, isSameMode))[0];
  }

  /**
   * Extract image patches <br>
   *
   * @param input Input array. Must be rank 4, with shape [minibatch, height, width, channels] (NUMERIC type)
   * @param kH Kernel height
   * @param kW Kernel width
   * @param sH Stride height
   * @param sW Stride width
   * @param rH Rate height
   * @param rW Rate width
   * @param sameMode If true: use same mode padding. If false
   * @return output The result is a 4D tensor which is indexed by batch, row, and column. (NUMERIC type)
   */
  public INDArray extractImagePatches(INDArray input, int kH, int kW, int sH, int sW, int rH,
      int rW, boolean sameMode) {
    NDValidation.validateNumerical("extractImagePatches", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches(input, kH, kW, sH, sW, rH, rW, sameMode))[0];
  }

  /**
   * im2col operation for use in 2D convolution operations. Outputs a 6d array with shape<br>
   * [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]   <br>
   *
   * @param in Input - rank 4 input with shape [minibatch, inputChannels, height, width] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output Im2Col output variable (NUMERIC type)
   */
  public INDArray im2Col(INDArray in, Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("im2Col", "in", in);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col(in, Conv2DConfig))[0];
  }

  /**
   * 2D convolution layer operation - local response normalization<br>
   *
   * @param input the inputs to lrn (NUMERIC type)
   * @param LocalResponseNormalizationConfig Configuration Object
   * @return output Result after Local Response Normalization (NUMERIC type)
   */
  public INDArray localResponseNormalization(INDArray input,
      LocalResponseNormalizationConfig LocalResponseNormalizationConfig) {
    NDValidation.validateNumerical("localResponseNormalization", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.LocalResponseNormalization(input, LocalResponseNormalizationConfig))[0];
  }

  /**
   * 2D Convolution layer operation - Max pooling on the input and outputs both max values and indices <br>
   *
   * @param input the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   */
  public INDArray[] maxPoolWithArgmax(INDArray input, Pooling2DConfig Pooling2DConfig) {
    NDValidation.validateNumerical("maxPoolWithArgmax", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPoolWithArgmax(input, Pooling2DConfig));
  }

  /**
   * 2D Convolution layer operation - max pooling 2d <br>
   *
   * @param input the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   * @return output Result after applying max pooling on the input (NUMERIC type)
   */
  public INDArray maxPooling2d(INDArray input, Pooling2DConfig Pooling2DConfig) {
    NDValidation.validateNumerical("maxPooling2d", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D(input, Pooling2DConfig))[0];
  }

  /**
   * 3D convolution layer operation - max pooling 3d operation.<br>
   *
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   *                         (shape [minibatch, channels, depth, height, width]) or NDHWC format
   *                         (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param Pooling3DConfig Configuration Object
   * @return output Result after applying max pooling on the input (NUMERIC type)
   */
  public INDArray maxPooling3d(INDArray input, Pooling3DConfig Pooling3DConfig) {
    NDValidation.validateNumerical("maxPooling3d", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling3D(input, Pooling3DConfig))[0];
  }

  /**
   * Separable 2D convolution operation with optional bias <br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                      (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param depthWeights Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param pointWeights Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]. May be null (NUMERIC type)
   * @param bias Optional bias, rank 1 with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of separable convolution 2d operation (NUMERIC type)
   */
  public INDArray separableConv2d(INDArray layerInput, INDArray depthWeights, INDArray pointWeights,
      INDArray bias, Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("separableConv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("separableConv2d", "depthWeights", depthWeights);
    NDValidation.validateNumerical("separableConv2d", "pointWeights", pointWeights);
    NDValidation.validateNumerical("separableConv2d", "bias", bias);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D(layerInput, depthWeights, pointWeights, bias, Conv2DConfig))[0];
  }

  /**
   * Separable 2D convolution operation with optional bias <br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                      (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param depthWeights Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param pointWeights Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]. May be null (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of separable convolution 2d operation (NUMERIC type)
   */
  public INDArray separableConv2d(INDArray layerInput, INDArray depthWeights, INDArray pointWeights,
      Conv2DConfig Conv2DConfig) {
    NDValidation.validateNumerical("separableConv2d", "layerInput", layerInput);
    NDValidation.validateNumerical("separableConv2d", "depthWeights", depthWeights);
    NDValidation.validateNumerical("separableConv2d", "pointWeights", pointWeights);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D(layerInput, depthWeights, pointWeights, null, Conv2DConfig))[0];
  }

  /**
   * Convolution 2d layer space to batch operation on 4d input.<br>
   * Increases input batch dimension by rearranging data from spatial dimensions into batch dimension <br>
   *
   * @param x Input variable. 4d input (NUMERIC type)
   * @param blocks Block size, in the height/width dimension (Size: Exactly(count=2))
   * @param paddingTop Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]] (Size: Exactly(count=2))
   * @param paddingBottom Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]] (Size: Exactly(count=2))
   * @return output Output variable (NUMERIC type)
   */
  public INDArray spaceToBatch(INDArray x, int[] blocks, int[] paddingTop, int... paddingBottom) {
    NDValidation.validateNumerical("spaceToBatch", "x", x);
    Preconditions.checkArgument(blocks.length == 2, "blocks has incorrect size/length. Expected: blocks.length == 2, got %s", blocks.length);
    Preconditions.checkArgument(paddingTop.length == 2, "paddingTop has incorrect size/length. Expected: paddingTop.length == 2, got %s", paddingTop.length);
    Preconditions.checkArgument(paddingBottom.length == 2, "paddingBottom has incorrect size/length. Expected: paddingBottom.length == 2, got %s", paddingBottom.length);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.SpaceToBatch(x, blocks, paddingTop, paddingBottom))[0];
  }

  /**
   * Convolution 2d layer space to depth operation on 4d input.<br>
   * Increases input channels (reduced spatial dimensions) by rearranging data into a larger channels dimension<br>
   * Example: if input has shape [mb, 2, 4, 4] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]<br>
   * = [mb, 2, 4, 4] <br>
   *
   * @param x the input to depth to space pooling 2d operation - 4d activations in NCHW format
   *                    (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param blockSize  Block size, in the height/width dimension
   * @param dataFormat Data format: "NCHW" or "NHWC"
   * @return output Output variable (NUMERIC type)
   */
  public INDArray spaceToDepth(INDArray x, int blockSize, DataFormat dataFormat) {
    NDValidation.validateNumerical("spaceToDepth", "x", x);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.SpaceToDepth(x, blockSize, dataFormat))[0];
  }

  /**
   * Upsampling layer for 2D inputs.<br>
   * scale is used for both height and width dimensions. <br>
   *
   * @param input Input in NCHW format (NUMERIC type)
   * @param scale The scale for both height and width dimensions.
   * @return output Upsampled input (NUMERIC type)
   */
  public INDArray upsampling2d(INDArray input, int scale) {
    NDValidation.validateNumerical("upsampling2d", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d(input, scale))[0];
  }

  /**
   * 2D Convolution layer operation - Upsampling 2d <br>
   *
   * @param input Input in NCHW format (NUMERIC type)
   * @param scaleH Scale to upsample in height dimension
   * @param scaleW Scale to upsample in width dimension
   * @param nchw If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format
   * @return output Upsampled input (NUMERIC type)
   */
  public INDArray upsampling2d(INDArray input, int scaleH, int scaleW, boolean nchw) {
    NDValidation.validateNumerical("upsampling2d", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d(input, scaleH, scaleW, nchw))[0];
  }

  /**
   * 3D Convolution layer operation - Upsampling 3d <br>
   *
   * @param input Input in NCHW format (NUMERIC type)
   * @param ncdhw If true: input is in NCDHW (minibatch, channels, depth, height, width) format. False: NDHWC format
   * @param scaleD Scale to upsample in depth dimension
   * @param scaleH Scale to upsample in height dimension
   * @param scaleW Scale to upsample in width dimension
   * @return output Upsampled input (NUMERIC type)
   */
  public INDArray upsampling3d(INDArray input, boolean ncdhw, int scaleD, int scaleH, int scaleW) {
    NDValidation.validateNumerical("upsampling3d", "input", input);
    return Nd4j.exec(new org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling3d(input, ncdhw, scaleD, scaleH, scaleW))[0];
  }
}
