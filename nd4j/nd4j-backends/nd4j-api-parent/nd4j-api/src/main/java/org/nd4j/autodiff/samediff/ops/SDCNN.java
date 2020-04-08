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

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.base.Preconditions;
import org.nd4j.enums.DataFormat;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.LocalResponseNormalizationConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling3DConfig;

public class SDCNN extends SDOps {
  public SDCNN(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * 2D Convolution layer operation - average pooling 2d<br>
   *
   * @param input the input to average pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   * @return output Result after applying average pooling on the input (NUMERIC type)
   */
  public SDVariable avgPooling2d(SDVariable input, Pooling2DConfig Pooling2DConfig) {
    SDValidation.validateNumerical("avgPooling2d", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D(sd,input, Pooling2DConfig).outputVariable();
  }

  /**
   * 2D Convolution layer operation - average pooling 2d<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the input to average pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   * @return output Result after applying average pooling on the input (NUMERIC type)
   */
  public SDVariable avgPooling2d(String name, SDVariable input, Pooling2DConfig Pooling2DConfig) {
    SDValidation.validateNumerical("avgPooling2d", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling2D(sd,input, Pooling2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable avgPooling3d(SDVariable input, Pooling3DConfig Pooling3DConfig) {
    SDValidation.validateNumerical("avgPooling3d", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling3D(sd,input, Pooling3DConfig).outputVariable();
  }

  /**
   * 3D convolution layer operation - average pooling 3d <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   *                         (shape [minibatch, channels, depth, height, width]) or NDHWC format
   *                         (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param Pooling3DConfig Configuration Object
   * @return output after applying average pooling on the input (NUMERIC type)
   */
  public SDVariable avgPooling3d(String name, SDVariable input, Pooling3DConfig Pooling3DConfig) {
    SDValidation.validateNumerical("avgPooling3d", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.AvgPooling3D(sd,input, Pooling3DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable batchToSpace(SDVariable x, int[] blocks, int[] croppingTop,
      int... croppingBottom) {
    SDValidation.validateNumerical("batchToSpace", "x", x);
    Preconditions.checkArgument(blocks.length == 2, "blocks has incorrect size/length. Expected: blocks.length == 2, got %s", blocks.length);
    Preconditions.checkArgument(croppingTop.length == 2, "croppingTop has incorrect size/length. Expected: croppingTop.length == 2, got %s", croppingTop.length);
    Preconditions.checkArgument(croppingBottom.length == 2, "croppingBottom has incorrect size/length. Expected: croppingBottom.length == 2, got %s", croppingBottom.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpace(sd,x, blocks, croppingTop, croppingBottom).outputVariable();
  }

  /**
   * Convolution 2d layer batch to space operation on 4d input.<br>
   * Reduces input batch dimension by rearranging data into a larger spatial dimensions<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable. 4d input (NUMERIC type)
   * @param blocks Block size, in the height/width dimension (Size: Exactly(count=2))
   * @param croppingTop  (Size: Exactly(count=2))
   * @param croppingBottom  (Size: Exactly(count=2))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable batchToSpace(String name, SDVariable x, int[] blocks, int[] croppingTop,
      int... croppingBottom) {
    SDValidation.validateNumerical("batchToSpace", "x", x);
    Preconditions.checkArgument(blocks.length == 2, "blocks has incorrect size/length. Expected: blocks.length == 2, got %s", blocks.length);
    Preconditions.checkArgument(croppingTop.length == 2, "croppingTop has incorrect size/length. Expected: croppingTop.length == 2, got %s", croppingTop.length);
    Preconditions.checkArgument(croppingBottom.length == 2, "croppingBottom has incorrect size/length. Expected: croppingBottom.length == 2, got %s", croppingBottom.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpace(sd,x, blocks, croppingTop, croppingBottom).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * col2im operation for use in 2D convolution operations. Outputs a 4d array with shape<br>
   * [minibatch, inputChannels, height, width]<br>
   *
   * @param in Input - rank 6 input with shape [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output Col2Im output variable (NUMERIC type)
   */
  public SDVariable col2Im(SDVariable in, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("col2Im", "in", in);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Col2Im(sd,in, Conv2DConfig).outputVariable();
  }

  /**
   * col2im operation for use in 2D convolution operations. Outputs a 4d array with shape<br>
   * [minibatch, inputChannels, height, width]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input - rank 6 input with shape [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output Col2Im output variable (NUMERIC type)
   */
  public SDVariable col2Im(String name, SDVariable in, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("col2Im", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Col2Im(sd,in, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable conv1d(SDVariable input, SDVariable weights, SDVariable bias,
      Conv1DConfig Conv1DConfig) {
    SDValidation.validateNumerical("conv1d", "input", input);
    SDValidation.validateNumerical("conv1d", "weights", weights);
    SDValidation.validateNumerical("conv1d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D(sd,input, weights, bias, Conv1DConfig).outputVariable();
  }

  /**
   * Conv1d operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the inputs to conv1d (NUMERIC type)
   * @param weights weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels] (NUMERIC type)
   * @param bias bias for conv1d op - rank 1 array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv1DConfig Configuration Object
   * @return output result of conv1d op (NUMERIC type)
   */
  public SDVariable conv1d(String name, SDVariable input, SDVariable weights, SDVariable bias,
      Conv1DConfig Conv1DConfig) {
    SDValidation.validateNumerical("conv1d", "input", input);
    SDValidation.validateNumerical("conv1d", "weights", weights);
    SDValidation.validateNumerical("conv1d", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D(sd,input, weights, bias, Conv1DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Conv1d operation.<br>
   *
   * @param input the inputs to conv1d (NUMERIC type)
   * @param weights weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels] (NUMERIC type)
   * @param Conv1DConfig Configuration Object
   * @return output result of conv1d op (NUMERIC type)
   */
  public SDVariable conv1d(SDVariable input, SDVariable weights, Conv1DConfig Conv1DConfig) {
    SDValidation.validateNumerical("conv1d", "input", input);
    SDValidation.validateNumerical("conv1d", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D(sd,input, weights, null, Conv1DConfig).outputVariable();
  }

  /**
   * Conv1d operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the inputs to conv1d (NUMERIC type)
   * @param weights weights for conv1d op - rank 3 array with shape [kernelSize, inputChannels, outputChannels] (NUMERIC type)
   * @param Conv1DConfig Configuration Object
   * @return output result of conv1d op (NUMERIC type)
   */
  public SDVariable conv1d(String name, SDVariable input, SDVariable weights,
      Conv1DConfig Conv1DConfig) {
    SDValidation.validateNumerical("conv1d", "input", input);
    SDValidation.validateNumerical("conv1d", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D(sd,input, weights, null, Conv1DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable conv2d(SDVariable layerInput, SDVariable weights, SDVariable bias,
      Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("conv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("conv2d", "weights", weights);
    SDValidation.validateNumerical("conv2d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(sd,layerInput, weights, bias, Conv2DConfig).outputVariable();
  }

  /**
   * 2D Convolution operation with optional bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param weights Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels] (NUMERIC type)
   * @param bias Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of conv2d op (NUMERIC type)
   */
  public SDVariable conv2d(String name, SDVariable layerInput, SDVariable weights, SDVariable bias,
      Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("conv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("conv2d", "weights", weights);
    SDValidation.validateNumerical("conv2d", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(sd,layerInput, weights, bias, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * 2D Convolution operation with optional bias<br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param weights Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of conv2d op (NUMERIC type)
   */
  public SDVariable conv2d(SDVariable layerInput, SDVariable weights, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("conv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("conv2d", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(sd,layerInput, weights, null, Conv2DConfig).outputVariable();
  }

  /**
   * 2D Convolution operation with optional bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param weights Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of conv2d op (NUMERIC type)
   */
  public SDVariable conv2d(String name, SDVariable layerInput, SDVariable weights,
      Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("conv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("conv2d", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2D(sd,layerInput, weights, null, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable conv3d(SDVariable input, SDVariable weights, SDVariable bias,
      Conv3DConfig Conv3DConfig) {
    SDValidation.validateNumerical("conv3d", "input", input);
    SDValidation.validateNumerical("conv3d", "weights", weights);
    SDValidation.validateNumerical("conv3d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D(sd,input, weights, bias, Conv3DConfig).outputVariable();
  }

  /**
   * Convolution 3D operation with optional bias <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   * (shape [minibatch, channels, depth, height, width]) or NDHWC format
   * (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param weights  Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]. (NUMERIC type)
   * @param bias  Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv3DConfig Configuration Object
   * @return output Conv3d output variable (NUMERIC type)
   */
  public SDVariable conv3d(String name, SDVariable input, SDVariable weights, SDVariable bias,
      Conv3DConfig Conv3DConfig) {
    SDValidation.validateNumerical("conv3d", "input", input);
    SDValidation.validateNumerical("conv3d", "weights", weights);
    SDValidation.validateNumerical("conv3d", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D(sd,input, weights, bias, Conv3DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable conv3d(SDVariable input, SDVariable weights, Conv3DConfig Conv3DConfig) {
    SDValidation.validateNumerical("conv3d", "input", input);
    SDValidation.validateNumerical("conv3d", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D(sd,input, weights, null, Conv3DConfig).outputVariable();
  }

  /**
   * Convolution 3D operation with optional bias <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   * (shape [minibatch, channels, depth, height, width]) or NDHWC format
   * (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param weights  Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels]. (NUMERIC type)
   * @param Conv3DConfig Configuration Object
   * @return output Conv3d output variable (NUMERIC type)
   */
  public SDVariable conv3d(String name, SDVariable input, SDVariable weights,
      Conv3DConfig Conv3DConfig) {
    SDValidation.validateNumerical("conv3d", "input", input);
    SDValidation.validateNumerical("conv3d", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D(sd,input, weights, null, Conv3DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable deconv2d(SDVariable layerInput, SDVariable weights, SDVariable bias,
      DeConv2DConfig DeConv2DConfig) {
    SDValidation.validateNumerical("deconv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("deconv2d", "weights", weights);
    SDValidation.validateNumerical("deconv2d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D(sd,layerInput, weights, bias, DeConv2DConfig).outputVariable();
  }

  /**
   * 2D deconvolution operation with optional bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format
   * (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param weights Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth] (NUMERIC type)
   * @param bias Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param DeConv2DConfig Configuration Object
   * @return output result of deconv2d op (NUMERIC type)
   */
  public SDVariable deconv2d(String name, SDVariable layerInput, SDVariable weights,
      SDVariable bias, DeConv2DConfig DeConv2DConfig) {
    SDValidation.validateNumerical("deconv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("deconv2d", "weights", weights);
    SDValidation.validateNumerical("deconv2d", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D(sd,layerInput, weights, bias, DeConv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable deconv2d(SDVariable layerInput, SDVariable weights,
      DeConv2DConfig DeConv2DConfig) {
    SDValidation.validateNumerical("deconv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("deconv2d", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D(sd,layerInput, weights, null, DeConv2DConfig).outputVariable();
  }

  /**
   * 2D deconvolution operation with optional bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format
   * (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param weights Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth] (NUMERIC type)
   * @param DeConv2DConfig Configuration Object
   * @return output result of deconv2d op (NUMERIC type)
   */
  public SDVariable deconv2d(String name, SDVariable layerInput, SDVariable weights,
      DeConv2DConfig DeConv2DConfig) {
    SDValidation.validateNumerical("deconv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("deconv2d", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D(sd,layerInput, weights, null, DeConv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable deconv3d(SDVariable input, SDVariable weights, SDVariable bias,
      DeConv3DConfig DeConv3DConfig) {
    SDValidation.validateNumerical("deconv3d", "input", input);
    SDValidation.validateNumerical("deconv3d", "weights", weights);
    SDValidation.validateNumerical("deconv3d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3D(sd,input, weights, bias, DeConv3DConfig).outputVariable();
  }

  /**
   * 3D CNN deconvolution operation with or without optional bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW) (NUMERIC type)
   * @param weights Weights array - shape [kD, kH, kW, oC, iC] (NUMERIC type)
   * @param bias Bias array - optional, may be null. If non-null, must have shape [outputChannels] (NUMERIC type)
   * @param DeConv3DConfig Configuration Object
   * @return output result of 3D CNN deconvolution operation (NUMERIC type)
   */
  public SDVariable deconv3d(String name, SDVariable input, SDVariable weights, SDVariable bias,
      DeConv3DConfig DeConv3DConfig) {
    SDValidation.validateNumerical("deconv3d", "input", input);
    SDValidation.validateNumerical("deconv3d", "weights", weights);
    SDValidation.validateNumerical("deconv3d", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3D(sd,input, weights, bias, DeConv3DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * 3D CNN deconvolution operation with or without optional bias<br>
   *
   * @param input Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW) (NUMERIC type)
   * @param weights Weights array - shape [kD, kH, kW, oC, iC] (NUMERIC type)
   * @param DeConv3DConfig Configuration Object
   * @return output result of 3D CNN deconvolution operation (NUMERIC type)
   */
  public SDVariable deconv3d(SDVariable input, SDVariable weights, DeConv3DConfig DeConv3DConfig) {
    SDValidation.validateNumerical("deconv3d", "input", input);
    SDValidation.validateNumerical("deconv3d", "weights", weights);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3D(sd,input, weights, null, DeConv3DConfig).outputVariable();
  }

  /**
   * 3D CNN deconvolution operation with or without optional bias<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW) (NUMERIC type)
   * @param weights Weights array - shape [kD, kH, kW, oC, iC] (NUMERIC type)
   * @param DeConv3DConfig Configuration Object
   * @return output result of 3D CNN deconvolution operation (NUMERIC type)
   */
  public SDVariable deconv3d(String name, SDVariable input, SDVariable weights,
      DeConv3DConfig DeConv3DConfig) {
    SDValidation.validateNumerical("deconv3d", "input", input);
    SDValidation.validateNumerical("deconv3d", "weights", weights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv3D(sd,input, weights, null, DeConv3DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable depthToSpace(SDVariable x, int blockSize, DataFormat dataFormat) {
    SDValidation.validateNumerical("depthToSpace", "x", x);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthToSpace(sd,x, blockSize, dataFormat).outputVariable();
  }

  /**
   * Convolution 2d layer batch to space operation on 4d input.<br>
   * Reduces input channels dimension by rearranging data into a larger spatial dimensions<br>
   * Example: if input has shape [mb, 8, 2, 2] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]<br>
   * = [mb, 2, 4, 4]<br>
   *
   * @param name name May be null. Name for the output variable
   * @param x the input to depth to space pooling 2d operation - 4d activations in NCHW format
   *                    (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param blockSize Block size, in the height/width dimension
   * @param dataFormat Data format: "NCHW" or "NHWC"
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable depthToSpace(String name, SDVariable x, int blockSize, DataFormat dataFormat) {
    SDValidation.validateNumerical("depthToSpace", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthToSpace(sd,x, blockSize, dataFormat).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable bias,
      Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("depthWiseConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("depthWiseConv2d", "depthWeights", depthWeights);
    SDValidation.validateNumerical("depthWiseConv2d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D(sd,layerInput, depthWeights, bias, Conv2DConfig).outputVariable();
  }

  /**
   * Depth-wise 2D convolution operation with optional bias <br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param depthWeights Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param bias Optional 1D bias array with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of depthwise conv2d op (NUMERIC type)
   */
  public SDVariable depthWiseConv2d(String name, SDVariable layerInput, SDVariable depthWeights,
      SDVariable bias, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("depthWiseConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("depthWiseConv2d", "depthWeights", depthWeights);
    SDValidation.validateNumerical("depthWiseConv2d", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D(sd,layerInput, depthWeights, bias, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Depth-wise 2D convolution operation with optional bias <br>
   *
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param depthWeights Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of depthwise conv2d op (NUMERIC type)
   */
  public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights,
      Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("depthWiseConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("depthWiseConv2d", "depthWeights", depthWeights);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D(sd,layerInput, depthWeights, null, Conv2DConfig).outputVariable();
  }

  /**
   * Depth-wise 2D convolution operation with optional bias <br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format (NUMERIC type)
   * @param depthWeights Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of depthwise conv2d op (NUMERIC type)
   */
  public SDVariable depthWiseConv2d(String name, SDVariable layerInput, SDVariable depthWeights,
      Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("depthWiseConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("depthWiseConv2d", "depthWeights", depthWeights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.DepthwiseConv2D(sd,layerInput, depthWeights, null, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable dilation2D(SDVariable df, SDVariable weights, int[] strides, int[] rates,
      boolean isSameMode) {
    SDValidation.validateNumerical("dilation2D", "df", df);
    SDValidation.validateNumerical("dilation2D", "weights", weights);
    Preconditions.checkArgument(strides.length == 2, "strides has incorrect size/length. Expected: strides.length == 2, got %s", strides.length);
    Preconditions.checkArgument(rates.length == 2, "rates has incorrect size/length. Expected: rates.length == 2, got %s", rates.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.Dilation2D(sd,df, weights, strides, rates, isSameMode).outputVariable();
  }

  /**
   * TODO doc string<br>
   *
   * @param name name May be null. Name for the output variable
   * @param df  (NUMERIC type)
   * @param weights df (NUMERIC type)
   * @param strides weights (Size: Exactly(count=2))
   * @param rates strides (Size: Exactly(count=2))
   * @param isSameMode isSameMode
   * @return output Computed the grayscale dilation of 4-D input and 3-D filters tensors. (NUMERIC type)
   */
  public SDVariable dilation2D(String name, SDVariable df, SDVariable weights, int[] strides,
      int[] rates, boolean isSameMode) {
    SDValidation.validateNumerical("dilation2D", "df", df);
    SDValidation.validateNumerical("dilation2D", "weights", weights);
    Preconditions.checkArgument(strides.length == 2, "strides has incorrect size/length. Expected: strides.length == 2, got %s", strides.length);
    Preconditions.checkArgument(rates.length == 2, "rates has incorrect size/length. Expected: rates.length == 2, got %s", rates.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.Dilation2D(sd,df, weights, strides, rates, isSameMode).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable extractImagePatches(SDVariable input, int kH, int kW, int sH, int sW, int rH,
      int rW, boolean sameMode) {
    SDValidation.validateNumerical("extractImagePatches", "input", input);
    return new org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches(sd,input, kH, kW, sH, sW, rH, rW, sameMode).outputVariable();
  }

  /**
   * Extract image patches <br>
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable extractImagePatches(String name, SDVariable input, int kH, int kW, int sH,
      int sW, int rH, int rW, boolean sameMode) {
    SDValidation.validateNumerical("extractImagePatches", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.image.ExtractImagePatches(sd,input, kH, kW, sH, sW, rH, rW, sameMode).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * im2col operation for use in 2D convolution operations. Outputs a 6d array with shape<br>
   * [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]   <br>
   *
   * @param in Input - rank 4 input with shape [minibatch, inputChannels, height, width] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output Im2Col output variable (NUMERIC type)
   */
  public SDVariable im2Col(SDVariable in, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("im2Col", "in", in);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col(sd,in, Conv2DConfig).outputVariable();
  }

  /**
   * im2col operation for use in 2D convolution operations. Outputs a 6d array with shape<br>
   * [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]   <br>
   *
   * @param name name May be null. Name for the output variable
   * @param in Input - rank 4 input with shape [minibatch, inputChannels, height, width] (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output Im2Col output variable (NUMERIC type)
   */
  public SDVariable im2Col(String name, SDVariable in, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("im2Col", "in", in);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Im2col(sd,in, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * 2D convolution layer operation - local response normalization<br>
   *
   * @param input the inputs to lrn (NUMERIC type)
   * @param LocalResponseNormalizationConfig Configuration Object
   * @return output Result after Local Response Normalization (NUMERIC type)
   */
  public SDVariable localResponseNormalization(SDVariable input,
      LocalResponseNormalizationConfig LocalResponseNormalizationConfig) {
    SDValidation.validateNumerical("localResponseNormalization", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.LocalResponseNormalization(sd,input, LocalResponseNormalizationConfig).outputVariable();
  }

  /**
   * 2D convolution layer operation - local response normalization<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the inputs to lrn (NUMERIC type)
   * @param LocalResponseNormalizationConfig Configuration Object
   * @return output Result after Local Response Normalization (NUMERIC type)
   */
  public SDVariable localResponseNormalization(String name, SDVariable input,
      LocalResponseNormalizationConfig LocalResponseNormalizationConfig) {
    SDValidation.validateNumerical("localResponseNormalization", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.LocalResponseNormalization(sd,input, LocalResponseNormalizationConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * 2D Convolution layer operation - Max pooling on the input and outputs both max values and indices <br>
   *
   * @param input the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   */
  public SDVariable[] maxPoolWithArgmax(SDVariable input, Pooling2DConfig Pooling2DConfig) {
    SDValidation.validateNumerical("maxPoolWithArgmax", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPoolWithArgmax(sd,input, Pooling2DConfig).outputVariables();
  }

  /**
   * 2D Convolution layer operation - Max pooling on the input and outputs both max values and indices <br>
   *
   * @param names names May be null. Arrays of names for the output variables.
   * @param input the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   */
  public SDVariable[] maxPoolWithArgmax(String[] names, SDVariable input,
      Pooling2DConfig Pooling2DConfig) {
    SDValidation.validateNumerical("maxPoolWithArgmax", "input", input);
    SDVariable[] out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPoolWithArgmax(sd,input, Pooling2DConfig).outputVariables();
    return sd.updateVariableNamesAndReferences(out, names);
  }

  /**
   * 2D Convolution layer operation - max pooling 2d <br>
   *
   * @param input the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   * @return output Result after applying max pooling on the input (NUMERIC type)
   */
  public SDVariable maxPooling2d(SDVariable input, Pooling2DConfig Pooling2DConfig) {
    SDValidation.validateNumerical("maxPooling2d", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D(sd,input, Pooling2DConfig).outputVariable();
  }

  /**
   * 2D Convolution layer operation - max pooling 2d <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                         (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param Pooling2DConfig Configuration Object
   * @return output Result after applying max pooling on the input (NUMERIC type)
   */
  public SDVariable maxPooling2d(String name, SDVariable input, Pooling2DConfig Pooling2DConfig) {
    SDValidation.validateNumerical("maxPooling2d", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling2D(sd,input, Pooling2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable maxPooling3d(SDVariable input, Pooling3DConfig Pooling3DConfig) {
    SDValidation.validateNumerical("maxPooling3d", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling3D(sd,input, Pooling3DConfig).outputVariable();
  }

  /**
   * 3D convolution layer operation - max pooling 3d operation.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input the input to average pooling 3d operation - 5d activations in NCDHW format
   *                         (shape [minibatch, channels, depth, height, width]) or NDHWC format
   *                         (shape [minibatch, depth, height, width, channels]) (NUMERIC type)
   * @param Pooling3DConfig Configuration Object
   * @return output Result after applying max pooling on the input (NUMERIC type)
   */
  public SDVariable maxPooling3d(String name, SDVariable input, Pooling3DConfig Pooling3DConfig) {
    SDValidation.validateNumerical("maxPooling3d", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.MaxPooling3D(sd,input, Pooling3DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights,
      SDVariable pointWeights, SDVariable bias, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("separableConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("separableConv2d", "depthWeights", depthWeights);
    SDValidation.validateNumerical("separableConv2d", "pointWeights", pointWeights);
    SDValidation.validateNumerical("separableConv2d", "bias", bias);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D(sd,layerInput, depthWeights, pointWeights, bias, Conv2DConfig).outputVariable();
  }

  /**
   * Separable 2D convolution operation with optional bias <br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                      (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param depthWeights Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param pointWeights Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]. May be null (NUMERIC type)
   * @param bias Optional bias, rank 1 with shape [outputChannels]. May be null. (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of separable convolution 2d operation (NUMERIC type)
   */
  public SDVariable separableConv2d(String name, SDVariable layerInput, SDVariable depthWeights,
      SDVariable pointWeights, SDVariable bias, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("separableConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("separableConv2d", "depthWeights", depthWeights);
    SDValidation.validateNumerical("separableConv2d", "pointWeights", pointWeights);
    SDValidation.validateNumerical("separableConv2d", "bias", bias);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D(sd,layerInput, depthWeights, pointWeights, bias, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights,
      SDVariable pointWeights, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("separableConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("separableConv2d", "depthWeights", depthWeights);
    SDValidation.validateNumerical("separableConv2d", "pointWeights", pointWeights);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D(sd,layerInput, depthWeights, pointWeights, null, Conv2DConfig).outputVariable();
  }

  /**
   * Separable 2D convolution operation with optional bias <br>
   *
   * @param name name May be null. Name for the output variable
   * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
   *                      (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param depthWeights Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier] (NUMERIC type)
   * @param pointWeights Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]. May be null (NUMERIC type)
   * @param Conv2DConfig Configuration Object
   * @return output result of separable convolution 2d operation (NUMERIC type)
   */
  public SDVariable separableConv2d(String name, SDVariable layerInput, SDVariable depthWeights,
      SDVariable pointWeights, Conv2DConfig Conv2DConfig) {
    SDValidation.validateNumerical("separableConv2d", "layerInput", layerInput);
    SDValidation.validateNumerical("separableConv2d", "depthWeights", depthWeights);
    SDValidation.validateNumerical("separableConv2d", "pointWeights", pointWeights);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D(sd,layerInput, depthWeights, pointWeights, null, Conv2DConfig).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable spaceToBatch(SDVariable x, int[] blocks, int[] paddingTop,
      int... paddingBottom) {
    SDValidation.validateNumerical("spaceToBatch", "x", x);
    Preconditions.checkArgument(blocks.length == 2, "blocks has incorrect size/length. Expected: blocks.length == 2, got %s", blocks.length);
    Preconditions.checkArgument(paddingTop.length == 2, "paddingTop has incorrect size/length. Expected: paddingTop.length == 2, got %s", paddingTop.length);
    Preconditions.checkArgument(paddingBottom.length == 2, "paddingBottom has incorrect size/length. Expected: paddingBottom.length == 2, got %s", paddingBottom.length);
    return new org.nd4j.linalg.api.ops.impl.transforms.custom.SpaceToBatch(sd,x, blocks, paddingTop, paddingBottom).outputVariable();
  }

  /**
   * Convolution 2d layer space to batch operation on 4d input.<br>
   * Increases input batch dimension by rearranging data from spatial dimensions into batch dimension <br>
   *
   * @param name name May be null. Name for the output variable
   * @param x Input variable. 4d input (NUMERIC type)
   * @param blocks Block size, in the height/width dimension (Size: Exactly(count=2))
   * @param paddingTop Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]] (Size: Exactly(count=2))
   * @param paddingBottom Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]] (Size: Exactly(count=2))
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable spaceToBatch(String name, SDVariable x, int[] blocks, int[] paddingTop,
      int... paddingBottom) {
    SDValidation.validateNumerical("spaceToBatch", "x", x);
    Preconditions.checkArgument(blocks.length == 2, "blocks has incorrect size/length. Expected: blocks.length == 2, got %s", blocks.length);
    Preconditions.checkArgument(paddingTop.length == 2, "paddingTop has incorrect size/length. Expected: paddingTop.length == 2, got %s", paddingTop.length);
    Preconditions.checkArgument(paddingBottom.length == 2, "paddingBottom has incorrect size/length. Expected: paddingBottom.length == 2, got %s", paddingBottom.length);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.transforms.custom.SpaceToBatch(sd,x, blocks, paddingTop, paddingBottom).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable spaceToDepth(SDVariable x, int blockSize, DataFormat dataFormat) {
    SDValidation.validateNumerical("spaceToDepth", "x", x);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.SpaceToDepth(sd,x, blockSize, dataFormat).outputVariable();
  }

  /**
   * Convolution 2d layer space to depth operation on 4d input.<br>
   * Increases input channels (reduced spatial dimensions) by rearranging data into a larger channels dimension<br>
   * Example: if input has shape [mb, 2, 4, 4] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]<br>
   * = [mb, 2, 4, 4] <br>
   *
   * @param name name May be null. Name for the output variable
   * @param x the input to depth to space pooling 2d operation - 4d activations in NCHW format
   *                    (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels]) (NUMERIC type)
   * @param blockSize  Block size, in the height/width dimension
   * @param dataFormat Data format: "NCHW" or "NHWC"
   * @return output Output variable (NUMERIC type)
   */
  public SDVariable spaceToDepth(String name, SDVariable x, int blockSize, DataFormat dataFormat) {
    SDValidation.validateNumerical("spaceToDepth", "x", x);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.SpaceToDepth(sd,x, blockSize, dataFormat).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Upsampling layer for 2D inputs.<br>
   * scale is used for both height and width dimensions. <br>
   *
   * @param input Input in NCHW format (NUMERIC type)
   * @param scale The scale for both height and width dimensions.
   * @return output Upsampled input (NUMERIC type)
   */
  public SDVariable upsampling2d(SDVariable input, int scale) {
    SDValidation.validateNumerical("upsampling2d", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d(sd,input, scale).outputVariable();
  }

  /**
   * Upsampling layer for 2D inputs.<br>
   * scale is used for both height and width dimensions. <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input in NCHW format (NUMERIC type)
   * @param scale The scale for both height and width dimensions.
   * @return output Upsampled input (NUMERIC type)
   */
  public SDVariable upsampling2d(String name, SDVariable input, int scale) {
    SDValidation.validateNumerical("upsampling2d", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d(sd,input, scale).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable upsampling2d(SDVariable input, int scaleH, int scaleW, boolean nchw) {
    SDValidation.validateNumerical("upsampling2d", "input", input);
    return new org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d(sd,input, scaleH, scaleW, nchw).outputVariable();
  }

  /**
   * 2D Convolution layer operation - Upsampling 2d <br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Input in NCHW format (NUMERIC type)
   * @param scaleH Scale to upsample in height dimension
   * @param scaleW Scale to upsample in width dimension
   * @param nchw If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format
   * @return output Upsampled input (NUMERIC type)
   */
  public SDVariable upsampling2d(String name, SDVariable input, int scaleH, int scaleW,
      boolean nchw) {
    SDValidation.validateNumerical("upsampling2d", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.layers.convolution.Upsampling2d(sd,input, scaleH, scaleW, nchw).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
