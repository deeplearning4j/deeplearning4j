/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;

import static org.nd4j.autodiff.samediff.ops.SDValidation.validateFloatingPoint;
import static org.nd4j.autodiff.samediff.ops.SDValidation.validateNumerical;

/**
 * SameDiff Convolutional Neural Network operations - CNN1d, 2d and 3d ops - as well as related functions.<br>
 * Accessible via {@link SameDiff#cnn()}<br>
 * See also {@link SDNN} (accessible via {@link SameDiff#nn()} for general neural network ops.<br>
 * See also {@link SDRNN} (accessible via {@link SameDiff#rnn()} for recurrent neural network ops.<br>
 *
 * @author Alex Black
 */
public class SDCNN extends SDOps {

    public SDCNN(SameDiff sameDiff) {
        super(sameDiff);
    }

    /**
     * 2D Convolution layer operation - average pooling 2d
     *
     * @param input           the input to average pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                        (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param pooling2DConfig the configuration for
     * @return Result after applying average pooling on the input
     */
    public SDVariable avgPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        return avgPooling2d(null, input, pooling2DConfig);
    }

    /**
     * 2D Convolution layer operation - average pooling 2d
     *
     * @param name            name of the operation in SameDiff
     * @param input           the input to average pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                        (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param pooling2DConfig the configuration
     * @return Result after applying average pooling on the input
     */
    public SDVariable avgPooling2d(String name, SDVariable input, Pooling2DConfig pooling2DConfig) {
        validateFloatingPoint("avgPooling2d", input);
        SDVariable ret = f().avgPooling2d(input, pooling2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * 3D convolution layer operation - average pooling 3d
     *
     * @param input           the input to average pooling 3d operation - 5d activations in NCDHW format
     *                        (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                        (shape [minibatch, depth, height, width, channels])
     * @param pooling3DConfig the configuration
     * @return Result after applying average pooling on the input
     */
    public SDVariable avgPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        return avgPooling3d(null, input, pooling3DConfig);
    }

    /**
     * 3D convolution layer operation - average pooling 3d
     *
     * @param name            name of the operation in SameDiff
     * @param input           the input to average pooling 3d operation - 5d activations in NCDHW format
     *                        (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                        (shape [minibatch, depth, height, width, channels])
     * @param pooling3DConfig the configuration
     * @return Result after applying average pooling on the input
     */
    public SDVariable avgPooling3d(String name, SDVariable input, Pooling3DConfig pooling3DConfig) {
        validateFloatingPoint("avgPooling3d", input);
        SDVariable ret = f().avgPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #batchToSpace(String, SDVariable, int[], int[][])
     */
    public SDVariable batchToSpace(SDVariable x, int[] blocks, int[][] crops) {
        return batchToSpace(null, x, blocks, crops);
    }

    /**
     * Convolution 2d layer batch to space operation on 4d input.
     * Reduces input batch dimension by rearranging data into a larger spatial dimensions
     *
     * @param name   Output variable name
     * @param x      Input variable. 4d input
     * @param blocks Block size, in the height/width dimension
     * @param crops  Optional 2d int[] array: values [[crop top, crop bottom], [crop left, crop right]]
     * @return Output variable
     * @see #spaceToBatch(String, SDVariable, int[], int[][])
     */
    public SDVariable batchToSpace(String name, SDVariable x, int[] blocks, int[][] crops) {
        validateNumerical("batchToSpace", x);
        SDVariable ret = f().batchToSpace(x, blocks, crops);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * col2im operation for use in 2D convolution operations. Outputs a 4d array with shape
     * [minibatch, inputChannels, height, width]
     *
     * @param in     Input - rank 6 input with shape [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]
     * @param config Convolution configuration for the col2im operation
     * @return Col2Im output variable
     */
    public SDVariable col2Im(SDVariable in, Conv2DConfig config) {
        return col2Im(null, in, config);
    }

    /**
     * col2im operation for use in 2D convolution operations. Outputs a 4d array with shape
     * [minibatch, inputChannels, height, width]
     *
     * @param name   Name of the output variable
     * @param in     Input - rank 6 input with shape [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]
     * @param config Convolution configuration for the col2im operation
     * @return Col2Im output variable
     */
    public SDVariable col2Im(String name, SDVariable in, Conv2DConfig config) {
        SDVariable ret = f().col2Im(in, config);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * 1D Convolution layer operation - Conv1d
     *
     * @param input        the input array/activations for the conv1d op
     * @param weights      weights for conv1d op - rank 3 array with values [kernelSize, inputChannels, outputChannels]
     * @param conv1DConfig the configuration
     * @return
     */
    public SDVariable conv1d(SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        return conv1d(null, input, weights, conv1DConfig);
    }

    /**
     * Conv1d operation.
     *
     * @param name         name of the operation in SameDiff
     * @param input        the inputs to conv1d
     * @param weights      weights for conv1d op - rank 3 array with values [kernelSize, inputChannels, outputChannels]
     * @param conv1DConfig the configuration
     * @return
     */
    public SDVariable conv1d(String name, SDVariable input, SDVariable weights, Conv1DConfig conv1DConfig) {
        validateFloatingPoint("conv1d", input);
        validateFloatingPoint("conv1d", weights);
        SDVariable ret = f().conv1d(input, weights, conv1DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * 2D Convolution operation (without bias)
     *
     * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                   (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param weights    Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels]
     * @param config     Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable conv2d(SDVariable layerInput, SDVariable weights, Conv2DConfig config) {
        return conv2d(layerInput, weights, null, config);
    }

    /**
     * 2D Convolution operation with optional bias
     *
     * @param layerInput the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                   (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param weights    Weights for the convolution operation. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, outputChannels]
     * @param bias       Optional 1D bias array with shape [outputChannels]. May be null.
     * @param config     Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable conv2d(SDVariable layerInput, SDVariable weights, SDVariable bias, Conv2DConfig config) {
        validateFloatingPoint("conv2d", "input", layerInput);
        validateFloatingPoint("conv2d", "weights", weights);
        validateFloatingPoint("conv2d", "bias", bias);
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = weights;
        if (bias != null)
            arr[2] = bias;
        return conv2d(arr, config);
    }

    /**
     * 2D Convolution operation with optional bias
     *
     * @param inputs an array with either 2 elements (layerInput, weights) or 3 elements (layerInput, weights, bias) as
     *               described in {@link #conv2d(SDVariable, SDVariable, SDVariable, Conv2DConfig)}
     * @param config Conv2DConfig configuration
     * @return result of convolution 2d operation
     */
    public SDVariable conv2d(SDVariable[] inputs, Conv2DConfig config) {
        return conv2d(null, inputs, config);
    }

    /**
     * 2D Convolution operation with optional bias
     *
     * @param name   Name of the output SDVariable
     * @param inputs an array with either 2 elements (layerInput, weights) or 3 elements (layerInput, weights, bias) as
     *               described in {@link #conv2d(SDVariable, SDVariable, SDVariable, Conv2DConfig)}
     * @param config Conv2DConfig configuration
     * @return result of convolution 2d operation
     */
    public SDVariable conv2d(String name, SDVariable[] inputs, Conv2DConfig config) {
        for(SDVariable v : inputs)
            validateNumerical("conv2d", v);
        SDVariable ret = f().conv2d(inputs, config);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Convolution 3D operation without bias
     *
     * @param input        the input to average pooling 3d operation - 5d activations in NCDHW format
     *                     (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                     (shape [minibatch, depth, height, width, channels])
     * @param weights      Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels].
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(SDVariable input, SDVariable weights, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, null, conv3DConfig);
    }

    /**
     * Convolution 3D operation with optional bias
     *
     * @param name         Name of the output variable
     * @param input        the input to average pooling 3d operation - 5d activations in NCDHW format
     *                     (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                     (shape [minibatch, depth, height, width, channels])
     * @param weights      Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels].
     * @param bias         Optional 1D bias array with shape [outputChannels]. May be null.
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(String name, SDVariable input, SDVariable weights, SDVariable bias, Conv3DConfig conv3DConfig) {
        validateFloatingPoint("conv3d", "input", input);
        validateFloatingPoint("conv3d", "weights", weights);
        validateFloatingPoint("conv3d", "bias", bias);
        SDVariable[] args;
        if (bias == null) {
            args = new SDVariable[]{input, weights};
        } else {
            args = new SDVariable[]{input, weights, bias};
        }
        SDVariable ret = f().conv3d(args, conv3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Convolution 3D operation with optional bias
     *
     * @param input        the input to average pooling 3d operation - 5d activations in NCDHW format
     *                     (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                     (shape [minibatch, depth, height, width, channels])
     * @param weights      Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels].
     * @param bias         Optional 1D bias array with shape [outputChannels]. May be null.
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(SDVariable input, SDVariable weights, SDVariable bias, Conv3DConfig conv3DConfig) {
        return conv3d(null, input, weights, bias, conv3DConfig);
    }

    /**
     * Convolution 3D operation without bias
     *
     * @param name         Name of the output variable
     * @param input        the input to average pooling 3d operation - 5d activations in NCDHW format
     *                     (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                     (shape [minibatch, depth, height, width, channels])
     * @param weights      Weights for conv3d. Rank 5 with shape [kernelDepth, kernelHeight, kernelWidth, inputChannels, outputChannels].
     * @param conv3DConfig the configuration
     * @return Conv3d output variable
     */
    public SDVariable conv3d(String name, SDVariable input, SDVariable weights, Conv3DConfig conv3DConfig) {
        return conv3d(name, input, weights, null, conv3DConfig);
    }

    /**
     * 2D deconvolution operation without bias
     *
     * @param layerInput     the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format
     *                       (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param weights        Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth].
     * @param deconv2DConfig DeConv2DConfig configuration
     * @return result of deconv2d op
     */
    public SDVariable deconv2d(SDVariable layerInput, SDVariable weights, DeConv2DConfig deconv2DConfig) {
        return deconv2d(layerInput, weights, null, deconv2DConfig);
    }

    /**
     * 2D deconvolution operation with optional bias
     *
     * @param layerInput     the input to deconvolution 2d operation - 4d CNN (image) activations in NCHW format
     *                       (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param weights        Weights for the 2d deconvolution operation. 4 dimensions with format [inputChannels, outputChannels, kernelHeight, kernelWidth].
     * @param bias           Optional 1D bias array with shape [outputChannels]. May be null.
     * @param deconv2DConfig DeConv2DConfig configuration
     * @return result of deconv2d op
     */
    public SDVariable deconv2d(SDVariable layerInput, SDVariable weights, SDVariable bias, DeConv2DConfig deconv2DConfig) {
        validateFloatingPoint("deconv2d", "input", layerInput);
        validateFloatingPoint("deconv2d", "weights", weights);
        validateFloatingPoint("deconv2d", "bias", bias);
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = weights;
        if (bias != null)
            arr[2] = bias;
        return deconv2d(arr, deconv2DConfig);
    }

    /**
     * 2D deconvolution operation with or without optional bias
     *
     * @param inputs         Inputs to the deconvolution 2d operation - input array of length 2 (layerInput, weights)
     *                       or length 3 (layerInput, weights, bias) as described in {@link #deconv2d(SDVariable[], DeConv2DConfig)}
     * @param deconv2DConfig the configuration
     * @return result of deconv2d op
     */
    public SDVariable deconv2d(SDVariable[] inputs, DeConv2DConfig deconv2DConfig) {
        return deconv2d(null, inputs, deconv2DConfig);
    }

    /**
     * 2D deconvolution operation with or without optional bias
     *
     * @param name           Name of the output variable
     * @param inputs         Inputs to the deconvolution 2d operation - input array of length 2 (layerInput, weights)
     *                       or length 3 (layerInput, weights, bias) as described in {@link #deconv2d(SDVariable[], DeConv2DConfig)}
     * @param deconv2DConfig the configuration
     * @return result of deconv2d op
     */
    public SDVariable deconv2d(String name, SDVariable[] inputs, DeConv2DConfig deconv2DConfig) {
        for(SDVariable v : inputs)
            validateNumerical("deconv2d", v);
        SDVariable ret = f().deconv2d(inputs, deconv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * 3D CNN deconvolution operation with or without optional bias
     *
     * @param name    Name of the output variable
     * @param input   Input array - shape [bS, iD, iH, iW, iC] (NDHWC) or [bS, iC, iD, iH, iW] (NCDHW)
     * @param weights Weights array - shape [kD, kH, kW, oC, iC]
     * @param bias    Bias array - optional, may be null. If non-null, must have shape [outputChannels]
     * @param config  Configuration
     */
    public SDVariable deconv3d(String name, SDVariable input, SDVariable weights, SDVariable bias, DeConv3DConfig config) {
        validateFloatingPoint("conv3d", input);
        validateFloatingPoint("conv3d", weights);
        validateFloatingPoint("conv3d", bias);
        SDVariable ret = f().deconv3d(input, weights, bias, config);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Convolution 2d layer batch to space operation on 4d input.<br>
     * Reduces input channels dimension by rearranging data into a larger spatial dimensions<br>
     * Example: if input has shape [mb, 8, 2, 2] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]
     * = [mb, 2, 4, 4]
     *
     * @param x          the input to depth to space pooling 2d operation - 4d activations in NCHW format
     *                   (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param blockSize  Block size, in the height/width dimension
     * @param dataFormat Data format: "NCHW" or "NHWC"
     * @return Output variable
     */
    public SDVariable depthToSpace(SDVariable x, int blockSize, String dataFormat) {
        return depthToSpace(null, x, blockSize, dataFormat);
    }

    /**
     * Convolution 2d layer batch to space operation on 4d input.<br>
     * Reduces input channels dimension by rearranging data into a larger spatial dimensions<br>
     * Example: if input has shape [mb, 8, 2, 2] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]
     * = [mb, 2, 4, 4]
     *
     * @param name       Output variable name
     * @param x          the input to depth to space pooling 2d operation - 4d activations in NCHW format
     *                   (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param blockSize  Block size, in the height/width dimension
     * @param dataFormat Data format: "NCHW" or "NHWC"
     * @return Output variable
     * @see #depthToSpace(String, SDVariable, int, String)
     */
    public SDVariable depthToSpace(String name, SDVariable x, int blockSize, String dataFormat) {
        SDVariable ret = f().depthToSpace(x, blockSize, dataFormat);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Depth-wise 2D convolution operation without bias
     *
     * @param layerInput   the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                     (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param depthWeights Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]
     * @param config       Conv2DConfig configuration
     * @return result of conv2d op
     */
    public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights, Conv2DConfig config) {
        return depthWiseConv2d(layerInput, depthWeights, null, config);
    }

    /**
     * Depth-wise 2D convolution operation with optional bias
     *
     * @param layerInput   the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                     (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param depthWeights Depth-wise conv2d weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]
     * @param bias         Optional 1D bias array with shape [outputChannels]. May be null.
     * @param config       Conv2DConfig configuration
     * @return result of depthwise conv2d op
     */
    public SDVariable depthWiseConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable bias, Conv2DConfig config) {
        validateFloatingPoint("depthwiseConv2d", "input", layerInput);
        validateFloatingPoint("depthwiseConv2d", "depth weights", depthWeights);
        validateFloatingPoint("depthwiseConv2d", "bias", bias);
        SDVariable[] arr = new SDVariable[bias == null ? 2 : 3];
        arr[0] = layerInput;
        arr[1] = depthWeights;
        if (bias != null)
            arr[2] = bias;
        return depthWiseConv2d(arr, config);
    }

    /**
     * Depth-wise convolution 2D operation.
     *
     * @param inputs            the inputs to depth-wise conv2d. An array with either 2 elements (layerInput, depthWeights)
     *                          or 3 elements (layerInput, depthWeights, bias) as described in
     *                          {@link #depthWiseConv2d(SDVariable, SDVariable, SDVariable, Conv2DConfig)}
     * @param depthConv2DConfig the configuration
     * @return result of depthwise conv2d op
     */
    public SDVariable depthWiseConv2d(SDVariable[] inputs, Conv2DConfig depthConv2DConfig) {
        return depthWiseConv2d(null, inputs, depthConv2DConfig);
    }

    /**
     * Depth-wise convolution 2D operation.
     *
     * @param name              name of the output variable
     * @param inputs            the inputs to depth-wise conv2d. An array with either 2 elements (layerInput, depthWeights)
     *                          or 3 elements (layerInput, depthWeights, bias) as described in
     *                          {@link #depthWiseConv2d(SDVariable, SDVariable, SDVariable, Conv2DConfig)}
     * @param depthConv2DConfig the configuration
     * @return result of depthwise conv2d op
     */
    public SDVariable depthWiseConv2d(String name, SDVariable[] inputs, Conv2DConfig depthConv2DConfig) {
        for(SDVariable v : inputs)
            validateFloatingPoint("depthWiseConv2d", v);
        SDVariable ret = f().depthWiseConv2d(inputs, depthConv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * TODO doc string
     *
     * @param df
     * @param weights
     * @param strides
     * @param rates
     * @param isSameMode
     * @return
     */
    public SDVariable dilation2D(SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        return dilation2D(null, df, weights, strides, rates, isSameMode);
    }

    /**
     * TODO doc string
     *
     * @param name
     * @param df
     * @param weights
     * @param strides
     * @param rates
     * @param isSameMode
     * @return
     */
    public SDVariable dilation2D(String name, SDVariable df, SDVariable weights, int[] strides,
                                 int[] rates, boolean isSameMode) {
        SDVariable ret = f().dilation2D(df, weights, strides, rates, isSameMode);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * Extract image patches
     *
     * @param name     Name of the output variable
     * @param input    Input array. Must be rank 4, with shape [minibatch, height, width, channels]
     * @param kH       Kernel height
     * @param kW       Kernel width
     * @param sH       Stride height
     * @param sW       Stride width
     * @param rH       Rate height
     * @param rW       Rate width
     * @param sameMode If true: use same mode padding. If false
     * @return
     */
    public SDVariable extractImagePatches(String name, SDVariable input, int kH, int kW, int sH, int sW, int rH, int rW, boolean sameMode) {
        SDVariable ret = f().extractImagePatches(input, kH, kW, sH, sW, rH, rW, sameMode);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * im2col operation for use in 2D convolution operations. Outputs a 6d array with shape
     * [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]
     *
     * @param in     Input - rank 4 input with shape [minibatch, inputChannels, height, width]
     * @param config Convolution configuration for the im2col operation
     * @return Im2Col output variable
     */
    public SDVariable im2Col(SDVariable in, Conv2DConfig config) {
        return im2Col(null, in, config);
    }

    /**
     * im2col operation for use in 2D convolution operations. Outputs a 6d array with shape
     * [minibatch, inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth]
     *
     * @param name   Name of the output variable
     * @param in     Input - rank 4 input with shape [minibatch, inputChannels, height, width]
     * @param config Convolution configuration for the im2col operation
     * @return Im2Col output variable
     */
    public SDVariable im2Col(String name, SDVariable in, Conv2DConfig config) {
        SDVariable ret = f().im2Col(in, config);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * 2D convolution layer operation - local response normalization
     *
     * @param inputs    the inputs to lrn
     * @param lrnConfig the configuration
     * @return
     */
    public SDVariable localResponseNormalization(SDVariable inputs, LocalResponseNormalizationConfig lrnConfig) {
        return localResponseNormalization(null, inputs, lrnConfig);
    }

    /**
     * 2D convolution layer operation - local response normalization
     *
     * @param name      name of the operation in SameDiff
     * @param input     the inputs to lrn
     * @param lrnConfig the configuration
     * @return
     */
    public SDVariable localResponseNormalization(String name, SDVariable input,
                                                 LocalResponseNormalizationConfig lrnConfig) {
        validateFloatingPoint("local response normalization", input);
        SDVariable ret = f().localResponseNormalization(input, lrnConfig);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * 2D Convolution layer operation - max pooling 2d
     *
     * @param input           the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                        (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param pooling2DConfig the configuration
     * @return Result after applying max pooling on the input
     */
    public SDVariable maxPooling2d(SDVariable input, Pooling2DConfig pooling2DConfig) {
        return maxPooling2d(null, input, pooling2DConfig);
    }

    /**
     * 2D Convolution layer operation - max pooling 2d
     *
     * @param name            name of the operation in SameDiff
     * @param input           the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                        (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param pooling2DConfig the configuration
     * @return Result after applying max pooling on the input
     */
    public SDVariable maxPooling2d(String name, SDVariable input, Pooling2DConfig pooling2DConfig) {
        validateNumerical("maxPooling2d", input);
        SDVariable ret = f().maxPooling2d(input, pooling2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * 3D convolution layer operation - max pooling 3d operation.
     *
     * @param input           the input to average pooling 3d operation - 5d activations in NCDHW format
     *                        (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                        (shape [minibatch, depth, height, width, channels])
     * @param pooling3DConfig the configuration
     * @return Result after applying max pooling on the input
     */
    public SDVariable maxPooling3d(SDVariable input, Pooling3DConfig pooling3DConfig) {
        return maxPooling3d(null, input, pooling3DConfig);
    }

    /**
     * 3D convolution layer operation - max pooling 3d operation.
     *
     * @param name            name of the operation in SameDiff
     * @param input           the input to average pooling 3d operation - 5d activations in NCDHW format
     *                        (shape [minibatch, channels, depth, height, width]) or NDHWC format
     *                        (shape [minibatch, depth, height, width, channels])
     * @param pooling3DConfig the configuration
     * @return Result after applying max pooling on the input
     */
    public SDVariable maxPooling3d(String name, SDVariable input, Pooling3DConfig pooling3DConfig) {
        validateNumerical("maxPooling3d", input);
        SDVariable ret = f().maxPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * Separable 2D convolution operation without bias
     *
     * @param layerInput   the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                     (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param depthWeights Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]
     * @param pointWeights Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]
     *                     May be null
     * @param config       Conv2DConfig configuration
     * @return result of separable convolution 2d operation
     */
    public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable pointWeights,
                                      Conv2DConfig config) {
        return separableConv2d(layerInput, depthWeights, pointWeights, null, config);
    }

    /**
     * Separable 2D convolution operation with optional bias
     *
     * @param layerInput   the input to max pooling 2d operation - 4d CNN (image) activations in NCHW format
     *                     (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param depthWeights Separable conv2d depth weights. 4 dimensions with format [kernelHeight, kernelWidth, inputChannels, depthMultiplier]
     * @param pointWeights Point weights, rank 4 with format [1, 1, inputChannels*depthMultiplier, outputChannels]
     *                     May be null
     * @param bias         Optional bias, rank 1 with shape [outputChannels]. May be null.
     * @param config       Conv2DConfig configuration
     * @return result of separable convolution 2d operation
     */
    public SDVariable separableConv2d(SDVariable layerInput, SDVariable depthWeights, SDVariable pointWeights,
                                      SDVariable bias, Conv2DConfig config) {
        validateFloatingPoint("separableConv2d", "input", layerInput);
        validateFloatingPoint("separableConv2d", "depthWeights", depthWeights);
        validateFloatingPoint("separableConv2d", "pointWeights", pointWeights);
        validateFloatingPoint("separableConv2d", "bias", bias);
        SDVariable[] arr = new SDVariable[bias == null ? 3 : 4];
        arr[0] = layerInput;
        arr[1] = depthWeights;
        arr[2] = pointWeights;
        if (bias != null)
            arr[3] = bias;
        return sconv2d(arr, config);
    }

    /**
     * Separable 2D convolution operation with/without optional bias
     *
     * @param inputs       the inputs to separable conv2 operation. Should be length 3 (layerInput, depthWeights, pointWeights)
     *                     or length 4 (layerInput, depthWeights, pointWeights, bias) as described in {@link #separableConv2d(SDVariable, SDVariable, SDVariable, SDVariable, Conv2DConfig)}
     * @param conv2DConfig the configuration
     * @return result of separable convolution 2d operation
     */
    public SDVariable sconv2d(SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        return sconv2d(null, inputs, conv2DConfig);
    }

    /**
     * Separable 2D convolution operation with/without optional bias
     *
     * @param name         name of the output variable
     * @param inputs       the inputs to separable conv2 operation. Should be length 3 (layerInput, depthWeights, pointWeights)
     *                     or length 4 (layerInput, depthWeights, pointWeights, bias) as described in {@link #separableConv2d(SDVariable, SDVariable, SDVariable, SDVariable, Conv2DConfig)}
     * @param conv2DConfig the configuration
     * @return result of separable convolution 2d operation
     */
    public SDVariable sconv2d(String name, SDVariable[] inputs, Conv2DConfig conv2DConfig) {
        for(SDVariable v : inputs)
            validateFloatingPoint("sconv2d", v);
        SDVariable ret = f().sconv2d(inputs, conv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * @see #spaceToBatch(String, SDVariable, int[], int[][])
     */
    public SDVariable spaceToBatch(SDVariable x, int[] blocks, int[][] padding) {
        return spaceToBatch(null, x, blocks, padding);
    }

    /**
     * Convolution 2d layer space to batch operation on 4d input.
     * Increases input batch dimension by rearranging data from spatial dimensions into batch dimension
     *
     * @param name    Output variable name
     * @param x       Input variable. 4d input
     * @param blocks  Block size, in the height/width dimension
     * @param padding Optional 2d int[] array for padding the result: values [[pad top, pad bottom], [pad left, pad right]]
     * @return Output variable
     * @see #batchToSpace(String, SDVariable, int[], int[][])
     */
    public SDVariable spaceToBatch(String name, SDVariable x, int[] blocks, int[][] padding) {
        SDVariable ret = f().spaceToBatch(x, blocks, padding);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #spaceToDepth(String, SDVariable, int, String)
     */
    public SDVariable spaceToDepth(SDVariable x, int blockSize, String dataFormat) {
        return spaceToDepth(null, x, blockSize, dataFormat);
    }

    /**
     * Convolution 2d layer space to depth operation on 4d input.<br>
     * Increases input channels (reduced spatial dimensions) by rearranging data into a larger channels dimension<br>
     * Example: if input has shape [mb, 2, 4, 4] and block size is 2, then output size is [mb, 8/(2*2), 2*2, 2*2]
     * = [mb, 2, 4, 4]
     *
     * @param name       Output variable name
     * @param x          the input to depth to space pooling 2d operation - 4d activations in NCHW format
     *                   (shape [minibatch, channels, height, width]) or NHWC format (shape [minibatch, height, width, channels])
     * @param blockSize  Block size, in the height/width dimension
     * @param dataFormat Data format: "NCHW" or "NHWC"
     * @return Output variable
     * @see #depthToSpace(String, SDVariable, int, String)
     */
    public SDVariable spaceToDepth(String name, SDVariable x, int blockSize, String dataFormat) {
        SDVariable ret = f().spaceToDepth(x, blockSize, dataFormat);
        return updateVariableNameAndReference(ret, name);
    }


    /**
     * 2D Convolution layer operation - Upsampling 2d with same scale for both dimensions. NCHW input format.
     *
     * @param input Input - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width])
     * @param scale Scale to upsample in both H and W dimensions
     * @return Upsampled input
     */
    public SDVariable upsampling2d(SDVariable input, int scale) {
        return upsampling2d(null, input, true, scale, scale);
    }

    /**
     * 2D Convolution layer operation - Upsampling 2d
     *
     * @param input  Input, in NCHW format
     * @param nchw   If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format
     * @param scaleH Scale to upsample in height dimension
     * @param scaleW Scale to upsample in width dimension
     * @return Upsampled input
     */
    public SDVariable upsampling2d(String name, SDVariable input, boolean nchw, int scaleH, int scaleW) {
        SDVariable ret = f().upsampling2d(input, nchw, scaleH, scaleW);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * 2D Convolution layer operation - Upsampling 2d with same scale for both dimensions. NCHW input format.
     *
     * @param input Input - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width])
     * @param scale Scale to upsample in both H and W dimensions
     * @return Upsampled input
     */
    public SDVariable upsampling2d(String name, SDVariable input, int scale) {
        return upsampling2d(name, input, true, scale, scale);
    }

    /**
     * 2D Convolution layer operation - Upsampling 2d
     *
     * @param input  Input - 4d CNN (image) activations in NCHW format (shape [minibatch, channels, height, width])
     *               or NHWC format (shape [minibatch, height, width, channels])
     * @param nchw   If true: input is in NCHW (minibatch, channels, height, width) format. False: NHWC format
     * @param scaleH Scale to upsample in height dimension
     * @param scaleW Scale to upsample in width dimension
     * @return Upsampled input
     */
    public SDVariable upsampling2d(SDVariable input, boolean nchw, int scaleH, int scaleW) {
        return upsampling2d(null, input, nchw, scaleH, scaleW);
    }
}
