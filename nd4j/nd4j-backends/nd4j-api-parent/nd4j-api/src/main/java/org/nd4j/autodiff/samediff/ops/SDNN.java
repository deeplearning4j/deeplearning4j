package org.nd4j.autodiff.samediff.ops;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.*;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.GRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.LSTMCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRU;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.SRUCell;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.GRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUCellConfiguration;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.SRUConfiguration;

public class SDNN extends SDOps {
    public SDNN(SameDiff sameDiff) {
        super(sameDiff);
    }

    //TODO also add some math ops here: tanh, sigmoid, etc

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
        SDVariable ret = f().avgPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Batch norm operation.
     *
     * @see #batchNorm(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, double, int...)
     */
    public SDVariable batchNorm(SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta, double epsilon, int... axis) {
        return batchNorm(null, input, mean, variance, gamma, beta, true, true, epsilon, axis);
    }

    /**
     * Batch normalization with optional application of gamma/beta args.
     * See {@link #batchNorm(String, SDVariable, SDVariable, SDVariable, SDVariable, SDVariable, double, int...)}
     */
    public SDVariable batchNorm(String name, SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta, boolean applyGamma, boolean applyBeta, double epsilon, int... axis) {
        SDVariable res = f().batchNorm(input, mean, variance, gamma, beta, applyGamma, applyBeta, epsilon, axis);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Neural network batch normalization operation.<br>
     * For details, see <a href="http://arxiv.org/abs/1502.03167">http://arxiv.org/abs/1502.03167</a>
     *
     * @param name     Name of the output variable
     * @param input    Input variable.
     * @param mean     Mean value. For 1d axis, this should match input.size(axis)
     * @param variance Variance value. For 1d axis, this should match input.size(axis)
     * @param gamma    Gamma value. For 1d axis, this should match input.size(axis)
     * @param beta     Beta value. For 1d axis, this should match input.size(axis)
     * @param epsilon  Epsilon constant for numerical stability (to avoid division by 0)
     * @param axis     For 2d CNN activations: 1 for NCHW format activations, or 3 for NHWC format activations.<br>
     *                 For 3d CNN activations: 1 for NCDHW format, 4 for NDHWC<br>
     *                 For 1d/RNN activations: 1 for NCW format, 2 for NWC
     * @return Output variable for batch normalization
     */
    public SDVariable batchNorm(String name, SDVariable input, SDVariable mean,
                                SDVariable variance, SDVariable gamma,
                                SDVariable beta, double epsilon, int... axis) {
        return batchNorm(name, input, mean, variance, gamma, beta, true, true, epsilon, axis);
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
        SDVariable ret = f().batchToSpace(x, blocks, crops);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * @see #biasAdd(String, SDVariable, SDVariable)
     */
    public SDVariable biasAdd(SDVariable input, SDVariable bias) {
        return biasAdd(null, input, bias);
    }

    /**
     * Bias addition operation: a special case of addition, typically used with CNN 4D activations and a 1D bias vector
     *
     * @param name  Name of the output variable
     * @param input 4d input variable
     * @param bias  1d bias
     * @return Output variable
     */
    public SDVariable biasAdd(String name, SDVariable input, SDVariable bias) {
        SDVariable ret = f().biasAdd(input, bias);
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
     * @param input                  Input
     * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
     * @return
     */
    public SDVariable dropout(SDVariable input, double inputRetainProbability) {
        return dropout(null, input, inputRetainProbability);
    }

    /**
     * @param input                  Input
     * @param inputRetainProbability Probability of retaining an input (set to 0 with probability 1-p)
     * @return
     */
    public SDVariable dropout(String name, SDVariable input, double inputRetainProbability) {
        SDVariable res = f().dropout(input, inputRetainProbability);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Element-wise exponential linear unit (ELU) function:<br>
     * out = x if x > 0<br>
     * out = a * (exp(x) - 1) if x <= 0<br>
     * with constant a = 1.0
     * <p>
     * See: <a href="http://arxiv.org/abs/1511.07289">http://arxiv.org/abs/1511.07289</a>
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable elu(SDVariable x) {
        return elu(null, x);
    }

    /**
     * Element-wise exponential linear unit (ELU) function:<br>
     * out = x if x > 0<br>
     * out = a * (exp(x) - 1) if x <= 0<br>
     * with constant a = 1.0
     * <p>
     * See: <a href="http://arxiv.org/abs/1511.07289">http://arxiv.org/abs/1511.07289</a>
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable elu(String name, SDVariable x) {
        SDVariable result = f().elu(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise derivative exponential linear unit (ELU) function, dOut/dIn given input.
     * {@link #elu(SDVariable)}
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable eluDerivative(SDVariable x) {
        return eluDerivative(null, x);
    }

    /**
     * Element-wise derivative exponential linear unit (ELU) function, dOut/dIn given input.
     * {@link #elu(SDVariable)}
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable eluDerivative(String name, SDVariable x) {
        SDVariable result = f().eluDerivative(x);
        return updateVariableNameAndReference(result, name);
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
     * GELU activation function - Gaussian Error Linear Units<br>
     * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a>
     * This method uses the sigmoid approximation
     *
     * @param x Input
     * @return Output variable - GELU applied to the input
     */
    public SDVariable gelu(SDVariable x) {
        return gelu(null, x);
    }

    /**
     * GELU activation function - Gaussian Error Linear Units<br>
     * For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a>
     * This method uses the sigmoid approximation
     *
     * @param name Name of the output variable. May be null.
     * @param x    Input
     * @return Output variable - GELU applied to the input
     */
    public SDVariable gelu(String name, SDVariable x) {
        SDVariable ret = f().gelu(x, false);    //Defaults to si
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * The gru cell
     *
     * @param configuration teh configuration to use
     * @return
     */
    public SDVariable gru(GRUCellConfiguration configuration) {
        return new GRUCell(this, configuration).outputVariables()[0];
    }

    /**
     * The gru cell
     *
     * @param baseName      the base name for the gru cell
     * @param configuration teh configuration to use
     * @return
     */
    public SDVariable gru(String baseName, GRUCellConfiguration configuration) {
        return new GRUCell(this, configuration).outputVariables(baseName)[0];
    }

    /**
     * Element-wise hard sigmoid function:<br>
     * out[i] = 0 if in[i] <= -2.5<br>
     * out[1] = 0.2*in[i]+0.5 if -2.5 < in[i] < 2.5<br>
     * out[i] = 1 if in[i] >= 2.5<br>
     *
     * @param in Input variable
     * @return Output variable
     */
    public SDVariable hardSigmoid(SDVariable in) {
        return hardSigmoid(null, in);
    }

    /**
     * Element-wise hard sigmoid function:<br>
     * out[i] = 0 if in[i] <= -2.5<br>
     * out[1] = 0.2*in[i]+0.5 if -2.5 < in[i] < 2.5<br>
     * out[i] = 1 if in[i] >= 2.5<br>
     *
     * @param name Name of the output variable
     * @param in   Input variable
     * @return Output variable
     */
    public SDVariable hardSigmoid(String name, SDVariable in) {
        SDVariable ret = f().hardSigmoid(in);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise hard tanh function:<br>
     * out[i] = -1 if in[i] <= -1<br>
     * out[1] = in[i] if -1 < in[i] < 1<br>
     * out[i] = 1 if in[i] >= 1<br>
     *
     * @param in Input variable
     * @return Output variable
     */
    public SDVariable hardTanh(SDVariable in) {
        return hardTanh(null, in);
    }

    /**
     * Element-wise hard tanh function:<br>
     * out[i] = -1 if in[i] <= -1<br>
     * out[1] = in[i] if -1 < in[i] < 1<br>
     * out[i] = 1 if in[i] >= 1<br>
     *
     * @param name Output variable name
     * @param in   Input variable
     * @return Output variable
     */
    public SDVariable hardTanh(String name, SDVariable in) {
        SDVariable result = f().hardTanh(in);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Derivative (dOut/dIn) of the element-wise hard Tanh function - {@link #hardTanh(SDVariable)}
     *
     * @param x Input
     * @return Output variable
     */
    public SDVariable hardTanhDerivative(SDVariable x) {
        return hardTanhDerivative(null, x);
    }

    /**
     * Derivative (dOut/dIn) of the element-wise hard Tanh function - {@link #hardTanh(SDVariable)}
     *
     * @param name Output variable name
     * @param x    Input
     * @return Output variable
     */
    public SDVariable hardTanhDerivative(String name, SDVariable x) {
        SDVariable result = f().hardTanhDerivative(x);
        return updateVariableNameAndReference(result, name);
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
     * Element-wise leaky ReLU function:<br>
     * out = x if x >= 0.0<br>
     * out = alpha * x if x < cutoff<br>
     * Alpha value is most commonly set to 0.01
     *
     * @param x     Input variable
     * @param alpha Cutoff - usually 0.0
     * @return Output variable
     */
    public SDVariable leakyRelu(SDVariable x, double alpha) {
        return leakyRelu(null, x, alpha);
    }

    /**
     * Element-wise leaky ReLU function:<br>
     * out = x if x >= 0.0<br>
     * out = alpha * x if x < cutoff<br>
     * Alpha value is most commonly set to 0.01
     *
     * @param x     Input variable
     * @param alpha Cutoff - usually 0.0
     * @return Output variable
     */
    public SDVariable leakyRelu(String name, SDVariable x, double alpha) {
        SDVariable result = f().leakyRelu(x, alpha);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Leaky ReLU derivative: dOut/dIn given input.<br>
     * See {@link #leakyRelu(String, SDVariable, double)}
     *
     * @param x     Input variable
     * @param alpha Alpha value
     * @return Output variable
     */
    public SDVariable leakyReluDerivative(String name, SDVariable x, double alpha) {
        SDVariable result = f().leakyReluDerivative(x, alpha);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #linear(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable linear(SDVariable input, SDVariable weights, SDVariable bias) {
        return linear(null, input, weights, bias);
    }

    /**
     * Linear layer operation: out = mmul(in,w) + bias<br>
     * Note that bias array is optional
     *
     * @param name    Name of the output variable
     * @param input   Input data
     * @param weights Weights variable
     * @param bias    Optional bias variable (may be null)
     * @return Output variable
     */
    public SDVariable linear(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        SDVariable res = f().xwPlusB(input, weights, bias);
        return updateVariableNameAndReference(res, name);
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
        SDVariable ret = f().localResponseNormalization(input, lrnConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))
     *
     * @param x Input Variable
     * @return Output variable
     */
    public SDVariable logSigmoid(SDVariable x) {
        return logSigmoid(null, x);
    }

    /**
     * Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))
     *
     * @param name Name of the output variable
     * @param x    Input Variable
     * @return Output variable
     */
    public SDVariable logSigmoid(String name, SDVariable x) {
        SDVariable ret = f().logSigmoid(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Log softmax activation
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable logSoftmax(SDVariable x) {
        return logSoftmax(null, x);
    }

    /**
     * Log softmax activation
     *
     * @param name Variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable logSoftmax(String name, SDVariable x) {
        SDVariable ret = f().logSoftmax(x);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * LSTM unit
     *
     * @param baseName      the base name for outputs
     * @param configuration the configuration to use
     * @return
     */
    public SDVariable lstm(String baseName, LSTMCellConfiguration configuration) {
        return new LSTMCell(this, configuration).outputVariables(baseName)[0];
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
        SDVariable ret = f().maxPooling3d(input, pooling3DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise rectified linear function with specified cutoff:<br>
     * out[i] = in[i] if in[i] >= cutoff
     * out[i] = 0 otherwise
     *
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu(SDVariable x, double cutoff) {
        return relu(null, x, cutoff);
    }

    /**
     * Element-wise rectified linear function with specified cutoff:<br>
     * out[i] = in[i] if in[i] >= cutoff
     * out[i] = 0 otherwise
     *
     * @param name   Output variable name
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu(String name, SDVariable x, double cutoff) {
        SDVariable result = f().relu(x, cutoff);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise "rectified linear 6" function with specified cutoff:<br>
     * out[i] = min(max(in, cutoff), 6)
     *
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu6(SDVariable x, double cutoff) {
        return relu6(null, x, cutoff);
    }

    /**
     * Element-wise "rectified linear 6" function with specified cutoff:<br>
     * out[i] = min(max(in, cutoff), 6)
     *
     * @param name   Output variable name
     * @param x      Input variable
     * @param cutoff Cutoff value. Usually 0
     * @return Output variable
     */
    public SDVariable relu6(String name, SDVariable x, double cutoff) {
        SDVariable result = f().relu6(x, cutoff);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @see #reluLayer(String, SDVariable, SDVariable, SDVariable)
     */
    public SDVariable reluLayer(SDVariable input, SDVariable weights, SDVariable bias) {
        return reluLayer(null, input, weights, bias);
    }

    /**
     * ReLU (Rectified Linear Unit) layer operation: out = relu(mmul(in,w) + bias)<br>
     * Note that bias array is optional
     *
     * @param name    Name of the output variable
     * @param input   Input data
     * @param weights Weights variable
     * @param bias    Optional bias variable (may be null)
     * @return Output variable
     */
    public SDVariable reluLayer(String name, SDVariable input, SDVariable weights, SDVariable bias) {
        SDVariable res = f().reluLayer(input, weights, bias);
        return updateVariableNameAndReference(res, name);
    }

    /**
     * Element-wise SeLU function - Scaled exponential Lineal Unit: see <a href="https://arxiv.org/abs/1706.02515">Self-Normalizing Neural Networks</a>
     * <br>
     * out[i] = scale * alpha * (exp(in[i])-1) if in[i]>0, or 0 if in[i] <= 0<br>
     * Uses default lcale and alpha values.
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable selu(SDVariable x) {
        return selu(null, x);
    }

    /**
     * Element-wise SeLU function - Scaled exponential Lineal Unit: see <a href="https://arxiv.org/abs/1706.02515">Self-Normalizing Neural Networks</a>
     * <br>
     * out[i] = scale * alpha * (exp(in[i])-1) if in[i]>0, or 0 if in[i] <= 0<br>
     * Uses default lcale and alpha values.
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable selu(String name, SDVariable x) {
        SDVariable ret = f().selu(x);
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
        SDVariable ret = f().sconv2d(inputs, conv2DConfig);
        return updateVariableNameAndReference(ret, name);
    }

    /**
     * Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))
     *
     * @param x Input Variable
     * @return Output variable
     */
    public SDVariable sigmoid(SDVariable x) {
        return sigmoid(null, x);
    }

    /**
     * Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))
     *
     * @param name Output variable name
     * @param x    Input Variable
     * @return Output variable
     */
    public SDVariable sigmoid(String name, SDVariable x) {
        SDVariable result = f().sigmoid(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut
     *
     * @param x   Input Variable
     * @param wrt Gradient at the output - dL/dOut. Must have same shape as the input
     * @return Output variable
     */
    public SDVariable sigmoidDerivative(SDVariable x, SDVariable wrt) {
        return sigmoidDerivative(null, x, wrt);
    }

    /**
     * Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut
     *
     * @param name Output variable name
     * @param x    Input Variable
     * @param wrt  Gradient at the output - dL/dOut. Must have same shape as the input
     * @return Output variable
     */
    public SDVariable sigmoidDerivative(String name, SDVariable x, SDVariable wrt) {
        SDVariable result = f().sigmoidDerivative(x, wrt);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Softmax activation
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softmax(SDVariable x) {
        return softmax(null, x);
    }

    /**
     * Softmax activation
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softmax(String name, SDVariable x) {
        SDVariable result = f().softmax(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * @param x
     * @return
     */
    public SDVariable softmaxDerivative(String name, SDVariable x, SDVariable wrt) {
        return softmaxDerivative(name, x, wrt, null);
    }

    public SDVariable softmaxDerivative(String name, SDVariable x, SDVariable wrt, Integer dimension) {
        SDVariable result = f().softmaxDerivative(x, wrt, dimension);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise softplus function: out = log(exp(x) + 1)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softplus(SDVariable x) {
        return softplus(null, x);
    }

    /**
     * Element-wise softplus function: out = log(exp(x) + 1)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable softplus(String name, SDVariable x) {
        SDVariable result = f().softplus(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise softsign function: out = x / (abs(x) + 1)
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable softsign(SDVariable x) {
        return softsign(null, x);
    }

    /**
     * Element-wise softsign function: out = x / (abs(x) + 1)
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable softsign(String name, SDVariable x) {
        SDVariable result = f().softsign(x);
        return updateVariableNameAndReference(result, name);
    }

    /**
     * Element-wise derivative (dOut/dIn) of the softsign function {@link #softsign(SDVariable)}
     *
     * @param x Input variable
     * @return Output varible
     */
    public SDVariable softsignDerivative(SDVariable x) {
        return softsignDerivative(null, x);
    }

    /**
     * Element-wise derivative (dOut/dIn) of the softsign function {@link #softsign(SDVariable)}
     *
     * @param name Output variable name
     * @param x    Input variable
     * @return Output varible
     */
    public SDVariable softsignDerivative(String name, SDVariable x) {
        SDVariable result = f().softsignDerivative(x);
        return updateVariableNameAndReference(result, name);
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
     * Simple recurrent unit
     *
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru(SRUConfiguration configuration) {
        return new SRU(this, configuration).outputVariables()[0];
    }

    /**
     * Simiple recurrent unit
     *
     * @param baseName      the base name to use for output variables
     * @param configuration the configuration for the sru
     * @return
     */
    public SDVariable sru(String baseName, SRUConfiguration configuration) {
        return new SRU(this, configuration).outputVariables(baseName)[0];
    }

    /**
     * An sru cell
     *
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell(SRUCellConfiguration configuration) {
        return new SRUCell(this, configuration).outputVariables()[0];
    }

    /**
     * An sru cell
     *
     * @param baseName      the base name to  use for the output variables
     * @param configuration the configuration for the sru cell
     * @return
     */
    public SDVariable sruCell(String baseName, SRUCellConfiguration configuration) {
        return new SRUCell(this, configuration).outputVariables(baseName)[0];
    }

    /**
     * Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0<br>
     * See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a>
     *
     * @param x Input variable
     * @return Output variable
     */
    public SDVariable swish(SDVariable x) {
        return swish(null, x);
    }

    /**
     * Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0<br>
     * See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a>
     *
     * @param name Name of the output variable
     * @param x    Input variable
     * @return Output variable
     */
    public SDVariable swish(String name, SDVariable x) {
        SDVariable ret = f().swish(x);
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
