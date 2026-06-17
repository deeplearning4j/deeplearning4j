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

package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.layers.SeparableConvolution1D;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.SeparableConvolutionParamInitializer;
import org.deeplearning4j.util.Convolution1DUtils;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.SConv2DDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

/**
 * 1D Separable Convolution layer implementation.
 * Uses the 2D separable convolution op internally by reshaping input.
 *
 * @author Adam Gibson
 */
public class SeparableConvolution1DLayer extends ConvolutionLayer {

    public SeparableConvolution1DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (epsilon.rank() != 3) {
            throw new DL4JInvalidInputException("Got rank " + epsilon.rank()
                    + " array as epsilon for SeparableConvolution1DLayer backprop with shape "
                    + Arrays.toString(epsilon.shape())
                    + ". Expected rank 3 array with shape [minibatchSize, channels, length]. " + layerId());
        }

        INDArray bias;
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, true, workspaceMgr);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, true, workspaceMgr);

        INDArray input = this.input.castTo(dataType);
        RNNFormat format = layerConf().getRnnDataFormat();
        boolean ncw = format == RNNFormat.NCW;

        if (!ncw) {
            input = input.permute(0, 2, 1); // NWC to NCW
            epsilon = epsilon.permute(0, 2, 1);
        }

        // Reshape to 4D: [batch, channels, length] -> [batch, channels, length, 1]
        long miniBatch = input.size(0);
        long inChannels = input.size(1);
        long inLength = input.size(2);

        INDArray input4d = input.reshape(miniBatch, inChannels, inLength, 1);
        INDArray epsilon4d = epsilon.reshape(epsilon.size(0), epsilon.size(1), epsilon.size(2), 1);

        long[] kernel = layerConf().getKernelSize();
        long[] stride = layerConf().getStride();
        long[] dilation = layerConf().getDilation();
        long[] pad;
        if (convolutionMode == ConvolutionMode.Same) {
            long outLength = Convolution1DUtils.getOutputSize(inLength, (int) kernel[0], (int) stride[0], 0,
                    convolutionMode, (int) dilation[0]);
            long[] samePad = ConvolutionUtils.getSameModeTopLeftPadding(
                    new long[]{outLength, 1}, new long[]{inLength, 1},
                    new long[]{kernel[0], 1}, new long[]{stride[0], 1}, new long[]{dilation[0], 1});
            pad = new long[]{samePad[0], 0};
        } else {
            pad = new long[]{layerConf().getPadding()[0], 0};
        }

        INDArray biasGradView = gradientViews.get(SeparableConvolutionParamInitializer.BIAS_KEY);
        INDArray depthWiseWeightGradView = gradientViews.get(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY);
        INDArray pointWiseWeightGradView = gradientViews.get(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY);

        long[] epsShape = new long[]{miniBatch, inChannels, inLength, 1};
        INDArray outEpsilon4d = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, depthWiseWeights.dataType(), epsShape, 'c');

        int sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        long[] args = {
                kernel[0], 1, stride[0], 1,
                pad[0], 0, dilation[0], 1, sameMode,
                0 // NCHW
        };

        // Compute delta
        IActivation afn = layerConf().getActivationFn();
        Pair<INDArray, INDArray> p = preOutput(true, true, workspaceMgr);
        INDArray z4d = p.getFirst().reshape(p.getFirst().size(0), p.getFirst().size(1), p.getFirst().size(2), 1);
        INDArray delta4d = afn.backprop(z4d, epsilon4d).getFirst();

        // Permute weights for libnd4j: [depthMult, nIn, kH, kW] -> [kH, kW, nIn, depthMult]
        depthWiseWeights = depthWiseWeights.permute(2, 3, 1, 0);
        pointWiseWeights = pointWiseWeights.permute(2, 3, 1, 0);
        INDArray opDepthWiseWeightGradView = depthWiseWeightGradView.permute(2, 3, 1, 0);
        INDArray opPointWiseWeightGradView = pointWiseWeightGradView.permute(2, 3, 1, 0);

        SConv2DDerivative op = new SConv2DDerivative();
        if (layerConf().hasBias()) {
            bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, true, workspaceMgr);
            op.addInputArgument(input4d, delta4d, depthWiseWeights, pointWiseWeights, bias);
            op.addOutputArgument(outEpsilon4d, opDepthWiseWeightGradView, opPointWiseWeightGradView, biasGradView);
        } else {
            op.addInputArgument(input4d, delta4d, depthWiseWeights, pointWiseWeights);
            op.addOutputArgument(outEpsilon4d, opDepthWiseWeightGradView, opPointWiseWeightGradView);
        }
        op.addIArgument(args);
        Nd4j.getExecutioner().exec(op);

        Gradient retGradient = new DefaultGradient();
        if (layerConf().hasBias()) {
            retGradient.setGradientFor(SeparableConvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, depthWiseWeightGradView, 'c');
        retGradient.setGradientFor(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, pointWiseWeightGradView, 'c');

        weightNoiseParams.clear();

        // Reshape back to 3D
        INDArray outEpsilon = outEpsilon4d.reshape(miniBatch, inChannels, inLength);

        if (!ncw) {
            outEpsilon = outEpsilon.permute(0, 2, 1); // NCW to NWC
        }

        outEpsilon = backpropDropOutIfPresent(outEpsilon);
        return new Pair<>(retGradient, outEpsilon);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);

        INDArray bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, training, workspaceMgr);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, training, workspaceMgr);

        INDArray input = this.input.castTo(dataType);
        RNNFormat format = layerConf().getRnnDataFormat();
        boolean ncw = format == RNNFormat.NCW;

        if (!ncw) {
            input = input.permute(0, 2, 1); // NWC to NCW
        }

        if (input.rank() != 3) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SeparableConvolution1D with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 3 array with shape [minibatchSize, channels, length]. " + layerId());
        }

        // Reshape to 4D: [batch, channels, length] -> [batch, channels, length, 1]
        long miniBatch = input.size(0);
        long inChannels = input.size(1);
        long inLength = input.size(2);

        INDArray input4d = input.reshape(miniBatch, inChannels, inLength, 1);

        long inDepth = depthWiseWeights.size(1);
        long outDepth = pointWiseWeights.size(0);

        if (inChannels != inDepth) {
            throw new DL4JInvalidInputException("Cannot do forward pass in SeparableConvolution1D layer: "
                    + "input channels does not match layer configuration. Input channels = " + inChannels
                    + ", expected = " + inDepth + ". " + layerId());
        }

        long[] kernel = layerConf().getKernelSize();
        long[] stride = layerConf().getStride();
        long[] dilation = layerConf().getDilation();

        long[] pad;
        long outLength;
        if (convolutionMode == ConvolutionMode.Same) {
            outLength = Convolution1DUtils.getOutputSize(inLength, (int) kernel[0], (int) stride[0], 0,
                    convolutionMode, (int) dilation[0]);
            long[] samePad = ConvolutionUtils.getSameModeTopLeftPadding(
                    new long[]{outLength, 1}, new long[]{inLength, 1},
                    new long[]{kernel[0], 1}, new long[]{stride[0], 1}, new long[]{dilation[0], 1});
            pad = new long[]{samePad[0], 0};
        } else {
            pad = new long[]{layerConf().getPadding()[0], 0};
            outLength = Convolution1DUtils.getOutputSize(inLength, (int) kernel[0], (int) stride[0], (int) pad[0],
                    convolutionMode, (int) dilation[0]);
        }

        long[] outShape = new long[]{miniBatch, outDepth, outLength, 1};
        INDArray output4d = workspaceMgr.create(ArrayType.ACTIVATIONS, depthWiseWeights.dataType(), outShape, 'c');

        int sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        long[] args = {
                kernel[0], 1, stride[0], 1,
                pad[0], 0, dilation[0], 1, sameMode,
                0 // NCHW
        };

        // Permute weights for libnd4j: [depthMult, nIn, kH, kW] -> [kH, kW, nIn, depthMult]
        depthWiseWeights = depthWiseWeights.permute(2, 3, 1, 0);
        pointWiseWeights = pointWiseWeights.permute(2, 3, 1, 0);

        INDArray[] opInputs;
        if (layerConf().hasBias()) {
            opInputs = new INDArray[]{input4d, depthWiseWeights, pointWiseWeights, bias};
        } else {
            opInputs = new INDArray[]{input4d, depthWiseWeights, pointWiseWeights};
        }

        SConv2D op = new SConv2D();
        op.addInputArgument(opInputs);
        op.addOutputArgument(output4d);
        op.addIArgument(args);
        Nd4j.getExecutioner().exec(op);

        // Reshape back to 3D
        INDArray output = output4d.reshape(miniBatch, outDepth, outLength);

        if (!ncw) {
            output = output.permute(0, 2, 1); // NCW to NWC
        }

        return new Pair<>(output, null);
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        applyDropOutIfNecessary(training, workspaceMgr);

        INDArray z = preOutput(training, false, workspaceMgr).getFirst();

        IActivation afn = layerConf().getActivationFn();

        INDArray activation = afn.getActivation(z, training);
        return activation;
    }

    @Override
    public SeparableConvolution1D layerConf() {
        return (SeparableConvolution1D) conf().getLayer();
    }
}
