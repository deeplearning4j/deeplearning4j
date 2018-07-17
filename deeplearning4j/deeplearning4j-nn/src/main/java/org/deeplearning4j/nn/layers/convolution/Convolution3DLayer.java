/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.Convolution3DParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.Convolution3DUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;

/**
 * 3D convolution layer implementation.
 *
 * @author Max Pumperla
 */
public class Convolution3DLayer extends ConvolutionLayer {

    public Convolution3DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public Convolution3DLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    void initializeHelper() {
        // no op
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {

        if (input.rank() != 5) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 5 array with shape [minibatchSize, channels, "
                    + "inputHeight, inputWidth, inputDepth]. "
                    + layerId());
        }

        INDArray weights = getParamWithNoise(Convolution3DParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        Convolution3D layerConfig = (Convolution3D) layerConf();

        boolean isNCDHW = layerConfig.getDataFormat() == Convolution3D.DataFormat.NCDHW;

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inD = (int) (isNCDHW ? input.size(2) : input.size(1));
        int inH = (int) (isNCDHW ? input.size(3) : input.size(2));
        int inW = (int) (isNCDHW ? input.size(4) : input.size(3));

        int outEpsChannels = (int) (isNCDHW ? weights.size(1) : weights.size(3));

        int[] dilation = layerConfig.getDilation();
        int[] kernel = layerConfig.getKernelSize();
        int[] strides = layerConfig.getStride();
        int[] pad;
        int[] outSize;

        if (convolutionMode == ConvolutionMode.Same) {
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            pad = Convolution3DUtils.get3DSameModeTopLeftPadding(
                    outSize, new int[]{inD, inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConfig.getPadding();
        }

        INDArray weightGradView = gradientViews.get(Convolution3DParamInitializer.WEIGHT_KEY);

        INDArray outEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD,
                miniBatch * outEpsChannels * inD * inH * inW);
        if (isNCDHW)
            outEpsilon = outEpsilon.reshape('c', miniBatch, outEpsChannels, inD, inH, inW);
        else
            outEpsilon = outEpsilon.reshape('c', miniBatch, inD, inH, inW, outEpsChannels);


        int[] intArgs = new int[]{
                kernel[0], kernel[1], kernel[2],
                strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2],
                dilation[0], dilation[1], dilation[2],
                convolutionMode == ConvolutionMode.Same ? 1 : 0,
                isNCDHW ? 0 : 1
        };

        INDArray delta;
        IActivation activation = layerConfig.getActivationFn();
        Pair<INDArray, INDArray> p = preOutput(true, true, workspaceMgr);

        delta = activation.backprop(p.getFirst(), epsilon).getFirst();

        INDArray bias;
        INDArray biasGradView = null;

        INDArray[] inputs;
        INDArray[] outputs;
        if (layerConfig.hasBias()) {
            biasGradView = gradientViews.get(Convolution3DParamInitializer.BIAS_KEY);
            bias = getParamWithNoise(Convolution3DParamInitializer.BIAS_KEY, true, workspaceMgr);
            inputs = new INDArray[]{input, weights, bias, delta};
            outputs = new INDArray[]{outEpsilon, weightGradView, biasGradView};
        } else {
            inputs = new INDArray[]{input, weights, delta};
            outputs = new INDArray[]{outEpsilon, weightGradView};
        }

        CustomOp op = DynamicCustomOp.builder("conv3dnew_bp")
                .addInputs(inputs)
                .addIntegerArguments(intArgs)
                .addOutputs(outputs)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        Gradient retGradient = new DefaultGradient();
        if (layerConfig.hasBias()) {
            retGradient.setGradientFor(Convolution3DParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(Convolution3DParamInitializer.WEIGHT_KEY, weightGradView, 'c');
        weightNoiseParams.clear();

        return new Pair<>(retGradient, outEpsilon);
    }


    @Override
    public INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return preOutput(training, false, workspaceMgr).getFirst();
    }

    protected Pair<INDArray, INDArray> preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {

        Convolution3D layerConfig = (Convolution3D) layerConf();

        ConvolutionMode mode = layerConfig.getConvolutionMode();
        boolean isNCDHW = layerConfig.getDataFormat() == Convolution3D.DataFormat.NCDHW;

        INDArray weights = getParamWithNoise(Convolution3DParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        if (input.rank() != 5) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Convolution3DLayer (layer name = " + layerName + ", layer index = "
                    + index + ") with shape " + Arrays.toString(input.shape()) + ". "
                    + "Expected rank 5 array with shape [minibatchSize, numChannels, inputHeight,"
                    + "inputWidth, inputDepth]."
                    + (input.rank() == 2
                    ? " (Wrong input type (see InputType.convolutionalFlat()) or wrong data type?)"
                    : "")
                    + " " + layerId());
        }

        // FIXME: int cast
        int miniBatch = (int) input.size(0);
        int inputChannels = (int) (isNCDHW ? input.size(1) : input.size(4));
        int inD =(int) (isNCDHW ? input.size(2) : input.size(1));
        int inH = (int) (isNCDHW ? input.size(3) : input.size(2));
        int inW = (int) (isNCDHW ? input.size(4) : input.size(3));

        int outWeightChannels = (int) (isNCDHW ? weights.size(0) : weights.size(4));
        int inWeightChannels = (int) (isNCDHW ? weights.size(1) : weights.size(3));

        if (inputChannels != inWeightChannels) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Cannot do forward pass in Convolution3D layer (layer name = "
                    + layerName
                    + ", layer index = " + index + "): number of input array channels does not match " +
                    "CNN layer configuration"
                    + " (data input channels = " + input.size(1)
                    + ", [minibatch, inputChannels, depth, height, width]="
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + inWeightChannels + ") "
                    + layerId());
        }


        int[] kernel = layerConfig.getKernelSize();
        int[] dilation = layerConfig.getDilation();
        int[] strides = layerConfig.getStride();

        int[] pad;
        int[] outSize;
        if (mode == ConvolutionMode.Same) {
            outSize = Convolution3DUtils.get3DOutputSize(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            int[] inSize = new int[]{inD, inH, inW};
            pad = Convolution3DUtils.get3DSameModeTopLeftPadding(outSize,
                    inSize, kernel, strides, dilation);
        } else {
            pad = layerConfig.getPadding();
            outSize = Convolution3DUtils.get3DOutputSize(input, kernel, strides, pad, convolutionMode, dilation, isNCDHW);
        }
        int outD = outSize[0];
        int outH = outSize[1];
        int outW = outSize[2];

        INDArray output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS,
                miniBatch*outWeightChannels*outD*outH*outW);
        if (isNCDHW)
            output = output.reshape('c', miniBatch, outWeightChannels, outD, outH, outW);
        else
            output = output.reshape('c', miniBatch, outD, outH, outW, outWeightChannels);

        int[] intArgs = new int[]{
                kernel[0], kernel[1], kernel[2],
                strides[0], strides[1], strides[2],
                pad[0], pad[1], pad[2],
                dilation[0], dilation[1], dilation[2],
                mode == ConvolutionMode.Same ? 1 : 0,
                isNCDHW ? 0 : 1
        };

        INDArray[] inputs;
        if (layerConfig.hasBias()) {
            INDArray bias = getParamWithNoise(Convolution3DParamInitializer.BIAS_KEY, training, workspaceMgr);
            inputs = new INDArray[]{input, weights, bias};
        } else {
            inputs = new INDArray[]{input, weights};
        }

        CustomOp op = DynamicCustomOp.builder("conv3dnew")
                .addInputs(inputs)
                .addIntegerArguments(intArgs)
                .addOutputs(output)
                .callInplace(false)
                .build();

        Nd4j.getExecutioner().exec(op);

        return new Pair<>(output, null);
    }
}