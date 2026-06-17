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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv3DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv3DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

import java.util.Arrays;

public class Convolution3DLayer extends ConvolutionLayer {

    public Convolution3DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
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

        INDArray input = this.input.castTo(dataType);
        INDArray weights = getParamWithNoise(Convolution3DParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        Convolution3D layerConfig = (Convolution3D) layerConf();

        boolean isNCDHW = layerConfig.getDataFormat() == Convolution3D.DataFormat.NCDHW;

        long miniBatch = input.size(0);
        int inD = (int) (isNCDHW ? input.size(2) : input.size(1));
        int inH = (int) (isNCDHW ? input.size(3) : input.size(2));
        int inW = (int) (isNCDHW ? input.size(4) : input.size(3));

        int outEpsChannels = (int) layerConf().getNIn();

        long[] dilation = layerConfig.getDilation();
        long[] kernel = layerConfig.getKernelSize();
        long[] strides = layerConfig.getStride();
        long[] pad;
        long[] outSize;

        if (convolutionMode == ConvolutionMode.Same) {
            outSize = Convolution3DUtils.get3DOutputSizeLong(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            pad = Convolution3DUtils.get3DSameModeTopLeftPaddingLong(
                    outSize, new long[]{inD, inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConfig.getPadding();
        }

        INDArray weightGradView = gradientViews.get(Convolution3DParamInitializer.WEIGHT_KEY);

        INDArray outEpsilon = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, weights.dataType(),
                miniBatch * outEpsChannels * inD * inH * inW);
        if (isNCDHW)
            outEpsilon = outEpsilon.reshape('c', miniBatch, outEpsChannels, inD, inH, inW);
        else
            outEpsilon = outEpsilon.reshape('c', miniBatch, inD, inH, inW, outEpsChannels);


        INDArray delta;
        IActivation activation = layerConfig.getActivationFn();
        Pair<INDArray, INDArray> p = preOutput(true, true, workspaceMgr);

        delta = activation.backprop(p.getFirst(), epsilon).getFirst();

        INDArray bias;
        INDArray biasGradView = null;

        //DL4J conv3d weights: val weightsShape = new long[]{outputDepth, inputDepth, kernel[0], kernel[1], kernel[2]};
        //libnd4j conv3d weights: [kD, kH, kW, iC, oC]
        weights = weights.permute(2, 3, 4, 1, 0);
        INDArray opWeightGradView = weightGradView.permute(2, 3, 4, 1, 0);

        INDArray[] inputs;
        INDArray[] outputs;
        if (layerConfig.hasBias()) {
            biasGradView = gradientViews.get(Convolution3DParamInitializer.BIAS_KEY);
            bias = getParamWithNoise(Convolution3DParamInitializer.BIAS_KEY, true, workspaceMgr);
            inputs = new INDArray[]{input, weights, bias, delta};
            outputs = new INDArray[]{outEpsilon, opWeightGradView, biasGradView};
        } else {
            inputs = new INDArray[]{input, weights, delta};
            outputs = new INDArray[]{outEpsilon, opWeightGradView};
        }

        Conv3DConfig bpConfig = Conv3DConfig.builder()
                .kD(kernel[0]).kH(kernel[1]).kW(kernel[2])
                .sD(strides[0]).sH(strides[1]).sW(strides[2])
                .pD(pad[0]).pH(pad[1]).pW(pad[2])
                .dD(dilation[0]).dH(dilation[1]).dW(dilation[2])
                .paddingMode(convolutionMode == ConvolutionMode.Same ? PaddingMode.SAME : PaddingMode.VALID)
                .dataFormat(isNCDHW ? Conv3DConfig.NCDHW : Conv3DConfig.NDHWC)
                .build();
        CustomOp op = new Conv3DDerivative(inputs, outputs, bpConfig);

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

        INDArray input = this.input.castTo(dataType);
        INDArray weights = getParamWithNoise(Convolution3DParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        if (input.rank() != 5) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Convolution3DLayer (layer name = " + layerName + ", layer index = "
                    + index + ") with shape " + Arrays.toString(input.shape()) + ". "
                    + "Expected rank 5 array with shape [minibatchSize, numChannels, inputHeight, "
                    + "inputWidth, inputDepth]."
                    + (input.rank() == 2
                    ? " (Wrong input type (see InputType.convolutionalFlat()) or wrong data type?)"
                    : "")
                    + " " + layerId());
        }

        long miniBatch = input.size(0);
        int inputChannels = (int) (isNCDHW ? input.size(1) : input.size(4));
        int inD =(int) (isNCDHW ? input.size(2) : input.size(1));
        int inH = (int) (isNCDHW ? input.size(3) : input.size(2));
        int inW = (int) (isNCDHW ? input.size(4) : input.size(3));

        int outWeightChannels = (int)layerConf().getNOut();
        int inWeightChannels = (int)layerConf().getNIn();

        if (inputChannels != inWeightChannels) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            long dataInCh = isNCDHW ? input.size(1) : input.size(4);
            String df;
            if(isNCDHW){
                df = ", dataFormat=NCDHW, [minibatch, inputChannels, depth, height, width]=";
            } else {
                df = ", dataFormat=NDHWC, [minibatch, depth, height, width, inputChannels]=";
            }
            throw new DL4JInvalidInputException("Cannot do forward pass in Convolution3D layer (layer name = "
                    + layerName
                    + ", layer index = " + index + "): number of input array channels does not match " +
                    "CNN layer configuration"
                    + " (data input channels = " + dataInCh
                    + df
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + inWeightChannels + ") "
                    + layerId());
        }


        long[] kernel = layerConfig.getKernelSize();
        long[] dilation = layerConfig.getDilation();
        long[] strides = layerConfig.getStride();

        long[] pad;
        long[] outSize;
        if (mode == ConvolutionMode.Same) {
            outSize = Convolution3DUtils.get3DOutputSizeLong(
                    input, kernel, strides, null, convolutionMode, dilation, isNCDHW);
            long[] inSize = {inD, inH, inW};
            pad = Convolution3DUtils.get3DSameModeTopLeftPaddingLong(outSize,
                    inSize, kernel, strides, dilation);
        } else {
            pad = layerConfig.getPadding();
            outSize = Convolution3DUtils.get3DOutputSizeLong(input, kernel, strides, pad, convolutionMode, dilation, isNCDHW);
        }
        long outD = outSize[0];
        long outH = outSize[1];
        long outW = outSize[2];

        INDArray output = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, weights.dataType(),miniBatch*outWeightChannels*outD*outH*outW);
        if (isNCDHW)
            output = output.reshape('c', miniBatch, outWeightChannels, outD, outH, outW);
        else
            output = output.reshape('c', miniBatch, outD, outH, outW, outWeightChannels);

        //DL4J conv3d weights: val weightsShape = new long[]{outputDepth, inputDepth, kernel[0], kernel[1], kernel[2]};
        //libnd4j conv3d weights: [kD, kH, kW, iC, oC]
        weights = weights.permute(2, 3, 4, 1, 0);

        INDArray[] inputs;
        if (layerConfig.hasBias()) {
            INDArray bias = getParamWithNoise(Convolution3DParamInitializer.BIAS_KEY, training, workspaceMgr);
            inputs = new INDArray[]{input, weights, bias};
        } else {
            inputs = new INDArray[]{input, weights};
        }

        Conv3DConfig fwdConfig = Conv3DConfig.builder()
                .kD(kernel[0]).kH(kernel[1]).kW(kernel[2])
                .sD(strides[0]).sH(strides[1]).sW(strides[2])
                .pD(pad[0]).pH(pad[1]).pW(pad[2])
                .dD(dilation[0]).dH(dilation[1]).dW(dilation[2])
                .paddingMode(mode == ConvolutionMode.Same ? PaddingMode.SAME : PaddingMode.VALID)
                .dataFormat(isNCDHW ? Conv3DConfig.NCDHW : Conv3DConfig.NDHWC)
                .build();
        CustomOp op = new Conv3D(inputs, new INDArray[]{output}, fwdConfig);

        Nd4j.getExecutioner().exec(op);

        return new Pair<>(output, null);
    }
}