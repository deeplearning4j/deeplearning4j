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

import lombok.SneakyThrows;
import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DeconvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.DeConv2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

public class Deconvolution2DLayer extends ConvolutionLayer {

    public Deconvolution2DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }


    @Override
    @SneakyThrows
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Deconvolution2DLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape " + layerConf().getCnn2dDataFormat().dimensionNames() + ". "
                    + layerId());
        }

        INDArray weights = getParamWithNoise(DeconvolutionParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        CNN2DFormat format = layerConf().getCnn2dDataFormat();
        boolean nchw = format == CNN2DFormat.NCHW;
        int hDim = nchw ? 2 : 1;
        int wDim = nchw ? 3 : 2;

        long miniBatch = input.size(0);
        long inH = input.size(hDim);
        long inW = input.size(wDim);

        long inDepth = weights.size(0);

        long kH = weights.size(2);
        long kW = weights.size(3);

        long[] dilation = layerConf().getDilation();
        long[] kernel = layerConf().getKernelSize();
        long[] strides = layerConf().getStride();
        long[] pad;
        if (convolutionMode == ConvolutionMode.Same) {
            long[] outSize = {epsilon.size(hDim), epsilon.size(wDim)};
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new long[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
        }

        INDArray biasGradView = gradientViews.get(DeconvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(DeconvolutionParamInitializer.WEIGHT_KEY);

        long[] epsShape = nchw ? new long[]{miniBatch, inDepth, inH, inW} : new long[]{miniBatch, inH, inW, inDepth};
        INDArray outEps = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, weights.dataType(), epsShape, 'c');

        INDArray delta;
        IActivation afn = layerConf().getActivationFn();
        Pair<INDArray, INDArray> p = preOutput4d(true, true, workspaceMgr);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst();

        //DL4J Deconv weights: [inputDepth, outputDepth, kH, kW]
        //libnd4j weights: [kH, kW, oC, iC]
        weights = weights.permute(2, 3, 1, 0);
        INDArray weightGradViewOp = weightGradView.permute(2, 3, 1, 0);

        INDArray[] opInputs;
        INDArray[] opOutputs;
        if(layerConf().hasBias()){
            INDArray bias = getParamWithNoise(DeconvolutionParamInitializer.BIAS_KEY, true, workspaceMgr);
            opInputs = new INDArray[]{input, weights, bias, delta};
            opOutputs = new INDArray[]{outEps, weightGradViewOp, biasGradView};
        } else {
            opInputs = new INDArray[]{input, weights, delta};
            opOutputs = new INDArray[]{outEps, weightGradViewOp};
        }
        DeConv2DConfig bpConfig = DeConv2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(strides[0]).sW(strides[1])
                .pH(pad[0]).pW(pad[1])
                .dH(dilation[0]).dW(dilation[1])
                .isSameMode(convolutionMode == ConvolutionMode.Same)
                .dataFormat(nchw ? DeConv2DConfig.NCHW : DeConv2DConfig.NHWC)
                .build();
        CustomOp op = new DeConv2DDerivative(opInputs, opOutputs, bpConfig);
        Nd4j.getExecutioner().exec(op);


        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()){
            retGradient.setGradientFor(DeconvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(DeconvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');
        weightNoiseParams.clear();

        return new Pair<>(retGradient, outEps);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput(boolean training , boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {

        INDArray bias = getParamWithNoise(DeconvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray weights = getParamWithNoise(DeconvolutionParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        //Input validation: expect rank 4 matrix
        if (input.rank() != 4) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to Deconvolution2D (layer name = " + layerName + ", layer index = "
                    + index + ") with shape " + Arrays.toString(input.shape()) + ". "
                    + "Expected rank 4 array with shape [minibatchSize, layerInputDepth, inputHeight, inputWidth]."
                    + (input.rank() == 2
                    ? " (Wrong input type (see InputType.convolutionalFlat()) or wrong data type?)"
                    : "")
                    + " " + layerId());
        }

        CNN2DFormat format = layerConf().getCnn2dDataFormat();
        boolean nchw = format == CNN2DFormat.NCHW;
        int cDim = nchw ? 1 : 3;
        int hDim = nchw ? 2 : 1;
        int wDim = nchw ? 3 : 2;

        long inDepth = weights.size(0);
        long outDepth = weights.size(1);

        if (input.size(cDim) != inDepth ) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";

            String s = "Cannot do forward pass in Deconvolution2D layer (layer name = " + layerName
                    + ", layer index = " + index + "): input array channels does not match CNN layer configuration"
                    + " (data format = " + format + ", data input channels = " + input.size(cDim) + ", "
                    + (nchw ? "[minibatch,inputDepth,height,width]" : "[minibatch,height,width,inputDepth]") + "="
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + inDepth + ") "
                    + layerId();

            int dimIfWrongFormat = format == CNN2DFormat.NHWC ? 1 : 3;
            if(input.size(dimIfWrongFormat) == inDepth){
                //User might have passed NCHW data to a NHWC net, or vice versa?
                s += "\n" + ConvolutionUtils.NCHW_NHWC_ERROR_MSG;
            }

            throw new DL4JInvalidInputException(s);
        }
        int kH = (int) weights.size(2);
        int kW = (int) weights.size(3);

        long[] dilation = layerConf().getDilation();
        long[] kernel = layerConf().getKernelSize();
        long[] strides = layerConf().getStride();

        long[] pad;
        long[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getDeconvolutionOutputSizeLong(input, kernel, strides, null, convolutionMode, dilation, format); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new long[] { input.size(hDim),  input.size(wDim)}, kernel,
                    strides, dilation );
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getDeconvolutionOutputSizeLong(input, kernel, strides, pad, convolutionMode, dilation, format); //Also performs validation
        }

        long outH = outSize[0];
        long outW = outSize[1];


        val miniBatch = input.size(0);
        long[] outShape = nchw ? new long[]{miniBatch, outDepth, outH, outW} : new long[]{miniBatch, outH, outW, outDepth};
        INDArray output = workspaceMgr.create(ArrayType.ACTIVATIONS, input.dataType(), outShape, 'c');

        //DL4J Deconv weights: [inputDepth, outputDepth, kH, kW]
        //libnd4j weights: [kH, kW, oC, iC]
        weights = weights.permute(2, 3, 1, 0);

        INDArray[] opInputs;
        if (layerConf().hasBias()) {
            opInputs = new INDArray[]{input, weights, bias};
        } else {
            opInputs = new INDArray[]{input, weights};
        }
        DeConv2DConfig fwdConfig = DeConv2DConfig.builder()
                .kH(kH).kW(kW)
                .sH(strides[0]).sW(strides[1])
                .pH(pad[0]).pW(pad[1])
                .dH(dilation[0]).dW(dilation[1])
                .isSameMode(convolutionMode == ConvolutionMode.Same)
                .dataFormat(nchw ? DeConv2DConfig.NCHW : DeConv2DConfig.NHWC)
                .build();
        CustomOp op = new DeConv2D(opInputs, new INDArray[]{output}, fwdConfig);
        Nd4j.getExecutioner().exec(op);

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
}