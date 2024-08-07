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
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.params.SeparableConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

public class SeparableConvolution2DLayer extends ConvolutionLayer {

    public SeparableConvolution2DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }



    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (input.rank() != 4) {
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SubsamplingLayer with shape " + Arrays.toString(input.shape())
                    + ". Expected rank 4 array with shape " + layerConf().getCnn2dDataFormat().dimensionNames() + ". "
                    + layerId());
        }
        INDArray bias;
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, true, workspaceMgr);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, true, workspaceMgr);

        INDArray input = this.input.castTo(dataType);

        CNN2DFormat format = layerConf().getCnn2dDataFormat();
        boolean nchw = format == CNN2DFormat.NCHW;

        long miniBatch = input.size(0);
        int inH = (int)input.size(nchw ? 2 : 1);
        int inW = (int)input.size(nchw ? 3 : 2);

        int inDepth = (int) depthWiseWeights.size(1);
        int kH = (int) depthWiseWeights.size(2);
        int kW = (int) depthWiseWeights.size(3);

        long[] dilation = layerConf().getDilation();
        long[] kernel = layerConf().getKernelSize();
        long[] strides = layerConf().getStride();
        long[] pad;
        if (convolutionMode == ConvolutionMode.Same) {
            long[] outSize = ConvolutionUtils.getOutputSize(input, kernel, strides, null, convolutionMode, dilation, format); //Also performs validation
            pad = ConvolutionUtils.getSameModeTopLeftPadding(outSize, new long[] {inH, inW}, kernel, strides, dilation);
        } else {
            pad = layerConf().getPadding();
            ConvolutionUtils.getOutputSize(input, kernel, strides, pad, convolutionMode, dilation, format); //Also performs validation
        }

        INDArray biasGradView = gradientViews.get(SeparableConvolutionParamInitializer.BIAS_KEY);
        INDArray depthWiseWeightGradView = gradientViews.get(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY);
        INDArray pointWiseWeightGradView = gradientViews.get(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY);

        long[] epsShape = nchw ? new long[]{miniBatch, inDepth, inH, inW} : new long[]{miniBatch, inH, inW, inDepth};
        INDArray outEpsilon = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, depthWiseWeights.dataType(), epsShape, 'c');

        int sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        long[] args = {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode,
                nchw ? 0 : 1
        };

        INDArray delta;
        IActivation afn = layerConf().getActivationFn();
        Pair<INDArray, INDArray> p = preOutput4d(true, true, workspaceMgr);
        delta = afn.backprop(p.getFirst(), epsilon).getFirst();

        //dl4j weights: depth [depthMultiplier, nIn, kH, kW], point [nOut, nIn * depthMultiplier, 1, 1]
        //libnd4j weights: depth [kH, kW, iC, mC], point [1, 1, iC*mC, oC]
        depthWiseWeights = depthWiseWeights.permute(2, 3, 1, 0);
        pointWiseWeights = pointWiseWeights.permute(2, 3, 1, 0);
        INDArray opDepthWiseWeightGradView = depthWiseWeightGradView.permute(2, 3, 1, 0);
        INDArray opPointWiseWeightGradView = pointWiseWeightGradView.permute(2, 3, 1, 0);

        CustomOp op;
        if(layerConf().hasBias()){
            bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, true, workspaceMgr);

            op = DynamicCustomOp.builder("sconv2d_bp")
                    .addInputs(input, delta, depthWiseWeights, pointWiseWeights, bias)
                    .addIntegerArguments(args)
                    .addOutputs(outEpsilon, opDepthWiseWeightGradView, opPointWiseWeightGradView, biasGradView)
                    .callInplace(false)
                    .build();
        } else {
            op = DynamicCustomOp.builder("sconv2d_bp")
                    .addInputs(input, delta, depthWiseWeights, pointWiseWeights)
                    .addIntegerArguments(args)
                    .addOutputs(outEpsilon, opDepthWiseWeightGradView, opPointWiseWeightGradView)
                    .callInplace(false)
                    .build();
        }
        Nd4j.getExecutioner().exec(op);

        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()){
            retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, depthWiseWeightGradView, 'c');
        retGradient.setGradientFor(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, pointWiseWeightGradView, 'c');

        weightNoiseParams.clear();

        outEpsilon = backpropDropOutIfPresent(outEpsilon);
        return new Pair<>(retGradient, outEpsilon);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput(boolean training , boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        INDArray bias = getParamWithNoise(SeparableConvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray depthWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.DEPTH_WISE_WEIGHT_KEY, training, workspaceMgr);
        INDArray pointWiseWeights =
                getParamWithNoise(SeparableConvolutionParamInitializer.POINT_WISE_WEIGHT_KEY, training, workspaceMgr);

        INDArray input = this.input.castTo(dataType);
        if(layerConf().getCnn2dDataFormat() == CNN2DFormat.NHWC) {
            input = input.permute(0,3,1,2).dup();
        }

        int chIdx =  1;
        int hIdx =  2;
        int wIdx = 3;

        if (input.rank() != 4) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";
            throw new DL4JInvalidInputException("Got rank " + input.rank()
                    + " array as input to SeparableConvolution2D (layer name = " + layerName + ", layer index = "
                    + index + ") with shape " + Arrays.toString(input.shape()) + ". "
                    + "Expected rank 4 array with shape " + layerConf().getCnn2dDataFormat().dimensionNames() + "."
                    + (input.rank() == 2
                    ? " (Wrong input type (see InputType.convolutionalFlat()) or wrong data type?)"
                    : "")
                    + " " + layerId());
        }

        long inDepth = depthWiseWeights.size(1);
        long outDepth = pointWiseWeights.size(0);

        if (input.size(chIdx) != inDepth) {
            String layerName = conf.getLayer().getLayerName();
            if (layerName == null)
                layerName = "(not named)";

            String s = "Cannot do forward pass in SeparableConvolution2D layer (layer name = " + layerName
                    + ", layer index = " + index + "): input array channels does not match CNN layer configuration"
                    + " (data format = " + layerConf().getCnn2dDataFormat() + ", data input channels = " + input.size(1) + ", [minibatch,inputDepth,height,width]="
                    + Arrays.toString(input.shape()) + "; expected" + " input channels = " + inDepth + ") "
                    + layerId();

            int dimIfWrongFormat = 1;
            if(input.size(dimIfWrongFormat) == inDepth){
                //User might have passed NCHW data to a NHWC net, or vice versa?
                s += "\n" + ConvolutionUtils.NCHW_NHWC_ERROR_MSG;
            }

            throw new DL4JInvalidInputException(s);
        }
        int kH = (int) depthWiseWeights.size(2);
        int kW = (int) depthWiseWeights.size(3);

        long[] dilation = layerConf().getDilation();
        long[] kernel = layerConf().getKernelSize();
        long[] strides = layerConf().getStride();

        long[] pad;
        long[] outSize;
        if (convolutionMode == ConvolutionMode.Same) {
            outSize = ConvolutionUtils.getOutputSize(
                    input,
                    kernel,
                    strides,
                    null,
                    convolutionMode,
                    dilation,
                    CNN2DFormat.NCHW); //Also performs validation, note: hardcoded due to above permute

            if (input.size(2) > Integer.MAX_VALUE || input.size(3) > Integer.MAX_VALUE) {
                throw new ND4JArraySizeException();
            }
            pad = ConvolutionUtils.getSameModeTopLeftPadding(
                    outSize,
                    new long[] {(int) input.size(hIdx), (int) input.size(wIdx)},
                    kernel,
                    strides,
                    dilation);
        } else {
            pad = layerConf().getPadding();
            outSize = ConvolutionUtils.getOutputSize(
                    input,
                    kernel,
                    strides,
                    pad,
                    convolutionMode,
                    dilation,
                    CNN2DFormat.NCHW); //Also performs validation, note hardcoded due to permute above
        }

        long outH = outSize[0];
        long outW = outSize[1];

        val miniBatch = input.size(0);
        long[] outShape = new long[]{miniBatch, outDepth, outH, outW};
        INDArray output = workspaceMgr.create(ArrayType.ACTIVATIONS, depthWiseWeights.dataType(), outShape, 'c');

        Integer sameMode = (convolutionMode == ConvolutionMode.Same) ? 1 : 0;

        long[] args = {
                kH, kW, strides[0], strides[1],
                pad[0], pad[1], dilation[0], dilation[1], sameMode,
                0
        };

        //dl4j weights: depth [depthMultiplier, nIn, kH, kW], point [nOut, nIn * depthMultiplier, 1, 1]
        //libnd4j weights: depth [kH, kW, iC, mC], point [1, 1, iC*mC, oC]
        depthWiseWeights = depthWiseWeights.permute(2, 3, 1, 0);
        pointWiseWeights = pointWiseWeights.permute(2, 3, 1, 0);

        INDArray[] opInputs;
        if (layerConf().hasBias()) {
            opInputs = new INDArray[]{input, depthWiseWeights, pointWiseWeights, bias};
        } else {
            opInputs = new INDArray[]{input, depthWiseWeights, pointWiseWeights};

        }

        CustomOp op = DynamicCustomOp.builder("sconv2d")
                .addInputs(opInputs)
                .addIntegerArguments(args)
                .addOutputs(output)
                .callInplace(false)
                .build();
        Nd4j.getExecutioner().exec(op);

        if(layerConf().getCnn2dDataFormat() == CNN2DFormat.NHWC) {
            output = output.permute(0,2,3,1); //NCHW to NHWC

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

        //String afn = conf.getLayer().getActivationFunction();
        IActivation afn = layerConf().getActivationFn();

        INDArray activation = afn.getActivation(z, training);
        return activation;
    }
}
