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
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CNN2DFormat;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.layers.Convolution1D;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.Convolution1DUtils;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1D;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv1DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv1DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.PaddingMode;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;
import java.util.List;

/**
 * 1D (temporal) convolutional layer. Currently, we just subclass off the
 * ConvolutionLayer and override the preOutput and backpropGradient methods.
 * Specifically, since this layer accepts RNN (not CNN) InputTypes, we
 * need to add a singleton fourth dimension before calling the respective
 * superclass method, then remove it from the result.
 *
 * This approach treats a multivariate time series with L timesteps and
 * P variables as an L x 1 x P image (L rows high, 1 column wide, P
 * channels deep). The kernel should be H<L pixels high and W=1 pixels
 * wide.
 *
 * TODO: We will eventually want to add a 1D-specific im2col method.
 *
 * @author dave@skymind.io
 */
public class Convolution1DLayer extends ConvolutionLayer {
    public Convolution1DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (epsilon.rank() != 3)
            throw new DL4JInvalidInputException("Got rank " + epsilon.rank()
                    + " array as epsilon for Convolution1DLayer backprop with shape "
                    + Arrays.toString(epsilon.shape())
                    + ". Expected rank 3 array with shape [minibatchSize, features, length]. " + layerId());
        Pair<INDArray,INDArray> fwd = preOutput(false,true,workspaceMgr);
        IActivation afn = layerConf().getActivationFn();
        INDArray delta = afn.backprop(fwd.getFirst(), epsilon).getFirst(); //TODO handle activation function params

        org.deeplearning4j.nn.conf.layers.Convolution1DLayer c = layerConf();
        Conv1DConfig conf = Conv1DConfig.builder()
                .k(c.getKernelSize()[0])
                .s(c.getStride()[0])
                .d(c.getDilation()[0])
                .p(c.getPadding()[0])
                .dataFormat(Conv1DConfig.NCW)
                .paddingMode(ConvolutionUtils.paddingModeForConvolutionMode(convolutionMode))
                .build();

        INDArray w = Convolution1DUtils.reshapeWeightArrayOrGradientForFormat(
                getParam(ConvolutionParamInitializer.WEIGHT_KEY),
                RNNFormat.NCW);

        INDArray[] inputArrs;
        INDArray[] outputArrs;
        INDArray wg = Convolution1DUtils.reshapeWeightArrayOrGradientForFormat(
                gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY),
                getRnnDataFormat());
        INDArray epsOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape());
        INDArray input = this.input.castTo(dataType);
        if(layerConf().getRnnDataFormat() == RNNFormat.NWC) {
            input = input.permute(0,2,1); //NHWC to NCHW
        }

        if(layerConf().hasBias()) {
            INDArray b = getParam(ConvolutionParamInitializer.BIAS_KEY);
            b = b.reshape(b.length());
            inputArrs = new INDArray[]{input, w, b, delta};
            INDArray bg = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
            bg = bg.reshape(bg.length());
            outputArrs = new INDArray[]{epsOut, wg, bg};
        } else {
            inputArrs = new INDArray[]{input, w, delta};
            outputArrs = new INDArray[]{epsOut, wg};
        }

        Conv1DDerivative op = new Conv1DDerivative(inputArrs, outputArrs, conf);
        Nd4j.exec(op);

        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()) {
            retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, gradientViews.get(ConvolutionParamInitializer.BIAS_KEY));
        }
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY), 'c');
        if (getRnnDataFormat() == RNNFormat.NWC) {
            epsOut = epsOut.permute(0, 2, 1);
        }
        return new Pair<>(retGradient, epsOut);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput4d(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        Pair<INDArray,INDArray> preOutput = super.preOutput(true, forBackprop, workspaceMgr);
        INDArray p3d = preOutput.getFirst();
        INDArray p = preOutput.getFirst().reshape(p3d.size(0), p3d.size(1), p3d.size(2), 1);
        preOutput.setFirst(p);
        return preOutput;
    }

    @Override
    protected Pair<INDArray,INDArray> preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);

        INDArray input = this.input.castTo(dataType);
        if(layerConf().getRnnDataFormat() == RNNFormat.NWC) {
            input = input.permute(0,2,1); //NHWC to NCHW
        }

        org.deeplearning4j.nn.conf.layers.Convolution1DLayer c = layerConf();
        Conv1DConfig conf = Conv1DConfig.builder()
                .k(c.getKernelSize()[0])
                .s(c.getStride()[0])
                .d(c.getDilation()[0])
                .p(c.getPadding()[0])
                .dataFormat(Conv1DConfig.NCW)
                .paddingMode(ConvolutionUtils.paddingModeForConvolutionMode(convolutionMode))
                .build();


        INDArray w = Convolution1DUtils.reshapeWeightArrayOrGradientForFormat(
                getParam(ConvolutionParamInitializer.WEIGHT_KEY)
                ,RNNFormat.NCW);


        INDArray[] inputs;
        if(layerConf().hasBias()) {
            INDArray b = getParam(ConvolutionParamInitializer.BIAS_KEY);
            b = b.reshape(b.length());
            inputs = new INDArray[]{input, w, b};
        } else {
            inputs = new INDArray[]{input, w};
        }

        Conv1D op = new Conv1D(inputs, null, conf);
        List<LongShapeDescriptor> outShape = op.calculateOutputShape();
        op.setOutputArgument(0, Nd4j.create(outShape.get(0), false));
        Nd4j.exec(op);
        INDArray output = op.getOutputArgument(0);

        if(getRnnDataFormat() == RNNFormat.NWC) {
            output = output.permute(0,2,1);
        }

        return new Pair<>(output, null);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray act4d = super.activate(training, workspaceMgr);
        INDArray act3d = act4d.rank() > 3 ?
                act4d.reshape(act4d.size(0), act4d.size(1), act4d.size(2)) : act4d;

        if(maskArray != null) {
            INDArray maskOut = feedForwardMaskArray(maskArray, MaskState.Active, (int)act3d.size(0)).getFirst();
            Preconditions.checkState(act3d.size(0) == maskOut.size(0) && act3d.size(2) == maskOut.size(1),
                    "Activations dimensions (0,2) and mask dimensions (0,1) don't match: Activations %s, Mask %s",
                    act3d.shape(), maskOut.shape());
            Broadcast.mul(act3d, maskOut, act3d, 0, 2);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, act3d);   //Should be zero copy most of the time
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        INDArray reduced = ConvolutionUtils.cnn1dMaskReduction(maskArray, layerConf().getKernelSize()[0],
                layerConf().getStride()[0], layerConf().getPadding()[0], layerConf().getDilation()[0],
                layerConf().getConvolutionMode());
        return new Pair<>(reduced, currentMaskState);
    }

    @Override
    public org.deeplearning4j.nn.conf.layers.Convolution1DLayer layerConf() {
        return (org.deeplearning4j.nn.conf.layers.Convolution1DLayer) conf().getLayer();
    }

    private RNNFormat getRnnDataFormat(){
        return layerConf().getRnnDataFormat();
    }
}
