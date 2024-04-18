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


import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.enums.WeightsFormat;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;


@Slf4j
public class ConvolutionLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {

    protected INDArray i2d;
    @Getter
    @Setter
    protected ConvolutionMode convolutionMode;
    private INDArray im2col2d;
    private INDArray lastZ;

    public ConvolutionLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
        convolutionMode = ((org.deeplearning4j.nn.conf.layers.ConvolutionLayer) conf().getLayer()).getConvolutionMode();
    }


    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        INDArray weights = getParamWithNoise(ConvolutionParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray bias = getParamWithNoise(ConvolutionParamInitializer.BIAS_KEY, true, workspaceMgr);

        INDArray input = this.input.castTo(dataType);       //No op if correct type
        if(epsilon.dataType() != dataType)
            epsilon = epsilon.castTo(dataType);


        if(layerConf().getCnn2dDataFormat() != CNN2DFormat.NCHW) {
            input = input.permute(0,3,1,2); //NHWC to NCHW
            epsilon = epsilon.permute(0,3,1,2); //NHWC to NCHW
        }


        long miniBatch = input.size(0);
        long inH = input.size(2);
        long inW =  input.size(3);

        long outDepth = weights.size(0);
        long inDepth = weights.size(1);
        long kH = weights.size(2);
        long kW =  weights.size(3);

        long[] dilation = layerConf().getDilation();
        long[] kernel = layerConf().getKernelSize();
        long[] strides = layerConf().getStride();
        long[] outSize;


        outSize = ConvolutionUtils.getOutputSizeLong(input.shape(), kernel, strides, null, convolutionMode, dilation, CNN2DFormat.NCHW); //Also performs validation

        long outH = outSize[0];
        long outW = outSize[1];


        INDArray biasGradView = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY); //4d, c order. Shape: [outDepth,inDepth,kH,kW]
        INDArray weightGradView2df = Shape
                .newShapeNoCopy(weightGradView, new long[]{outDepth, inDepth * kH * kW}, false).transpose();



        INDArray delta;
        IActivation afn = layerConf().getActivationFn();



        delta = afn.backprop(lastZ, epsilon).getFirst(); //TODO handle activation function params



        //Do im2col, but with order [miniB,outH,outW,depthIn,kH,kW]; but need to input [miniBatch,channels,kH,kW,outH,outW] given the current im2col implementation
        //To get this: create an array of the order we want, permute it to the order required by im2col implementation, and then do im2col on that
        //to get old order from required order: permute(0,3,4,5,1,2)
        INDArray im2col2d = this.im2col2d; //Re-use im2col2d array from forward pass if available; recalculate if not

        OpContext ctx = Nd4j.getExecutioner().buildContext();
        ctx.addIntermediateResult(im2col2d);

        INDArray epsOut = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, epsilon.dataType(), input.shape());

        Conv2DDerivative conv2DDerivative = Conv2DDerivative.derivativeBuilder()
                .config(Conv2DConfig.builder()
                        .dH((int) strides[0])
                        .dW((int) strides[1])
                        .kH((int) kernel[0])
                        .kW((int) kernel[1])
                        .sH((int) strides[0])
                        .sW((int) strides[1])
                        .weightsFormat(WeightsFormat.OIYX)
                        .paddingMode(ConvolutionUtils.paddingModeForConvolutionMode(layerConf().getConvolutionMode()))
                        .dataFormat(ConvolutionUtils.getFormatForLayer(layerConf()).name())
                        .build())
                .build();

        if(bias != null) {
            conv2DDerivative.addInputArgument(input, weights, bias, delta);
            conv2DDerivative.addOutputArgument(epsOut, weightGradView2df, biasGradView);
        } else {
            conv2DDerivative.addInputArgument(input, weights, delta);
            conv2DDerivative.addOutputArgument(epsOut, weightGradView2df);
        }



        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()) {
            retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradView);
        }
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradView, 'c');

        weightNoiseParams.clear();

        if(layerConf().hasBias()) {
            retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, gradientViews.get(ConvolutionParamInitializer.BIAS_KEY));
        }
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY), 'c');

        try {
         /*   ctx.close();
            im2col2d.close();
            lastZ.close();
            lastZ = null;
            this.im2col2d = null;*/
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        return new Pair<>(retGradient, workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD,epsOut));
    }

    /**
     * preOutput4d: Used so that ConvolutionLayer subclasses (such as Convolution1DLayer) can maintain their standard
     * non-4d preOutput method, while overriding this to return 4d activations (for use in backprop) without modifying
     * the public API
     */
    protected Pair<INDArray, INDArray> preOutput4d(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        return preOutput(training, forBackprop, workspaceMgr);
    }



    /**
     * PreOutput method that also returns the im2col2d array (if being called for backprop), as this can be re-used
     * instead of being calculated again.
     *
     * @param training    Train or test time (impacts dropout)
     * @param forBackprop If true: return the im2col2d array for re-use during backprop. False: return null for second
     *                    pair entry. Note that it may still be null in the case of CuDNN and the like.
     * @return            Pair of arrays: preOutput (activations) and optionally the im2col2d array
     */
    protected Pair<INDArray, INDArray> preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);


        INDArray bias = getParamWithNoise(ConvolutionParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray weights = getParamWithNoise(ConvolutionParamInitializer.WEIGHT_KEY, training, workspaceMgr);

        long miniBatch = input.size(0);
        long outDepth = weights.size(0);
        long inDepth = weights.size(1);

        long kH = weights.size(2);
        long kW = weights.size(3);


        Conv2DConfig config = Conv2DConfig.builder()
                .dH(layerConf().getDilation()[0])
                .dW(layerConf().getDilation()[1])
                .kH(layerConf().getKernelSize()[0])
                .kW(layerConf().getKernelSize()[1])
                .sH(layerConf().getStride()[0])
                .sW(layerConf().getStride()[1])
                .pH(layerConf().getPadding()[0])
                .pW(layerConf().getPadding()[1])
                .weightsFormat(WeightsFormat.OIYX)
                .paddingMode(ConvolutionUtils.paddingModeForConvolutionMode(layerConf().getConvolutionMode()))
                .dataFormat(ConvolutionUtils.getFormatForLayer(layerConf()).name())
                .build();

        //initialize a context and inject it for pulling out the im2col forward pass.
        OpContext ctx = Nd4j.getExecutioner().injectNewContext();
        INDArray z  = Nd4j.cnn().conv2d(input,weights,bias,config);
        INDArray im2col = ctx.getIntermediateResult(0);
        Nd4j.getExecutioner().clearOpContext();
        long outH = im2col.size(-2);
        long outW = im2col.size(-1);
        INDArray im2col2d = im2col.reshape(miniBatch * outH * outW, inDepth * kH * kW);
        this.lastZ = z;
        this.im2col2d = im2col2d;
        return new Pair<>(workspaceMgr.leverageTo(ArrayType.ACTIVATIONS,z), forBackprop ? im2col2d : null);
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot perform forward pass with null input " + layerId());
        }

        if (cacheMode == null)
            cacheMode = CacheMode.NONE;

        applyDropOutIfNecessary(training, workspaceMgr);

        INDArray z = preOutput(training, false, workspaceMgr).getFirst();
        // we do cache only if cache workspace exists. Skip otherwise
        if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE) && workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
            try (MemoryWorkspace wsB = workspaceMgr.notifyScopeBorrowed(ArrayType.FF_CACHE)) {
                preOutput = z.unsafeDuplication();
            }
        }

        IActivation afn = layerConf().getActivationFn();
        INDArray activation = afn.getActivation(z, training);
        return activation;
    }

    @Override
    public boolean hasBias() {
        return layerConf().hasBias();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setParams(INDArray params) {
        //Override, as base layer does f order parameter flattening by default
        setParams(params, 'c');
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        if (maskArray == null) {
            //For same mode (with stride 1): output activations size is always same size as input activations size -> mask array is same size
            return new Pair<>(maskArray, currentMaskState);
        }

        INDArray outMask = ConvolutionUtils.cnn2dMaskReduction(maskArray, layerConf().getKernelSize(), layerConf().getStride(),
                layerConf().getPadding(), layerConf().getDilation(), layerConf().getConvolutionMode());
        return new Pair<>(outMask, currentMaskState);
    }

}
