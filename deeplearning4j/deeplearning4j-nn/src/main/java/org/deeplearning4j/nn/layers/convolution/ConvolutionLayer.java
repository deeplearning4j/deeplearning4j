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
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ops.impl.layers.convolution.Conv2DDerivative;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;


@Slf4j
public class ConvolutionLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {

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


        long[] kernel = layerConf().getKernelSize();
        long[] strides = layerConf().getStride();


        INDArray biasGradView = gradientViews.get(ConvolutionParamInitializer.BIAS_KEY);
        INDArray weightGradView = gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY).reshape(weights.shape()); //4d, c order. Shape: [outDepth,inDepth,kH,kW]



        INDArray delta;
        IActivation afn = layerConf().getActivationFn();



        INDArray lastZDup = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, lastZ);
        if(lastZDup.dataType() != dataType)
            lastZDup = lastZDup.castTo(dataType);
        delta = afn.backprop(lastZDup, epsilon).getFirst(); //TODO handle activation function params
        if(delta.dataType() != dataType)
            delta = delta.castTo(dataType);



        INDArray epsOut = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, epsilon.dataType(), input.shape());
        CNN2DFormat format = ConvolutionUtils.getFormatForLayer(layerConf());

        Conv2DDerivative conv2DDerivative = Conv2DDerivative.derivativeBuilder()
                .config(Conv2DConfig.builder()
                        .dH(layerConf().getDilation()[0])
                        .dW(layerConf().getDilation()[1])
                        .kH((int) kernel[0])
                        .kW((int) kernel[1])
                        .sH((int) strides[0])
                        .sW((int) strides[1])
                        .pH(layerConf().getPadding()[0])
                        .pW(layerConf().getPadding()[1])
                        .weightsFormat(ConvolutionUtils.getWeightFormat(format))
                        .paddingMode(ConvolutionUtils.paddingModeForConvolutionMode(layerConf().getConvolutionMode()))
                        .dataFormat(ConvolutionUtils.getFormatForLayer(layerConf()).name())
                        .build())
                .build();

        if(bias != null) {
            conv2DDerivative.addInputArgument(input, weights, bias, delta);
            conv2DDerivative.addOutputArgument(epsOut, weightGradView, biasGradView);
        } else {
            conv2DDerivative.addInputArgument(input, weights, delta);
            conv2DDerivative.addOutputArgument(epsOut, weightGradView);
        }

        Nd4j.getExecutioner().exec(conv2DDerivative);

        Gradient retGradient = new DefaultGradient();
        if(layerConf().hasBias()) {
            retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, gradientViews.get(ConvolutionParamInitializer.BIAS_KEY));
        }
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, gradientViews.get(ConvolutionParamInitializer.WEIGHT_KEY), 'c');

        weightNoiseParams.clear();

        // Clean up forward pass cache
        if (lastZ != null) {
            lastZ.close();
            lastZ = null;
        }

        return new Pair<>(retGradient, workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsOut));
    }

    /**
     * preOutput4d: Used so that ConvolutionLayer subclasses (such as Convolution1DLayer) can maintain their standard
     * non-4d preOutput method, while overriding this to return 4d activations (for use in backprop) without modifying
     * the public API
     */
    @SneakyThrows
    protected Pair<INDArray, INDArray> preOutput4d(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) throws Exception {
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
        long outDepth = layerConf().getNOut();
        long inDepth = layerConf().getNIn();

        long kH = layerConf().getKernelSize()[0];
        long kW = layerConf().getKernelSize()[1];

        CNN2DFormat format = ConvolutionUtils.getFormatForLayer(layerConf());

        Conv2DConfig config = Conv2DConfig.builder()
                .dH(layerConf().getDilation()[0])
                .dW(layerConf().getDilation()[1])
                .kH(layerConf().getKernelSize()[0])
                .kW(layerConf().getKernelSize()[1])
                .sH(layerConf().getStride()[0])
                .sW(layerConf().getStride()[1])
                .pH(layerConf().getPadding()[0])
                .pW(layerConf().getPadding()[1])
                .weightsFormat(ConvolutionUtils.getWeightFormat(format))
                .paddingMode(ConvolutionUtils.paddingModeForConvolutionMode(layerConf().getConvolutionMode()))
                .dataFormat(format.name())
                .build();

        INDArray z = Nd4j.cnn().conv2d(input, weights, bias, config);

        // Store z for backward pass activation function derivative
        try(MemoryWorkspace ws1 = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            this.lastZ = z.dup();
        }

        INDArray leveragedRet = workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, z);
        return new Pair<>(leveragedRet, null);
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
