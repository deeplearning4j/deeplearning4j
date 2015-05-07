/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Convolutional and pooling
 * layer
 *
 * @author Adam Gibson
 */
public class ConvolutionDownSampleLayer extends BaseLayer {
    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    /**
     * Create a layer from a configuration
     * @param conf
     */
    public ConvolutionDownSampleLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ConvolutionDownSampleLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    @Override
    public Pair<Gradient, Gradient> backWard(Gradient ixes, Gradient deltas, INDArray activation, String previousActivation) {
        int[] stride = conf().getStride();
        INDArray errors = Nd4j.zerosLike(getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS));

        return super.backWard(ixes, deltas, activation, previousActivation);
    }

    @Override
    public INDArray activate() {
        INDArray W = getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);
        if(W.shape()[1] != input.shape()[1])
            throw new IllegalStateException("Input size at dimension 1 must be same as the filter size");
        final INDArray b = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);

        INDArray convolution = Convolution.conv2d(input,W, Convolution.Type.FULL);
        if(convolution.shape().length < 4) {
            int[] newShape = new int[4];
            for(int i = 0; i < newShape.length; i++)
                newShape[i] = 1;
            int lengthDiff = 4 - convolution.shape().length;
            for(int i = lengthDiff; i < 4; i++)
                newShape[i] = convolution.shape()[i - lengthDiff];
            convolution = convolution.reshape(newShape);

        }

        final INDArray pooled = getPool(convolution);
        final INDArray bias = b.dimShuffle(new Object[]{'x', 0, 'x', 'x'}, new int[4], new boolean[]{true});
        final INDArray broadCasted = bias.broadcast(pooled.shape());
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), pooled));
    }


    private INDArray getPool(INDArray convolution) {
        INDArray pooled = null;
        switch (conf.getConvolutionType()) {
            case MAX:
                pooled = Transforms.maxPool(convolution, conf.getStride(),false);
                break;
            case SUM:
                pooled = Transforms.sumPooling(convolution,conf.getStride());
                break;
            case AVG:
                pooled = Transforms.avgPooling(convolution,conf.getStride());
                break;
            case NONE:
                return convolution;


        }

        return pooled;
    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public INDArray transform(INDArray data) {
        return activate(data);
    }



    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient gradient() {
        return new DefaultGradient();
    }

    @Override
    public void fit() {
        //no-op
    }

    @Override
    public void fit(INDArray input) {
        //no-op
    }
}
