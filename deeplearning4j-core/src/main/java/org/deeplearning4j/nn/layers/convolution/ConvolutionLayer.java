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

import com.google.common.primitives.Ints;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

/**
 * Convolution layer
 *
 * @author Adam Gibson
 */
public class ConvolutionLayer extends BaseLayer {

    public ConvolutionLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ConvolutionLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public double l2Magnitude() {
        return Transforms.pow(getParam(ConvolutionParamInitializer.WEIGHT_KEY), 2).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public double l1Magnitude() {
        return Transforms.abs(getParam(ConvolutionParamInitializer.WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, Gradient gradient, Layer layer) {
        INDArray gy = gradient.getGradientFor(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray biasGradient = gradient.getGradientFor(ConvolutionParamInitializer.BIAS_KEY);
        getParam(ConvolutionParamInitializer.BIAS_KEY).addi(gy.sum(0,2,3));
        INDArray gcol = Nd4j.tensorMmul(getParam(ConvolutionParamInitializer.WEIGHT_KEY), gy.slice(0), new int[][]{{0, 1}});
        gcol = Nd4j.rollAxis(gcol,3);
        INDArray z = preOutput(input());
        INDArray weightGradient =  Convolution.conv2d(gcol, z, conf.getConvolutionType());
        Gradient retGradient = new DefaultGradient();
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradient);
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, biasGradient);
        return new Pair<>(retGradient,weightGradient);
    }

    public INDArray createFeatureMaps() {
        // Creates number of feature maps wanted (depth) in the convolution layer = number kernels
        return Convolution.im2col(input, conf.getKernelSize(), conf.getStride(), conf.getPadding());
    }

    public INDArray calculateActivation(INDArray featureMaps, INDArray kernelWeights, INDArray bias) {
        INDArray activation = Nd4j.tensorMmul(featureMaps, kernelWeights, new int[][]{{1, 2, 3}, {1, 2, 3}});
        activation = activation.reshape(activation.size(0), activation.size(1), activation.size(2), activation.size(3));
        bias = bias.broadcast(activation.shape()).reshape(activation.shape());
        activation.addi(bias);
        return activation;
    }

    @Override
    public INDArray activate(boolean training) {
        if(conf.getDropOut() > 0.0 && !conf.isUseDropConnect() && training) {
            input = Dropout.applyDropout(input,conf.getDropOut(),dropoutMask);
        }
        INDArray kernelWeights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(ConvolutionParamInitializer.BIAS_KEY);
//        kernelWeights = kernelWeights.dup().reshape(Ints.concat(kernelWeights.shape(), new int[] {1, 1}));

        if(conf.getDropOut() > 0 && conf.isUseDropConnect()) {
            kernelWeights = kernelWeights.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(kernelWeights.shape()));
        }
        INDArray featureMaps = createFeatureMaps();
        INDArray activation = calculateActivation(featureMaps, kernelWeights, bias);
        return Nd4j.rollAxis(activation, 3, 1);
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        if(x == null)
            throw new IllegalArgumentException("No null input allowed");

        setInput(x,training);
        applyDropOutIfNecessary(x,training);
        INDArray b = getParam(ConvolutionParamInitializer.BIAS_KEY);
        INDArray W = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        if(conf.isUseDropConnect() && training) {
            if (conf.getDropOut() > 0) {
                W = Dropout.applyDropConnect(this, ConvolutionParamInitializer.WEIGHT_KEY);
            }
        }

        INDArray ret = x.mmul(W).addiRowVector(b);
        return ret;

    }

    @Override
    public Layer transpose(){
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }


}
