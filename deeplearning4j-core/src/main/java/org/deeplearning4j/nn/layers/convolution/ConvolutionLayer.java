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
    protected INDArray col; // vectorized input

    public ConvolutionLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ConvolutionLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    public void setCol(INDArray col) {this.col = col;}

    @Override
    public double calcL2() {
    	if(!conf.isUseRegularization() || conf.getL2() <= 0.0 ) return 0.0;
        return 0.5 * conf.getL2() * Transforms.pow(getParam(ConvolutionParamInitializer.WEIGHT_KEY), 2).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public double calcL1() {
    	if(!conf.isUseRegularization() || conf.getL1() <= 0.0 ) return 0.0;
        return conf.getL1() * Transforms.abs(getParam(ConvolutionParamInitializer.WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    public INDArray calculateDelta(INDArray epsilon) {
        INDArray z = preOutput(true);
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getActivationFunction(), z).derivative());

        return epsilon.mmul(activationDerivative);

    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, Gradient gradient, Layer layer) {
        // TODO - how to handle transpose?
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);

        // gy, Note: epsilon should be reshaped to a tensor when passed in
        INDArray delta = calculateDelta(epsilon);

        Gradient retGradient = new DefaultGradient();

        // TODO do we roll delta for biasGradient? Note chainer adds bias to existing biasGradient for layer. Do we want to do this?
        //gb += gy[0].sum(axis=(0, 2, 3)) - add delta to bias or just pass in delta?
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, delta.sum(0, 2, 3));

        // TODO Note chainer adds weightGradient to existing weightGradient for layer. Do we want to do this?
        // gW += np.tensordot(gy[0], col, ([0, 2, 3], [0, 4, 5]))
        INDArray weightGradient = Nd4j.tensorMmul(delta, col, new int[][] {{0, 2, 3},{0, 4, 5}});
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradient);

        //gcol = tensorMmul(W, gy[0], (0, 1))
        INDArray nextEpsilon = Nd4j.tensorMmul(weights, delta.slice(0), new int[][]{{0, 1}});
        // TODO reshape epsilon?
//        epsilon.reshape(epsilon.size(0), epsilon.size(1), epsilon.size(2), epsilon.size(3));
        nextEpsilon = Nd4j.rollAxis(nextEpsilon, 3);
        nextEpsilon = Convolution.col2im(nextEpsilon, conf.getStride(), conf.getPadding(), inputHeight, inputWidth);
        return new Pair<>(retGradient,nextEpsilon);
    }

    public INDArray createFeatureMapColumn() {
        return Convolution.im2col(input, conf.getKernelSize(), conf.getStride(), conf.getPadding());
    }

    public INDArray preOutput(boolean training) {
        INDArray Weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(ConvolutionParamInitializer.BIAS_KEY);
        if(conf.isUseDropConnect() && training) {
            if (conf.getDropOut() > 0) {
                Weights = Dropout.applyDropConnect(this, ConvolutionParamInitializer.WEIGHT_KEY);
            }
        }

        INDArray z = Nd4j.tensorMmul(col, Weights, new int[][]{{1, 2, 3}, {1, 2, 3}});
        // TODO check shape and confirm correct approach
        z = z.reshape(z.size(0), z.size(1), z.size(2), z.size(3));
        bias = bias.broadcast(z.shape()).reshape(z.shape());
        z.addi(bias);
        return Nd4j.rollAxis(z, 3, 1);
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");
        applyDropOutIfNecessary(input, training);

        col = createFeatureMapColumn();
        INDArray z = preOutput(training);
        INDArray activation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), z));
        return activation;
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
