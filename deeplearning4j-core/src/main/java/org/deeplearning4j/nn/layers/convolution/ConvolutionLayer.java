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
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;


/**
 * Convolution layer
 *
 * @author Adam Gibson
 */
public class ConvolutionLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ConvolutionLayer> {
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
        //TODO figure out if need to sum over axes otherwise delete
    	if(!conf.isUseRegularization() || conf.getLayer().getL2() <= 0.0 ) return 0.0;
        return 0.5 * conf.getLayer().getL2() * Transforms.pow(getParam(ConvolutionParamInitializer.WEIGHT_KEY), 2).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public double calcL1() {
        //TODO figure out if need to sum over axes otherwise delete
    	if(!conf.isUseRegularization() || conf.getLayer().getL1() <= 0.0 ) return 0.0;
        return conf.getLayer().getL1() * Transforms.abs(getParam(ConvolutionParamInitializer.WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    public INDArray calculateDelta(INDArray epsilon) {
        INDArray z = preOutput(true);
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), z).derivative());
        if(!Arrays.equals(z.shape(),activationDerivative.shape()))
            throw new IllegalStateException("Shapes must be same");
        return epsilon.muli(activationDerivative);

    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        int inputHeight = input().size(-2);
        int inputWidth = input().size(-1);
        INDArray weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);

        // gy, Note: epsilon should be reshaped to a tensor when passed in
        INDArray delta = calculateDelta(epsilon);

        Gradient retGradient = new DefaultGradient();

        //gb = gy[0].sum(axis=(0, 2, 3))
        retGradient.setGradientFor(ConvolutionParamInitializer.BIAS_KEY, delta.sum(0, 2, 3));

        // gW = np.tensordot(gy[0], col, ([0, 2, 3], [0, 4, 5]))
        INDArray weightGradient = Nd4j.tensorMmul(delta, col, new int[][] {{0, 2, 3},{0, 4, 5}});
        retGradient.setGradientFor(ConvolutionParamInitializer.WEIGHT_KEY, weightGradient);

        //gcol = tensorMmul(W, gy[0], (0, 1))
        INDArray nextEpsilon = Nd4j.tensorMmul(weights, delta, new int[][] {{0}, {1}});

        nextEpsilon = Nd4j.rollAxis(nextEpsilon, 3);
        nextEpsilon = Convolution.col2im(nextEpsilon, layerConf().getStride(), layerConf().getPadding(), inputHeight, inputWidth);
        return new Pair<>(retGradient,nextEpsilon);
    }

    public INDArray preOutput(boolean training) {
        INDArray Weights = getParam(ConvolutionParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(ConvolutionParamInitializer.BIAS_KEY);
        if(conf.isUseDropConnect() && training) {
            if (conf.getLayer().getDropOut() > 0) {
                Weights = Dropout.applyDropConnect(this, ConvolutionParamInitializer.WEIGHT_KEY);
            }
        }

        INDArray z = Nd4j.tensorMmul(col, Weights, new int[][]{{1, 2, 3}, {1, 2, 3}});
        BroadcastOp op = new BroadcastAddOp(z,bias,z,3);
        Nd4j.getExecutioner().exec(op);

        return Nd4j.rollAxis(z, 3, 1);
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");
        applyDropOutIfNecessary(training);

        col = Convolution.im2col(input, layerConf().getKernelSize(), layerConf().getStride(), layerConf().getPadding());
        INDArray z = preOutput(training);
        // TODO add switch here to use bn if included
        INDArray activation = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), z));
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
    public void fit(INDArray input) {}

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }


}
