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
import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Convolution layer
 *
 * @author Adam Gibson
 */
public class ConvolutionLayer implements Layer {

    private NeuralNetConfiguration conf;
    private Map<String,INDArray> params;
    protected ParamInitializer paramInitializer;
    private List<IterationListener> listeners = new ArrayList<>();
    protected int index = 0;
    private INDArray dropoutMask;

    public ConvolutionLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
    }


    @Override
    public int getIndex() {
        return index;
    }

    @Override
    public void setIndex(int index) {
        this.index = index;
    }

    @Override
    public double l2Magnitude() {
        return Transforms.pow(getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS), 2).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public double l1Magnitude() {
        return Transforms.abs(getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS)).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    @Override
    public Gradient error(INDArray input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        INDArray deriv = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getActivationFunction(), activate(input)).derivative());
        return deriv;
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradient errorSignal(Gradient error, INDArray input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, Gradient gradient, Layer layer) {
        INDArray gy = gradient.getGradientFor(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);
        INDArray biasGradient = gradient.getGradientFor(ConvolutionParamInitializer.CONVOLUTION_BIAS);
        getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS).addi(gy.sum(0,2,3));
        INDArray gcol = Nd4j.tensorMmul(getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS), gy.slice(0), new int[][]{{0, 1}});
        gcol = Nd4j.rollAxis(gcol,3);
        INDArray z = preOutput(input());
        INDArray weightGradient =  Convolution.conv2d(gcol, z, conf.getConvolutionType());
        Gradient retGradient = new DefaultGradient();
        retGradient.setGradientFor(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS, weightGradient);
        retGradient.setGradientFor(ConvolutionParamInitializer.CONVOLUTION_BIAS,biasGradient);
        return new Pair<>(retGradient,weightGradient);
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException()
                ;

    }

    @Override
    public INDArray activationMean() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void update(Gradient gradient) {

    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray preOutput(INDArray x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return null;
    }

    @Override
    public INDArray activate(boolean training) {
        return null;
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        if(conf.getDropOut() > 0.0 && !conf.isUseDropConnect() && training) {
            input = Dropout.applyDropout(input,conf.getDropOut(),dropoutMask);
        }


        // Activations
        INDArray bias = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);

        INDArray kernelWeights = getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);

        kernelWeights = kernelWeights.dup().reshape(Ints.concat(kernelWeights.shape(), new int[] {1, 1}));
        if(conf.getDropOut() > 0 && conf.isUseDropConnect()) {
            kernelWeights = kernelWeights.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(kernelWeights.shape()));
        }

        // Creates number of feature maps wanted (depth) in the convolution layer = number kernels
        INDArray convolved = Convolution.im2col(input, conf.getKernelSize(), conf.getStride(), conf.getPadding());
        INDArray activation = Nd4j.tensorMmul(convolved, kernelWeights, new int[][]{{1, 2, 3}, {1, 2, 3}});
        activation = activation.reshape(activation.size(0),activation.size(1),activation.size(2),activation.size(3));
        bias = bias.broadcast(activation.shape()).reshape(activation.shape());
        activation.addi(bias);
        return Nd4j.rollAxis(activation, 3, 1);
    }

    @Override
    public INDArray activate() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activate(INDArray input) {
        return activate(input,true);
    }


    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Collection<IterationListener> getListeners() {
        return null;
    }


    @Override
    public void setListeners(IterationListener... listeners) {
        for(IterationListener l : listeners)
            this.listeners.add(l);
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        this.listeners = new ArrayList<>(listeners);
    }

    @Override
    public void fit() {
        throw new UnsupportedOperationException();

    }

    @Override
    public double score() {
        return 0;
    }

    @Override
    public void computeGradientAndScore() {

    }

    @Override
    public void accumulateScore(double accum) {

    }

    @Override
    public INDArray transform(INDArray data) {
        return activate(data);
    }

    /**
     * Returns the parameters of the neural network
     *
     * @return the parameters of the neural network
     */
    @Override
    public INDArray params() {
        List<INDArray> ret = new ArrayList<>();
        for(String s : params.keySet())
            ret.add(params.get(s));
        return Nd4j.toFlattened(ret);
    }

    @Override
    public int numParams() {
        int ret = 0;
        for(INDArray val : params.values())
            ret += val.length();
        return ret;
    }

    @Override
    public void setParams(INDArray params) {

    }




    @Override
    public void fit(INDArray data) {

    }

    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int batchSize() {
        return 0;
    }

    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public INDArray input() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void validateInput() {

    }

    @Override
    public ConvexOptimizer getOptimizer() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray getParam(String param) {
        return params.get(param);
    }

    @Override
    public void initParams() {
        paramInitializer.init(paramTable(),conf());
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return params;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.params = paramTable;
    }

    @Override
    public void setParam(String key, INDArray val) {
        this.params.put(key,val);
    }

    @Override
    public void clear() {

    }
}
