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
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

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
    public Gradient backwardGradient(INDArray z, Layer nextLayer, Gradient nextGradient, INDArray activation) {
        INDArray gy = nextGradient.getGradientFor(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);
        INDArray biasGradient = nextGradient.getGradientFor(ConvolutionParamInitializer.CONVOLUTION_BIAS);
        getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS).addi(gy.sum(0,2,3));
        INDArray gcol = Nd4j.tensorMmul(getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS), gy.slice(0), new int[][]{{0, 1}});
        gcol = Nd4j.rollAxis(gcol,3);
        INDArray weightGradient =  Convolution.conv2d(gcol,z, Convolution.Type.VALID);
        Gradient retGradient = new DefaultGradient();
        retGradient.setGradientFor(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS, weightGradient);
        retGradient.setGradientFor(ConvolutionParamInitializer.CONVOLUTION_BIAS,biasGradient);
        return retGradient;
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray activationMean() {
        throw new UnsupportedOperationException();
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
    public INDArray activate() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activate(INDArray input) {
        if(conf.getDropOut() > 0.0 && !conf.isUseDropConnect()) {
            input = input.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(input.shape()));
        }
        //number of feature maps for the weights
        int currentFeatureMaps = ConvolutionUtils.numFeatureMap(conf);
        //number of channels of the input
        int inputChannels = ConvolutionUtils.numChannels(input.shape());
        INDArray ret = Nd4j.create(Ints.concat(new int[]{input.slices(),currentFeatureMaps},conf.getFeatureMapSize()));
        INDArray bias = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);
        INDArray filters = getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);
        if(conf.getDropOut() > 0 && conf.isUseDropConnect()) {
            filters = filters.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(filters.shape()));
        }

        for(int i = 0; i < currentFeatureMaps; i++) {
            INDArray featureMap = Nd4j.create(Ints.concat(new int[]{input.slices(), 1}, conf.getFeatureMapSize()));
            for(int j = 0; j <  inputChannels; j++) {
                INDArray convolved = Nd4j.getConvolution().convn(input, filters.slice(i).slice(j), Convolution.Type.VALID);
                featureMap.addi(convolved.broadcast(featureMap.shape()));
            }

            featureMap.addi(bias.getDouble(i));
            INDArray activationForSlice = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), featureMap));
            ret.put(new NDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),new NDArrayIndex(new int[]{i}),NDArrayIndex.all()},activationForSlice);
        }
        return ret;
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
    public Pair<Gradient, Gradient> backWard(Gradient errors, Gradient deltas, INDArray activation, String previousActivation) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Collection<IterationListener> getIterationListeners() {
        return listeners;
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
    public void setScore() {

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
        return null;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return null;
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
        return null;
    }

    @Override
    public void validateInput() {

    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return null;
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
