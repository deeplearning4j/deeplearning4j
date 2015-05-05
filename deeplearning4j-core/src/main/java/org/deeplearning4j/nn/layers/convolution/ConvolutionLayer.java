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
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
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
    private NeuralNetConfiguration previousConf;
    protected ParamInitializer paramInitializer;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();

    public ConvolutionLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public Collection<IterationListener> getIterationListeners() {
        return iterationListeners;
    }
    
    @Override
    public void setIterationListeners(Collection<IterationListener> listeners) {
        this.iterationListeners = listeners != null ? listeners : new ArrayList<IterationListener>();
    }
    
    @Override
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    @Override
    public Gradient error(INDArray input) {
        return null;
    }



    @Override
    public INDArray derivativeActivation(INDArray input) {
        INDArray deriv = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getActivationFunction(), activate(input)).derivative());
        return deriv;
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        return null;
    }

    @Override
    public Gradient errorSignal(Gradient error, INDArray input) {
        return null;
    }

    @Override
    public Gradient backwardGradient(INDArray activation, Gradient errorSignal) {
        return null;
    }

    @Override
    public void merge(Layer layer, int batchSize) {

    }

    @Override
    public INDArray activationMean() {
        return null;
    }

    @Override
    public INDArray preOutput(INDArray x) {
        return null;
    }

    @Override
    public INDArray activate() {
        return null;
    }

    @Override
    public INDArray activate(INDArray input) {
        int featureMaps = ConvolutionUtils.numFeatureMap(conf);
        int inputFeatureMaps = ConvolutionUtils.numFeatureMap(input.shape());
        INDArray ret = Nd4j.create(Ints.concat(new int[]{input.slices(),featureMaps},conf.getFeatureMapSize()));
        INDArray bias = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);
        INDArray filters = getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);

        for(int i = 0; i < featureMaps; i++) {

            INDArray featureMap = Nd4j.create(Ints.concat(new int[]{input.slices(), 1}, conf.getFeatureMapSize()));
            for(int j = 0; j <  inputFeatureMaps; j++) {
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
        return null;
    }

    @Override
    public Layer clone() {
        return null;
    }

    @Override
    public Pair<Gradient, Gradient> backWard(Gradient errors, Gradient deltas, INDArray activation, String previousActivation) {
        return null;
    }

    @Override
    public void fit() {

    }

    @Override
    public void update(Gradient gradient) {

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
        List<String> gradientList = conf.variables();
        int length = 0;
        for(String s : gradientList)
            length += getParam(s).length();
        if(params.length() != length)
            throw new IllegalArgumentException("Unable to set parameters: must be of length " + length);
        int idx = 0;
        for(int i = 0; i < gradientList.size(); i++) {
            INDArray param = getParam(gradientList.get(i));
            INDArray get = params.get(NDArrayIndex.interval(idx, idx + param.length()));
            if(param.length() != get.length())
                throw new IllegalStateException("Parameter " + gradientList.get(i) + " should have been of length " + param.length() + " but was " + get.length());
            param.assign(get.reshape(param.shape()));
            idx += param.length();
        }

        setScore();

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
