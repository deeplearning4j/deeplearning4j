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
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;

/**
 * Convolution layer
 *
 * @author Adam Gibson
 */
public class ConvolutionLayer implements Layer {

   private NeuralNetConfiguration conf;
   private Map<String,INDArray> params;
   protected Collection<IterationListener> iterationListeners = new ArrayList<>();
   
    public Collection<IterationListener> getIterationListeners() {
        return iterationListeners;
    }

    public void setIterationListeners(Collection<IterationListener> listeners) {
        this.iterationListeners = listeners != null ? listeners : new ArrayList<IterationListener>();
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
        INDArray ret = Nd4j.create(conf.getFilterSize());
        INDArray bias = getParam(ConvolutionParamInitializer.CONVOLUTION_BIAS);
        INDArray filters = getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS);

        for(int i = 0; i < featureMaps; i++) {
            INDArray featureMap = Nd4j.create(ArrayUtil.replace(conf.getFilterSize(),1,1));
            for(int j = 0; j < featureMap.slices(); j++) {
                featureMap.addi(Nd4j.getConvolution().convn(input.slice(j),filters.slice(i).slice(j), Convolution.Type.VALID));
            }
            featureMap.addi(bias.slice(i));
            ret.putSlice(i,Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(),featureMap)));
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
        return null;
    }

    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public int numParams() {
        return 0;
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
