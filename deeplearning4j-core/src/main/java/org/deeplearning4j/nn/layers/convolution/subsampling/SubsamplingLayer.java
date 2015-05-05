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

package org.deeplearning4j.nn.layers.convolution.subsampling;

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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.util.ConvolutionUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Subsampling layer.
 *
 * Used for downsampling a convolution
 *
 * @author Adam Gibson
 */
public class SubsamplingLayer implements Layer {
    //fmSize = floor(self.layers{lL-1}.fmSize./stride);
    //aka the feature map size relative to a convolution layer
    private NeuralNetConfiguration conf;
    private Layer convLayer;
    protected ParamInitializer paramInitializer;
    private Map<String,INDArray> params;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
    
    public SubsamplingLayer(NeuralNetConfiguration conf) {
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
        INDArray ret = null;
        int numFeatureMaps = ConvolutionUtils.numFeatureMap(conf);
        for(int i = 0; i < input.slices(); i++) {
            INDArray downSampled = Transforms.downSample(input.slice(i),conf.getStride());
            if(ret == null) {
                ret = Nd4j.create(Ints.concat(new int[]{input.slices(),numFeatureMaps},downSampled.shape()));
            }
            ret.putSlice(i, downSampled);
        }
        return ret;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Layer clone() {
        return null;
    }

    @Override
    public Pair<Gradient, Gradient> backWard(Gradient errors, Gradient deltas, INDArray activation, String previousActivation) {
        INDArray ret = Nd4j.create(conf.getFilterSize());
        int[] filterSize = conf.getFilterSize();
        int currLayerFeatureMaps = ConvolutionUtils.numFeatureMap(conf);
        int forwardLayerFeatureMaps = ConvolutionUtils.numFeatureMap(convLayer.conf());
        if(filterSize.length < 4)
            throw new IllegalStateException("Illegal filter size found ");

        //activation is the forward layers convolution
        for(int i = 0; i < forwardLayerFeatureMaps; i++) {
            for(int j = 0; j < currLayerFeatureMaps; j++) {
                INDArray featureMapError  = Nd4j.create(filterSize[0], 1, filterSize[filterSize.length - 2], filterSize[filterSize.length - 1]);
                //rotated filter for convolution
                INDArray rotatedFilter = Nd4j.rot(convLayer.getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS).get(NDArrayIndex.all(),NDArrayIndex.all()).slice(i).slice(j));
                //forward error for the particular slice
                INDArray forwardError = activation.slice(j);
                featureMapError.addi(Nd4j.getConvolution().convn(forwardError,rotatedFilter, Convolution.Type.FULL));
                ret.putSlice(i,featureMapError);
            }




        }

        Gradient ret2 = new DefaultGradient();
        ret2.gradientForVariable().put(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS,ret);
        return new Pair<>(ret2,ret2);
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
            INDArray get = params.get(NDArrayIndex.interval(idx,idx + param.length()));
            if(param.length() != get.length())
                throw new IllegalStateException("Parameter " + gradientList.get(i) + " should have been of length " + param.length() + " but was " + get.length());
            param.assign(get.reshape(param.shape()));
            idx += param.length();
        }

        setScore();

    }

    @Override
    public void initParams() {
        paramInitializer.init(paramTable(),conf());
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
