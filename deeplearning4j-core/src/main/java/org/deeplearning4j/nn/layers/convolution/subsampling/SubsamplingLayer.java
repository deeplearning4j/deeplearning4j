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
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
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
import org.deeplearning4j.util.ConvolutionUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.layers.SubsamplingLayer.PoolingType;


/**
 * Subsampling layer.
 *
 * Used for downsampling a convolution
 *
 * @author Adam Gibson
 */
public class SubsamplingLayer implements Layer {
    private NeuralNetConfiguration conf;
    private Layer convLayer;
    protected ParamInitializer paramInitializer;
    private Map<String,INDArray> params;
    protected int index = 0;
    protected INDArray input;
    private INDArray dropoutMask;

    public SubsamplingLayer(NeuralNetConfiguration conf) {
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
        return 0;
    }

    @Override
    public double l1Magnitude() {
        return 0;
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
        INDArray z = preOutput(input, true);

        INDArray error = Nd4j.create(Ints.concat(new int[]{conf.getNIn(),conf.getNOut()},conf.getKernelSize()));

        if(layer.conf().getPoolingType() == PoolingType.AVG) {
            //TODO tile - change code
            int[] filterSize = conf.getKernelSize();
            int currLayerFeatureMaps = ConvolutionUtils.numFeatureMap(conf);
            int forwardLayerFeatureMaps = ConvolutionUtils.numFeatureMap(convLayer.conf());
            if (filterSize.length < 4)
                throw new IllegalStateException("Illegal filter size found ");

            //activation is the forward layers convolution
            for (int i = 0; i < forwardLayerFeatureMaps; i++) {
                for (int j = 0; j < currLayerFeatureMaps; j++) {
                    INDArray featureMapError = Nd4j.create(filterSize[0], 1, filterSize[filterSize.length - 2], filterSize[filterSize.length - 1]);
                    //rotated filter for convolution
                    INDArray rotatedFilter = Nd4j.rot(convLayer.getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS).get(NDArrayIndex.all(), NDArrayIndex.all()).slice(i).slice(j));
                    //forward error for the particular slice
                    INDArray forwardError = z.slice(j);
                    featureMapError.addi(Nd4j.getConvolution().convn(forwardError, rotatedFilter, Convolution.Type.FULL));
                    error.putSlice(i, featureMapError);
                }
            }
        }


        else if(layer.conf().getPoolingType() == PoolingType.MAX){
            //TODO rotation - change code
            int[] filterSize = conf.getKernelSize();
            int currLayerFeatureMaps = ConvolutionUtils.numFeatureMap(conf);
            int forwardLayerFeatureMaps = ConvolutionUtils.numFeatureMap(convLayer.conf());
            if (filterSize.length < 4)
                throw new IllegalStateException("Illegal filter size found ");

            //activation is the forward layers convolution
            for (int i = 0; i < forwardLayerFeatureMaps; i++) {
                for (int j = 0; j < currLayerFeatureMaps; j++) {
                    INDArray featureMapError = Nd4j.create(filterSize[0], 1, filterSize[filterSize.length - 2], filterSize[filterSize.length - 1]);
                    //rotated filter for convolution
                    INDArray rotatedFilter = Nd4j.rot(convLayer.getParam(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS).get(NDArrayIndex.all(), NDArrayIndex.all()).slice(i).slice(j));
                    //forward error for the particular slice
                    INDArray forwardError = z.slice(j);
                    featureMapError.addi(Nd4j.getConvolution().convn(forwardError, rotatedFilter, Convolution.Type.FULL));
                    error.putSlice(i, featureMapError);
                }
            }

        } else {
            throw new IllegalArgumentException("Convolution type is not average and max");
        }

        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(ConvolutionParamInitializer.CONVOLUTION_WEIGHTS, error);
        return new Pair<>(ret,z);

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
    public void update(Gradient gradient) {

    }

    @Override
    public INDArray preOutput(INDArray x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        return activate(x,training);
    }

    @Override
    public INDArray activate(boolean training) {
        return activate(this.input,training);
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        if(training && conf.getDropOut() > 0) {
            this.dropoutMask = Dropout.applyDropout(input,conf.getDropOut(),dropoutMask);
        }

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
    public INDArray activate() {
        throw new UnsupportedOperationException();
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
        throw new UnsupportedOperationException();
    }

    @Override
    public Collection<IterationListener> getListeners() {
        return null;
    }

    @Override
    public void setListeners(IterationListener... listeners) {

    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {

    }

    @Override
    public void fit() {

    }

    @Override
    public void update(INDArray gradient, String paramType) {

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
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int batchSize() {
        return input.size(0);
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
        return input;
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
