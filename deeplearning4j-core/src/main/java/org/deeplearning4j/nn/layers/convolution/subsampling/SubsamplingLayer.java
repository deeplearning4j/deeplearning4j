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

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.util.ConvolutionUtils;

import java.util.Map;

/**
 * Created by agibsoncccc on 4/29/15.
 */
public class SubsamplingLayer implements Layer {
    //the bias for a sub sampling layer. the size is the number of feature maps as a 1d tensor
    private INDArray b;
    //fmSize = floor(self.layers{lL-1}.fmSize./stride);
    //aka the feature map size relative to a convolution layer
    private int[] subSampledSize;
    private NeuralNetConfiguration conf;
    private Layer convLayer;


    @Override
    public Gradient error(INDArray input) {
        return null;
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        return null;
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
        INDArray ret = Nd4j.create(subSampledSize);
        for(int i = 0; i < input.slices(); i++) {
            ret.putSlice(i, Transforms.downSample(input.slice(i),conf.getStride()));
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
        INDArray ret = Nd4j.create(subSampledSize);
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
        return null;
    }

    @Override
    public void initParams() {

    }

    @Override
    public Map<String, INDArray> paramTable() {
        return null;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {

    }

    @Override
    public void setParam(String key, INDArray val) {

    }

    @Override
    public void clear() {

    }
}
