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

package org.deeplearning4j.nn.params;

import org.canova.api.conf.Configuration;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Static weight initializer with just a weight matrix and a bias
 * @author Adam Gibson
 */
public class DefaultParamInitializer implements ParamInitializer {

    public final static String WEIGHT_KEY = "W";
    public final static String BIAS_KEY = "b";

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        return nIn*nOut + nOut;     //weights + bias
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramsView) {
        if(!(conf.getLayer() instanceof org.deeplearning4j.nn.conf.layers.FeedForwardLayer))
            throw new IllegalArgumentException("unsupported layer type: " + conf.getLayer().getClass().getName());

        int length = numParams(conf,true);
        if(paramsView.length() != length) throw new IllegalStateException("Expected params view of length " + length + ", got length " + paramsView.length());

        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();

        int nWeightParams = nIn*nOut;
        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0,nWeightParams));
        INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nWeightParams, nWeightParams + nOut));


        params.put(WEIGHT_KEY,createWeightMatrix(conf, weightView));
        params.put(BIAS_KEY,createBias(conf, biasView));
        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        int nIn = layerConf.getNIn();
        int nOut = layerConf.getNOut();
        int nWeightParams = nIn*nOut;

        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0,nWeightParams)).reshape('f',nIn,nOut);
        INDArray biasView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nWeightParams, nWeightParams + nOut));    //Already a row vector

        Map<String,INDArray> out = new LinkedHashMap<>();
        out.put(WEIGHT_KEY, weightGradientView);
        out.put(BIAS_KEY, biasView);

        return out;
    }


    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasParamView) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        INDArray ret =  Nd4j.valueArrayOf(layerConf.getNOut(), layerConf.getBiasInit());
        biasParamView.assign(ret);
        return biasParamView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightParamView) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();

        Distribution dist = Distributions.createDistribution(layerConf.getDist());
        INDArray ret =  WeightInitUtil.initWeights(
                layerConf.getNIn(),
                layerConf.getNOut(),
                layerConf.getWeightInit(),
                dist,
                weightParamView);
        return ret;
    }



}
