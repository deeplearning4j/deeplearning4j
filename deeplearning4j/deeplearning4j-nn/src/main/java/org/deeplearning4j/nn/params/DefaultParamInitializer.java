/*-
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

import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * Static weight initializer with just a weight matrix and a bias
 * @author Adam Gibson
 */
public class DefaultParamInitializer implements ParamInitializer {

    private static final DefaultParamInitializer INSTANCE = new DefaultParamInitializer();

    public static DefaultParamInitializer getInstance() {
        return INSTANCE;
    }

    public final static String WEIGHT_KEY = "W";
    public final static String BIAS_KEY = "b";

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        FeedForwardLayer layerConf = (FeedForwardLayer) l;
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        return (nIn * nOut + (hasBias(l) ? nOut : 0)); //weights + bias
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        if(hasBias(layer)){
            return Arrays.asList(WEIGHT_KEY, BIAS_KEY);
        } else {
            return weightKeys(layer);
        }

    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Collections.singletonList(WEIGHT_KEY);
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        if(hasBias(layer)){
            return Collections.singletonList(BIAS_KEY);
        } else {
            return Collections.emptyList();
        }
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return BIAS_KEY.equals(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        if (!(conf.getLayer() instanceof org.deeplearning4j.nn.conf.layers.FeedForwardLayer))
            throw new IllegalArgumentException("unsupported layer type: " + conf.getLayer().getClass().getName());

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

        val length = numParams(conf);
        if (paramsView.length() != length)
            throw new IllegalStateException(
                            "Expected params view of length " + length + ", got length " + paramsView.length());

        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();

        val nWeightParams = nIn * nOut;
        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nWeightParams));

        params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
        conf.addVariable(WEIGHT_KEY);


        if(hasBias(layerConf)){
            INDArray biasView = paramsView.get(NDArrayIndex.point(0),
                    NDArrayIndex.interval(nWeightParams, nWeightParams + nOut));
            params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
            conf.addVariable(BIAS_KEY);
        }

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        val nWeightParams = nIn * nOut;

        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nWeightParams))
                        .reshape('f', nIn, nOut);

        Map<String, INDArray> out = new LinkedHashMap<>();
        out.put(WEIGHT_KEY, weightGradientView);

        if(hasBias(layerConf)){
            INDArray biasView = gradientView.get(NDArrayIndex.point(0),
                    NDArrayIndex.interval(nWeightParams, nWeightParams + nOut)); //Already a row vector
            out.put(BIAS_KEY, biasView);
        }

        return out;
    }


    protected INDArray createBias(NeuralNetConfiguration conf, INDArray biasParamView, boolean initializeParameters) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        return createBias(layerConf.getNOut(), layerConf.getBiasInit(), biasParamView, initializeParameters);
    }

    protected INDArray createBias(long nOut, double biasInit, INDArray biasParamView, boolean initializeParameters) {
        if (initializeParameters) {
            INDArray ret = Nd4j.valueArrayOf(new long[] {1, nOut}, biasInit);
            biasParamView.assign(ret);
        }
        return biasParamView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightParamView,
                    boolean initializeParameters) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();

        if (initializeParameters) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            return createWeightMatrix(layerConf.getNIn(), layerConf.getNOut(), layerConf.getWeightInit(), dist,
                            weightParamView, true);
        } else {
            return createWeightMatrix(layerConf.getNIn(), layerConf.getNOut(), null, null, weightParamView, false);
        }
    }

    protected INDArray createWeightMatrix(long nIn, long nOut, WeightInit weightInit, Distribution dist,
                    INDArray weightParamView, boolean initializeParameters) {
        val shape = new long[] {nIn, nOut};

        if (initializeParameters) {
            INDArray ret = WeightInitUtil.initWeights(nIn, //Fan in
                            nOut, //Fan out
                            shape, weightInit, dist, weightParamView);
            return ret;
        } else {
            return WeightInitUtil.reshapeWeights(shape, weightParamView);
        }
    }

    protected boolean hasBias(Layer layer){
        if(layer instanceof BaseOutputLayer ) {
            return ((BaseOutputLayer) layer).hasBias();
        } else if(layer instanceof DenseLayer){
            return ((DenseLayer)layer).hasBias();
        } else if(layer instanceof EmbeddingLayer){
            return ((EmbeddingLayer)layer).hasBias();
        }  else if(layer instanceof EmbeddingSequenceLayer){
            return ((EmbeddingSequenceLayer)layer).hasBias();
        }
        return true;
    }
}
