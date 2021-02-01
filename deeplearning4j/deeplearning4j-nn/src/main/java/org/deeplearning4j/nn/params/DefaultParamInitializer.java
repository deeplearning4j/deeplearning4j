/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    public final static String GAIN_KEY = "g";

    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        FeedForwardLayer layerConf = (FeedForwardLayer) l;
        val nIn = layerConf.getNIn();
        val nOut = layerConf.getNOut();
        return (nIn * nOut + (hasBias(l) ? nOut : 0) + (hasLayerNorm(l) ? nOut : 0)); //weights + bias + gain
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        final ArrayList<String> keys = new ArrayList<>(3);
        keys.addAll(weightKeys(layer));
        keys.addAll(biasKeys(layer));
        return keys;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        if(hasLayerNorm(layer)){
            return Arrays.asList(WEIGHT_KEY, GAIN_KEY);
        }
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
        return WEIGHT_KEY.equals(key) || (hasLayerNorm(layer) && GAIN_KEY.equals(key));
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
        INDArray weightView = paramsView.get(NDArrayIndex.interval(0,0,true), NDArrayIndex.interval(0, nWeightParams));

        params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
        conf.addVariable(WEIGHT_KEY);

        long offset = nWeightParams;
        if(hasBias(layerConf)){
            INDArray biasView = paramsView.get(NDArrayIndex.interval(0,0,true),
                    NDArrayIndex.interval(offset, offset + nOut));
            params.put(BIAS_KEY, createBias(conf, biasView, initializeParams));
            conf.addVariable(BIAS_KEY);
            offset += nOut;
        }

        if(hasLayerNorm(layerConf)){
            INDArray gainView = paramsView.get(NDArrayIndex.interval(0,0,true),
                    NDArrayIndex.interval(offset, offset + nOut));
            params.put(GAIN_KEY, createGain(conf, gainView, initializeParams));
            conf.addVariable(GAIN_KEY);
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

        INDArray weightGradientView = gradientView.get(NDArrayIndex.interval(0,0,true), NDArrayIndex.interval(0, nWeightParams))
                        .reshape('f', nIn, nOut);

        Map<String, INDArray> out = new LinkedHashMap<>();
        out.put(WEIGHT_KEY, weightGradientView);

        long offset = nWeightParams;
        if(hasBias(layerConf)){
            INDArray biasView = gradientView.get(NDArrayIndex.interval(0,0,true),
                    NDArrayIndex.interval(offset, offset + nOut)); //Already a row vector
            out.put(BIAS_KEY, biasView);
            offset += nOut;
        }

        if(hasLayerNorm(layerConf)){
            INDArray gainView = gradientView.get(NDArrayIndex.interval(0,0,true),
                    NDArrayIndex.interval(offset, offset + nOut)); //Already a row vector
            out.put(GAIN_KEY, gainView);
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
            biasParamView.assign(biasInit);
        }
        return biasParamView;
    }

    protected INDArray createGain(NeuralNetConfiguration conf, INDArray gainParamView, boolean initializeParameters) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();
        return createGain(layerConf.getNOut(), layerConf.getGainInit(), gainParamView, initializeParameters);
    }

    protected INDArray createGain(long nOut, double gainInit, INDArray gainParamView, boolean initializeParameters) {
        if (initializeParameters) {
            gainParamView.assign(gainInit);
        }
        return gainParamView;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightParamView,
                    boolean initializeParameters) {
        org.deeplearning4j.nn.conf.layers.FeedForwardLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) conf.getLayer();

        if (initializeParameters) {
            return createWeightMatrix(layerConf.getNIn(), layerConf.getNOut(), layerConf.getWeightInitFn(),
                            weightParamView, true);
        } else {
            return createWeightMatrix(layerConf.getNIn(), layerConf.getNOut(), null, weightParamView, false);
        }
    }

    protected INDArray createWeightMatrix(long nIn, long nOut, IWeightInit weightInit,
                                          INDArray weightParamView, boolean initializeParameters) {
        val shape = new long[] {nIn, nOut};

        if (initializeParameters) {
            INDArray ret = weightInit.init(nIn, //Fan in
                            nOut, //Fan out
                            shape, IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, weightParamView);
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

    protected boolean hasLayerNorm(Layer layer){
        if(layer instanceof DenseLayer){
            return ((DenseLayer) layer).hasLayerNorm();
        }
        return false;
    }
}
