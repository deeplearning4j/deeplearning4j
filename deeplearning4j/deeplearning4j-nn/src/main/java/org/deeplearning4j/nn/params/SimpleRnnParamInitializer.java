/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.params;

import lombok.val;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class SimpleRnnParamInitializer implements ParamInitializer {

    private static final SimpleRnnParamInitializer INSTANCE = new SimpleRnnParamInitializer();

    public static SimpleRnnParamInitializer getInstance(){
        return INSTANCE;
    }

    public static final String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public static final String RECURRENT_WEIGHT_KEY = "RW";
    public static final String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    private static final List<String> PARAM_KEYS = Collections.unmodifiableList(Arrays.asList(WEIGHT_KEY, RECURRENT_WEIGHT_KEY, BIAS_KEY));
    private static final List<String> WEIGHT_KEYS = Collections.unmodifiableList(Arrays.asList(WEIGHT_KEY, RECURRENT_WEIGHT_KEY));
    private static final List<String> BIAS_KEYS = Collections.singletonList(BIAS_KEY);


    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer layer) {
        SimpleRnn c = (SimpleRnn)layer;
        val nIn = c.getNIn();
        val nOut = c.getNOut();
        return nIn * nOut + nOut * nOut + nOut;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return PARAM_KEYS;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return WEIGHT_KEYS;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return BIAS_KEYS;
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return WEIGHT_KEY.equals(key) || RECURRENT_WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return BIAS_KEY.equals(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        SimpleRnn c = (SimpleRnn)conf.getLayer();
        val nIn = c.getNIn();
        val nOut = c.getNOut();

        Map<String,INDArray> m;

        if (initializeParams) {
            Distribution dist = Distributions.createDistribution(c.getDist());

            m = getSubsets(paramsView, nIn, nOut, false);
            INDArray w = WeightInitUtil.initWeights(nIn, nOut, new long[]{nIn, nOut}, c.getWeightInit(), dist, 'f', m.get(WEIGHT_KEY));
            m.put(WEIGHT_KEY, w);

            WeightInit rwInit;
            Distribution rwDist = dist;
            if (c.getWeightInitRecurrent() != null) {
                rwInit = c.getWeightInitRecurrent();
                if(c.getDistRecurrent() != null) {
                    rwDist = Distributions.createDistribution(c.getDistRecurrent());
                }
            } else {
                rwInit = c.getWeightInit();
            }

            INDArray rw = WeightInitUtil.initWeights(nOut, nOut, new long[]{nOut, nOut}, rwInit, rwDist, 'f', m.get(RECURRENT_WEIGHT_KEY));
            m.put(RECURRENT_WEIGHT_KEY, rw);
        } else {
            m = getSubsets(paramsView, nIn, nOut, true);
        }

        conf.addVariable(WEIGHT_KEY);
        conf.addVariable(RECURRENT_WEIGHT_KEY);
        conf.addVariable(BIAS_KEY);

        return m;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        SimpleRnn c = (SimpleRnn)conf.getLayer();
        val nIn = c.getNIn();
        val nOut = c.getNOut();

        return getSubsets(gradientView, nIn, nOut, true);
    }

    private static Map<String,INDArray> getSubsets(INDArray in, long nIn, long nOut, boolean reshape){
        long pos = nIn * nOut;
        INDArray w = in.get(point(0), interval(0, pos));
        INDArray rw = in.get(point(0), interval(pos, pos + nOut * nOut));
        pos += nOut * nOut;
        INDArray b = in.get(point(0), interval(pos, pos + nOut));

        if(reshape){
            w = w.reshape('f', nIn, nOut);
            rw = rw.reshape('f', nOut, nOut);
        }

        Map<String,INDArray> m = new LinkedHashMap<>();
        m.put(WEIGHT_KEY, w);
        m.put(RECURRENT_WEIGHT_KEY, rw);
        m.put(BIAS_KEY, b);
        return m;
    }
}
