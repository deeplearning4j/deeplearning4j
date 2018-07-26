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
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * PReLU weight initializer. PReLU layer has weights of input shape (excluding mini-batch
 * dimension).
 *
 * @author Max Pumperla
 */
public class PReLUParamInitializer implements ParamInitializer {

    public final static String WEIGHT_KEY = "W";
    private long[] weightShape;
    private long[] sharedAxes;

    public PReLUParamInitializer(long[] shape, long[] sharedAxes) {
        this.weightShape = shape;
        this.sharedAxes = sharedAxes;
        // Set shared axes to 1, broadcasting will take place on c++ level.
        if (sharedAxes != null) {
            for (long axis: sharedAxes) {
                weightShape[(int)axis - 1] = 1;
            }
        }
    }


    public static PReLUParamInitializer getInstance(long[] shape, long[] sharedAxes) {
        return new PReLUParamInitializer(shape, sharedAxes);
    }


    @Override
    public long numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public long numParams(Layer l) {
        return numParams(weightShape);
    }

    private long numParams(long[] shape) {
        long flattened = 1;
        for(long value : shape) {
            flattened *= value;
        }
        return flattened;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
            return weightKeys(layer);
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Collections.singletonList(WEIGHT_KEY);
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return false;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        if (!(conf.getLayer() instanceof FeedForwardLayer))
            throw new IllegalArgumentException("unsupported layer type: " + conf.getLayer().getClass().getName());

        Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());

        val length = numParams(conf);
        if (paramsView.length() != length)
            throw new IllegalStateException(
                            "Expected params view of length " + length + ", got length " + paramsView.length());

        INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length));

        params.put(WEIGHT_KEY, createWeightMatrix(conf, weightView, initializeParams));
        conf.addVariable(WEIGHT_KEY);

        return params;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {

        val length = numParams(conf);
        INDArray weightGradientView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, length))
                        .reshape('f', weightShape);
        Map<String, INDArray> out = new LinkedHashMap<>();
        out.put(WEIGHT_KEY, weightGradientView);

        return out;
    }


    protected INDArray createWeightMatrix(NeuralNetConfiguration conf, INDArray weightParamView,
                    boolean initializeParameters) {

        FeedForwardLayer layerConf = (FeedForwardLayer) conf.getLayer();
        if (initializeParameters) {
            Distribution dist = Distributions.createDistribution(layerConf.getDist());
            return WeightInitUtil.initWeights(layerConf.getNIn(), layerConf.getNOut(),
                    weightShape, layerConf.getWeightInit(), dist, weightParamView);
        } else {
            return WeightInitUtil.reshapeWeights(weightShape, weightParamView);
        }
    }

}
