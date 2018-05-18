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

package org.deeplearning4j.nn.layers;


import lombok.val;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;


/**
 * Baseline class for any Neural Network used
 * as a layer in a deep network *
 * @author Adam Gibson
 *
 */
public abstract class BasePretrainNetwork<LayerConfT extends org.deeplearning4j.nn.conf.layers.BasePretrainNetwork>
                extends BaseLayer<LayerConfT> {


    public BasePretrainNetwork(NeuralNetConfiguration conf) {
        super(conf);
    }

    public BasePretrainNetwork(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }


    /**
     * Corrupts the given input by doing a binomial sampling
     * given the corruption level
     * @param x the input to corrupt
     * @param corruptionLevel the corruption value
     * @return the binomial sampled corrupted input
     */
    public INDArray getCorruptedInput(INDArray x, double corruptionLevel) {
        INDArray corrupted = Nd4j.getDistributions().createBinomial(1, 1 - corruptionLevel).sample(x.shape());
        corrupted.muli(x);
        return corrupted;
    }


    protected Gradient createGradient(INDArray wGradient, INDArray vBiasGradient, INDArray hBiasGradient) {
        Gradient ret = new DefaultGradient(gradientsFlattened);
        // The order of the following statements matter! The gradient is being flattened and applied to
        // flattened params in this order.
        // The arrays neeed to be views, with the current Updater implementation

        //TODO: optimize this, to do it would the assigns
        INDArray wg = gradientViews.get(PretrainParamInitializer.WEIGHT_KEY);
        wg.assign(wGradient);

        INDArray hbg = gradientViews.get(PretrainParamInitializer.BIAS_KEY);
        hbg.assign(hBiasGradient);

        INDArray vbg = gradientViews.get(PretrainParamInitializer.VISIBLE_BIAS_KEY);
        vbg.assign(vBiasGradient);

        ret.gradientForVariable().put(PretrainParamInitializer.WEIGHT_KEY, wg);
        ret.gradientForVariable().put(PretrainParamInitializer.BIAS_KEY, hbg);
        ret.gradientForVariable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY, vbg);

        return ret;
    }

    @Override
    public int numParams(boolean backwards) {
        return super.numParams(backwards);
    }

    /**
     * Sample the hidden distribution given the visible
     * @param v the visible to sample from
     * @return the hidden mean and sample
     */
    public abstract Pair<INDArray, INDArray> sampleHiddenGivenVisible(INDArray v);

    /**
     * Sample the visible distribution given the hidden
     * @param h the hidden to sample from
     * @return the mean and sample
     */
    public abstract Pair<INDArray, INDArray> sampleVisibleGivenHidden(INDArray h);

    @Override
    protected void setScoreWithZ(INDArray z) {
        if (input == null || z == null)
            throw new IllegalStateException("Cannot calculate score without input and labels " + layerId());
        ILossFunction lossFunction = layerConf().getLossFunction().getILossFunction();

        //double score = lossFunction.computeScore(input, z, layerConf().getActivationFunction(), maskArray, false);
        double score = lossFunction.computeScore(input, z, layerConf().getActivationFn(), maskArray, false);
        score += calcL1(false) + calcL2(false);
        score /= getInputMiniBatchSize();

        this.score = score;
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        if (!backpropParamsOnly)
            return params;
        Map<String, INDArray> map = new LinkedHashMap<>();
        map.put(PretrainParamInitializer.WEIGHT_KEY, params.get(PretrainParamInitializer.WEIGHT_KEY));
        map.put(PretrainParamInitializer.BIAS_KEY, params.get(PretrainParamInitializer.BIAS_KEY));
        return map;
    }

    public INDArray params() {
        List<INDArray> list = new ArrayList<>(2);
        for (Map.Entry<String, INDArray> entry : params.entrySet()) {
            list.add(entry.getValue());
        }
        return Nd4j.toFlattened('f', list);
    }

    /**The number of parameters for the model, for backprop (i.e., excluding visible bias)
     * @return the number of parameters for the model (ex. visible bias)
     */
    public int numParams() {
        int ret = 0;
        for (Map.Entry<String, INDArray> entry : params.entrySet()) {
            ret += entry.getValue().length();
        }
        return ret;
    }

    @Override
    public void setParams(INDArray params) {
        if (params == paramsFlattened)
            return; //No op

        //SetParams has two different uses: during pretrain vs. backprop.
        //pretrain = 3 sets of params (inc. visible bias); backprop = 2

        List<String> parameterList = conf.variables();
        long paramLength = 0;
        for (String s : parameterList) {
            val len = getParam(s).length();
            paramLength += len;
        }

        if (params.length() != paramLength) {
            throw new IllegalArgumentException("Unable to set parameters: must be of length " + paramLength
                            + ", got params of length " + params.length() + " " + layerId());
        }

        // Set for backprop and only W & hb
        paramsFlattened.assign(params);

    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        Pair<Gradient, INDArray> result = super.backpropGradient(epsilon, workspaceMgr);
        ((DefaultGradient) result.getFirst()).setFlattenedGradient(gradientsFlattened);

        //During backprop, visible bias gradients are set to 0 - this is necessary due to the gradient view mechanics
        // that DL4J uses
        INDArray vBiasGradient = gradientViews.get(PretrainParamInitializer.VISIBLE_BIAS_KEY);
        result.getFirst().gradientForVariable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY, vBiasGradient);
        vBiasGradient.assign(0);

        weightNoiseParams.clear();

        return result;
    }


    @Override
    public double calcL2(boolean backpropParamsOnly) {
        double l2Sum = super.calcL2(true);
        if (backpropParamsOnly)
            return l2Sum;
        if (conf.getL2ByParam(PretrainParamInitializer.VISIBLE_BIAS_KEY) > 0) {
            double l2Norm = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).norm2Number().doubleValue();
            l2Sum += 0.5 * conf.getL2ByParam(PretrainParamInitializer.VISIBLE_BIAS_KEY) * l2Norm * l2Norm;
        }
        return l2Sum;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        double l1Sum = super.calcL1(true);
        if (conf.getL1ByParam(PretrainParamInitializer.VISIBLE_BIAS_KEY) > 0) {
            l1Sum += conf.getL1ByParam(PretrainParamInitializer.VISIBLE_BIAS_KEY)
                            * getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY).norm1Number().doubleValue();
        }
        return l1Sum;
    }

}
