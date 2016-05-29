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

package org.deeplearning4j.nn.layers;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossCalculation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;


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
        INDArray corrupted = Nd4j.getDistributions().createBinomial(1,1 - corruptionLevel).sample(x.shape());
        corrupted.muli(x);
        return corrupted;
    }


    protected Gradient createGradient(INDArray wGradient,INDArray vBiasGradient,INDArray hBiasGradient) {
        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY,vBiasGradient);
        ret.gradientForVariable().put(PretrainParamInitializer.BIAS_KEY,hBiasGradient);
        ret.gradientForVariable().put(PretrainParamInitializer.WEIGHT_KEY, wGradient);
        return ret;
    }

    @Override
    public int numParams(boolean backwards) {
        if(!backwards)
            return super.numParams(backwards);
        int ret = 0;
        for(String s : paramTable().keySet()) {
            if(!s.equals(PretrainParamInitializer.VISIBLE_BIAS_KEY)) {
                ret += getParam(s).length();
            }
        }

        return ret;
    }

    /**
     * Sample the hidden distribution given the visible
     * @param v the visible to sample from
     * @return the hidden mean and sample
     */
    public abstract Pair<INDArray,INDArray> sampleHiddenGivenVisible(INDArray v);

    /**
     * Sample the visible distribution given the hidden
     * @param h the hidden to sample from
     * @return the mean and sample
     */
    public abstract Pair<INDArray,INDArray> sampleVisibleGivenHidden(INDArray h);

    @Override
    protected void setScoreWithZ(INDArray z) {
        if (layerConf().getLossFunction() == LossFunctions.LossFunction.CUSTOM) {
            LossFunction create = Nd4j.getOpFactory().createLossFunction(layerConf().getCustomLossFunction(), input, z);
            create.exec();
            score = create.getFinalResult().doubleValue();
        }

        else {
            score = LossCalculation.builder()
                    .l1(calcL1()).l2(calcL2())
                    .labels(input).z(z).lossFunction(layerConf().getLossFunction())
                    .miniBatch(conf.isMiniBatch()).miniBatchSize(input.size(0))
                    .useRegularization(conf.isUseRegularization()).build().score();
        }
    }

    public INDArray paramsBackprop(){
        List<INDArray> list = new ArrayList<>(2);
        for(Map.Entry<String,INDArray> entry : params.entrySet()){
            if(!PretrainParamInitializer.VISIBLE_BIAS_KEY.equals(entry.getKey())) list.add(entry.getValue());
        }
        return Nd4j.toFlattened('f', list);
    }

    /**The number of parameters for the model, for backprop (i.e., excluding visible bias)
     * @return the number of parameters for the model (ex. visible bias)
     */
    public int numParamsBackprop() {
        int ret = 0;
        for(Map.Entry<String,INDArray> entry : params.entrySet()){
            if(PretrainParamInitializer.VISIBLE_BIAS_KEY.equals(entry.getKey())) continue;
            ret += entry.getValue().length();
        }
        return ret;
    }

    @Override
    public void setParams(INDArray params) {
        if(params == paramsFlattened) return;   //No op

        //SetParams has two different uses: during pretrain vs. backprop.
        //pretrain = 3 sets of params (inc. visible bias); backprop = 2

        List<String> parameterList = conf.variables();
        int lengthPretrain = 0;
        int lengthBackprop = 0;
        for(String s : parameterList) {
            int len = getParam(s).length();
            lengthPretrain += len;
            if(!PretrainParamInitializer.VISIBLE_BIAS_KEY.equals(s)) lengthBackprop += len;
        }

        boolean pretrain = params.length() == lengthPretrain;
        if( !pretrain && params.length() != lengthBackprop ) {
            throw new IllegalArgumentException("Unable to set parameters: must be of length " + lengthPretrain + " for pretrain, "
                + " or " + lengthBackprop + " for backprop. Is: " + params.length());
        }

        if(!pretrain){
            paramsFlattened.assign(params);
            return;
        }

        int idx = 0;
        Set<String> paramKeySet = this.params.keySet();
        for(String s : paramKeySet) {
            INDArray param = getParam(s);
            INDArray get = params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, idx + param.length()));
            if(param.length() != get.length())
                throw new IllegalStateException("Parameter " + s + " should have been of length " + param.length() + " but was " + get.length());
            param.assign(get.reshape('f',param.shape()));  //Use assign due to backprop params being a view of a larger array
            idx += param.length();

        }

    }

}
