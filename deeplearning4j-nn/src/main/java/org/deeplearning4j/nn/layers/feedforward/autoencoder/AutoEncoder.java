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

package org.deeplearning4j.nn.layers.feedforward.autoencoder;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 *  Autoencoder.
 * Add Gaussian noise to input and learn
 * a reconstruction function.
 *
 * @author Adam Gibson
 *
 */
public class AutoEncoder extends BasePretrainNetwork<org.deeplearning4j.nn.conf.layers.AutoEncoder>  {

    private static final long serialVersionUID = -6445530486350763837L;

    public AutoEncoder(NeuralNetConfiguration conf) {
        super(conf);
    }

    public AutoEncoder(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Pair<INDArray, INDArray> sampleHiddenGivenVisible(
            INDArray v) {
        setInput(v);
        INDArray ret = encode(true);
        return new Pair<>(ret,ret);
    }

    @Override
    public Pair<INDArray, INDArray> sampleVisibleGivenHidden(
            INDArray h) {
        INDArray ret = decode(h);
        return new Pair<>(ret,ret);
    }

    // Encode
    public INDArray encode(boolean training) {
        if(conf.getLayer().getDropOut() > 0 && training) {
            Dropout.applyDropout(input, conf.getLayer().getDropOut());
        }

        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);

        INDArray preAct = input.mmul(W).addiRowVector(hBias);

        INDArray ret = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), preAct));

        return ret;
    }

    // Decode
    public INDArray decode(INDArray y) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);
        INDArray preAct = y.mmul(W.transposei());
        preAct.addiRowVector(vBias);
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), preAct));

    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        setInput(input);
        return encode(training);
    }

    @Override
    public INDArray activate(INDArray input) {
        setInput(input);
        return encode(true);
    }

    @Override
    public INDArray activate(boolean training) {
        return decode(encode(training));
    }

    @Override
    public INDArray activate() {
        return decode(encode(false));
    }

    @Override
    public void computeGradientAndScore() {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);

        double corruptionLevel = layerConf().getCorruptionLevel();

        INDArray corruptedX = corruptionLevel > 0 ? getCorruptedInput(input, corruptionLevel) : input;
        setInput(corruptedX);
        INDArray y = encode(true);

        INDArray z = decode(y);
        INDArray visibleLoss =  input.sub(z);
        INDArray hiddenLoss = layerConf().getSparsity() == 0 ? visibleLoss.mmul(W).muli(y).muli(y.rsub(1)) :
                visibleLoss.mmul(W).muli(y).muli(y.add(-layerConf().getSparsity()));


        INDArray wGradient = corruptedX.transposei().mmul(hiddenLoss).addi(visibleLoss.transposei().mmul(y));

        INDArray hBiasGradient = hiddenLoss.sum(0);
        INDArray vBiasGradient = visibleLoss.sum(0);

        gradient = createGradient(wGradient, vBiasGradient, hBiasGradient);
        setScoreWithZ(z);

    }


}
