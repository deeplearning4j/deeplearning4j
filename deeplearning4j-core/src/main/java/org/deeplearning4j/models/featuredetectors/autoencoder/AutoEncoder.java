/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.models.featuredetectors.autoencoder;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
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
public class AutoEncoder extends BasePretrainNetwork  {

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
        INDArray ret = encode(v);
        return new Pair<>(ret,ret);
    }

    @Override
    public Pair<INDArray, INDArray> sampleVisibleGivenHidden(
            INDArray h) {
        INDArray ret = decode(h);
        return new Pair<>(ret,ret);
    }

    // Encode
    public INDArray encode(INDArray x) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray hBias = getParam(PretrainParamInitializer.BIAS_KEY);

        INDArray preAct;
        if(conf.isConcatBiases()) {
            INDArray concat = Nd4j.hstack(W,hBias.transposei());
            preAct =  x.mmul(concat);
        }
        else {
          preAct = x.mmul(W).addiRowVector(hBias);
        }

        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), preAct));
    }

    // Decode
    public INDArray decode(INDArray y) {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);
        INDArray vBias = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);

        if(conf.isConcatBiases()) {
            //row already accounted for earlier
            INDArray preAct = y.mmul(W.transposei());
            preAct = Nd4j.hstack(preAct,Nd4j.ones(preAct.rows(),1));
            return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), preAct));

        }
        else {
            INDArray preAct = y.mmul(W.transposei());
            preAct.addiRowVector(vBias);
            return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), preAct));

        }

    }

    @Override
    public INDArray transform(INDArray x) {
        INDArray y = encode(x);
        return decode(y);
    }

    @Override
    public  Gradient gradient() {
        INDArray W = getParam(PretrainParamInitializer.WEIGHT_KEY);

        double corruptionLevel = conf.getCorruptionLevel();

        INDArray corruptedX = corruptionLevel > 0 ? getCorruptedInput(input, corruptionLevel) : input;
        INDArray y = encode(corruptedX);

        INDArray z = decode(y);
        INDArray visibleLoss =  input.sub(z);
        INDArray hiddenLoss = conf.getSparsity() == 0 ? visibleLoss.mmul(W).muli(y).muli(y.rsub(1)) :
                visibleLoss.mmul(W).muli(y).muli(y.add(-conf.getSparsity()));


        INDArray wGradient = corruptedX.transposei().mmul(hiddenLoss).addi(visibleLoss.transposei().mmul(y));

        INDArray hBiasGradient = hiddenLoss.mean(0);
        INDArray vBiasGradient = visibleLoss.mean(0);

        return createGradient(wGradient, vBiasGradient, hBiasGradient);
    }
}
