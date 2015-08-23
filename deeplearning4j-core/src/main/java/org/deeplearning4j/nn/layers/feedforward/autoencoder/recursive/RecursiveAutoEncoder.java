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

package org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive;



import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.RecursiveParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

/**
 *
 * Recursive AutoEncoder.
 * Uses back propagation through structure.
 *
 * @author Adam Gibson
 */
public class RecursiveAutoEncoder extends BaseLayer<org.deeplearning4j.nn.conf.layers.RecursiveAutoEncoder> {
    private INDArray currInput = null,
                     allInput = null,
                     visibleLoss = null,
                     hiddenLoss = null,
                     cLoss = null,
                     bLoss = null,
                     y = null;
    double currScore = 0.0;
    public RecursiveAutoEncoder(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Type type() {
        return Type.RECURSIVE;
    }

    @Override
    public double score() {
        return currScore;
    }

    @Deprecated
    private double scoreSnapShot() {
        return 0.5 * pow(y.sub(allInput),2).mean(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public void computeGradientAndScore() {
        gradient();
        score = 0.5 * pow(y.sub(allInput),2).mean(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public INDArray activate(INDArray data) {
        INDArray w = getParam(RecursiveParamInitializer.ENCODER_WEIGHT_KEY);
        INDArray c = getParam(RecursiveParamInitializer.DECODER_WEIGHT_KEY);
        INDArray inputTimesW = data.mmul(w);
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), inputTimesW.addiRowVector(c)));
    }


    public INDArray decode(INDArray input) {
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), input.mmul(params.get(RecursiveParamInitializer.DECODER_WEIGHT_KEY).addiRowVector(params.get(RecursiveParamInitializer.HIDDEN_BIAS_KEY)))));
    }


    @Override
    public void iterate(INDArray input) {

    }

    @Override
    public Gradient gradient() {
       /**
         * Going up the tree involves repeated calculations
        * using the output of the previous autoencoder
         * for the next.
         * This starts with a base case at x[0] and x[1]
        * and expands to subsequent layers.
         *
         * The error is the sum going up the tree.
         */
        currScore = 0.0;
        for(int i = 0; i < input.rows(); i++) {
            INDArray combined = currInput == null ? Nd4j.concat(0,input.slice(i),input.slice(i + 1)) : Nd4j.concat(0,input.slice(i),currInput);
            //combine first 2: aka base case
            if(i == 0) {
                i++;
            }

            currInput = combined;
            allInput = combined;
            INDArray encoded = activate(combined);
            y = decode(encoded);

            INDArray currVisibleLoss = currInput.sub(y);
            INDArray currHiddenLoss = currVisibleLoss.mmul(getParam(RecursiveParamInitializer.ENCODER_WEIGHT_KEY)).muli(encoded).muli(encoded.rsub(1));

            INDArray hiddenGradient = y.transpose().mmul(currHiddenLoss);
            INDArray visibleGradient = encoded.transpose().mmul(currVisibleLoss);

            if(visibleLoss == null)
                visibleLoss = visibleGradient;
            else
                visibleLoss.addi(visibleGradient);



            if(hiddenLoss == null)
                hiddenLoss = hiddenGradient;
            else
                hiddenLoss.addi(hiddenGradient);

            INDArray currCLoss = currVisibleLoss.isMatrix() ? currVisibleLoss.mean(0) : currVisibleLoss;
            INDArray currBLoss = currHiddenLoss.isMatrix() ? currHiddenLoss.mean(0) : currHiddenLoss;


            if(cLoss == null)
                cLoss = currCLoss;
            else
                cLoss.addi(currCLoss);
            if(bLoss == null)
                bLoss = currBLoss;
            else
                bLoss.addi(currBLoss);
            currInput = encoded;
            currScore += scoreSnapShot();
        }

        return createGradient(hiddenLoss,visibleLoss,cLoss,bLoss);
    }



}
