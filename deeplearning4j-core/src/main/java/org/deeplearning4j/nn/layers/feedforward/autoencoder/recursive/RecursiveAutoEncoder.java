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
                     vbLoss = null,
                     bLoss = null,
                     y = null,
                     z = null;
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

    public INDArray encode(boolean training) {
        INDArray w = getParam(RecursiveParamInitializer.ENCODER_WEIGHT_KEY);
        INDArray b = getParam(RecursiveParamInitializer.HIDDEN_BIAS_KEY);
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), currInput.mmul(w).addiRowVector(b)));
    }

    public INDArray decode(INDArray activation) {
        INDArray U = getParam(RecursiveParamInitializer.DECODER_WEIGHT_KEY);
        INDArray vb = getParam(RecursiveParamInitializer.VISIBLE_BIAS_KEY);
        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), activation.mmul(U).addiRowVector(vb)));
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
    public void iterate(INDArray input) {
    }

    // TODO review code below to confirm computation
    @Override
    public void computeGradientAndScore() {
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
            y = encode(true);
            z = decode(y);

            INDArray currVisibleLoss = currInput.sub(z);
            INDArray currHiddenLoss = currVisibleLoss.mmul(getParam(RecursiveParamInitializer.ENCODER_WEIGHT_KEY)).muli(y).muli(y.rsub(1));

            INDArray hiddenGradient = z.transpose().mmul(currHiddenLoss);
            INDArray visibleGradient = y.transpose().mmul(currVisibleLoss);

            if(visibleLoss == null)
                visibleLoss = visibleGradient;
            else
                visibleLoss.addi(visibleGradient);

            if(hiddenLoss == null)
                hiddenLoss = hiddenGradient;
            else
                hiddenLoss.addi(hiddenGradient);

            INDArray currVBLoss = currVisibleLoss.isMatrix() ? currVisibleLoss.mean(0) : currVisibleLoss;
            INDArray currBLoss = currHiddenLoss.isMatrix() ? currHiddenLoss.mean(0) : currHiddenLoss;

            if(vbLoss == null)
                vbLoss = currVBLoss;
            else
                vbLoss.addi(currVBLoss);
            if(bLoss == null)
                bLoss = currBLoss;
            else
                bLoss.addi(currBLoss);
            // TODO is this following line needed - it  needs to be size that maps to the input for the next iteration
//            currInput = encoded;

            currScore += 0.5 * pow(z.sub(allInput),2).mean(Integer.MAX_VALUE).getDouble(0);
        }

        gradient = createGradient(hiddenLoss,visibleLoss,bLoss,vbLoss);
        score = currScore;
    }

}
