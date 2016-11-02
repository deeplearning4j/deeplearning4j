/*
 *
 *  * Copyright 2016 Skymind,Inc.
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

package org.deeplearning4j.nn.layers.feedforward.embedding;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.Collections;

/**
 * Embedding layer: feed-forward layer that expects single column input, each scalar value being interpreted as an
 * integer - the class index. Typical usage includes transforming word IDs into dense vectors, when performing NLP tasks.
 * Each example may either be:
 * <ul>
 * <li>a scalar, e.g. the word index in a vocabulary. In this case the input is expected to have shape [batchSize, 1]</li>
 * <li>a column vector, e.g. the word indexes for a sequence of words (a document). In this case the input is
 * expected to have shape [batchSize, numWords, 1]</li>
 * </ul>
 * Any other input shape will cause an exception.
 * <p>
 * This embedding layer is equivalent to a dense layer using 1-hot representation as input (either [batchSize, numClasses],
 * or [batchSize, numWords, numClasses]) but is more efficient when the number of classes (vocabulary size) is large.
 * <br>
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding for each input token.
 *
 * @author Alex Black
 */
public class EmbeddingLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.EmbeddingLayer> {
    public EmbeddingLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {

        //If this layer is layer L, then epsilon is (w^(L+1)*(d^(L+1))^T) (or equivalent)
        INDArray z = preOutput(input);
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), z).derivative());
        INDArray delta = epsilon.muli(activationDerivative);

        if (maskArray != null) {
            delta.muliColumnVector(maskArray);
        }

        INDArray weightGradients = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);
        weightGradients.assign(0);

        INDArray linearisedInput = Nd4j.toFlattened(input);
        int[] indexes = new int[linearisedInput.length()];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = linearisedInput.getInt(i);
            weightGradients.getRow(indexes[i]).addi(delta.getRow(i));
        }

        INDArray biasGradientsView = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        INDArray biasGradients = delta.sum(0);
        biasGradientsView.assign(biasGradients);    //TODO do this without the assign...

        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGradients);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGradientsView);

        return new Pair<>(ret, null);    //Don't bother returning epsilons: no layer below this one...
    }

    @Override
    public INDArray preOutput(boolean training) {
        if (input.shape().length < 2 || input.shape().length > 3 || input.shape()[input.shape().length - 1] != 1) {
            throw new IllegalArgumentException("Input should be of shape [batchSize, 1] for single token examples, " +
                    "or [batchSize, numTokens, 1] for multi-token examples. However, got shape " + Arrays.toString(input.shape()));
        }

        INDArray flatInput = Nd4j.toFlattened(input);
        int[] indexes = new int[flatInput.length()];
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = flatInput.getInt(i);
        }

        INDArray weights = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray rows = Nd4j.pullRows(weights, 1, indexes, 'c');
        rows.addiRowVector(bias);

        final int[] outputShape = input.shape().clone();
        outputShape[outputShape.length - 1] = weights.columns();
        rows = rows.reshape(outputShape);
        return rows;
    }

    @Override
    public INDArray activate(boolean training) {
        INDArray rows = preOutput(training);

        INDArray ret = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), rows));
        if (maskArray != null) {
            ret.muliColumnVector(maskArray);
        }
        return ret;
    }

    @Override
    protected void applyDropOutIfNecessary(boolean training) {
        throw new UnsupportedOperationException("Dropout not supported with EmbeddingLayer");
    }

}
