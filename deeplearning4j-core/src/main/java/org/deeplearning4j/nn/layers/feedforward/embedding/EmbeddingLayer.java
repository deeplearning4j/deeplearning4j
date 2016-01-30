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

/**Embedding layer: feed-forward layer that expects single integers per example as input (class numbers, in range 0 to numClass-1)
 * as input. This input has shape [numExamples,1] instead of [numExamples,numClasses] for the equivalent one-hot representation.
 * Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot representation for the input; however,
 * it can be much more efficient with a large number of classes (as a dense layer + one-hot input does a matrix multiply
 * with all but one value being zero).<br>
 * <b>Note</b>: can only be used as the first layer for a network<br>
 * <b>Note 2</b>: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding for each example.
 * @author Alex Black
 */
public class EmbeddingLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.EmbeddingLayer> {
    public EmbeddingLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon){

        //If this layer is layer L, then epsilon is (w^(L+1)*(d^(L+1))^T) (or equivalent)
        INDArray z = preOutput(input);
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), z).derivative());
        INDArray delta = epsilon.muli(activationDerivative);

        if(maskArray != null){
            delta.muliColumnVector(maskArray);
        }

        INDArray weights = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray weightGradients = Nd4j.zeros(weights.shape());

        int[] indexes = new int[input.length()];
        for( int i=0; i<indexes.length; i++ ){
            indexes[i] = input.getInt(i,0);

            weightGradients.getRow(indexes[i]).addi(delta.getRow(i));
        }

        INDArray biasGradients = delta.sum(0);

        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGradients);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGradients);

        return new Pair<>(ret,null);    //Don't bother returning epsilons: no layer below this one...
    }

    @Override
    public INDArray preOutput(boolean training){
        if(input.columns() != 1){
            //Assume shape is [numExamples,1], and each entry is an integer index
            throw new IllegalStateException("Cannot do forward pass for embedding layer with input more than one column. "
                    + "Expected input shape: [numExamples,1] with each entry being an integer index");
        }

        int[] indexes = new int[input.length()];
        for( int i=0; i<indexes.length; i++ ) indexes[i] = input.getInt(i,0);

        INDArray weights = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray bias = getParam(DefaultParamInitializer.BIAS_KEY);

        //INDArray rows = weights.getRows(indexes);
        INDArray rows = Nd4j.create(indexes.length,weights.size(1));
        for( int i=0; i<indexes.length; i++ ){
            rows.getRow(i).assign(weights.getRow(indexes[i]));
        }
        rows.addiRowVector(bias);

        return rows;
    }

    @Override
    public INDArray activate(boolean training){
        INDArray rows = preOutput(training);

        INDArray ret =  Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), rows));
        if(maskArray != null){
            ret.muliColumnVector(maskArray);
        }
        return ret;
    }

    @Override
    protected void applyDropOutIfNecessary(boolean training){
        throw new UnsupportedOperationException("Dropout not supported with EmbeddingLayer");
    }

}
