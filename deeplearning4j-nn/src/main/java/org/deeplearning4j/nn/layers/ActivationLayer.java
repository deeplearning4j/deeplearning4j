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
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Activation Layer
 *
 * Used to apply activation on input and corresponding derivative on epsilon.
 * Decouples activation from the layer type and ideal for cases when applying
 * BatchNormLayer. For example, use "identity" activation on the layer prior to BatchNorm and
 * apply this layer after the BatchNorm.
 */
public class ActivationLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.ActivationLayer> {

    public ActivationLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ActivationLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public double calcL2() {
        return 0;
    }

    @Override
    public double calcL1() {
        return 0;
    }

    @Override
    public Type type() {
        return Type.FEED_FORWARD;
    }

    @Override
    public void fit(INDArray input) {}

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), input).derivative());
        INDArray delta = epsilon.muli(activationDerivative);

        if(maskArray != null){
            delta.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();
        return new Pair<>(ret,delta);
    }

    @Override
    public INDArray activate(boolean training) {
        if(input == null)
            throw new IllegalArgumentException("No null input allowed");
        applyDropOutIfNecessary(training);

        return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), input));

    }

    @Override
    public Layer transpose(){
        throw new UnsupportedOperationException("Not yet implemented");
    }


    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray params(){
        return null;
    }

}
