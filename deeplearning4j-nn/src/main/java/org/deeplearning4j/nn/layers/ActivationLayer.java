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


import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;


/**
 * Activation Layer
 *
 * Used to apply activation on input and corresponding derivative on epsilon.
 * Decouples activation from the layer type and ideal for cases when applying
 * BatchNormLayer. For example, use "identity" activation on the layer prior to BatchNorm and
 * apply this layer after the BatchNorm.
 */
public class ActivationLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.ActivationLayer> {

    public ActivationLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public ActivationLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
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
        INDArray delta = layerConf().getActivationFn().backprop(input.dup(), epsilon).getFirst(); //TODO handle activation function params

        if (maskArray != null) {
            delta.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();
        return new Pair<>(ret, delta);
    }

    @Override
    public INDArray activate(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot do forward pass with null input " + layerId());
        }
        applyDropOutIfNecessary(training, getIterationCount(), getEpochCount());

        INDArray in;
        if (training) {
            //dup required: need to keep original input for backprop
            in = input.dup();
        } else {
            in = input;
        }
        //return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), in));
        return layerConf().getActivationFn().getActivation(in, training);

    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported - " + layerId());
    }

    @Override
    public Layer clone() {
        return new ActivationLayer(conf.clone());
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }


    @Override
    public Gradient calcGradient(Gradient layerError, INDArray indArray) {
        throw new UnsupportedOperationException("Not supported - " + layerId());
    }

    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException("Not supported - " + layerId());
    }

    @Override
    public INDArray activationMean() {
        return activate(false);
    }


    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray preOutput(boolean training) {
        return null;
    }

}
