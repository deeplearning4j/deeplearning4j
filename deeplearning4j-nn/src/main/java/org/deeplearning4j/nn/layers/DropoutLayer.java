/*-
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
package org.deeplearning4j.nn.layers;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

/**
 * Created by davekale on 12/7/16.
 */
public class DropoutLayer extends BaseLayer<org.deeplearning4j.nn.conf.layers.DropoutLayer> {

    public DropoutLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public DropoutLayer(NeuralNetConfiguration conf, INDArray input) {
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
        INDArray delta = epsilon.dup();

        if (maskArray != null) {
            delta.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();
        return new Pair<>(ret, delta);
    }

    @Override
    public INDArray preOutput(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot perform forward pass with null input " + layerId());
        }
        applyDropOutIfNecessary(training);

        if (maskArray != null) {
            input.muliColumnVector(maskArray);
        }

        return input;
    }

    @Override
    public INDArray activate(boolean training) {
        INDArray z = preOutput(training);
        return z;
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported - " + layerId());
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray params() {
        return null;
    }
}
