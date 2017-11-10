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


import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
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

    public ActivationLayer(org.deeplearning4j.nn.conf.layers.ActivationLayer conf) {
        super(conf);
    }

    @Override
    public Gradients backpropGradient(Gradients gradients) {
        INDArray epsilon = gradients.get(0);
        INDArray delta = layerConf().getActivationFn().backprop(input.get(0).dup(), epsilon).getFirst(); //TODO handle activation function params

        if (input.getMask(0) != null) {
            delta.muliColumnVector(input.getMask(0));
        }

        Gradient ret = new DefaultGradient();

        Gradients g = GradientsFactory.getInstance().create(delta, ret);
        return backpropPreprocessor(g);
    }

    @Override
    public Activations activate(boolean training) {
        if (input == null) {
            throw new IllegalArgumentException("Cannot do forward pass with null input " + layerId());
        }
        applyPreprocessorIfNecessary(training);
        applyDropOutIfNecessary(training);

        INDArray input = this.input.get(0);

        INDArray in;
        if (training) {
            //dup required: need to keep original input for backprop
            in = input.dup();
        } else {
            in = input;
        }
        //return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getLayer().getActivationFunction(), in));
        INDArray act = layerConf().getActivationFn().getActivation(in, training);
        return ActivationsFactory.getInstance().create(act);
    }

    @Override
    public Layer clone() {
        return new ActivationLayer((org.deeplearning4j.nn.conf.layers.ActivationLayer)conf.clone());
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //No op
    }


    @Override
    public INDArray params() {
        return null;
    }

}
