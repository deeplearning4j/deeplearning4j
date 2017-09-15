/*-
 *
 *  * Copyright 2017 Skymind,Inc.
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
package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Zero padding 1D layer for convolutional neural networks.
 * Allows padding to be done separately for left and right boundaries.
 *
 * @author Max Pumperla
 */
public class ZeroPadding1DLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer> {

    private int[] padding; // [padLeft, padRight]

    public ZeroPadding1DLayer(NeuralNetConfiguration conf) {
        super(conf);
        this.padding = ((org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer) conf.getLayer()).getPadding();
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
    public Gradients backpropGradient(Gradients epsilon) {
        int[] inShape = input.shape();

        INDArray epsNext = epsilon.get(0).get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2]));

        return GradientsFactory.getInstance().create(epsNext, null);
    }


    @Override
    public Activations activate(boolean training) {
        int[] inShape = input.shape();
        int paddedOut = inShape[2] + padding[0] + padding[1];
        int[] outShape = new int[] {inShape[0], inShape[1], paddedOut};

        INDArray out = Nd4j.create(outShape);
        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2])}, input);

        return ActivationsFactory.getInstance().create(out);
    }

    @Override
    public Layer clone() {
        return new ZeroPadding1DLayer(conf.clone());
    }
}
