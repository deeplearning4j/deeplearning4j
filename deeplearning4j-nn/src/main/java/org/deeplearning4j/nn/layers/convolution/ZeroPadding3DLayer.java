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
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

/**
 * Zero padding 3D layer for convolutional neural networks.
 * Allows padding to be done separately for left and right boundaries
 * in all three spatial input dimensions.
 *
 * @author Max Pumperla
 */
public class ZeroPadding3DLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.ZeroPadding3DLayer> {

    private int[] padding; // [padLeft1, padRight1, padLeft2, padRight2, padLeft3, padRight3]

    public ZeroPadding3DLayer(NeuralNetConfiguration conf) {
        super(conf);
        this.padding = ((org.deeplearning4j.nn.conf.layers.ZeroPadding3DLayer) conf.getLayer()).getPadding();
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
    public Type type() {
        return Type.CONVOLUTIONAL;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        int[] inShape = input.shape();

        INDArray epsNext = epsilon.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                NDArrayIndex.interval(padding[2], padding[2] + inShape[3]),
                NDArrayIndex.interval(padding[4], padding[4] + inShape[4]));

        epsNext = workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsNext);
        return new Pair<>((Gradient) new DefaultGradient(), epsNext);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        int[] inShape = input.shape();
        int outD = inShape[2] + padding[0] + padding[1];
        int outH = inShape[3] + padding[2] + padding[3];
        int outW = inShape[4] + padding[4] + padding[5];
        int[] outShape = new int[] {inShape[0], inShape[1], outD, outH, outW};

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, outShape, 'c');

        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2]),
                NDArrayIndex.interval(padding[2], padding[2] + inShape[3]),
                NDArrayIndex.interval(padding[4], padding[4] + inShape[4])},
                input);

        return out;
    }

    @Override
    public Layer clone() {
        return new ZeroPadding3DLayer(conf.clone());
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }
}
