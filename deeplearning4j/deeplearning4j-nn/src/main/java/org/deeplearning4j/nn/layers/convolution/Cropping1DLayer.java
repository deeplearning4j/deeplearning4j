/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Zero cropping layer for 1D convolutional neural networks.
 * Allows cropping to be done separately for top/bottom
 *
 * @author Max Pumperla
 */
public class Cropping1DLayer extends AbstractLayer<Cropping1D> {

    private int[] cropping; //[padTop, padBottom]

    public Cropping1DLayer(NeuralNetConfiguration conf) {
        super(conf);
        this.cropping = ((org.deeplearning4j.nn.conf.layers.convolutional.Cropping1D) conf.getLayer()).getCropping();
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
        val inShape = input.shape();
        INDArray epsNext = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, inShape, 'c');
        INDArray epsNextSubset = inputSubset(epsNext, ArrayType.ACTIVATION_GRAD, workspaceMgr);
        epsNextSubset.assign(epsilon);
        return new Pair<>((Gradient) new DefaultGradient(), epsNext);
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        return inputSubset(input, ArrayType.ACTIVATIONS, workspaceMgr);
    }

    @Override
    public Layer clone() {
        return new Cropping2DLayer(conf.clone());
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0;
    }

    private INDArray inputSubset(INDArray from, ArrayType arrayType, LayerWorkspaceMgr workspaceMgr){
        return workspaceMgr.leverageTo(arrayType, from.get(all(), all(), interval(cropping[0], from.size(2)-cropping[1])));
    }
}
