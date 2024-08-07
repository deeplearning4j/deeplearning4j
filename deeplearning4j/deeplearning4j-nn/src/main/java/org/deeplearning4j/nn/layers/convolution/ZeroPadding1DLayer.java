/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers.convolution;

import lombok.val;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.common.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

public class ZeroPadding1DLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.ZeroPadding1DLayer> {

    private int[] padding; // [padLeft, padRight]

    public ZeroPadding1DLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
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
    public Type type() {
        return Type.RECURRENT;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        val inShape = input.shape();

        INDArray epsNext = epsilon.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2]));

        return new Pair<>(new DefaultGradient(), workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsNext));
    }


    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        val inShape = input.shape();
        val paddedOut = inShape[2] + padding[0] + padding[1];
        val outShape = new long[] {inShape[0], inShape[1], paddedOut};

        INDArray out = workspaceMgr.create(ArrayType.ACTIVATIONS, dataType, outShape, 'c');
        out.put(new INDArrayIndex[] {NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(padding[0], padding[0] + inShape[2])}, input);

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS,out);
    }

    @Override
    public Layer clone() {
        return new ZeroPadding1DLayer(conf.clone(), dataType);
    }

    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly){
        return 0;
    }
}
