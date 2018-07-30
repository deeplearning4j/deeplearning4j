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

package org.deeplearning4j.nn.layers.util;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.nn.workspace.ArrayType;

import java.util.Arrays;

/**
 * MaskLayer applies the mask array to the forward pass activations, and backward pass gradients, passing through
 * this layer. It can be used with 2d (feed-forward), 3d (time series) or 4d (CNN) activations.
 *
 * @author Alex Black
 */
public class MaskLayer extends AbstractLayer<org.deeplearning4j.nn.conf.layers.util.MaskLayer> {
    private Gradient emptyGradient = new DefaultGradient();

    public MaskLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException("Not supported");
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
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        return new Pair<>(emptyGradient, applyMask(epsilon, maskArray, workspaceMgr, ArrayType.ACTIVATION_GRAD));
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return applyMask(input, maskArray, workspaceMgr, ArrayType.ACTIVATIONS);
    }

    private static INDArray applyMask(INDArray input, INDArray maskArray, LayerWorkspaceMgr workspaceMgr, ArrayType type){
        if(maskArray == null){
            return workspaceMgr.leverageTo(type, input);
        }
        switch (input.rank()){
            case 2:
                if(!maskArray.isColumnVectorOrScalar() || maskArray.size(0) != input.size(0)){
                    throw new IllegalStateException("Expected column vector for mask with 2d input, with same size(0)" +
                            " as input. Got mask with shape: " + Arrays.toString(maskArray.shape()) +
                            ", input shape = " + Arrays.toString(input.shape()));
                }
                return workspaceMgr.leverageTo(type, input.mulColumnVector(maskArray));
            case 3:
                //Time series input, shape [Minibatch, size, tsLength], Expect rank 2 mask
                if(maskArray.rank() != 2 || input.size(0) != maskArray.size(0) || input.size(2) != maskArray.size(1)){
                    throw new IllegalStateException("With 3d (time series) input with shape [minibatch, size, sequenceLength]=" +
                            Arrays.toString(input.shape()) + ", expected 2d mask array with shape [minibatch, sequenceLength]." +
                            " Got mask with shape: "+ Arrays.toString(maskArray.shape()));
                }
                INDArray fwd = workspaceMgr.createUninitialized(type, input.shape(), 'f');
                Broadcast.mul(input, maskArray, fwd, 0, 2);
                return fwd;
            case 4:
                //CNN input. Expect column vector to be shape [mb,1,h,1], [mb,1,1,w], or [mb,1,h,w]
                int[] dimensions = new int[4];
                int count = 0;
                for(int i=0; i<4; i++ ){
                    if(input.size(i) == maskArray.size(i)){
                        dimensions[count++] = i;
                    }
                }
                if(count < 4){
                    dimensions = Arrays.copyOfRange(dimensions, 0, count);
                }

                INDArray fwd2 = workspaceMgr.createUninitialized(type, input.shape(), 'c');
                Broadcast.mul(input, maskArray, fwd2, dimensions);
                return fwd2;
            default:
                throw new RuntimeException("Expected rank 2 to 4 input. Got rank " + input.rank() + " with shape "
                        + Arrays.toString(input.shape()));
        }
    }

}
