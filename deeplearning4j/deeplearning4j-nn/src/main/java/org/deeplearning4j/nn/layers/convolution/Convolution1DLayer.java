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

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.util.ConvolutionUtils;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Broadcast;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import java.util.Arrays;

/**
 * 1D (temporal) convolutional layer. Currently, we just subclass off the
 * ConvolutionLayer and override the preOutput and backpropGradient methods.
 * Specifically, since this layer accepts RNN (not CNN) InputTypes, we
 * need to add a singleton fourth dimension before calling the respective
 * superclass method, then remove it from the result.
 *
 * This approach treats a multivariate time series with L timesteps and
 * P variables as an L x 1 x P image (L rows high, 1 column wide, P
 * channels deep). The kernel should be H<L pixels high and W=1 pixels
 * wide.
 *
 * TODO: We will eventually want to add a 1D-specific im2col method.
 *
 * @author dave@skymind.io
 */
public class Convolution1DLayer extends ConvolutionLayer {
    public Convolution1DLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public Convolution1DLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if (epsilon.rank() != 3)
            throw new DL4JInvalidInputException("Got rank " + epsilon.rank()
                            + " array as epsilon for Convolution1DLayer backprop with shape "
                            + Arrays.toString(epsilon.shape())
                            + ". Expected rank 3 array with shape [minibatchSize, features, length]. " + layerId());

        if(maskArray != null){
            INDArray maskOut = feedForwardMaskArray(maskArray, MaskState.Active, (int)epsilon.size(0)).getFirst();
            Preconditions.checkState(epsilon.size(0) == maskOut.size(0) && epsilon.size(2) == maskOut.size(1),
                    "Activation gradients dimensions (0,2) and mask dimensions (0,1) don't match: Activation gradients %s, Mask %s",
                    epsilon.shape(), maskOut.shape());
            Broadcast.mul(epsilon, maskOut, epsilon, 0, 2);
        }

        // add singleton fourth dimension to input and next layer's epsilon
        epsilon = epsilon.reshape(epsilon.size(0), epsilon.size(1), epsilon.size(2), 1);
        INDArray origInput = input;
        input = input.reshape(input.size(0), input.size(1), input.size(2), 1);

        // call 2D ConvolutionLayer's backpropGradient method
        Pair<Gradient, INDArray> gradientEpsNext = super.backpropGradient(epsilon, workspaceMgr);
        INDArray epsNext = gradientEpsNext.getSecond();

        // remove singleton fourth dimension from input and current epsilon
        epsNext = epsNext.reshape(epsNext.size(0), epsNext.size(1), epsNext.size(2));
        input = origInput;

        return new Pair<>(gradientEpsNext.getFirst(), epsNext);
    }

    @Override
    protected Pair<INDArray, INDArray> preOutput4d(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        Pair<INDArray,INDArray> preOutput = super.preOutput(true, forBackprop, workspaceMgr);
        INDArray p3d = preOutput.getFirst();
        INDArray p = preOutput.getFirst().reshape(p3d.size(0), p3d.size(1), p3d.size(2), 1);
        preOutput.setFirst(p);
        return preOutput;
    }

    @Override
    protected Pair<INDArray,INDArray> preOutput(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        INDArray origInput = input;
        input = input.reshape(input.size(0), input.size(1), input.size(2), 1);

        // call 2D ConvolutionLayer's activate method
        Pair<INDArray,INDArray> preOutput = super.preOutput(training, forBackprop, workspaceMgr);

        // remove singleton fourth dimension from output activations
        input = origInput;
        INDArray p4d = preOutput.getFirst();
        INDArray p = preOutput.getFirst().reshape(p4d.size(0), p4d.size(1), p4d.size(2));
        preOutput.setFirst(p);

        return preOutput;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr){
        INDArray act4d = super.activate(training, workspaceMgr);
        INDArray act3d = act4d.reshape(act4d.size(0), act4d.size(1), act4d.size(2));

        if(maskArray != null){
            INDArray maskOut = feedForwardMaskArray(maskArray, MaskState.Active, (int)act3d.size(0)).getFirst();
            Preconditions.checkState(act3d.size(0) == maskOut.size(0) && act3d.size(2) == maskOut.size(1),
                    "Activations dimensions (0,2) and mask dimensions (0,1) don't match: Activations %s, Mask %s",
                    act3d.shape(), maskOut.shape());
            Broadcast.mul(act3d, maskOut, act3d, 0, 2);
        }

        return workspaceMgr.leverageTo(ArrayType.ACTIVATIONS, act3d);   //Should be zero copy most of the time
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                                                          int minibatchSize) {
        INDArray reduced = ConvolutionUtils.cnn1dMaskReduction(maskArray, layerConf().getKernelSize()[0],
                layerConf().getStride()[0], layerConf().getPadding()[0], layerConf().getDilation()[0],
                layerConf().getConvolutionMode());
        return new Pair<>(reduced, currentMaskState);
    }
}
