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

package org.deeplearning4j.nn.api;


import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.Collection;

/**
 * Interface for a layer of a neural network.
 * This has an activation function, an input and output size,
 * weights, and a bias
 *
 * @author Adam Gibson
 */
public interface Layer extends Serializable, Cloneable, Model, Trainable {

    enum Type {
        FEED_FORWARD, RECURRENT, CONVOLUTIONAL, CONVOLUTIONAL3D,
        SUBSAMPLING, UPSAMPLING, RECURSIVE, MULTILAYER, NORMALIZATION
    }

    enum TrainingMode {
        TRAIN, TEST
    }

    /**
     * This method sets given CacheMode for current layer
     *
     * @param mode
     */
    void setCacheMode(CacheMode mode);

    /**
     * Calculate the l2 regularization term<br>
     * 0.0 if regularization is not used. Or 0.5 * l2Coeff * l2Magnitude otherwise.<br>
     * Note that this does not divide by mini-batch size
     *
     * @param backpropOnlyParams If true: calculate L2 based on backprop params only. If false: calculate
     *                           based on all params (including pretrain params, if any)
     * @return the l2 regularization term for this layer.
     */
    double calcL2(boolean backpropOnlyParams);

    /**
     * Calculate the l1 regularization term<br>
     * 0.0 if regularization is not used. Or l1Coeff * l1Magnitude otherwise.<br>
     * Note that this does not divide by mini-batch size
     *
     * @param backpropOnlyParams If true: calculate L1 based on backprop params only. If false: calculate
     *                           based on all params (including pretrain params, if any)
     * @return the l1 regularization term for this layer.
     */
    double calcL1(boolean backpropOnlyParams);

    /**
     * Returns the layer type
     *
     * @return
     */
    Type type();


    /**
     * Calculate the gradient relative to the error in the next layer
     *
     * @param epsilon      w^(L+1)*delta^(L+1). Or, equiv: dC/da, i.e., (dC/dz)*(dz/da) = dC/da, where C
     *                     is cost function a=sigma(z) is activation.
     * @param workspaceMgr Workspace manager
     * @return Pair<Gradient   ,   INDArray> where Gradient is gradient for this layer, INDArray is epsilon (activation gradient)
     * needed by next layer, but before element-wise multiply by sigmaPrime(z). So for standard feed-forward layer, if this layer is
     * L, then return.getSecond() == dL/dIn = (w^(L)*(delta^(L))^T)^T. Note that the returned array should be placed in the
     * {@link org.deeplearning4j.nn.workspace.ArrayType#ACTIVATION_GRAD} workspace via the workspace manager
     */
    Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr);


    /**
     * Perform forward pass and return the activations array with the last set input
     *
     * @param training     training or test mode
     * @param workspaceMgr Workspace manager
     * @return the activation (layer output) of the last specified input. Note that the returned array should be placed
     * in the {@link org.deeplearning4j.nn.workspace.ArrayType#ACTIVATIONS} workspace via the workspace manager
     */
    INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr);

    /**
     * Perform forward pass and return the activations array with the specified input
     *
     * @param input    the input to use
     * @param training train or test mode
     * @param mgr      Workspace manager.
     * @return Activations array. Note that the returned array should be placed in the
     * {@link org.deeplearning4j.nn.workspace.ArrayType#ACTIVATIONS} workspace via the workspace manager
     */
    INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr mgr);

    /**
     * Get the iteration listeners for this layer.
     */
    Collection<TrainingListener> getListeners();

    /**
     * Set the iteration listeners for this layer.
     */
    void setListeners(TrainingListener... listeners);

    /**
     * Set the iteration listeners for this layer.
     */
    void setListeners(Collection<TrainingListener> listeners);

    /**
     * Set the layer index.
     */
    void setIndex(int index);

    /**
     * Get the layer index.
     */
    int getIndex();

    /**
     * @return The current iteration count (number of parameter updates) for the layer/network
     */
    int getIterationCount();

    /**
     * @return The current epoch count (number of training epochs passed) for the layer/network
     */
    int getEpochCount();

    /**
     * Set the current iteration count (number of parameter updates) for the layer/network
     */
    void setIterationCount(int iterationCount);

    /**
     * Set the current epoch count (number of epochs passed ) for the layer/network
     */
    void setEpochCount(int epochCount);

    /**
     * Set the layer input.
     */
    void setInput(INDArray input, LayerWorkspaceMgr workspaceMgr);

    /**
     * Set current/last input mini-batch size.<br>
     * Used for score and gradient calculations. Mini batch size may be different from
     * getInput().size(0) due to reshaping operations - for example, when using RNNs with
     * DenseLayer and OutputLayer. Called automatically during forward pass.
     */
    void setInputMiniBatchSize(int size);

    /**
     * Get current/last input mini-batch size, as set by setInputMiniBatchSize(int)
     *
     * @see Layer#setInputMiniBatchSize(int)
     */
    int getInputMiniBatchSize();

    /**
     * Set the mask array. Note: In general, {@link #feedForwardMaskArray(INDArray, MaskState, int)} should be used in
     * preference to this.
     *
     * @param maskArray Mask array to set
     */
    void setMaskArray(INDArray maskArray);


    INDArray getMaskArray();

    /**
     * Returns true if the layer can be trained in an unsupervised/pretrain manner (AE, VAE, etc)
     *
     * @return true if the layer can be pretrained (using fit(INDArray), false otherwise
     */
    boolean isPretrainLayer();


    void clearNoiseWeightParams();

    /**
     * A performance optimization: mark whether the layer is allowed to modify its input array in-place. In many cases,
     * this is totally safe - in others, the input array will be shared by multiple layers, and hence it's not safe to
     * modify the input array.
     * This is usually used by ops such as dropout.
     * @param allow If true: the input array is safe to modify. If false: the input array should be copied before it
     *              is modified (i.e., in-place modifications are un-safe)
     */
    void allowInputModification(boolean allow);


    /**
     * Feed forward the input mask array, setting in in the layer as appropriate. This allows different layers to
     * handle masks differently - for example, bidirectional RNNs and normal RNNs operate differently with masks (the
     * former sets activations to 0 outside of the data present region (and keeps the mask active for future layers like
     * dense layers), whereas normal RNNs don't zero out the activations/errors )instead relying on backpropagated error
     * arrays to handle the variable length case.<br>
     * This is also used for example for networks that contain global pooling layers, arbitrary preprocessors, etc.
     *
     * @param maskArray        Mask array to set
     * @param currentMaskState Current state of the mask - see {@link MaskState}
     * @param minibatchSize    Current minibatch size. Needs to be known as it cannot always be inferred from the activations
     *                         array due to reshaping (such as a DenseLayer within a recurrent neural network)
     * @return New mask array after this layer, along with the new mask state.
     */
    Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize);

    /**
     * @return Get the layer helper, if any
     */
    LayerHelper getHelper();
}
