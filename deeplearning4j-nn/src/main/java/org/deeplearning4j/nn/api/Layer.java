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

package org.deeplearning4j.nn.api;


import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;
import java.util.Collection;

/**
 * Interface for a layer of a neural network.
 * This has an activation function, an input and output size,
 * weights, and a bias
 *
 * @author Adam Gibson
 */
public interface Layer extends Serializable, Cloneable, Model {

    enum Type {
        FEED_FORWARD, RECURRENT, CONVOLUTIONAL, SUBSAMPLING, RECURSIVE, MULTILAYER, NORMALIZATION
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

    /**Calculate the l2 regularization term<br>
     * 0.0 if regularization is not used. Or 0.5 * l2Coeff * l2Magnitude otherwise.<br>
     * Note that this does not divide by mini-batch size
     * @param backpropOnlyParams If true: calculate L2 based on backprop params only. If false: calculate
     *                           based on all params (including pretrain params, if any)
     * @return the l2 regularization term for this layer.
     */
    double calcL2(boolean backpropOnlyParams);

    /**Calculate the l1 regularization term<br>
     * 0.0 if regularization is not used. Or l1Coeff * l1Magnitude otherwise.<br>
     * Note that this does not divide by mini-batch size
     * @param backpropOnlyParams If true: calculate L1 based on backprop params only. If false: calculate
     *                           based on all params (including pretrain params, if any)
     * @return the l1 regularization term for this layer.
     */
    double calcL1(boolean backpropOnlyParams);

    /**
     * Returns the layer type
     * @return
     */
    Type type();

    /**
     * Calculate error with respect to the
     * current layer.
     *
     * This gradient will contain the error signal
     * @param input the gradient for the forward layer
     *              If this is the final layer, it will start
     *              with the error from the output.
     *              This is on the user to initialize.
     * @return the gradient wrt the parameters
     * on the current layer
     * @deprecated As of 0.7.3 - Feb 2017. No longer used.
     */
    @Deprecated
    Gradient error(INDArray input);



    /**
     * Take the derivative of the given input
     * based on the activation
     * @param input the input to take the derivative of
     * @return the derivative of the action
     * @deprecated As of 0.7.3 - Feb 2017. No longer used.
     */
    @Deprecated
    INDArray derivativeActivation(INDArray input);


    /**
     * Calculate the gradient
     * @param layerError the layer error
     * @param indArray
     * @return the gradient
     * @deprecated As of 0.7.3 - Feb 2017. No longer used.
     */
    @Deprecated
    Gradient calcGradient(Gradient layerError, INDArray indArray);


    /**Calculate the gradient relative to the error in the next layer
     * @param epsilon w^(L+1)*delta^(L+1). Or, equiv: dC/da, i.e., (dC/dz)*(dz/da) = dC/da, where C 
     * 	is cost function a=sigma(z) is activation.
     * @return Pair<Gradient,INDArray> where Gradient is gradient for this layer, INDArray is epsilon needed by next
     *  layer, but before element-wise multiply by sigmaPrime(z). So for standard feed-forward layer, if this layer is
     *  L, then return.getSecond() == (w^(L)*(delta^(L))^T)^T
     */
    Pair<Gradient, INDArray> backpropGradient(INDArray epsilon);


    /**
     * Parameter averaging
     * @param layer the layer to merge
     * @param batchSize the batch size to merge on
     * @deprecated As of 0.7.3 - Feb 2017. No longer used. Merging (for parameter averaging) done via alternative means
     */
    @Deprecated
    void merge(Layer layer, int batchSize);


    /**
     * Calculate the mean representation
     * for the activation for this layer
     * @return the activation mean for this layer
     * @deprecated As of 0.7.3 - Feb 2017. No longer used.
     */
    @Deprecated
    INDArray activationMean();

    /**
     * Raw activations
     * @param x the input to transform
     * @return the raw activation
     * for this layer
     */
    INDArray preOutput(INDArray x);



    /**
     * Raw activations
     * @param x the input to transform
     * @return the raw activation
     * for this layer
     */
    INDArray preOutput(INDArray x, TrainingMode training);


    /**
     * Trigger an activation with the last specified input
     * @param training  training or test mode
     * @return the activation of the last specified input
     */
    INDArray activate(TrainingMode training);

    /**
     * Initialize the layer with the given input
     * and return the activation for this layer
     * given this input
     * @param input the input to use
     * @param training  train or test mode
     * @return
     */
    INDArray activate(INDArray input, TrainingMode training);


    /**
     * Raw activations
     * @param x the input to transform
     * @return the raw activation
     * for this layer
     */
    INDArray preOutput(INDArray x, boolean training);


    /**
     * Trigger an activation with the last specified input
     * @param training  training or test mode
     * @return the activation of the last specified input
     */
    INDArray activate(boolean training);

    /**
     * Initialize the layer with the given input
     * and return the activation for this layer
     * given this input
     * @param input the input to use
     * @param training  train or test mode
     * @return
     */
    INDArray activate(INDArray input, boolean training);

    /**
     * Trigger an activation with the last specified input
     * @return the activation of the last specified input
     */
    INDArray activate();

    /**
     * Initialize the layer with the given input
     * and return the activation for this layer
     * given this input
     * @param input the input to use
     * @return
     */
    INDArray activate(INDArray input);

    /**
     * Return a transposed copy of the weights/bias
     * (this means reverse the number of inputs and outputs on the weights)
     *
     * @return the transposed layer
     */
    Layer transpose();

    /**
     * Clone the layer
     * @return
     */
    Layer clone();



    /**
     * Get the iteration listeners for this layer.
     */
    Collection<IterationListener> getListeners();

    /**
     * Set the iteration listeners for this layer.
     */
    void setListeners(IterationListener... listeners);

    /**
     * Set the iteration listeners for this layer.
     */
    void setListeners(Collection<IterationListener> listeners);

    /**
     * Set the layer index.
     */
    void setIndex(int index);

    /**
     * Get the layer index.
     */
    int getIndex();

    int getIterationCount();

    int getEpochCount();

    void setIterationCount(int iterationCount);

    void setEpochCount(int epochCount);

    /**
     * Get the layer input.
     */
    void setInput(INDArray input);

    /** Set current/last input mini-batch size.<br>
     * Used for score and gradient calculations. Mini batch size may be different from
     * getInput().size(0) due to reshaping operations - for example, when using RNNs with
     * DenseLayer and OutputLayer. Called automatically during forward pass.
     */
    void setInputMiniBatchSize(int size);

    /** Get current/last input mini-batch size, as set by setInputMiniBatchSize(int)
     * @see Layer#setInputMiniBatchSize(int)
     */
    int getInputMiniBatchSize();

    /**
     * Set the mask array. Note: In general, {@link #feedForwardMaskArray(INDArray, MaskState, int)} should be used in
     * preference to this.
     * @param maskArray Mask array to set
     */
    void setMaskArray(INDArray maskArray);


    INDArray getMaskArray();

    /**
     * Returns true if the layer can be trained in an unsupervised/pretrain manner (VAE, RBMs etc)
     *
     * @return true if the layer can be pretrained (using fit(INDArray), false otherwise
     */
    boolean isPretrainLayer();


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
     * @return                 New mask array after this layer, along with the new mask state.
     */
    Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize);
}
