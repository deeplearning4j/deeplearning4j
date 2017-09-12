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


import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

/**
 * Interface for a layer of a neural network.
 * This has an activation function, an input and output size,
 * weights, and a bias
 *
 * @author Adam Gibson
 */
public interface Layer extends Serializable, Cloneable {

    enum Type {
        FEED_FORWARD, RECURRENT, CONVOLUTIONAL, SUBSAMPLING, RECURSIVE, MULTILAYER, NORMALIZATION
    }

    // ----- Input Related Methods -----

    int numInputs();

    int numOutputs();

    /**
     * The current inputs batch size
     * @return the current inputs batch size
     */
    int batchSize();

    /**
     * The input/feature matrix for the model
     * @return the input/feature matrix for the model
     */
    INDArray input();


    // ----- Parameter Methods -----
    /**
     * Parameters of the model (if any)
     * @return the parameters of the model
     */
    INDArray params();

    /**
     * the number of parameters for the model
     * @return the number of parameters for the model
     *
     */
    int numParams();

    /**
     * the number of parameters for the model
     * @return the number of parameters for the model
     *
     */
    int numParams(boolean backwards);

    /**
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally
     * relative to the expected ordering of the model
     * @param params the parameters for the model
     */
    void setParams(INDArray params);

    /**
     * Set the initial parameters array as a view of the full (backprop) network parameters
     * NOTE: this is intended to be used internally in MultiLayerNetwork and ComputationGraph, not by users.
     * @param params a 1 x nParams row vector that is a view of the larger (MLN/CG) parameters array
     */
    void setParamsViewArray(INDArray params);

    /**
     * Get the parameter
     * @param param the key of the parameter
     * @return the parameter vector/matrix with that particular key
     */
    INDArray getParam(String param);

    /**
     * The param table
     * @return
     */
    Map<String, INDArray> paramTable();

    /**
     * Table of parameters by key, for backprop
     * For many models (dense layers, etc) - all parameters are backprop parameters
     * @param backpropParamsOnly If true, return backprop params only. If false: return all params (equivalent to
     *                           paramsTable())
     */
    Map<String, INDArray> paramTable(boolean backpropParamsOnly);

    /**
     * Setter for the param table
     * @param paramTable
     */
    void setParamTable(Map<String, INDArray> paramTable);


    /**
     * Set the parameter with a new ndarray
     * @param key the key to se t
     * @param val the new ndarray
     */
    void setParam(String key, INDArray val);

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





    // ----- Forward Pass Methods -----

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
     * Initialize the layer with the given input
     * and return the activation for this layer
     * given this input
     * @param input the input to use
     * @return
     */
    INDArray activate(INDArray input);


    /**
     * Get the layer input.
     */
    void setInput(INDArray input);
    void setInput(int inputNumber, INDArray input);

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



    // ----- Gradient Methods -----

    /**Calculate the gradient relative to the error in the next layer
     * @param epsilon w^(L+1)*delta^(L+1). Or, equiv: dC/da, i.e., (dC/dz)*(dz/da) = dC/da, where C
     * 	is cost function a=sigma(z) is activation.
     * @return Pair<Gradient,INDArray> where Gradient is gradient for this layer, INDArray is epsilon needed by next
     *  layer, but before element-wise multiply by sigmaPrime(z). So for standard feed-forward layer, if this layer is
     *  L, then return.getSecond() == (w^(L)*(delta^(L))^T)^T
     */
    Pair<Gradient, INDArray> backpropGradient(INDArray epsilon);

    /**
     * @return the gradient for this model
     */
    Gradient gradient();


    INDArray getGradientsViewArray();

    /**
     * Set the gradients array as a view of the full (backprop) network parameters
     * NOTE: this is intended to be used internally in MultiLayerNetwork and ComputationGraph, not by users.
     * @param gradients a 1 x nParams row vector that is a view of the larger (MLN/CG) gradients array
     */
    void setBackpropGradientsViewArray(INDArray gradients);

    /**
     * Update layer weights and biases with gradient change
     */
    void update(Gradient gradient);

    /**
     * Perform one update  applying the gradient
     * @param gradient the gradient to apply
     */
    void update(INDArray gradient, String paramType);


    // ----- General Methods -----

    /**
     * Clear input
     */
    void clear();


    /**
     * Apply any constraints to the model
     */
    void applyConstraints(int iteration, int epoch);

    /**
     * The configuration for the neural network
     * @return the configuration for the neural network
     */
    NeuralNetConfiguration conf();

    /**
     * Setter for the configuration
     * @param conf
     */
    void setConf(NeuralNetConfiguration conf);

    /**
     * This method sets given CacheMode for current layer
     *
     * @param mode
     */
    void setCacheMode(CacheMode mode);

    /**
     * Returns the layer type
     * @return
     */
    Type type();

    /**
     * Clone the layer
     * @return
     */
    Layer clone();

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
     * Returns true if the layer can be trained in an unsupervised/pretrain manner (VAE, RBMs etc)
     *
     * @return true if the layer can be pretrained (using fit(INDArray), false otherwise
     */
    boolean isPretrainLayer();


    void clearNoiseWeightParams();
}
