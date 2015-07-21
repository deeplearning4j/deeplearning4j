/*
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


import org.deeplearning4j.berkeley.Pair;
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
public interface Layer extends Serializable,Cloneable,Model {

    enum Type {
       FEED_FORWARD,RECURRENT,CONVOLUTIONAL,RECURSIVE
    }



    /**
     * The l2 magnitude for the weights
     * @return the l2 magnitude for the weights
     */
    double l2Magnitude();

    /**
     * The l1 magnitude for the weights
     * @return the l1 magnitude for the weights
     */
    double l1Magnitude();

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
     */
    Gradient error(INDArray input);



    /**
     * Take the derivative of the given input
     * based on the activation
     * @param input the input to take the derivative of
     * @return the derivative of the action
     */
    INDArray derivativeActivation(INDArray input);


    /**
     * Calculate the gradient
     * @param layerError the layer error
     * @param indArray
     * @return the gradient
     */
    Gradient calcGradient(Gradient layerError, INDArray indArray);


    /**
     * Error signal for this layer
     *
     * Using the amount of error
     * caused by this layer
     * calculate the error signal used
     * as input in to the next layer.
     * This is used for actually calculating the
     * gradient of the layer
     *
     * @param error
     * @param input
     * @return
     */
    @Deprecated
    Gradient errorSignal(Gradient error, INDArray input);

    /**Calculate the gradient relative to the error in the next layer
     * @param epsilon w^(L+1)*delta^(L+1). Or, equiv: dC/da, i.e., (dC/dz)/(dz/da) = dC/da, where C 
     * 	is cost function a=sigma(z) is activation. Division is element-wise.
     * @return Pair<Gradient,INDArray> where Gradient is gradient for this layer, INDArray is epsilon needed by next
     *  layer, but before element-wise multiply by sigmaPrime(z). So for standard feed-forward layer, if this layer is
     *  L, then return.getSecond() == (w^(L)*(delta^(L))^T)^T
     */
    Pair<Gradient,INDArray> backwardGradient(INDArray epsilon);


    /**
     * Parameter averaging
     * @param layer the layer to merge
     * @param batchSize the batch size to merge on
     */
    void merge(Layer layer,int batchSize);


    /**
     * Calculate the mean representation
     * for the activation for this layer
     * @return the activation mean for this layer
     */
    INDArray activationMean();


    /**
     * Update layer weights and biases with gradient change
     */

    void update(INDArray gradient, String paramType);

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
    INDArray preOutput(INDArray x,boolean training);


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
    INDArray activate(INDArray input,boolean training);

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
     * Propagate errors backwards for a particular layer.
     * This calculates the node error for a given true output.
     * @param errors the errors to propagate
     * @param deltas the previous deltas
     * @param previousActivation the previous layer's activation
     * @param activation  the activation from the previous layer
     */
    @Deprecated
    Pair<Gradient, Gradient> backWard(Gradient errors, Gradient deltas, INDArray activation,String previousActivation);


    /**
     * Get the iteration listeners for this layer.
     */
    Collection<IterationListener> getIterationListeners();

    /**
     * Set the iteration listeners for this layer.
     */
    void setListeners(IterationListener...listeners);
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

}
