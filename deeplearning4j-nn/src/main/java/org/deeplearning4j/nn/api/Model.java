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

import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;

/**
 * A Model is meant for predicting something from data.
 * Note that this is not like supervised learning where
 * there are labels attached to the examples.
 *
 */
public interface Model extends Layer {

    /**
     * Init the model
     */
    void init();


    /**
     * Set the IterationListeners for the ComputationGraph (and all layers in the network)
     */
    void setListeners(Collection<IterationListener> listeners);


    /**
     * Set the IterationListeners for the ComputationGraph (and all layers in the network)
     */
    void setListeners(IterationListener... listeners);

    /**
     * This method ADDS additional IterationListener to existing listeners
     *
     * @param listener
     */
    void addListeners(IterationListener... listener);


    /**
     * Get the iteration listeners for this layer.
     */
    Collection<IterationListener> getListeners();

    /**
     * All models have a fit method
     */
    void fit();


    /**
     * The score for the model
     * @return the score for the model
     */
    double score();


    /**
     * Update the score
     */
    void computeGradientAndScore();


    /**
     * Fit the model to the given data
     * @param data the data to fit the model to
     */
    void fit(INDArray data);

    /**
     * Get the gradient and score
     * @return the gradient and score
     */
    Pair<Gradient, Double> gradientAndScore();


    /**
     * Returns this models optimizer
     * @return this models optimizer
     */
    ConvexOptimizer getOptimizer();


}
