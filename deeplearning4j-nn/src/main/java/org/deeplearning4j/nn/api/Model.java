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

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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

    Activations getLabels();


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
     * The score for the model
     * @return the score for the model
     */
    double score();


    Pair<Gradients, Double> computeGradientAndScore(DataSet dataSet);

    Pair<Gradients, Double> computeGradientAndScore(MultiDataSet dataSet);

    /**
     * Update the score
     */
    Pair<Gradients, Double> computeGradientAndScore(Activations input, Activations labels);


    void fit(Activations data);

    /**
     * Train the model based on the datasetiterator
     * @param iter the iterator to train on
     */
    void fit(DataSetIterator iter);


    /**
     * Fit the model
     * @param examples the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     */
    void fit(INDArray examples, INDArray labels);

    /**
     * Fit the model
     * @param data the data to train on
     */
    void fit(DataSet data);


    /**
     * Returns this models optimizer
     * @return this models optimizer
     */
    ConvexOptimizer getOptimizer();

    OptimizationConfig getOptimizationConfig();

}
