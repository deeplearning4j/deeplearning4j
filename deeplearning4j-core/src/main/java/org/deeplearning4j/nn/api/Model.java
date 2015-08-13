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
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * A Model is meant for predicting something from data.
 * Note that this is not like supervised learning where
 * there are labels attached to the examples.
 *
 */
public interface Model {

    /**
     * All models have a fit method
     */
    void fit();

    /**
     * Perform one update  applying the gradient
     * @param gradient the gradient to apply
     */
    void update(INDArray gradient, String paramType);


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
     * Sets a rolling tally for the score. This is useful for mini batch learning when
     * you are accumulating error across a dataset.
     * @param accum the amount to accum
     */
    void accumulateScore(double accum);


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
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally
     * relative to the expected ordering of the model
     * @param params the parameters for the model
     */
    void setParams(INDArray params);



    /**
     * Fit the model to the given data
     * @param data the data to fit the model to
     */
    void fit(INDArray data);


    /**
     * Run one iteration
     * @param input the input to iterate on
     */
    void iterate(INDArray input);


    /**
     * Calculate a gradient
     * @return the gradient for this model
     */
    Gradient gradient();

    /**
     * Get the gradient and score
     * @return the gradient and score
     */
    Pair<Gradient,Double> gradientAndScore();

    /**
     * The current inputs batch size
     * @return the current inputs batch size
     */
    int batchSize();


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
     * The input/feature matrix for the model
     * @return the input/feature matrix for the model
     */
    INDArray input();


    /**
     * Validate the input
     */
    void validateInput();

    /**
     * Returns this models optimizer
     * @return this models optimizer
     */
    ConvexOptimizer getOptimizer();

    /**
     * Get the parameter
     * @param param the key of the parameter
     * @return the parameter vector/matrix with that particular key
     */
    INDArray getParam(String param);

    /**
     * Initialize the parameters
     */
    void initParams();

    /**
     * The param table
     * @return
     */
    Map<String,INDArray> paramTable();

    /**
     * Setter for the param table
     * @param paramTable
     */
    void setParamTable(Map<String,INDArray> paramTable);


    /**
     * Set the parameter with a new ndarray
     * @param key the key to se t
     * @param val the new ndarray
     */
    void setParam(String key,INDArray val);

    /**
     * Clear input
     */
    void clear();


}
