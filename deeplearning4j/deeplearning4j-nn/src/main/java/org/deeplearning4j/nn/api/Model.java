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

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;
import java.util.Map;

/**
 * A Model is meant for predicting something from data.
 * Note that this is not like supervised learning where
 * there are labels attached to the examples.
 *
 */
public interface Model {

    /**
     * Init the model
     */
    void init();


    /**
     * Set the trainingListeners for the ComputationGraph (and all layers in the network)
     */
    void setListeners(Collection<TrainingListener> listeners);


    /**
     * Set the trainingListeners for the ComputationGraph (and all layers in the network)
     */
    void setListeners(TrainingListener... listeners);

    /**
     * This method ADDS additional TrainingListener to existing listeners
     *
     * @param listener
     */
    void addListeners(TrainingListener... listener);


    /**
     * All models have a fit method
     */
    @Deprecated
    void fit();

    /**
     * Update layer weights and biases with gradient change
     */
    void update(Gradient gradient);

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
    void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr);

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


    INDArray getGradientsViewArray();

    /**
     * Set the gradients array as a view of the full (backprop) network parameters
     * NOTE: this is intended to be used internally in MultiLayerNetwork and ComputationGraph, not by users.
     * @param gradients a 1 x nParams row vector that is a view of the larger (MLN/CG) gradients array
     */
    void setBackpropGradientsViewArray(INDArray gradients);

    /**
     * Fit the model to the given data
     * @param data the data to fit the model to
     */
    void fit(INDArray data, LayerWorkspaceMgr workspaceMgr);


    /**
     * Get the gradient. Note that this method will not calculate the gradient, it will rather return the gradient
     * that has been computed before.
     * For calculating the gradient, see {@link Model#computeGradientAndScore(LayerWorkspaceMgr)} } .
     * @return the gradient for this model, as calculated before
     */
    Gradient gradient();

    /**
     * Get the gradient and score
     * @return the gradient and score
     */
    Pair<Gradient, Double> gradientAndScore();

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
     * @deprecated As of 0.7.3 - Feb 2017. No longer used, most implementations are unsupported or no-op.
     */
    @Deprecated
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
     * @deprecated As of 0.7.3 - Feb 2017. Not used; neural network params are initialized by the parameter initializaters.
     *  Furthermore, most implementations are unsupported or no-op.
     */
    @Deprecated
    void initParams();

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

    /**
     * Clear input
     */
    void clear();


    /**
     * Apply any constraints to the model
     */
    void applyConstraints(int iteration, int epoch);
}
