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

package org.deeplearning4j.optimize.api;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;
import org.nd4j.linalg.learning.GradientUpdater;

import java.io.Serializable;
import java.util.Map;

/**
 * Convex optimizer.
 * @author Adam Gibson
 */
public interface ConvexOptimizer extends Serializable {
    /**
     * The score for the optimizer so far
     * @return the score for this optimizer so far
     */
    double score();

    Updater getUpdater();

    ComputationGraphUpdater getComputationGraphUpdater();

    void setUpdater(Updater updater);

    void setUpdaterComputationGraph(ComputationGraphUpdater updater);

    NeuralNetConfiguration getConf();

    /**
     * The gradient and score for this optimizer
     * @return the gradient and score for this optimizer
     */
    Pair<Gradient,Double> gradientAndScore();

    /**
     * Calls optimize
     * @return whether the convex optimizer
     * converted or not
     */
    boolean optimize();


    /**
     * The batch size for the optimizer
     * @return
     */
    int batchSize();

    /**
     * Set the batch size for the optimizer
     * @param batchSize
     */
    void setBatchSize(int batchSize);

    /**
     * Pre preProcess a line before an iteration
     */
    void preProcessLine();

    /**
     * After the step has been made, do an action
     * @param line
     * */
    void postStep(INDArray line);

    /**
     * Based on the gradient and score
     * setup a search state
     * @param pair the gradient and score
     */
    void setupSearchState(Pair<Gradient, Double> pair);

    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param model the model with the parameters to update
     * @param batchSize batchSize for update
     * @paramType paramType to update
     */
    void updateGradientAccordingToParams(Gradient gradient, Model model, int batchSize);

    /**
     * Check termination conditions
     * setup a search state
     * @param gradient layer gradients
     * @param iteration what iteration the optimizer is on
     */
    boolean checkTerminalConditions(INDArray gradient, double oldScore, double score, int iteration);
}
