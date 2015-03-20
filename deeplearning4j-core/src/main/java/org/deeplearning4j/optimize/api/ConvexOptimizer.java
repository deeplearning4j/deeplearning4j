/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.optimize.api;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;

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
     * Pre process a line before an iteration
     * @param line
     */
    void preProcessLine(INDArray line);

    /**
     * After the step has been made, do an action
     */
    void postStep();

    /**
     * Based on the gradient and score
     * setup a search state
     * @param pair the gradient and score
     */
    void setupSearchState(Pair<Gradient, Double> pair);

    /**
     * The adagrad in this model
     * @return the adagrad in this model
     */
    AdaGrad getAdaGrad();

    /**
     * Return the ada grad look up table
     * @return the ada grad for variables
     */
    Map<String,AdaGrad> adaGradForVariables();

    /**
     * Get adagrad for a variable
     * @param variable
     * @return
     */
    AdaGrad getAdaGradForVariable(String variable);


    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param params the parameters to update
     */
    void updateGradientAccordingToParams(Gradient gradient, Model params, int batchSize);


    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     */
    void updateGradientAccordingToParams(INDArray gradient, INDArray params, int batchSize);
}
