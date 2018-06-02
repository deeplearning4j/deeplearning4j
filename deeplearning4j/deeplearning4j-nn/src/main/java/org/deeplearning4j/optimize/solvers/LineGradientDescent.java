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

package org.deeplearning4j.optimize.solvers;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collection;

/**
 * Stochastic Gradient Descent with Line Search
 * @author Adam Gibson
 *
 */
public class LineGradientDescent extends BaseOptimizer {
    private static final long serialVersionUID = 6336124657542062284L;

    public LineGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction,
                    Collection<TrainingListener> trainingListeners, Model model) {
        super(conf, stepFunction, trainingListeners, model);
    }

    @Override
    public void preProcessLine() {
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);
        searchState.put(SEARCH_DIR, gradient.dup());
    }

    @Override
    public void postStep(INDArray gradient) {
        double norm2 = Nd4j.getBlasWrapper().level1().nrm2(gradient);
        if (norm2 > stepMax)
            searchState.put(SEARCH_DIR, gradient.dup().muli(stepMax / norm2));
        else
            searchState.put(SEARCH_DIR, gradient.dup());
        searchState.put(GRADIENT_KEY, gradient.dup());
    }

}
