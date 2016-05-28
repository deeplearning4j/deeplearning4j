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

package org.deeplearning4j.optimize.solvers;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

/**
 * Stochastic Gradient Descent
 * Standard fix step size
 * No line search
 * @author Adam Gibson
 */
public class StochasticGradientDescent extends BaseOptimizer {


    public StochasticGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
    }

    public StochasticGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction, Collection<IterationListener> iterationListeners, Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, iterationListeners, terminationConditions, model);
    }


    @Override
    public boolean optimize() {
        for(int i = 0; i < conf.getNumIterations(); i++) {

            Pair<Gradient,Double> pair = gradientAndScore();
            Gradient gradient = pair.getFirst();

            INDArray params = model.params();
            stepFunction.step(params,gradient.gradient());
            //Note: model.params() is always in-place for MultiLayerNetwork and ComputationGraph, hence no setParams is necessary there
            //However: for pretrain layers, params are NOT a view. Thus a setParams call is necessary
            //But setParams should be a no-op for MLN and CG
            model.setParams(params);

            for(IterationListener listener : iterationListeners)
                listener.iterationDone(model, i);

            checkTerminalConditions(pair.getFirst().gradient(), oldScore, score, i);


        }
        return true;
    }

    @Override
    public void preProcessLine() {
    }

    @Override
    public void postStep(INDArray gradient) {
    }
}
