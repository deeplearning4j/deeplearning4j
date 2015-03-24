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

package org.arbiter.optimize.solvers;

import org.arbiter.nn.api.Model;
import org.arbiter.nn.conf.BaseNeuralNetConfiguration;
import org.arbiter.nn.gradient.Gradient;
import org.arbiter.optimize.api.IterationListener;
import org.arbiter.optimize.api.StepFunction;
import org.arbiter.berkeley.Pair;
import org.arbiter.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * No line search gradient descent
 * @author Adam Gibson
 */
public class IterationGradientDescent extends BaseOptimizer {


    public IterationGradientDescent(BaseNeuralNetConfiguration conf, StepFunction stepFunction,
                                    Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
    }

    public IterationGradientDescent(BaseNeuralNetConfiguration conf, StepFunction stepFunction,
                                    Collection<IterationListener> iterationListeners,
                                    Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, iterationListeners, terminationConditions, model);
    }


    @Override
    public boolean optimize() {
        for(int i = 0; i < conf.getNumIterations(); i++) {
            Pair<Gradient,Double> score = gradientAndScore();
            model.update(score.getFirst()); // this line causing very bad optimization results
            for(IterationListener listener : conf.getListeners())
                listener.iterationDone(model,i);
        }
        return true;
    }

    @Override
    public void preProcessLine(INDArray line) {
          if(conf.isConstrainGradientToUnitNorm())
              line.divi(line.norm2(Integer.MAX_VALUE));
    }

    @Override
    public void postStep() {

    }
}
