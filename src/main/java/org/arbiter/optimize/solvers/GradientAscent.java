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


import java.util.Collection;
import org.arbiter.berkeley.Pair;
import org.arbiter.nn.api.Model;
import org.arbiter.nn.conf.BaseNeuralNetConfiguration;
import org.arbiter.nn.gradient.Gradient;
import org.arbiter.optimize.api.IterationListener;
import org.arbiter.optimize.api.StepFunction;
import org.arbiter.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Vectorized Stochastic Gradient Ascent
 *
 * @author Adam Gibson
 */
public class GradientAscent extends BaseOptimizer {


  public GradientAscent(BaseNeuralNetConfiguration conf, StepFunction stepFunction,
                        Collection<IterationListener> iterationListeners, Model model) {
    super(conf, stepFunction, iterationListeners, model);
  }

  public GradientAscent(BaseNeuralNetConfiguration conf, StepFunction stepFunction,
                        Collection<IterationListener> iterationListeners,
                        Collection<TerminationCondition> terminationConditions, Model model) {
    super(conf, stepFunction, iterationListeners, terminationConditions, model);
  }


  @Override
  public void preProcessLine(INDArray line) {
    double norm2 = line.norm2(Integer.MAX_VALUE).getDouble(0);
    if (norm2 > stpMax) {
      line.muli(stpMax / norm2);
    }
  }

  @Override
  public void postStep() {
    //no-op
  }

  @Override
  public void setupSearchState(Pair<Gradient, Double> pair) {
    super.setupSearchState(pair);
  }


}
