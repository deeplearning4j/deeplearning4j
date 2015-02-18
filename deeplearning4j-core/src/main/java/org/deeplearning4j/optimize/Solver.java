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

package org.deeplearning4j.optimize;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.solvers.*;
import java.util.Collection;

/**
 * Generic purpose solver
 * @author Adam Gibson
 */
public class Solver {
    private NeuralNetConfiguration conf;
    private Collection<IterationListener> listeners;
    private Model model;
    private ConvexOptimizer optimizer;


    public void optimize() {
        if(optimizer == null)
            optimizer = getOptimizer();
        optimizer.optimize();

    }

    public ConvexOptimizer getOptimizer() {
        OptimizationAlgorithm algo = conf.getOptimizationAlgo();
        switch(algo) {
            case LBFGS:
                return new LBFGS(conf,conf.getStepFunction(),listeners,model);
            case GRADIENT_DESCENT:
                return new GradientAscent(conf,conf.getStepFunction(),listeners,model);
            case HESSIAN_FREE:
                return new StochasticHessianFree(conf,conf.getStepFunction(),listeners,model);
            case CONJUGATE_GRADIENT:
                return new ConjugateGradient(conf,conf.getStepFunction(),listeners,model);
            case ITERATION_GRADIENT_DESCENT:
                return new IterationGradientDescent(conf,conf.getStepFunction(),listeners,model);
        }

        throw new IllegalStateException("No optimizer found");
    }

    public static class Builder {
        private NeuralNetConfiguration conf;
        private Model model;


        public Builder configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }




        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        public Solver build() {
            Solver solver = new Solver();
            solver.conf = conf;
            solver.model = model;
            solver.listeners = conf.getListeners();
            return solver;
        }
    }


}
