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

package org.deeplearning4j.optimize;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.solvers.ConjugateGradient;
import org.deeplearning4j.optimize.solvers.LBFGS;
import org.deeplearning4j.optimize.solvers.LineGradientDescent;
import org.deeplearning4j.optimize.solvers.StochasticGradientDescent;
import org.deeplearning4j.optimize.stepfunctions.StepFunctions;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Generic purpose solver
 * @author Adam Gibson
 */
public class Solver {
    private NeuralNetConfiguration conf;
    private Collection<TrainingListener> listeners;
    private Model model;
    private ConvexOptimizer optimizer;
    private StepFunction stepFunction;

    public void optimize(LayerWorkspaceMgr workspaceMgr) {
        initOptimizer();

        optimizer.optimize(workspaceMgr);
    }

    public void initOptimizer() {
        if (optimizer == null) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                optimizer = getOptimizer();
            }
        }
    }

    public ConvexOptimizer getOptimizer() {
        if (optimizer != null)
            return optimizer;
        switch (conf.getOptimizationAlgo()) {
            case LBFGS:
                optimizer = new LBFGS(conf, stepFunction, listeners, model);
                break;
            case LINE_GRADIENT_DESCENT:
                optimizer = new LineGradientDescent(conf, stepFunction, listeners, model);
                break;
            case CONJUGATE_GRADIENT:
                optimizer = new ConjugateGradient(conf, stepFunction, listeners, model);
                break;
            case STOCHASTIC_GRADIENT_DESCENT:
                optimizer = new StochasticGradientDescent(conf, stepFunction, listeners, model);
                break;
            default:
                throw new IllegalStateException("No optimizer found");
        }
        return optimizer;
    }

    public void setListeners(Collection<TrainingListener> listeners) {
        this.listeners = listeners;
        if (optimizer != null)
            optimizer.setListeners(listeners);
    }

    public static class Builder {
        private NeuralNetConfiguration conf;
        private Model model;
        private List<TrainingListener> listeners = new ArrayList<>();

        public Builder configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }

        public Builder listener(TrainingListener... listeners) {
            this.listeners.addAll(Arrays.asList(listeners));
            return this;
        }

        public Builder listeners(Collection<TrainingListener> listeners) {
            this.listeners.addAll(listeners);
            return this;
        }

        public Builder model(Model model) {
            this.model = model;
            return this;
        }

        public Solver build() {
            Solver solver = new Solver();
            solver.conf = conf;
            solver.stepFunction = StepFunctions.createStepFunction(conf.getStepFunction());
            solver.model = model;
            solver.listeners = listeners;
            return solver;
        }
    }


}
