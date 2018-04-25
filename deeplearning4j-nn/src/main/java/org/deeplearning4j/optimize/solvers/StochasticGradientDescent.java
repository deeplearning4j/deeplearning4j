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

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;

/**
 * Stochastic Gradient Descent
 * Standard fix step size
 * No line search
 * @author Adam Gibson
 */
@Slf4j
public class StochasticGradientDescent extends BaseOptimizer {


    public StochasticGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction,
                    Collection<TrainingListener> trainingListeners, Model model) {
        super(conf, stepFunction, trainingListeners, model);
    }

    public StochasticGradientDescent(NeuralNetConfiguration conf, StepFunction stepFunction,
                    Collection<TrainingListener> trainingListeners,
                    Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, trainingListeners, terminationConditions, model);
    }


    @Override
    public boolean optimize(LayerWorkspaceMgr workspaceMgr) {
        Pair<Gradient, Double> pair = gradientAndScore(workspaceMgr);

        Gradient gradient = pair.getFirst();

        INDArray params = model.params();

        // if optimizer has GradientsAccumulator defined - go for it
        if (accumulator != null) {
            // we're propagating current update
            accumulator.storeUpdate(gradient.gradient());

            // and getting (possible) pending update from accumulator
            //INDArray pendingUpdate = accumulator.getUpdate();
            //stepFunction.step(params, pendingUpdate);
            accumulator.applyUpdate(stepFunction, params, gradient.gradient());

            // if there's no update available - just go on then
        } else {
            // if accumulator isn't used - we just to for direct updates application
            stepFunction.step(params, gradient.gradient());
        }

        //Note: model.params() is always in-place for MultiLayerNetwork and ComputationGraph, hence no setParams is necessary there
        //However: for pretrain layers, params are NOT a view. Thus a setParams call is necessary
        //But setParams should be a no-op for MLN and CG
        model.setParams(params);

        int iterationCount = BaseOptimizer.getIterationCount(model);
        int epochCount = BaseOptimizer.getEpochCount(model);
        try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            for (TrainingListener listener : trainingListeners)
                listener.iterationDone(model, iterationCount, epochCount);
        }

        checkTerminalConditions(pair.getFirst().gradient(), oldScore, score, iterationCount);

        BaseOptimizer.incrementIterationCount(model, 1);
        applyConstraints(model);
        return true;
    }

    @Override
    public void preProcessLine() {}

    @Override
    public void postStep(INDArray gradient) {}
}
