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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.InvalidStepException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationConfig;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ModelConfig;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.*;
import org.deeplearning4j.optimize.solvers.accumulation.GradientsAccumulator;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.deeplearning4j.optimize.terminations.EpsTermination;
import org.deeplearning4j.optimize.terminations.ZeroDirection;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base optimizer
 * @author Adam Gibson
 */
@Slf4j
public abstract class BaseOptimizer implements ConvexOptimizer {

    protected OptimizationConfig conf;
    @Getter
    protected StepFunction stepFunction;
    protected Collection<TerminationCondition> terminationConditions = new ArrayList<>();
    protected Model model;
    protected BackTrackLineSearch lineMaximizer;
    protected Updater updater;
    protected ComputationGraphUpdater computationGraphUpdater;
    protected double step;
    private int batchSize;
    protected double score, oldScore;
    protected double stepMax = Double.MAX_VALUE;
    public final static String GRADIENT_KEY = "g";
    public final static String SCORE_KEY = "score";
    public final static String PARAMS_KEY = "params";
    public final static String SEARCH_DIR = "searchDirection";
    protected Map<String, Object> searchState = new ConcurrentHashMap<>();


    protected GradientsAccumulator accumulator;

    /**
     *
     * @param conf
     * @param stepFunction
     * @param model
     */
    public BaseOptimizer(OptimizationConfig conf, StepFunction stepFunction, Model model) {
        this(conf, stepFunction, Arrays.asList(new ZeroDirection(), new EpsTermination()), model);
    }


    /**
     *
     * @param conf
     * @param stepFunction
     * @param terminationConditions
     * @param model
     */
    public BaseOptimizer(OptimizationConfig conf, StepFunction stepFunction,
                         Collection<TerminationCondition> terminationConditions, Model model) {
        this.conf = conf;
        this.stepFunction = (stepFunction != null ? stepFunction : getDefaultStepFunctionForOptimizer(this.getClass()));
        this.terminationConditions = terminationConditions;
        this.model = model;
        lineMaximizer = new BackTrackLineSearch(model, this.stepFunction, this);
        lineMaximizer.setStepMax(stepMax);
        lineMaximizer.setMaxIterations(conf.getMaxNumLineSearchIterations());
    }

    @Override
    public void setGradientsAccumulator(GradientsAccumulator accumulator) {
        this.accumulator = accumulator;
    }

    @Override
    public GradientsAccumulator getGradientsAccumulator() {
        return accumulator;
    }

    @Override
    public Updater getUpdater() {
        if (updater == null) {
            updater = UpdaterCreator.getUpdater(model);
        }
        return updater;
    }

    @Override
    public void setUpdater(Updater updater) {
        this.updater = updater;
    }



    @Override
    public ComputationGraphUpdater getComputationGraphUpdater() {
        if (computationGraphUpdater == null && model instanceof ComputationGraph) {
            computationGraphUpdater = new ComputationGraphUpdater((ComputationGraph) model);
        }
        return computationGraphUpdater;
    }

    @Override
    public void setUpdaterComputationGraph(ComputationGraphUpdater updater) {
        this.computationGraphUpdater = updater;
    }

    @Override
    public OptimizationConfig getConf() {
        return conf;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore(Activations input, Activations labels) {
        oldScore = score;
        Pair<Gradients, Double> pair = model.computeGradientAndScore(input, labels);

        if (model.getListeners() != null && model.getListeners().size() > 0) {
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                for (IterationListener l : model.getListeners()) {
                    if (l instanceof TrainingListener) {
                        ((TrainingListener) l).onGradientCalculation(model, pair.getFirst());
                    }
                }
            }
        }

        score = pair.getSecond();
        updateGradientAccordingToParams(pair.getFirst().getParameterGradients(), model, model.getInputMiniBatchSize());
        return new Pair<>(pair.getFirst().getParameterGradients(), score);
    }

    /**
     * Optimize call. This runs the optimizer.
     * @return whether it converged or not
     */
    // TODO add flag to allow retaining state between mini batches and when to apply updates
    @Override
    public boolean optimize() {
        //validate the input before training
        INDArray gradient;
        INDArray searchDirection;
        INDArray parameters;

        //Get the original input/labels
        Activations input = model.getInput();
        Activations labels = model.getLabels();

        Pair<Gradient, Double> pair = gradientAndScore(input, labels);
        if (searchState.isEmpty()) {
            searchState.put(GRADIENT_KEY, pair.getFirst().gradient());
            setupSearchState(pair); //Only do this once
        } else {
            searchState.put(GRADIENT_KEY, pair.getFirst().gradient());
        }

        //pre existing termination conditions
        /*
         * Commented out for now; this has been problematic for testing/debugging
         * Revisit & re-enable later. */
        for (TerminationCondition condition : terminationConditions) {
            if (condition.terminate(0.0, 0.0, new Object[] {pair.getFirst().gradient()})) {
                log.info("Hit termination condition " + condition.getClass().getName());
                return true;
            }
        }

        //calculate initial search direction
        preProcessLine();
        gradient = (INDArray) searchState.get(GRADIENT_KEY);
        searchDirection = (INDArray) searchState.get(SEARCH_DIR);
        parameters = (INDArray) searchState.get(PARAMS_KEY);

        //perform one line search optimization
        try {
            step = lineMaximizer.optimize(input, labels, parameters, gradient, searchDirection);
        } catch (InvalidStepException e) {
            log.warn("Invalid step...continuing another iteration: {}", e.getMessage());
            step = 0.0;
        }

        //Update parameters based on final/best step size returned by line search:
        if (step != 0.0) {
            // TODO: inject accumulation use here
            stepFunction.step(parameters, searchDirection, step); //Calculate params. given step size
            model.setParams(parameters);
        } else {
            log.debug("Step size returned by line search is 0.0.");
        }

        pair = gradientAndScore(input, labels);

        //updates searchDirection
        postStep(pair.getFirst().gradient());

        //invoke listeners
        int iterationCount = BaseOptimizer.getIterationCount(model);
        int epochCount = BaseOptimizer.getEpochCount(model);
        if(model.getListeners() != null ) {
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                for (IterationListener listener : model.getListeners())
                    listener.iterationDone(model, iterationCount, epochCount);
            }
        }


        //check for termination conditions based on absolute change in score
        checkTerminalConditions(pair.getFirst().gradient(), oldScore, score, model.getIterationCount());
        incrementIterationCount(model, 1);
        applyConstraints(model);
        return true;
    }

    protected void postFirstStep(INDArray gradient) {
        //no-op
    }

    @Override
    public boolean checkTerminalConditions(INDArray gradient, double oldScore, double score, int i) {
        for (TerminationCondition condition : terminationConditions) {
            //log.info("terminations: {}", condition);
            if (condition.terminate(score, oldScore, new Object[] {gradient})) {
                log.debug("Hit termination condition on iteration {}: score={}, oldScore={}, condition={}", i, score,
                                oldScore, condition);
                return true;
            }
        }
        return false;
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    @Override
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }


    /**
     * Pre preProcess to setup initial searchDirection approximation
     */
    @Override
    public void preProcessLine() {
        //no-op
    }

    /**
     * Post step to update searchDirection with new gradient and parameter information
     */
    @Override
    public void postStep(INDArray gradient) {
        //no-op
    }


    @Override
    public void updateGradientAccordingToParams(Gradient gradient, Model model, int batchSize) {
        if (model instanceof ComputationGraph) {
            ComputationGraph graph = (ComputationGraph) model;
            if (computationGraphUpdater == null) {
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    computationGraphUpdater = new ComputationGraphUpdater(graph);
                }
            }
            computationGraphUpdater.update(gradient, getIterationCount(model), getEpochCount(model), batchSize);
        } else {
            if (updater == null) {
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    updater = UpdaterCreator.getUpdater(model);
                }
            }
            Layer layer = (Layer) model;

            updater.update(layer, gradient, getIterationCount(model), getEpochCount(model), batchSize);
        }
    }

    /**
     * Setup the initial search state
     * @param pair
     */
    @Override
    public void setupSearchState(Pair<Gradient, Double> pair) {
        INDArray gradient = pair.getFirst().gradient(); //.gradient(conf.variables());
        INDArray params = model.params().dup(); //Need dup here: params returns an array that isn't a copy (hence changes to this are problematic for line search methods)
        searchState.put(GRADIENT_KEY, gradient);
        searchState.put(SCORE_KEY, pair.getSecond());
        searchState.put(PARAMS_KEY, params);
    }


    public static StepFunction getDefaultStepFunctionForOptimizer(Class<? extends ConvexOptimizer> optimizerClass) {
        if (optimizerClass == StochasticGradientDescent.class) {
            return new NegativeGradientStepFunction();
        } else {
            return new NegativeDefaultStepFunction();
        }
    }

    public static int getIterationCount(Model model) {
        if (model instanceof MultiLayerNetwork) {
            return ((MultiLayerNetwork) model).getLayerWiseConfigurations().getIterationCount();
        } else if (model instanceof ComputationGraph) {
            return ((ComputationGraph) model).getConfiguration().getIterationCount();
        } else {
            return model.getOptimizationConfig().getIterationCount();
        }
    }

    public static void incrementIterationCount(Model model, int incrementBy) {
        if (model instanceof MultiLayerNetwork) {
            MultiLayerConfiguration conf = ((MultiLayerNetwork) model).getLayerWiseConfigurations();
            conf.setIterationCount(conf.getIterationCount() + incrementBy);
        } else if (model instanceof ComputationGraph) {
            ComputationGraphConfiguration conf = ((ComputationGraph) model).getConfiguration();
            conf.setIterationCount(conf.getIterationCount() + incrementBy);
        } else {
            model.getOptimizationConfig().setIterationCount(model.getOptimizationConfig().getIterationCount() + incrementBy);
        }
    }

    public static int getEpochCount(Model model){
        if (model instanceof MultiLayerNetwork) {
            return ((MultiLayerNetwork) model).getLayerWiseConfigurations().getEpochCount();
        } else if (model instanceof ComputationGraph) {
            return ((ComputationGraph) model).getConfiguration().getEpochCount();
        } else {
            return model.getOptimizationConfig().getEpochCount();
        }
    }

    public static void applyConstraints(Model model){
        int iter = getIterationCount(model);
        int epoch = getEpochCount(model);
        model.applyConstraints(iter, epoch);
    }

}
