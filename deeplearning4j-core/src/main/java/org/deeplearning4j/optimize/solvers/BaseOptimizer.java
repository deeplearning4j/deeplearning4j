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

import com.google.common.annotations.VisibleForTesting;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.exception.InvalidStepException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.deeplearning4j.optimize.stepfunctions.DefaultStepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.deeplearning4j.optimize.terminations.EpsTermination;
import org.deeplearning4j.optimize.terminations.ZeroDirection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base optimizer
 * @author Adam Gibson
 */
public abstract class BaseOptimizer implements ConvexOptimizer {

    protected NeuralNetConfiguration conf;
    protected int iteration = 0;
    protected static final Logger log = LoggerFactory.getLogger(BaseOptimizer.class);
    protected StepFunction stepFunction;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
    protected Collection<TerminationCondition> terminationConditions = new ArrayList<>();
    protected Model model;
    protected BackTrackLineSearch lineMaximizer;
    protected Updater updater;
    protected ComputationGraphUpdater computationGraphUpdater;
    protected double step;
    private int batchSize;
    protected double score,oldScore;
    protected double stepMax = Double.MAX_VALUE;
    public final static String GRADIENT_KEY = "g";
    public final static String SCORE_KEY = "score";
    public final static String PARAMS_KEY = "params";
    public final static String SEARCH_DIR = "searchDirection";
    protected Map<String,Object> searchState = new ConcurrentHashMap<>();

    /**
     *
     * @param conf
     * @param stepFunction
     * @param iterationListeners
     * @param model
     */
    public BaseOptimizer(NeuralNetConfiguration conf,StepFunction stepFunction,Collection<IterationListener> iterationListeners,Model model) {
        this(conf, stepFunction, iterationListeners, Arrays.asList(new ZeroDirection(), new EpsTermination()), model);
    }


    /**
     *
     * @param conf
     * @param stepFunction
     * @param iterationListeners
     * @param terminationConditions
     * @param model
     */
    public BaseOptimizer(NeuralNetConfiguration conf,StepFunction stepFunction,Collection<IterationListener> iterationListeners,Collection<TerminationCondition> terminationConditions,Model model) {
        this.conf = conf;
        this.stepFunction = (stepFunction != null ? stepFunction : getDefaultStepFunctionForOptimizer(this.getClass()));
        this.iterationListeners = iterationListeners != null ? iterationListeners : new ArrayList<IterationListener>();
        this.terminationConditions = terminationConditions;
        this.model = model;
        lineMaximizer = new BackTrackLineSearch(model,this.stepFunction,this);
        lineMaximizer.setStepMax(stepMax);
        lineMaximizer.setMaxIterations(conf.getMaxNumLineSearchIterations());

    }


    @Override
    public double score() {
        model.computeGradientAndScore();
        return model.score();
    }

    @Override
    public Updater getUpdater() {
        if(updater == null) {
            updater = UpdaterCreator.getUpdater(model);
        }
        return updater;
    }

    @Override
    public void setUpdater(Updater updater){
        this.updater = updater;
    }



    @Override
    public ComputationGraphUpdater getComputationGraphUpdater() {
        if(computationGraphUpdater == null && model instanceof ComputationGraph){
            computationGraphUpdater = new ComputationGraphUpdater((ComputationGraph)model);
        }
        return computationGraphUpdater;
    }

    @Override
    public void setUpdaterComputationGraph(ComputationGraphUpdater updater) {
        this.computationGraphUpdater = updater;
    }

    @Override
    public NeuralNetConfiguration getConf() { return conf; }

    @Override
    public Pair<Gradient,Double> gradientAndScore() {
        oldScore = score;
        model.computeGradientAndScore();
        Pair<Gradient,Double> pair = model.gradientAndScore();
        score = pair.getSecond();
        updateGradientAccordingToParams(pair.getFirst(), model, model.batchSize());
        return pair;
    }

    /**
     * Optimize call. This runs the optimizer.
     * @return whether it converged or not
     */
    // TODO add flag to allow retaining state between mini batches and when to apply updates
    @Override
    public  boolean optimize() {
        //validate the input before training
        INDArray gradient;
        INDArray searchDirection;
        INDArray parameters = null;
        model.validateInput();
        Pair<Gradient,Double> pair = gradientAndScore();
        if(searchState.isEmpty()){
            searchState.put(GRADIENT_KEY, pair.getFirst().gradient());
            setupSearchState(pair);		//Only do this once
        } else {
            searchState.put(GRADIENT_KEY, pair.getFirst().gradient());
        }

        //pre existing termination conditions
        /*
         * Commented out for now; this has been problematic for testing/debugging
         * Revisit & re-enable later. */
        for(TerminationCondition condition : terminationConditions){
            if(condition.terminate(0.0,0.0,new Object[]{pair.getFirst().gradient()})) {
                log.info("Hit termination condition " + condition.getClass().getName());
                return true;
            }
        }

        //calculate initial search direction
        preProcessLine();

        for(int i = 0; i < conf.getNumIterations(); i++) {
            gradient = (INDArray) searchState.get(GRADIENT_KEY);
            searchDirection = (INDArray) searchState.get(SEARCH_DIR);
            parameters = (INDArray) searchState.get(PARAMS_KEY);

            //perform one line search optimization
            try {
                step = lineMaximizer.optimize(parameters, gradient, searchDirection);
            } catch (InvalidStepException e) {
                log.warn("Invalid step...continuing another iteration: {}",e.getMessage());
                step = 0.0;
            }

            //Update parameters based on final/best step size returned by line search:
            if(step != 0.0) {
                stepFunction.step(parameters, searchDirection, step);	//Calculate params. given step size
                model.setParams(parameters);
            }else {
                log.debug("Step size returned by line search is 0.0.");
            }

            pair = gradientAndScore();

            //updates searchDirection
            postStep(pair.getFirst().gradient());

            //invoke listeners for debugging
            for(IterationListener listener : iterationListeners)
                listener.iterationDone(model,i);

            //check for termination conditions based on absolute change in score
            checkTerminalConditions(pair.getFirst().gradient(), oldScore, score, i);
            this.iteration++;
        }
        return true;
    }

    protected  void postFirstStep(INDArray gradient) {
        //no-op
    }

    @Override
    public boolean checkTerminalConditions(INDArray gradient, double oldScore, double score, int i){
        for(TerminationCondition condition : terminationConditions){
            if(condition.terminate(score,oldScore,new Object[]{gradient})){
                log.debug("Hit termination condition on iteration {}: score={}, oldScore={}, condition={}", i, score, oldScore, condition);
                if(condition instanceof EpsTermination && conf.getLayer() != null && conf.getLayer().getLrScoreBasedDecay() != 0.0) {
                    model.applyLearningRateScoreDecay();
                }
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
    public  void preProcessLine() {
        //no-op
    }
    /**
     * Post step to update searchDirection with new gradient and parameter information
     */
    @Override
    public  void postStep(INDArray gradient) {
        //no-op
    }


    @Override
    public void updateGradientAccordingToParams(Gradient gradient, Model model, int batchSize) {
        if(model instanceof ComputationGraph){
            //TODO: find a better way of doing this...
            ComputationGraph graph = (ComputationGraph)model;
            if(computationGraphUpdater == null){
                computationGraphUpdater = new ComputationGraphUpdater(graph);
            }
            computationGraphUpdater.update(graph, gradient, iteration, batchSize);

        } else {

            if (updater == null)
                updater = UpdaterCreator.getUpdater(model);
            Layer layer = (Layer) model;
            updater.update(layer, gradient, iteration, batchSize);
        }
    }

    /**
     * Setup the initial search state
     * @param pair
     */
    @Override
    public  void setupSearchState(Pair<Gradient, Double> pair) {
        INDArray gradient = pair.getFirst().gradient(conf.variables());
        INDArray params = model.params();
        searchState.put(GRADIENT_KEY,gradient);
        searchState.put(SCORE_KEY,pair.getSecond());
        searchState.put(PARAMS_KEY,params);
    }


    public static StepFunction getDefaultStepFunctionForOptimizer( Class<? extends ConvexOptimizer> optimizerClass ){
        log.warn("Objective function automatically set to minimize. Set stepFunction in neural net configuration to change default settings.");
        if( optimizerClass == StochasticGradientDescent.class ){
            return new NegativeGradientStepFunction();
        } else {
            return new NegativeDefaultStepFunction();
        }
    }

}
