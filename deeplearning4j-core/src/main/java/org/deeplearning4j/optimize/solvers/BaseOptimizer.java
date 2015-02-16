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

package org.deeplearning4j.optimize.solvers;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.exception.InvalidStepException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.GradientAdjustment;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.deeplearning4j.optimize.terminations.EpsTermination;
import org.deeplearning4j.optimize.terminations.ZeroDirection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;
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
    protected AdaGrad adaGrad;
    protected int iteration = 0;
    protected static final Logger log = LoggerFactory.getLogger(BaseOptimizer.class);
    protected StepFunction stepFunction;
    private Collection<IterationListener> iterationListeners = new ArrayList<>();
    protected Collection<TerminationCondition> terminationConditions = new ArrayList<>();
    protected Model model;
    protected BackTrackLineSearch lineMaximizer;
    protected double step;
    private int batchSize = 10;
    protected double score,oldScore;
    protected double stpMax = Double.MAX_VALUE;
    public final static String GRADIENT_KEY = "g";
    public final static String SCORE_KEY = "score";
    public final static String PARAMS_KEY = "params";
    protected Map<String,AdaGrad> adaGradForVariable = new ConcurrentHashMap<>();
    protected Map<String,Object> searchState = new ConcurrentHashMap<>();

    /**
     *
     * @param conf
     * @param stepFunction
     * @param iterationListeners
     * @param model
     */
    public BaseOptimizer(NeuralNetConfiguration conf,StepFunction stepFunction,Collection<IterationListener> iterationListeners,Model model) {
        this(conf,stepFunction,iterationListeners, Arrays.asList(new ZeroDirection(),new EpsTermination()),model);
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
        this.stepFunction = stepFunction;
        this.iterationListeners = iterationListeners;
        this.terminationConditions = terminationConditions;
        this.model = model;
        lineMaximizer = new BackTrackLineSearch(model,stepFunction,this);
        lineMaximizer.setStpmax(stpMax);
        lineMaximizer.setMaxIterations(conf.getNumLineSearchIterations());

    }

    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     */
    @Override
    public void updateGradientAccordingToParams(INDArray gradient, INDArray params, int batchSize) {
        GradientAdjustment.updateGradientAccordingToParams(conf,0,adaGrad,gradient,params,batchSize);
    }

    @Override
    public double score() {
        model.setScore();
        return model.score();
    }


    @Override
    public Pair<Gradient,Double> gradientAndScore() {
        model.setScore();
        Pair<Gradient,Double> pair = model.gradientAndScore();
        updateGradientAccordingToParams(pair.getFirst(),model,model.batchSize());
        return pair;
    }


    /**
     * Optimize call. This runs the optimizer.
     * @return whether it converged or not
     */
    @Override
    public  boolean optimize() {
        //validate the input before training
        model.validateInput();
        Pair<Gradient,Double> pair = gradientAndScore();
        setupSearchState(pair);
        //get initial score
        score = pair.getSecond();
        //check initial gradient
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);

        //pre existing termination conditions
         for(TerminationCondition condition : terminationConditions)
            if(condition.terminate(0.0,0.0,new Object[]{gradient})) {
                log.info("Hit termination condition " + condition.getClass().getName());
                return true;
            }
        //some algorithms do pre processing of gradient and
        //need to test possible directions. (LBFGS)
        boolean testLineSearch = preFirstStepProcess(gradient);
        if(testLineSearch) {
            //ensure we can take a step
            try {
                INDArray params = (INDArray) searchState.get(PARAMS_KEY);
                step = lineMaximizer.optimize(step,params,gradient);
            } catch (InvalidStepException e) {
                log.warn("Invalid step...continuing another iteration");

            }
            gradient = (INDArray) searchState.get(GRADIENT_KEY);
            postFirstStep(gradient);

            if(step == 0.0) {
                log.warn("Unable to step in direction");
                return false;
            }
        }


        for(int i = 0; i < conf.getNumIterations(); i++) {
            //line normalization where relevant
            preProcessLine(gradient);
            //perform one step
            try {
                INDArray params = (INDArray) searchState.get(PARAMS_KEY);
                step = lineMaximizer.optimize(step,params,gradient);
            } catch (InvalidStepException e) {
                log.warn("Invalid step...continuing another iteration");
            }


            //invoke listeners for debugging
            for(IterationListener listener : iterationListeners)
                listener.iterationDone(model,i);


            //record old score for deltas and other termination conditions
            oldScore = score;
            pair = gradientAndScore();
            setupSearchState(pair);
            //check for termination conditions based on absolute change in score
            for(TerminationCondition condition : terminationConditions)
                if(condition.terminate(score,oldScore,new Object[]{gradient}))
                    return true;

            //post step updates to other search parameters
            postStep();

            //check for termination conditions based on absolute change in score
            for(TerminationCondition condition : terminationConditions)
                if(condition.terminate(score,oldScore,new Object[]{gradient}))
                    return true;


        }

        return true;
    }





    protected  void postFirstStep(INDArray gradient) {
        //no-op
    }

    protected  boolean preFirstStepProcess(INDArray gradient) {
        //no-op
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
     * Pre process the line (scaling and the like)
     * @param line the line to pre process
     */
    @Override
    public  void preProcessLine(INDArray line) {
        //no-op
    }
    /**
     * Post step (conjugate gradient among other methods needs this)

     */
    @Override
    public  void postStep() {
        //no-op
    }

    @Override
    public AdaGrad getAdaGrad() {
        return adaGrad;
    }

    @Override
    public Map<String, AdaGrad> adaGradForVariables() {
        return adaGradForVariable;
    }

    @Override
    public AdaGrad getAdaGradForVariable(String variable) {
        return adaGradForVariable.get(variable);
    }

    @Override
    public void updateGradientAccordingToParams(Gradient gradient, Model params, int batchSize) {
         GradientAdjustment.updateGradientAccordingToParams(conf,iteration,gradient,batchSize,adaGradForVariable,params);
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
        score = pair.getSecond();

    }




}
