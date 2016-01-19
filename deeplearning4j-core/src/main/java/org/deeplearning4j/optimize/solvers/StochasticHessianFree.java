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
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Hessian Free Optimization
 * by Ryan Kiros http://www.cs.toronto.edu/~rkiros/papers/shf13.pdf
 * @author Adam Gibson
 */
public class StochasticHessianFree extends BaseOptimizer {
    private static final Logger logger = LoggerFactory.getLogger(StochasticHessianFree.class);

    boolean converged = false;
    double initialStepSize = 1f;
    double tolerance = 1e-5f;
    double gradientTolerance = 0f;
    private MultiLayerNetwork network;
    int maxIterations = 10000;
    /* decay, current gradient/direction/current point in vector space,preCondition on conjugate gradient,current parameters */
    private INDArray ch,gradient,xi;
    private double pi = 0.5f;
    private double decrease = 0.99f;
    private double boost = 1.0f / decrease;
    /* current score, step size */
    private double score;

    public StochasticHessianFree(NeuralNetConfiguration conf, StepFunction stepFunction,
                                 Collection<IterationListener> iterationListeners, Model model) {
        super(conf, stepFunction, iterationListeners, model);
        setup();
    }

    public StochasticHessianFree(NeuralNetConfiguration conf, StepFunction stepFunction,
                                 Collection<IterationListener> iterationListeners,
                                 Collection<TerminationCondition> terminationConditions, Model model) {
        super(conf, stepFunction, iterationListeners, terminationConditions, model);
        setup();
    }


    void setup() {
        if(!(model instanceof MultiLayerNetwork))
            return;
        network = (MultiLayerNetwork) model;
        xi = network.pack();
        ch = Nd4j.zeros(1, xi.length());
    }


    public boolean isConverged() {
        return converged;
    }

    /* run conjugate gradient for numIterations */
    public Pair<List<Integer>,List<INDArray>> conjGradient(INDArray b,INDArray x0,INDArray preCon,int numIterations) {
        List<Integer> is = new ArrayList<>();
        List<INDArray> xs = new ArrayList<>();
        //in the pseudo code the gradient is b
        //x0 is ch
        INDArray r = network.getBackPropRGradient(x0).subi(b);
        INDArray y = r.div(preCon);
        double deltaNew =  r.mul(y).sum(Integer.MAX_VALUE).getDouble(0);
        INDArray p = y.neg();

        for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {

            INDArray Ap = network.getBackPropRGradient(p);
            //log.info("Ap sum at iteration " + iterationCount + " is " + Ap.sum());
            //think Ax + b, this is the curvature
            double pAp =  Ap.mul(p).sum(Integer.MAX_VALUE).getDouble(0);
            if(pAp < 0) {
                log.info("Negative slope: " + pAp + " breaking");
            }

            //step size
            double alpha = deltaNew / pAp;
            //step
            x0.addi(p.mul(alpha));

            //conjugate gradient
            INDArray rNew = r.addi(Ap.mul(alpha));
            INDArray yNew = rNew.div(preCon);
            double deltaOld = deltaNew;
            deltaNew =  rNew.mul(yNew).sum(Integer.MAX_VALUE).getDouble(0);
            double beta = deltaNew / deltaOld;
            p = yNew.neg().addi(p.mul(beta));

            r = rNew;
            //append to the steps taken
            is.add(iterationCount);
            xs.add(x0.dup());

        }
        return new Pair<>(is,xs);
    }

    //setup baseline conjugate gradient and run it for n iterations
    private Triple<INDArray,List<INDArray>,INDArray> runConjugateGradient(INDArray preCon,int numIterations) {
        Pair<List<Integer>,List<INDArray>> cg = conjGradient(gradient,ch,preCon,numIterations);
        ch = cg.getSecond().get(cg.getSecond().size() - 1);
        return new Triple<>(ch,cg.getSecond(),ch);
    }


    /**
     * Search with the proposed objective
     * @param newScore the new score to start with
     * @param params the params of the proposed step
     * @return the rate to step by
     */
    public double lineSearch(double newScore,INDArray params,INDArray p) {
        double rate = 1.0f;
        double c = 1e-2f;
        int j = 0;
        int numSearches = 60;
        while(j < numSearches) {
            if(10 % numSearches == 0) {
                log.info("Iteration " + j + " on line search with current rate of " + rate);
            }
            //converged
            if(newScore <=   gradient.mul(p).muli(score + c * rate).sum(Integer.MAX_VALUE).getDouble(0)) {
                break;
            }
            else {
                rate *= 0.8f;
                j++;
            }

            //explore in this direction and obtain a score
            newScore = network.score(params.add(p.mul(rate)));
        }

        if(j == numSearches) {
            rate = 0.0f;
            log.info("Went too far...reverting rate to 0");
        }


        return rate;

    }



    /**
     * Iterate through the current list of gradients
     * and backtrack upon an optimal step
     * that improves the current score
     * @param chs the proposed changes
     * @return the new changed path and score for that path
     */
    public Pair<INDArray,Double> cgBackTrack(List<INDArray> chs,INDArray p) {
        INDArray params = network.params();
        double score = network.score(p.add(params));
        double currMin = network.score();
        int i = chs.size() - 2;

        for(; i > 0; i--) {
            double score2 = network.score(params.add(chs.get(i)));
            if(score2 < score || score2 < currMin) {
                i++;
                score = score2;
                break;
            }


        }

        if(i < 0)
            i = 0;

        return new Pair<>(chs.get(i),score);
    }

    @Override
    public boolean optimize() {
        if(!(model instanceof MultiLayerNetwork))
            return true;

        if (converged)
            return true;

        score = network.score();

        xi = network.params();

       for(int i = 0; i < conf.getNumIterations(); i++) {
           //initial gradient, precon/conjugate gradient conditioner
           Pair<INDArray,INDArray> backPropGradient = network.getBackPropGradient2();

           gradient = backPropGradient.getFirst().neg();
           //log.info("Gradient sum " + gradient.sum());

           INDArray preCon = backPropGradient.getSecond();

           if(ch == null)
               setup();

           ch.muli(pi);

           Triple<INDArray,List<INDArray>,INDArray>  cg = runConjugateGradient(preCon,conf.getNumIterations());

           INDArray searchDirection = cg.getFirst();

           Pair<INDArray,Double> cgBackTrack = cgBackTrack(cg.getSecond(),searchDirection);

           searchDirection = cgBackTrack.getFirst();

           double rho = network.reductionRatio(cgBackTrack.getFirst(), network.score(), cgBackTrack.getSecond(), gradient);
           double newScore = network.score(cgBackTrack.getFirst());

           double step = lineSearch(newScore,gradient,searchDirection);
           network.dampingUpdate(rho,boost,decrease);

           INDArray proposedUpdate = xi.add(searchDirection.mul(1.0f * step));
           network.setParameters(proposedUpdate);
           log.info("Score at iteration " + i + " was " + newScore);
       }

        return true;
    }





}
