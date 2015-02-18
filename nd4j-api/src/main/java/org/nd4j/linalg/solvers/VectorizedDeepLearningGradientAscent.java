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

package org.nd4j.linalg.solvers;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.solvers.api.IterationListener;
import org.nd4j.linalg.solvers.api.OptimizableByGradientValueMatrix;
import org.nd4j.linalg.solvers.api.OptimizerMatrix;
import org.nd4j.linalg.solvers.api.TrainingEvaluator;
import org.nd4j.linalg.solvers.exception.InvalidStepException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Vectorized Stochastic Gradient Ascent
 *
 * @author Adam Gibson
 */
public class VectorizedDeepLearningGradientAscent implements OptimizerMatrix {


    static final double initialStepSize = 0.2f;
    private static Logger logger = LoggerFactory.getLogger(VectorizedDeepLearningGradientAscent.class);
    // "eps" is a small number to rectify the special case of converging
    // to exactly zero function value
    final double eps = 1.0e-10f;
    boolean converged = false;
    OptimizableByGradientValueMatrix optimizable;
    double tolerance = 0.00001f;
    int maxIterations = 200;
    VectorizedBackTrackLineSearch lineMaximizer;
    double stpmax = 100;
    double step = initialStepSize;
    TrainingEvaluator eval;
    private IterationListener listener;
    private double maxStep = 1.0f;

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function, double initialStepSize) {
        this.optimizable = function;
        this.lineMaximizer = new VectorizedBackTrackLineSearch(function);
        lineMaximizer.setAbsTolx(tolerance);
        // Alternative:
        //this.lineMaximizer = new GradientBracketLineOptimizer (function);

    }

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function, IterationListener listener) {
        this(function, 0.01f);
        this.listener = listener;

    }

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function, double initialStepSize, IterationListener listener) {
        this(function, initialStepSize);
        this.listener = listener;


    }

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function) {
        this(function, 0.01f);
    }


    @Override
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public OptimizableByGradientValueMatrix getOptimizable() {
        return this.optimizable;
    }

    public boolean isConverged() {
        return converged;
    }


    public VectorizedBackTrackLineSearch getLineMaximizer() {
        return lineMaximizer;
    }

	/* Tricky: this is now applyTransformToDestination at GradientAscent construction time.  How to applyTransformToDestination it later?
     * What to pass as an argument here?  The lineMaximizer needs the function at the time of its construction!
	  public void setLineMaximizer (LineOptimizer.ByGradient lineMaximizer)
	  {
	    this.lineMaximizer = lineMaximizer;
	  }*/


    /**
     * Sets the tolerance in the convergence test:
     * 2.0*|value-old_value| <= tolerance*(|value|+|old_value|+eps)
     * Default value is 0.001.
     *
     * @param tolerance tolerance for convergence test
     */
    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    public double getInitialStepSize() {
        return initialStepSize;
    }

    public void setInitialStepSize(double initialStepSize) {
        step = initialStepSize;
    }

    public double getStpmax() {
        return stpmax;
    }

    public void setStpmax(double stpmax) {
        this.stpmax = stpmax;
    }

    public boolean optimize() {
        return optimize(maxIterations);
    }

    public boolean optimize(int numIterations) {
        int iterations;
        double fret;
        double fp = optimizable.getValue();
        INDArray xi = optimizable.getValueGradient(0);

        for (iterations = 0; iterations < numIterations; iterations++) {
            logger.info("At iteration " + iterations + ", cost = " + fp + ", scaled = " + maxStep + " step = " + step + ", gradient infty-norm = " + xi.normmax(Integer.MAX_VALUE));
            boolean calledEpochDone = false;
            // Ensure step not too large
            optimizable.setCurrentIteration(iterations);
            double sum = (double) xi.norm2(Integer.MAX_VALUE).element();
            if (sum > stpmax) {
                logger.info("*** Step 2-norm " + sum + " greater than max " + stpmax + "  Scaling...");
                xi.muli(stpmax / sum);
            }
            try {
                step = lineMaximizer.optimize(xi, iterations, step);

            } catch (InvalidStepException e) {
                logger.warn("Error during computation", e);
                continue;

            }
            fret = optimizable.getValue();
            if (2.0 * Math.abs(fret - fp) <= tolerance * (Math.abs(fret) + Math.abs(fp) + eps)) {
                logger.info("Gradient Ascent: Value difference " + Math.abs(fret - fp) + " below " +
                        "tolerance; saying converged.");
                converged = true;
                if (listener != null) {
                    listener.iterationDone(iterations);
                    calledEpochDone = true;
                }
                return true;
            }

            fp = fret;

            xi = optimizable.getValueGradient(iterations);


            if (listener != null && !calledEpochDone) {
                listener.iterationDone(iterations);
            }
            if (eval != null && eval.shouldStop(iterations)) {
                return true;
            }

        }
        return false;
    }

    public void setMaxStepSize(double v) {
        maxStep = v;
    }

}
