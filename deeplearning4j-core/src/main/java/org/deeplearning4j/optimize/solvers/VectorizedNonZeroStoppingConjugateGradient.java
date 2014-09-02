/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

/**
 @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package org.deeplearning4j.optimize.solvers;


import org.deeplearning4j.exception.InvalidStepException;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.optimize.api.LineOptimizerMatrix;
import org.deeplearning4j.optimize.api.NeuralNetEpochListener;
import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.util.OptimizerMatrix;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;



/**
 * Modified based on cc.mallet.optimize.ConjugateGradient <p/>
 * no termination when zero tolerance 
 *
 * @author Adam Gibson
 * @since 2013-08-25
 */

// Conjugate Gradient, Polak and Ribiere version
// from "Numeric Recipes in C", Section 10.6.

public class VectorizedNonZeroStoppingConjugateGradient implements OptimizerMatrix {
    private static Logger logger = LoggerFactory.getLogger(VectorizedNonZeroStoppingConjugateGradient.class);

    boolean converged = false;
    OptimizableByGradientValueMatrix optimizable;
    VectorizedBackTrackLineSearch lineMaximizer;
    TrainingEvaluator eval;
    float initialStepSize = 1;
    float tolerance = 1e-5f;
    float gradientTolerance = 1e-5f;
    int maxIterations = 10000;
    private String myName = "";
    private NeuralNetEpochListener listener;

    // The state of a conjugate gradient search
    float fp, gg, gam, dgg, step, fret;
    INDArray xi, g, h;
    int j, iterations;




    // "eps" is a small number to recitify the special case of converging
    // to exactly zero function value
    final float eps = 1.0e-10f;

    public VectorizedNonZeroStoppingConjugateGradient(OptimizableByGradientValueMatrix function, float initialStepSize) {
        this.initialStepSize = initialStepSize;
        this.optimizable = function;
        this.lineMaximizer = new VectorizedBackTrackLineSearch(function);
        lineMaximizer.setAbsTolx(tolerance);
        // Alternative:
        //this.lineMaximizer = new GradientBracketLineOptimizer (function);

    }

    public VectorizedNonZeroStoppingConjugateGradient(OptimizableByGradientValueMatrix function,NeuralNetEpochListener listener) {
        this(function, 0.01f);
        this.listener = listener;

    }

    public VectorizedNonZeroStoppingConjugateGradient(OptimizableByGradientValueMatrix function, float initialStepSize,NeuralNetEpochListener listener) {
        this(function,initialStepSize);
        this.listener = listener;


    }

    public VectorizedNonZeroStoppingConjugateGradient(OptimizableByGradientValueMatrix function) {
        this(function, 0.01f);
    }


    public boolean isConverged() {
        return converged;
    }



    public void setLineMaximizer(LineOptimizerMatrix lineMaximizer) {
        this.lineMaximizer = (VectorizedBackTrackLineSearch) lineMaximizer;
    }

    public void setInitialStepSize(float initialStepSize) {
        this.initialStepSize = initialStepSize;
    }

    public float getInitialStepSize() {
        return this.initialStepSize;
    }

    public float getStepSize() {
        return step;
    }

    public boolean optimize() {
        return optimize(maxIterations);
    }

    public void setTolerance(float t) {
        tolerance = t;
    }

    public boolean optimize(int numIterations) {
        myName = Thread.currentThread().getName();
        if (converged)
            return true;
        long last = System.currentTimeMillis();
        if (xi == null) {
            fp = optimizable.getValue();
            assert !Float.isNaN(fp) && !Float.isInfinite(fp);
            xi = optimizable.getValueGradient(0);
            g = xi.dup();
            h = xi.dup();
            iterations = 0;
        }

        long curr = 0;
        for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {
            curr = System.currentTimeMillis();
            logger.info(myName + " ConjugateGradient: At iteration " + iterations + ", cost = " + fp + " -"
                    + (curr - last));
            last = curr;
            optimizable.setCurrentIteration(iterationCount);
            try {
                step = lineMaximizer.optimize(xi, iterationCount,step);
            } catch (InvalidStepException e) {
                logger.warn("Breaking: negative slope");
            }

            fret = optimizable.getValue();
            xi = optimizable.getValueGradient(iterationCount);

            // This termination provided by "Numeric Recipes in C".
            if ((0 < tolerance) && (2.0 * Math.abs(fret - fp) <= tolerance * (Math.abs(fret) + Math.abs(fp) + eps))) {
                logger.info("ConjugateGradient converged: old value= " + fp + " new value= " + fret + " tolerance="
                        + tolerance);
                converged = true;
                return true;
            }

            fp = fret;

            // This termination provided by McCallum
            float twoNorm = (float) xi.norm2(Integer.MAX_VALUE).element();
            if (twoNorm < gradientTolerance) {
                logger.info("ConjugateGradient converged: gradient two norm " + twoNorm + ", less than "
                        + gradientTolerance);
                converged = true;
                if(listener != null) {
                    listener.iterationDone(iterationCount);
                }
                return true;
            }

            dgg = gg = 0.0f;
            gg = Transforms.pow(g, 2).sum(Integer.MAX_VALUE).get(0);
            dgg =  xi.mul(xi.sub(g)).sum(Integer.MAX_VALUE).get(0);
            gam = dgg / gg;


            g = xi.dup();
            h = xi.add(h.mul(gam));


            LinAlgExceptions.assertValidNum(h);

            // gdruck
            // Mallet line search algorithms stop search whenever
            // a step is found that increases the value significantly.
            // ConjugateGradient assumes that line maximization finds something
            // close
            // to the maximum in that direction. In tests, sometimes the
            // direction suggested by CG was downhill. Consequently, here I am
            // setting the search direction to the gradient if the slope is
            // negative or 0.
            if (NDArrays.getBlasWrapper().dot(xi, h) > 0) {
                xi = h.dup();
            } else {
                logger.warn("Reverting back to GA");
                h = xi.dup();
            }

            iterations++;
            if (iterations > maxIterations) {
                logger.info("Passed max number of iterations");
                converged = true;
                if(listener != null) {
                    listener.iterationDone(iterationCount);
                }
                return true;
            }



            if(listener != null) {
                listener.iterationDone(iterationCount);
            }

            if(eval != null && eval.shouldStop(iterations)) {
                return true;
            }

        }
        return false;
    }

    /**
     * Sets the training evaluator
     *
     * @param eval the evaluator to use
     */
    @Override
    public void setTrainingEvaluator(TrainingEvaluator eval) {
        this.eval = eval;
    }

    public void reset() {
        xi = null;
    }

    public int getMaxIterations() {
        return maxIterations;
    }

    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public INDArray getH() {
        return h;
    }

    public void setH(INDArray h) {
        this.h = h;
    }

    public INDArray getG() {
        return g;
    }

    public void setG(INDArray g) {
        this.g = g;
    }

    public INDArray getXi() {
        return xi;
    }

    public void setXi(INDArray xi) {
        this.xi = xi;
    }

    public float getFret() {
        return fret;
    }

    public void setFret(float fret) {
        this.fret = fret;
    }

    public float getStep() {
        return step;
    }

    public void setStep(float step) {
        this.step = step;
    }

    public float getDgg() {
        return dgg;
    }

    public void setDgg(float dgg) {
        this.dgg = dgg;
    }

    public float getGam() {
        return gam;
    }

    public void setGam(float gam) {
        this.gam = gam;
    }

    public float getGg() {
        return gg;
    }

    public void setGg(float gg) {
        this.gg = gg;
    }

    public float getFp() {
        return fp;
    }

    public void setFp(float fp) {
        this.fp = fp;
    }
}
