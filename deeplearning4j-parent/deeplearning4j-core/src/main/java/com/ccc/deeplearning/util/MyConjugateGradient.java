/* Copyright (C) 2002 Univ. of Massachusetts Amherst, Computer Science Dept.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
   http://www.cs.umass.edu/~mccallum/mallet
   This software is provided under the terms of the Common Public License,
   version 1.0, as published by http://www.opensource.org.  For further
   information, see the file `LICENSE' included with this distribution. */

/** 
 @author Andrew McCallum <a href="mailto:mccallum@cs.umass.edu">mccallum@cs.umass.edu</a>
 */

package com.ccc.deeplearning.util;

import java.util.logging.Logger;

import cc.mallet.optimize.BackTrackLineSearch;
import cc.mallet.optimize.LineOptimizer;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;
import cc.mallet.optimize.OptimizerEvaluator;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.MalletLogger;

import com.ccc.deeplearning.optimize.NeuralNetEpochListener;

/**
 * Modified based on cc.mallet.optimize.ConjugateGradient <p/>
 * no termination when zero tolerance 
 * 
 * @author GuoDing
 * @since 2013-08-25
 */

// Conjugate Gradient, Polak and Ribiere version
// from "Numeric Recipes in C", Section 10.6.

public class MyConjugateGradient implements Optimizer {
	private static Logger logger = MalletLogger.getLogger(MyConjugateGradient.class.getName());

	boolean converged = false;
	Optimizable.ByGradientValue optimizable;
	LineOptimizer.ByGradient lineMaximizer;

	double initialStepSize = 1;
	double tolerance = 0.0001;
	double gradientTolerance = 0.001;
	int maxIterations = 1000;
	private String myName = "";
	private NeuralNetEpochListener listener;

	// "eps" is a small number to recitify the special case of converging
	// to exactly zero function value
	final double eps = 1.0e-10;
	private OptimizerEvaluator.ByGradient eval;

	public MyConjugateGradient(Optimizable.ByGradientValue function, double initialStepSize) {
		this.initialStepSize = initialStepSize;
		this.optimizable = function;
		this.lineMaximizer = new BackTrackLineSearch(function);
		BackTrackLineSearch l = (BackTrackLineSearch) this.lineMaximizer;
		l.setAbsTolx(eps);
		// Alternative:
		//this.lineMaximizer = new GradientBracketLineOptimizer (function);

	}

	public MyConjugateGradient(Optimizable.ByGradientValue function,NeuralNetEpochListener listener) {
		this(function, 0.01);
		this.listener = listener;

	}

	public MyConjugateGradient(Optimizable.ByGradientValue function, double initialStepSize,NeuralNetEpochListener listener) {
		this(function,initialStepSize);
		this.listener = listener;


	}

	public MyConjugateGradient(Optimizable.ByGradientValue function) {
		this(function, 0.01);
	}


	public Optimizable getOptimizable() {
		return this.optimizable;
	}

	public boolean isConverged() {
		return converged;
	}

	public void setEvaluator(OptimizerEvaluator.ByGradient eval) {
		this.eval = eval;
	}

	public void setLineMaximizer(LineOptimizer.ByGradient lineMaximizer) {
		this.lineMaximizer = lineMaximizer;
	}

	public void setInitialStepSize(double initialStepSize) {
		this.initialStepSize = initialStepSize;
	}

	public double getInitialStepSize() {
		return this.initialStepSize;
	}

	public double getStepSize() {
		return step;
	}

	// The state of a conjugate gradient search
	double fp, gg, gam, dgg, step, fret;
	double[] xi, g, h;
	int j, iterations;

	public boolean optimize() {
		return optimize(maxIterations);
	}

	public void setTolerance(double t) {
		tolerance = t;
	}

	public boolean optimize(int numIterations) {
		myName = Thread.currentThread().getName();
		if (converged)
			return true;
		int n = optimizable.getNumParameters();
		long last = System.currentTimeMillis();
		if (xi == null) {
			fp = optimizable.getValue();
			xi = new double[n];
			g = new double[n];
			h = new double[n];
			optimizable.getValueGradient(xi);
			System.arraycopy(xi, 0, g, 0, n);
			System.arraycopy(xi, 0, h, 0, n);
			step = initialStepSize;
			iterations = 0;
		}

		long curr = 0;
		for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {
			curr = System.currentTimeMillis();
			if(listener != null)
				listener.epochDone(iterationCount);

			logger.info(myName + " ConjugateGradient: At iteration " + iterations + ", cost = " + fp + " -"
					+ (curr - last));
			last = curr;
			try {
				step = lineMaximizer.optimize(xi, step);
			} catch (Throwable e) {
				logger.info(e.getMessage());
			}
			fret = optimizable.getValue();
			optimizable.getValueGradient(xi);

			// This termination provided by "Numeric Recipes in C".
			if ((0 < tolerance) && (2.0 * Math.abs(fret - fp) <= tolerance * (Math.abs(fret) + Math.abs(fp) + eps))) {
				logger.info("ConjugateGradient converged: old value= " + fp + " new value= " + fret + " tolerance="
						+ tolerance);
				converged = true;
				return true;
			}
			fp = fret;

			// This termination provided by McCallum
			double twoNorm = MatrixOps.twoNorm(xi);
			if (twoNorm < gradientTolerance) {
				logger.info("ConjugateGradient converged: gradient two norm " + twoNorm + ", less than "
						+ gradientTolerance);
				converged = true;
				return true;
			}

			dgg = gg = 0.0;
			for (j = 0; j < xi.length; j++) {
				gg += g[j] * g[j];
				dgg += xi[j] * (xi[j] - g[j]);
			}
			gam = dgg / gg;

			for (j = 0; j < xi.length; j++) {
				g[j] = xi[j];
				h[j] = xi[j] + gam * h[j];
			}
			assert (!MatrixOps.isNaN(h));

			// gdruck
			// Mallet line search algorithms stop search whenever
			// a step is found that increases the value significantly.
			// ConjugateGradient assumes that line maximization finds something
			// close
			// to the maximum in that direction. In tests, sometimes the
			// direction suggested by CG was downhill. Consequently, here I am
			// setting the search direction to the gradient if the slope is
			// negative or 0.
			if (MatrixOps.dotProduct(xi, h) > 0) {
				MatrixOps.set(xi, h);
			} else {
				logger.warning("Reverting back to GA");
				MatrixOps.set(h, xi);
			}

			iterations++;
			if (iterations > maxIterations) {
				logger.info("Too many iterations in ConjugateGradient.java");
				converged = true;
				return true;
			}

			if (eval != null) {
				eval.evaluate(optimizable, iterations);
			}
		}
		return false;
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
}
