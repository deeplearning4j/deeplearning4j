package org.deeplearning4j.util;

import java.util.logging.Logger;

import org.deeplearning4j.optimize.NeuralNetEpochListener;

import cc.mallet.optimize.BackTrackLineSearch;
import cc.mallet.optimize.GradientAscent;
import cc.mallet.optimize.LineOptimizer;
import cc.mallet.optimize.Optimizable;
import cc.mallet.optimize.Optimizer;
import cc.mallet.optimize.OptimizerEvaluator;
import cc.mallet.optimize.Optimizable.ByGradientValue;
import cc.mallet.optimize.OptimizerEvaluator.ByGradient;
import cc.mallet.types.MatrixOps;
import cc.mallet.util.MalletLogger;

public class DeepLearningGradientAscent implements Optimizer {




	public DeepLearningGradientAscent (Optimizable.ByGradientValue function,NeuralNetEpochListener listener)
	{
		optimizable = function;
		lineMaximizer = new BackTrackLineSearch(function);
		this.listener = listener;
	}
	
	
	private static Logger logger = MalletLogger.getLogger(GradientAscent.class.getName());

	
	private NeuralNetEpochListener listener;
	boolean converged = false;
	Optimizable.ByGradientValue optimizable;
	private double maxStep = 1.0;
	private OptimizerEvaluator.ByGradient eval;

	static final double initialStepSize = 0.2;
	double tolerance = 0.00001;
	int maxIterations = 200;
	LineOptimizer.ByGradient lineMaximizer;
	double stpmax = 100;

	// "eps" is a small number to rectify the special case of converging
	// to exactly zero function value
	final double eps = 1.0e-10;
	double step = initialStepSize;



	public Optimizable getOptimizable () { return this.optimizable; }
	public boolean isConverged () { return converged; }


	public LineOptimizer.ByGradient getLineMaximizer ()
	{
		return lineMaximizer;
	}

	/* Tricky: this is now set at GradientAscent construction time.  How to set it later?  
	 * What to pass as an argument here?  The lineMaximizer needs the function at the time of its construction!
	  public void setLineMaximizer (LineOptimizer.ByGradient lineMaximizer)
	  {
	    this.lineMaximizer = lineMaximizer;
	  }*/


	/**
	 * Sets the tolerance in the convergence test:
	 * 2.0*|value-old_value| <= tolerance*(|value|+|old_value|+eps)
	 * Default value is 0.001.
	 * @param tolerance tolerance for convergence test
	 */
	public void setTolerance(double tolerance) {
		this.tolerance = tolerance;
	}

	public double getInitialStepSize ()
	{
		return initialStepSize;
	}

	public void setInitialStepSize (double initialStepSize)
	{
		step = initialStepSize;
	}

	public double getStpmax ()
	{
		return stpmax;
	}

	public void setStpmax (double stpmax)
	{
		this.stpmax = stpmax;
	}

	public boolean optimize ()
	{
		return optimize (maxIterations);
	}

	public boolean optimize (int numIterations)
	{
		int iterations;
		double fret;
		double fp = optimizable.getValue ();
		double[] xi = new double [optimizable.getNumParameters()];
		optimizable.getValueGradient(xi);

		for (iterations = 0; iterations < numIterations; iterations++) {
			logger.info ("At iteration "+ iterations +", cost = "+ fp  +", scaled = "+ maxStep +" step = "+step+", gradient infty-norm = "+MatrixOps.infinityNorm (xi));

			// Ensure step not too large
			double sum = MatrixOps.twoNorm (xi);
			if (sum > stpmax) {
				logger.info ("*** Step 2-norm "+sum+" greater than max "+stpmax+"  Scaling...");
				MatrixOps.timesEquals (xi,stpmax/sum);
			}

			step = lineMaximizer.optimize (xi, step);
			fret = optimizable.getValue ();
			if (2.0 * Math.abs(fret-fp) <= tolerance * (Math.abs(fret) + Math.abs(fp) + eps)) {
				logger.info ("Gradient Ascent: Value difference " + Math.abs( fret-fp ) +" below " +
						"tolerance; saying converged.");
				converged = true;
				if(listener != null) {
					listener.epochDone(iterations);
				}
				return true;
			}
			
			fp = fret;
			
			optimizable.getValueGradient(xi);
			
			
			if (eval != null) {
				eval.evaluate (optimizable, iterations);
			}
			if(listener != null) {
				listener.epochDone(iterations);
			}
		}
		return false;
	}

	public void setMaxStepSize (double v)
	{
		maxStep = v;
	}

	public void setEvaluator (OptimizerEvaluator.ByGradient eval)
	{
		this.eval = eval;
	}

}
