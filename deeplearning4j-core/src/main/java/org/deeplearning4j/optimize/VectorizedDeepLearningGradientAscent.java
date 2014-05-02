package org.deeplearning4j.optimize;


import org.deeplearning4j.util.OptimizerMatrix;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Vectorized Stochastic Gradient Ascent
 * @author Adam Gibson
 *
 */
public class VectorizedDeepLearningGradientAscent implements OptimizerMatrix {


	private NeuralNetEpochListener listener;
	boolean converged = false;
	OptimizableByGradientValueMatrix optimizable;
	private double maxStep = 1.0;

	static final double initialStepSize = 0.2;
	double tolerance = 0.00001;
	int maxIterations = 200;
	VectorizedBackTrackLineSearch lineMaximizer;
	double stpmax = 100;
	private static Logger logger = LoggerFactory.getLogger(VectorizedDeepLearningGradientAscent.class);
	// "eps" is a small number to rectify the special case of converging
	// to exactly zero function value
	final double eps = 1.0e-10;
	double step = initialStepSize;


	public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function, double initialStepSize) {
		this.optimizable = function;
		this.lineMaximizer = new VectorizedBackTrackLineSearch(function);
		lineMaximizer.setAbsTolx(tolerance);
		// Alternative:
		//this.lineMaximizer = new GradientBracketLineOptimizer (function);

	}

	public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function,NeuralNetEpochListener listener) {
		this(function, 0.01);
		this.listener = listener;

	}

	public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function, double initialStepSize,NeuralNetEpochListener listener) {
		this(function,initialStepSize);
		this.listener = listener;


	}

	public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function) {
		this(function, 0.01);
	}


    @Override
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }

    public OptimizableByGradientValueMatrix getOptimizable () { return this.optimizable; }
	public boolean isConverged () { return converged; }


	public  VectorizedBackTrackLineSearch  getLineMaximizer ()
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
		DoubleMatrix xi = optimizable.getValueGradient();

		for (iterations = 0; iterations < numIterations; iterations++) {
			logger.info ("At iteration "+ iterations +", cost = "+ fp  +", scaled = "+ maxStep +" step = "+step+", gradient infty-norm = "+ xi.normmax());
            boolean calledEpochDone = false;
			// Ensure step not too large
			double sum = xi.norm2();
			if (sum > stpmax) {
				logger.info ("*** Step 2-norm "+sum+" greater than max "+stpmax+"  Scaling...");
				xi.muli(stpmax / sum);
			}
			try {
				step = lineMaximizer.optimize (xi, step);

			}catch(Exception e) {
				logger.warn("Error during computation",e);
				continue;

			}
			fret = optimizable.getValue ();
			if (2.0 * Math.abs(fret-fp) <= tolerance * (Math.abs(fret) + Math.abs(fp) + eps)) {
				logger.info ("Gradient Ascent: Value difference " + Math.abs( fret-fp ) +" below " +
						"tolerance; saying converged.");
				converged = true;
				if(listener != null) {
					listener.epochDone(iterations);
                    calledEpochDone = true;
				}
				return true;
			}

			fp = fret;

			xi = optimizable.getValueGradient();


			if(listener != null && !calledEpochDone) {
				listener.epochDone(iterations);
			}
		}
		return false;
	}

	public void setMaxStepSize (double v)
	{
		maxStep = v;
	}



}
