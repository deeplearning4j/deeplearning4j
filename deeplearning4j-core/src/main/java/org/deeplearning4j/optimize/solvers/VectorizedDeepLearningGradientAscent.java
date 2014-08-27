package org.deeplearning4j.optimize.solvers;


import org.deeplearning4j.exception.InvalidStepException;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.optimize.api.NeuralNetEpochListener;
import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.util.OptimizerMatrix;
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
    private float maxStep = 1.0f;

    static final float initialStepSize = 0.2f;
    float tolerance = 0.00001f;
    int maxIterations = 200;
    VectorizedBackTrackLineSearch lineMaximizer;
    float stpmax = 100;
    private static Logger logger = LoggerFactory.getLogger(VectorizedDeepLearningGradientAscent.class);
    // "eps" is a small number to rectify the special case of converging
    // to exactly zero function value
    final float eps = 1.0e-10f;
    float step = initialStepSize;
    TrainingEvaluator eval;

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function, float initialStepSize) {
        this.optimizable = function;
        this.lineMaximizer = new VectorizedBackTrackLineSearch(function);
        lineMaximizer.setAbsTolx(tolerance);
        // Alternative:
        //this.lineMaximizer = new GradientBracketLineOptimizer (function);

    }

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function,NeuralNetEpochListener listener) {
        this(function, 0.01f);
        this.listener = listener;

    }

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function, float initialStepSize,NeuralNetEpochListener listener) {
        this(function,initialStepSize);
        this.listener = listener;


    }

    public VectorizedDeepLearningGradientAscent(OptimizableByGradientValueMatrix function) {
        this(function, 0.01f);
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
     * @param tolerance tolerance for convergence test
     */
    public void setTolerance(float tolerance) {
        this.tolerance = tolerance;
    }

    public float getInitialStepSize ()
    {
        return initialStepSize;
    }

    public void setInitialStepSize (float initialStepSize)
    {
        step = initialStepSize;
    }

    public float getStpmax ()
    {
        return stpmax;
    }

    public void setStpmax (float stpmax)
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
        float fret;
        float fp = optimizable.getValue ();
        INDArray xi = optimizable.getValueGradient(0);

        for (iterations = 0; iterations < numIterations; iterations++) {
            logger.info ("At iteration "+ iterations +", cost = "+ fp  +", scaled = "+ maxStep +" step = "+step+", gradient infty-norm = "+ xi.normmax(Integer.MAX_VALUE));
            boolean calledEpochDone = false;
            // Ensure step not too large
            optimizable.setCurrentIteration(iterations);
            float sum = (float) xi.norm2(Integer.MAX_VALUE).element();
            if (sum > stpmax) {
                logger.info ("*** Step 2-norm "+sum+" greater than max " + stpmax + "  Scaling...");
                xi.muli(stpmax / sum);
            }
            try {
                step = lineMaximizer.optimize (xi,iterations, step);

            }catch(InvalidStepException e) {
                logger.warn("Error during computation",e);
                continue;

            }
            fret = optimizable.getValue ();
            if (2.0 * Math.abs(fret-fp) <= tolerance * (Math.abs(fret) + Math.abs(fp) + eps)) {
                logger.info ("Gradient Ascent: Value difference " + Math.abs( fret-fp ) +" below " +
                        "tolerance; saying converged.");
                converged = true;
                if(listener != null) {
                    listener.iterationDone(iterations);
                    calledEpochDone = true;
                }
                return true;
            }

            fp = fret;

            xi = optimizable.getValueGradient(iterations);


            if(listener != null && !calledEpochDone) {
                listener.iterationDone(iterations);
            }
            if(eval != null && eval.shouldStop(iterations)) {
                return true;
            }

        }
        return false;
    }

    public void setMaxStepSize (float v)
    {
        maxStep = v;
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
}
