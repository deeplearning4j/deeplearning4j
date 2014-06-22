package org.deeplearning4j.optimize;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.OptimizerMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Hessian Free Optimization
 */
public class HessianFreeOptimizerMatrix implements OptimizerMatrix {
    private static Logger logger = LoggerFactory.getLogger(VectorizedNonZeroStoppingConjugateGradient.class);

    boolean converged = false;
    OptimizableByGradientValueMatrix optimizable;
    VectorizedBackTrackLineSearch lineMaximizer;
    TrainingEvaluator eval;
    double initialStepSize = 1;
    double tolerance = 1e-5;
    double gradientTolerance = 1e-5;
    private BaseMultiLayerNetwork network;
    int maxIterations = 10000;
    private String myName = "";
    private NeuralNetEpochListener listener;


    // The state of a conjugate gradient search
    /*
       fp is the current objective score
       gg is the gradient squared
       fret is the best score
     */
    double fp, gg, gam, dgg, step, fret;
    /*

     xi is the current step
     g is the gradient
     h is direction by which to minimize
     */
    DoubleMatrix xi, g, h;
    int iterations;



    // "eps" is a small number to recitify the special case of converging
    // to exactly zero function value
    final double eps = 1.0e-10;

    public HessianFreeOptimizerMatrix(OptimizableByGradientValueMatrix function, double initialStepSize) {
        this.initialStepSize = initialStepSize;
        this.optimizable = function;
        this.lineMaximizer = new VectorizedBackTrackLineSearch(function);
        lineMaximizer.setAbsTolx(tolerance);
        // Alternative:
        //this.lineMaximizer = new GradientBracketLineOptimizer (function);

    }

    public HessianFreeOptimizerMatrix(OptimizableByGradientValueMatrix function,NeuralNetEpochListener listener) {
        this(function, 0.01);
        this.listener = listener;

    }

    public HessianFreeOptimizerMatrix(OptimizableByGradientValueMatrix function, double initialStepSize,NeuralNetEpochListener listener) {
        this(function,initialStepSize);
        this.listener = listener;


    }

    public HessianFreeOptimizerMatrix(OptimizableByGradientValueMatrix function) {
        this(function, 0.01);
    }


    public boolean isConverged() {
        return converged;
    }



    public void setLineMaximizer(LineOptimizerMatrix lineMaximizer) {
        this.lineMaximizer = (VectorizedBackTrackLineSearch) lineMaximizer;
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
        long last = System.currentTimeMillis();


        if (xi == null) {
            fp = optimizable.getValue();
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
            } catch (Throwable e) {
                logger.warn("Error during computation",e);
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

            //update current best
            fp = fret;

            // This termination provided by McCallum
            double twoNorm = xi.norm2();
            if (twoNorm < gradientTolerance) {
                logger.info("ConjugateGradient converged: gradient two norm " + twoNorm + ", less than "
                        + gradientTolerance);
                converged = true;
                if(listener != null) {
                    listener.iterationDone(iterationCount);
                }
                return true;
            }


            dgg = gg = 0.0;
            gg = MatrixFunctions.pow(g, 2).sum();
            dgg = xi.mul(xi.sub(g)).sum();
            gam = dgg / gg;

            //current gradient
            g = xi.dup();
            //points on which to minimize
            h = xi.add(h.mul(gam));


            assert (!MatrixUtil.isNaN(h));

            // gdruck
            // Mallet line search algorithms stop search whenever
            // a step is found that increases the value significantly.
            // ConjugateGradient assumes that line maximization finds something
            // close
            // to the maximum in that direction. In tests, sometimes the
            // direction suggested by CG was downhill. Consequently, here I am
            // setting the search direction to the gradient if the slope is
            // negative or 0.
            if (SimpleBlas.dot(xi, h) > 0) {
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
}
