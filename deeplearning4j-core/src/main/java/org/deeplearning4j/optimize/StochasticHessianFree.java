package org.deeplearning4j.optimize;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.util.OptimizerMatrix;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Hessian Free Optimization
 * by Ryan Kiros http://www.cs.toronto.edu/~rkiros/papers/shf13.pdf
 * @author Adam Gibson
 */
public class StochasticHessianFree implements OptimizerMatrix {
    private static Logger logger = LoggerFactory.getLogger(StochasticHessianFree.class);

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
    //conjugate gradient decay
    private DoubleMatrix ch;
    private NeuralNetEpochListener listener;
    private BaseMultiLayerNetwork multiLayerNetwork;
    private double pi = 0.5;
    private double decrease = 0.99;
    private double boost = 1.0 / decrease;

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

    public StochasticHessianFree(OptimizableByGradientValueMatrix function, double initialStepSize,BaseMultiLayerNetwork network) {
        this.initialStepSize = initialStepSize;
        this.optimizable = function;
        this.lineMaximizer = new VectorizedBackTrackLineSearch(function);
        lineMaximizer.setAbsTolx(tolerance);
        this.network = network;
        // Alternative:
        //this.lineMaximizer = new GradientBracketLineOptimizer (function);

    }

    public StochasticHessianFree(OptimizableByGradientValueMatrix function, NeuralNetEpochListener listener,BaseMultiLayerNetwork network) {
        this(function, 0.01,network);
        this.listener = listener;

    }

    public StochasticHessianFree(OptimizableByGradientValueMatrix function, double initialStepSize, NeuralNetEpochListener listener,BaseMultiLayerNetwork network) {
        this(function,initialStepSize,network);
        this.listener = listener;


    }

    public StochasticHessianFree(OptimizableByGradientValueMatrix function,BaseMultiLayerNetwork network) {
        this(function, 0.01,network);
        this.network = network;
    }


    void setup() {
        ch = DoubleMatrix.zeros(1,optimizable.getNumParameters());
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

        List<DoubleMatrix> is = new ArrayList<>();
        List<DoubleMatrix> xs = new ArrayList<>();

        xi = optimizable.getValueGradient(0);

        Pair<DoubleMatrix,DoubleMatrix> backward = multiLayerNetwork.getBackPropRGradient2();
        DoubleMatrix gradient = backward.getFirst().neg();
        DoubleMatrix precon = backward.getSecond();

        DoubleMatrix r = multiLayerNetwork.getBackPropRGradient().sub(xi);

        long curr = 0;
        for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {
            curr = System.currentTimeMillis();
            logger.info(myName + " ConjugateGradient: At iteration " + iterations + ", cost = " + fp + " -"
                    + (curr - last));
            last = curr;
            double oldScore = network.score();

            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(optimizable,listener);
            g.optimize(numIterations);
            if(network != null) {
                double rho = network.reductionRatio(g.getH(),network.score(),oldScore,g.getG());
                VectorizedBackTrackLineSearch search = new VectorizedBackTrackLineSearch(optimizable);
                double newPoint = search.optimize(g.getXi(),1000,network.score());
                this.fp = newPoint;


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
