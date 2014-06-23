package org.deeplearning4j.optimize;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;

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


    private Pair<List<Integer>,List<DoubleMatrix>> conjGradient(int numIterations) {
        List<Integer> is = new ArrayList<>();
        List<DoubleMatrix> xs = new ArrayList<>();

        xi = optimizable.getValueGradient(0);

        Pair<DoubleMatrix,DoubleMatrix> backward = multiLayerNetwork.getBackPropRGradient2(xi);
        DoubleMatrix gradient = backward.getFirst().neg();
        DoubleMatrix precon = backward.getSecond();

        DoubleMatrix r = multiLayerNetwork.getBackPropRGradient(ch).sub(xi);
        DoubleMatrix y = r.div(precon);
        DoubleMatrix p = y.neg();
        DoubleMatrix x = xi;
        double deltaNew = r.mul(y).sum();

        long curr = 0;
        for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {
            DoubleMatrix Ap = network.getBackPropRGradient(p);
            double pAp = p.mul(Ap).sum();
            if (pAp < 0)
                throw new IllegalStateException("Negative curature!");

            double alpha = deltaNew / pAp;

            x.addi(p.mul(alpha));

            //conjugate gradient
            DoubleMatrix rNew = r.add(alpha).mul(Ap);
            DoubleMatrix yNew = rNew.div(precon);
            double deltaOld = deltaNew;
            deltaNew = rNew.mul(yNew).sum();
            double beta = deltaNew / deltaOld;
            p = yNew.neg().add(beta).mul(p);
            r = rNew;
            y = yNew;
            is.add(iterationCount);
            xs.add(x);


            curr = System.currentTimeMillis();


            if (listener != null) {
                listener.iterationDone(iterationCount);
            }

        }

        return new Pair<>(is,xs);
    }

    //setup baseline conjugate gradietn and run it for n iterations
    private Triple<DoubleMatrix,List<DoubleMatrix>,DoubleMatrix> runConjugateGradient(int numIterations) {
        Pair<List<Integer>,List<DoubleMatrix>> cg = conjGradient(numIterations);
        DoubleMatrix ch = cg.getSecond().get(cg.getSecond().size() - 1);
        DoubleMatrix p = ch;

        return new Triple<>(p,cg.getSecond(),ch);
    }



    public Pair<DoubleMatrix,Double> cgBackTrack(DoubleMatrix p,List<DoubleMatrix> chs,DoubleMatrix x,DoubleMatrix y) {
        double score = network.score(p.add(xi));
        int i = chs.size() - 2;
        for(; i >= 0; i--) {
            double score2 = network.score(xi.add(chs.get(i)));
            if(score2 < score) {
                i++;
                break;
            }
            score = score2;

        }

        return new Pair<>(chs.get(i),score);
    }



    public boolean optimize(int numIterations) {
        myName = Thread.currentThread().getName();
        if (converged)
            return true;
        Triple<DoubleMatrix,List<DoubleMatrix>,DoubleMatrix>  cg = runConjugateGradient(numIterations);
        Pair<DoubleMatrix,Double> cgBackTrack = cgBackTrack(cg.getFirst(),cg.getSecond(),network.getInput(),network.getLabels());
        double rho = network.reductionRatio(cgBackTrack.getFirst(),network.score(),cgBackTrack.getSecond(),xi);
        VectorizedBackTrackLineSearch l = new VectorizedBackTrackLineSearch(optimizable);
        double step = l.optimize(cg.getFirst(),numIterations,rho);
        network.dampingUpdate(rho,boost,decrease);
        xi.addi(cgBackTrack.getFirst().mul(step * network.getLearningRateUpdate()));
        return true;
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
