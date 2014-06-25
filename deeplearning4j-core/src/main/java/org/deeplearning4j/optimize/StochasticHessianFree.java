package org.deeplearning4j.optimize;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.util.MatrixUtil;
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
    private static Logger log = LoggerFactory.getLogger(StochasticHessianFree.class);
    //conjugate gradient decay
    private DoubleMatrix ch,gradient,p;
    private NeuralNetEpochListener listener;
    private double pi = 0.5;
    private double decrease = 0.99;
    private double boost = 1.0 / decrease;
    private double f = 1.0;
    private double score,step;

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
        xi = network.pack();
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


    private Pair<List<Integer>,List<DoubleMatrix>> conjGradient(int numIterations,DoubleMatrix grad,DoubleMatrix preCon) {
        List<Integer> is = new ArrayList<>();
        List<DoubleMatrix> xs = new ArrayList<>();


        if(xi == null)
            setup();

        //this is the issue
        DoubleMatrix r = network.getBackPropRGradient(ch).sub(grad);
        DoubleMatrix y = r.div(preCon);


        p = y.neg();

        DoubleMatrix x = ch.dup();
        double deltaNew = r.mul(y).sum();

        for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {


            DoubleMatrix Ap = network.getBackPropRGradient(p);

            if(MatrixUtil.isNaN(Ap)) {
                log.warn("Bad Ap; NaN breaking");
                break;
            }

            double pAp = p.mul(Ap).sum();
            if(pAp < 0)
                log.warn("Negative curve!");

            log.info("Iteration " + iterationCount);
            double alpha = deltaNew / pAp;


            x.addi(p.mul(alpha));

            //conjugate gradient
            DoubleMatrix rNew = r.add(Ap.mul(alpha));
            DoubleMatrix yNew = rNew.div(preCon);
            double deltaOld = deltaNew;
            deltaNew = rNew.mul(yNew).sum();
            double beta = deltaNew / deltaOld;
            p = yNew.neg().add(p.mul(beta));
            r = rNew;
            y = yNew;
            is.add(iterationCount);
            xs.add(x);

            if (listener != null) {
                listener.iterationDone(iterationCount);
            }

        }

        return new Pair<>(is,xs);
    }

    //setup baseline conjugate gradient and run it for n iterations
    private Triple<DoubleMatrix,List<DoubleMatrix>,DoubleMatrix> runConjugateGradient(DoubleMatrix grad,DoubleMatrix preCon,int numIterations) {
        Pair<List<Integer>,List<DoubleMatrix>> cg = conjGradient(numIterations,grad,preCon);
        if(!cg.getSecond().isEmpty())
            ch = cg.getSecond().get(cg.getSecond().size() - 1);
            p = ch.dup();

        return new Triple<>(p,cg.getSecond(),ch);
    }



    public Pair<DoubleMatrix,Double> cgBackTrack(DoubleMatrix p,List<DoubleMatrix> chs) {
        DoubleMatrix params = network.params();
        double score = network.score(p.add(params));
        int i = chs.size() - 2;
        for(; i > 0; i--) {
            double score2 = network.score(params.add(chs.get(i)));
            if(score2 < score) {
                i++;
                break;
            }
            score = score2;

        }

        if(chs.isEmpty()) {
            log.warn("No chs; exiting with old ch value");
        }
        DoubleMatrix ret = !chs.isEmpty() && i >= 0 ? chs.get(i) : ch;
        return new Pair<>(ret,score);
    }



    public boolean optimize(int numIterations) {
        myName = Thread.currentThread().getName();
        if (converged)
            return true;
        score = network.score();
        xi = optimizable.getParameters();
        BaseMultiLayerNetwork revert = network.clone();

        for(int i = 0; i < numIterations; i++) {
            //investigate way params are being referenced/saved, this is probably the root of the erotic behavior below


            //grad,precon
            Pair<DoubleMatrix,DoubleMatrix> backPropGradient = network.getBackPropGradient2();
            backPropGradient.setFirst(backPropGradient.getFirst().negi());
            gradient = backPropGradient.getFirst();
            if(ch == null)
                setup();

            ch.muli(pi);

            Triple<DoubleMatrix,List<DoubleMatrix>,DoubleMatrix>  cg = runConjugateGradient(gradient, backPropGradient.getSecond(), numIterations);
            p = cg.getFirst();
            Pair<DoubleMatrix,Double> cgBackTrack = cgBackTrack(cg.getFirst(),cg.getSecond());
            p = cgBackTrack.getFirst();
            double rho = network.reductionRatio(cgBackTrack.getFirst(),network.score(),cgBackTrack.getSecond(),gradient);

            VectorizedBackTrackLineSearchMinimum l = new VectorizedBackTrackLineSearchMinimum(optimizable);
            DoubleMatrix params = network.params();
            try {
                double step = l.optimize(cg.getFirst(),numIterations,gradient,params);
                network.dampingUpdate(rho,boost,decrease);
                params.addi(p.mul(f * step));

            }catch(Exception e) {
                log.warn("Rejected update; continuing");
            }



            if(network.score() < score) {
                score = network.score();
                xi = params.dup();
                revert = network.clone();
                network.setParameters(params);
                log.info("New score " + score);

            }

            else {
                network.update(revert);
                log.info("Reverting to score " + network.score());

            }

        }

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
