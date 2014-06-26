package org.deeplearning4j.optimize;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.OptimizerMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
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
    TrainingEvaluator eval;
    double initialStepSize = 1;
    double tolerance = 1e-5;
    double gradientTolerance = 1e-5;
    private BaseMultiLayerNetwork network;
    int maxIterations = 10000;
    private String myName = "";
    private static Logger log = LoggerFactory.getLogger(StochasticHessianFree.class);
    /* decay, current gradient/direction/current point in vector space,preCondition on conjugate gradient,current parameters */
    private DoubleMatrix ch,gradient,p,preCon,xi;
    private NeuralNetEpochListener listener;
    private double pi = 0.5;
    private double decrease = 0.99;
    private double boost = 1.0 / decrease;
    private double f = 1.0;
    /* current score, step size */
    private double score,step;



    public StochasticHessianFree(OptimizableByGradientValueMatrix function, double initialStepSize,BaseMultiLayerNetwork network) {
        this.initialStepSize = initialStepSize;
        this.optimizable = function;
        this.network = network;

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


    public boolean optimize() {
        return optimize(maxIterations);
    }

    public void setTolerance(double t) {
        tolerance = t;
    }


    /* run conjugate gradient for numIterations */
    private Pair<List<Integer>,List<DoubleMatrix>> conjGradient(int numIterations) {
        List<Integer> is = new ArrayList<>();
        List<DoubleMatrix> xs = new ArrayList<>();

        //in the pseudo code the gradient is b
        //x0 is ch
        DoubleMatrix r = network.getBackPropRGradient(ch).subi(gradient);
        DoubleMatrix y = r.div(preCon);
        log.info("Precon mean " + preCon.mean());
        log.info("R  mean " + r.mean());
        log.info("Y mean " + y.mean());
        log.info("Gradient mean " + gradient.mean());
        p = y.neg();
        double mean = p.mean();

        //initial x
        DoubleMatrix x = ch;
        double deltaNew = r.mul(y).sum();

        for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {


            DoubleMatrix Ap = network.getBackPropRGradient(p);

            log.info("P mean for iteration " + iterationCount + " is " + p.mean());


            //think Ax + b, this is the curvature
            double pAp = Ap.mul(p).sum();
            log.info("pAp for iteration " + iterationCount +   " is " + pAp);
            if(pAp < 0)
                log.warn("Negative curve!");


            double val = 0.5 * SimpleBlas.dot(ch.neg().addi(r).transpose(), x);

            log.info("Iteration on conjugate gradient " + iterationCount + " with value " + val);

            //step size
            double alpha = deltaNew / pAp;

            //step
            x.addi(p.mul(alpha));

            //conjugate gradient
            DoubleMatrix rNew = r.add(Ap.mul(alpha));
            DoubleMatrix yNew = rNew.div(preCon);
            double deltaOld = deltaNew;
            deltaNew = rNew.mul(yNew).sum();
            double beta = deltaNew / deltaOld;
            log.info("Beta for iteration " + iterationCount + " is " + beta);
            p = yNew.neg().add(p.mul(beta));
            r = rNew;
            val = 0.5 * SimpleBlas.dot(ch.neg().addi(r).transpose(), x);
            log.info("Value for " + iterationCount + " is " + val);
            //append to the steps taken
            is.add(iterationCount);
            xs.add(x);



        }

        return new Pair<>(is,xs);
    }

    //setup baseline conjugate gradient and run it for n iterations
    private Triple<DoubleMatrix,List<DoubleMatrix>,DoubleMatrix> runConjugateGradient(int numIterations) {
        Pair<List<Integer>,List<DoubleMatrix>> cg = conjGradient(numIterations);
        ch = cg.getSecond().get(cg.getSecond().size() - 1);
        p = ch.dup();

        return new Triple<>(ch,cg.getSecond(),ch);
    }


    /**
     * Search with the proposed objective
     * @param newScore the new score to start with
     * @param params the params of the proposed step
     * @return the rate to step by
     */
    public double lineSearch(double newScore,DoubleMatrix params) {
        double rate = 1.0;
        double c = 1e-2;
        int j = 0;

        while(j < 60) {
            //converged
            if(newScore <= gradient.mul(p).mul(score + c * rate).sum())
                break;
            else
                rate *= 0.8;
            //explore in this direction and obtain a score
            newScore = network.score(params.add(p.mul(rate)));
        }

        if(j == 60)
            rate = 0.0;



        return rate;

    }



    /**
     * Iterate through the current set of gradients
     * and backtrack upon an optimal step
     * that improves the current score
     * @param p the point to start at
     * @param chs the proposed changes
     * @return the new changed path and score for that path
     */
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


            //initial gradient, precon/conjugate gradient conditioner
            Pair<DoubleMatrix,DoubleMatrix> backPropGradient = network.getBackPropGradient2();

            gradient = backPropGradient.getFirst().neg();
            log.info("Gradient mean " + gradient.mean());
            preCon = backPropGradient.getSecond();
            log.info("precon mean " + preCon.mean());

            if(ch == null)
                setup();

            ch.muli(pi);

            Triple<DoubleMatrix,List<DoubleMatrix>,DoubleMatrix>  cg = runConjugateGradient(numIterations);

            p = cg.getFirst();

            Pair<DoubleMatrix,Double> cgBackTrack = cgBackTrack(cg.getFirst(),cg.getSecond());

            p = cgBackTrack.getFirst();

            double rho = network.reductionRatio(cgBackTrack.getFirst(),network.score(),cgBackTrack.getSecond(),gradient);
            DoubleMatrix revertParams = xi.dup();
            double newScore = network.score(xi);
            try {
                step = lineSearch(newScore,ch);
                network.dampingUpdate(rho,boost,decrease);
                xi.addi(p.mul(f * step));

            }catch(Exception e) {
                log.warn("Rejected update; continuing");
            }


            newScore = network.score(xi);
            if(newScore < score) {
                score = newScore;
                revert = network.clone();
                network.setParameters(xi.dup());
                log.info("New score " + score);

            }

            else {
                xi = revertParams;
                network.update(revert);
                log.info("Reverting to score " + score + " from  " + newScore);

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
