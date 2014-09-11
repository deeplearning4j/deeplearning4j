package org.deeplearning4j.optimize.solvers;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.optimize.api.NeuralNetEpochListener;
import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.util.OptimizerMatrix;

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
    Float initialStepSize = 1f;
    Float tolerance = 1e-5f;
    Float gradientTolerance = 0f;
    private BaseMultiLayerNetwork network;
    int maxIterations = 10000;
    private String myName = "";
    private static Logger log = LoggerFactory.getLogger(StochasticHessianFree.class);
    /* decay, current gradient/direction/current point in vector space,preCondition on conjugate gradient,current parameters */
    private INDArray ch,gradient,xi;
    private NeuralNetEpochListener listener;
    private Float pi = 0.5f;
    private Float decrease = 0.99f;
    private Float boost = 1.0f / decrease;
    private Float f = 1.0f;
    /* current score, step size */
    private Float score,step;



    public StochasticHessianFree(OptimizableByGradientValueMatrix function, Float initialStepSize,BaseMultiLayerNetwork network) {
        this.initialStepSize = initialStepSize;
        this.optimizable = function;
        this.network = network;

    }

    public StochasticHessianFree(OptimizableByGradientValueMatrix function, NeuralNetEpochListener listener,BaseMultiLayerNetwork network) {
        this(function, 0.01f,network);
        this.listener = listener;

    }

    public StochasticHessianFree(OptimizableByGradientValueMatrix function, Float initialStepSize, NeuralNetEpochListener listener,BaseMultiLayerNetwork network) {
        this(function,initialStepSize,network);
        this.listener = listener;


    }

    public StochasticHessianFree(OptimizableByGradientValueMatrix function,BaseMultiLayerNetwork network) {
        this(function, 0.01f,network);
        this.network = network;
    }


    void setup() {
        ch = Nd4j.zeros(1,optimizable.getNumParameters());
        xi = network.pack();
    }


    public boolean isConverged() {
        return converged;
    }


    public boolean optimize() {
        return optimize(maxIterations);
    }

    public void setTolerance(Float t) {
        tolerance = t;
    }


    /* run conjugate gradient for numIterations */
    public Pair<List<Integer>,List<INDArray>> conjGradient(INDArray b,INDArray x0,INDArray preCon,int numIterations) {
        List<Integer> is = new ArrayList<>();
        List<INDArray> xs = new ArrayList<>();
        //log.info("B sum " + b.sum());
        //in the pseudo code the gradient is b
        //x0 is ch
        INDArray r = network.getBackPropRGradient(x0).subi(b);
        INDArray y = r.div(preCon);
        Float deltaNew = (Float) r.mul(y).sum(Integer.MAX_VALUE).element();
        INDArray p = y.neg();
        //initial x
        INDArray x = x0;


        for (int iterationCount = 0; iterationCount < numIterations; iterationCount++) {
            //log.info("P sum at iteration " + iterationCount + " is " + p.sum());
            //log.info("R sum at iteration " + iterationCount + " is " + r.sum());

            INDArray Ap = network.getBackPropRGradient(p);
            //log.info("Ap sum at iteration " + iterationCount + " is " + Ap.sum());
            //think Ax + b, this is the curvature
            Float pAp = (Float) Ap.mul(p).sum(Integer.MAX_VALUE).element();
            if(pAp < 0) {
                log.info("Negative slope: " + pAp + " breaking");
            }


            Float val = 0.5f * Nd4j.getBlasWrapper().dot(b.neg().add(r).transpose(), x);

            log.info("Iteration on conjugate gradient " + iterationCount + " with value " + val);





            //step size
            Float alpha = deltaNew / pAp;
            //step
            x.addi(p.mul(alpha));

           //conjugate gradient
            INDArray rNew = r.add(Ap.mul(alpha));
            INDArray yNew = rNew.div(preCon);
            Float deltaOld = deltaNew;
            deltaNew = (Float) rNew.mul(yNew).sum(Integer.MAX_VALUE).element();
            Float beta = deltaNew / deltaOld;
            p = yNew.neg().add(p.mul(beta));

            r = rNew;
            //append to the steps taken
            is.add(iterationCount);
            xs.add(x.dup());



        }

        return new Pair<>(is,xs);
    }

    //setup baseline conjugate gradient and run it for n iterations
    private Triple<INDArray,List<INDArray>,INDArray> runConjugateGradient(INDArray preCon,int numIterations) {
        Pair<List<Integer>,List<INDArray>> cg = conjGradient(gradient,ch,preCon,numIterations);
        ch = cg.getSecond().get(cg.getSecond().size() - 1);
        return new Triple<>(ch,cg.getSecond(),ch);
    }


    /**
     * Search with the proposed objective
     * @param newScore the new score to start with
     * @param params the params of the proposed step
     * @return the rate to step by
     */
    public Float lineSearch(Float newScore,INDArray params,INDArray p) {
        Float rate = 1.0f;
        Float c = 1e-2f;
        int j = 0;
        int numSearches = 60;
        while(j < numSearches) {
            if(10 % numSearches == 0) {
                log.info("Iteration " + j + " on line search with current rate of " + rate);
            }
            //converged
            if(newScore <= (Float) gradient.mul(p).mul(score + c * rate).sum(Integer.MAX_VALUE).element()) {
                break;
            }
            else {
                rate *= 0.8f;
                j++;
            }

            //explore in this direction and obtain a score
            newScore = network.score(params.add(p.mul(rate)));
        }

        if(j == numSearches) {
            rate = 0.0f;
            log.info("Went too far...reverting rate to 0");
        }


        return rate;

    }



    /**
     * Iterate through the current applyTransformToDestination of gradients
     * and backtrack upon an optimal step
     * that improves the current score
     * @param chs the proposed changes
     * @return the new changed path and score for that path
     */
    public Pair<INDArray,Float> cgBackTrack(List<INDArray> chs,INDArray p) {
        INDArray params = network.params();
        Float score = network.score(p.add(params));
        Float currMin = network.score();
        int i = chs.size() - 2;

        for(; i > 0; i--) {
            Float score2 = network.score(params.add(chs.get(i)));
            if(score2 < score || score2 < currMin) {
                i++;
                score = score2;
                log.info("Breaking on new score " + score2 + " with iteration " + i + " with current minimum of " + currMin);
                break;
            }

            log.info("Trial " + i + " with trial score of " + score2);

        }

        if(i < 0)
            i = 0;

        return new Pair<>(chs.get(i),score);
    }



    public boolean optimize(int numIterations) {
        myName = Thread.currentThread().getName();
        if (converged)
            return true;

        score = network.score();

        xi = network.params();




        //initial gradient, precon/conjugate gradient conditioner
        Pair<INDArray,INDArray> backPropGradient = network.getBackPropGradient2();

        gradient = backPropGradient.getFirst().neg();
        //log.info("Gradient sum " + gradient.sum());

        INDArray preCon = backPropGradient.getSecond();

        if(ch == null)
            setup();

        ch.muli(pi);

        Triple<INDArray,List<INDArray>,INDArray>  cg = runConjugateGradient(preCon,numIterations);

        INDArray p = cg.getFirst();

        Pair<INDArray,Float> cgBackTrack = cgBackTrack(cg.getSecond(),p);

        p = cgBackTrack.getFirst();

        Float rho = network.reductionRatio(cgBackTrack.getFirst(), network.score(), cgBackTrack.getSecond(), gradient);
        Float newScore = network.score(cgBackTrack.getFirst());

        step = lineSearch(newScore,gradient,p);
        network.dampingUpdate(rho,boost,decrease);

        INDArray proposedUpdate = xi.add(p.mul(f * step));
        network.setParameters(proposedUpdate);

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

    /**
     * The tolerance for change when running
     *
     * @param tolerance
     */
    @Override
    public void setTolerance(float tolerance) {

    }
}
