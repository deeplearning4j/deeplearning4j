package org.deeplearning4j.optimize.solvers;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.exception.InvalidStepException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TerminationCondition;
import org.deeplearning4j.optimize.terminations.EpsTermination;
import org.deeplearning4j.optimize.terminations.ZeroDirection;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.AdaGrad;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Base optimizer
 * @author Adam Gibson
 */
public abstract class BaseOptimizer   {



    protected NeuralNetConfiguration conf;
    protected AdaGrad adaGrad;
    protected int iteration = 0;
    protected static Logger log = LoggerFactory.getLogger(BaseOptimizer.class);
    protected StepFunction stepFunction;
    private Collection<IterationListener> iterationListeners = new ArrayList<>();
    protected Collection<TerminationCondition> terminationConditions = new ArrayList<>();
    protected Model model;
    protected BackTrackLineSearch lineMaximizer;
    protected double step;
    private int batchSize = 10;
    protected double score,oldScore;
    protected double stpMax = Double.MAX_VALUE;
    public final static String GRADIENT_KEY = "g";
    public final static String SCORE_KEY = "score";
    public final static String PARAMS_KEY = "params";

    protected Map<String,Object> searchState = new HashMap<>();

    public BaseOptimizer(NeuralNetConfiguration conf,StepFunction stepFunction,Collection<IterationListener> iterationListeners,Model model) {
        this(conf,stepFunction,iterationListeners, Arrays.asList(new ZeroDirection(),new EpsTermination()),model);
    }



    public BaseOptimizer(NeuralNetConfiguration conf,StepFunction stepFunction,Collection<IterationListener> iterationListeners,Collection<TerminationCondition> terminationConditions,Model model) {
        this.conf = conf;
        this.stepFunction = stepFunction;
        this.iterationListeners = iterationListeners;
        this.terminationConditions = terminationConditions;
        this.model = model;
        lineMaximizer = new BackTrackLineSearch(model,stepFunction,this);
        lineMaximizer.setStpmax(stpMax);

    }

    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     */
    public void updateGradientAccordingToParams(INDArray gradient,INDArray params,int batchSize) {
        if(adaGrad == null)
            adaGrad = new AdaGrad(1,gradient.length());


        //reset adagrad history
        if(iteration != 0 && conf.getResetAdaGradIterations() > 0 &&  iteration % conf.getResetAdaGradIterations() == 0) {
            adaGrad.historicalGradient = null;

            log.info("Resetting adagrad");
        }

        //change up momentum after so many iterations if specified
        double momentum = conf.getMomentum();
        if(conf.getMomentumAfter() != null && !conf.getMomentumAfter().isEmpty()) {
            int key = conf.getMomentumAfter().keySet().iterator().next();
            if(iteration >= key) {
                momentum = conf.getMomentumAfter().get(key);
            }
        }


        gradient = adaGrad.getGradient(gradient);
        if (conf.isUseAdaGrad())
            gradient.assign(adaGrad.getGradient(gradient));

        else
            gradient.muli(conf.getLr());





        if (momentum > 0)
            gradient.addi(gradient.mul(momentum).addi(gradient.mul(1 - momentum)));

        //simulate post gradient application  and apply the difference to the gradient to decrease the change the gradient has
        if(conf.isUseRegularization() && conf.getL2() > 0)
            if(conf.isUseAdaGrad())
                gradient.subi(params.mul(conf.getL2()));



        if(conf.isConstrainGradientToUnitNorm())
            gradient.divi(gradient.norm2(Integer.MAX_VALUE));


        gradient.divi(batchSize);


    }


    public Pair<Gradient,Double> gradientAndScore() {
        Pair<Gradient,Double> pair = model.gradientAndScore();
        return pair;
    }

    public  boolean optimize() {
        Pair<Gradient,Double> pair = gradientAndScore();
        setupSearchState(pair);
        //get initial score
        score = pair.getSecond();
        //check initial gradient
        INDArray gradient = (INDArray) searchState.get(GRADIENT_KEY);

        //pre existing termination conditions
        for(TerminationCondition condition : terminationConditions)
            if(condition.terminate(0.0,0.0,new Object[]{gradient}))
                return true;

        //some algorithms do pre processing of gradient and
        //need to test possible directions. (LBFGS)
        boolean testLineSearch = preFirstStepProcess(gradient);
        if(testLineSearch) {
            //ensure we can take a step
            try {
                INDArray params = (INDArray) searchState.get(PARAMS_KEY);
                step = lineMaximizer.optimize(gradient,conf.getNumIterations(),step,params,gradient);
            } catch (InvalidStepException e) {
                e.printStackTrace();
            }
            gradient = (INDArray) searchState.get(GRADIENT_KEY);
            postFirstStep(gradient);

            if(step == 0.0) {
                log.warn("Unable to step in direction");
                return false;
            }
        }


        for(int i = 0; i < conf.getNumIterations(); i++) {
            //line normalization where relevant
            preProcessLine(gradient);
            //perform one step
            try {
                INDArray params = (INDArray) searchState.get(PARAMS_KEY);
                step = lineMaximizer.optimize(gradient,conf.getNumIterations(),step,params,gradient);
            } catch (InvalidStepException e) {
                e.printStackTrace();
            }


            //invoke listeners for debugging
            for(IterationListener listener : iterationListeners)
                listener.iterationDone(i);


            //record old score for deltas and other termination conditions
            oldScore = score;
            pair = gradientAndScore();
            //update the gradient after step
            score = pair.getSecond();
            //new gradient post step
            gradient = pair.getFirst().gradient();
            //update gradient
            searchState.put(GRADIENT_KEY,gradient);
            //update score
            searchState.put(SCORE_KEY,score);

            //check for termination conditions based on absolute change in score
            for(TerminationCondition condition : terminationConditions)
                if(condition.terminate(score,oldScore,new Object[]{gradient}))
                    return true;

            //post step updates to other search parameters
            postStep();


            log.info("Score at iteration " + i + " is " + score);
            //check for termination conditions based on absolute change in score
            for(TerminationCondition condition : terminationConditions)
                if(condition.terminate(score,oldScore,new Object[]{gradient}))
                    return true;


        }

        return true;
    }



    public double score() {
        return (double) searchState.get(SCORE_KEY);
    }

    protected  void postFirstStep(INDArray gradient) {
        //no-op
    }

    protected  boolean preFirstStepProcess(INDArray gradient) {
        //no-op
        return false;
    }


    public int batchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }




    /**
     * Pre process the line (scaling and the like)
     * @param line the line to pre process
     */
    public  void preProcessLine(INDArray line) {
        //no-op
    }
    /**
     * Post step (conjugate gradient among other methods needs this)

     */
    public  void postStep() {
        //no-op
    }

    /**
     * Setup the initial search state
     * @param pair
     */
    public  void setupSearchState(Pair<Gradient, Double> pair) {
        INDArray gradient = pair.getFirst().gradient();
        INDArray params = model.params();
        updateGradientAccordingToParams(gradient,params,batchSize());
        searchState.put(GRADIENT_KEY,gradient);
        searchState.put(SCORE_KEY,pair.getSecond());
        searchState.put(PARAMS_KEY,params);
    }




}
