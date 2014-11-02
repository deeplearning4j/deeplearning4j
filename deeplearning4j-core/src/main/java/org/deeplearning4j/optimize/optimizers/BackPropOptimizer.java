package org.deeplearning4j.optimize.optimizers;

import org.deeplearning4j.optimize.stepfunctions.BackPropStepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.optimize.solvers.StochasticHessianFree;
import org.deeplearning4j.optimize.solvers.VectorizedDeepLearningGradientAscent;
import org.deeplearning4j.optimize.solvers.VectorizedNonZeroStoppingConjugateGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Optimizes via back prop gradients
 * @author Adam Gibson
 */
public class BackPropOptimizer implements Serializable,OptimizableByGradientValueMatrix {

    private BaseMultiLayerNetwork network;
    private int length = -1;
    private double lr  = 1e-1f;
    private int iterations = 1000;
    private static Logger log = LoggerFactory.getLogger(BackPropOptimizer.class);
    private int currentIteration = -1;

    public BackPropOptimizer(BaseMultiLayerNetwork network,double lr,int epochs) {
        this.network = network;
        this.lr = lr;
        this.iterations = epochs;
    }

    @Override
    public void setCurrentIteration(int value) {
        this.currentIteration = value;
    }

    public void optimize(TrainingEvaluator eval,boolean lineSearch) {
        if(!lineSearch) {
            log.info("BEGIN BACKPROP WITH SCORE OF " + network.score());

            //store a copy of the network for when binary cross entropy gets
            //worse after an iteration
            //sgd style; only iterate a certain number of iterations
            if(network.forceNumIterations()) {
                for(int i = 0; i < iterations; i++) {
                    if(i % network.getDefaultConfiguration().getResetAdaGradIterations() == 0)
                        network.getOutputLayer().getAdaGrad().historicalGradient = null;
                    network.backPropStep();
                    log.info("Iteration " + i + " error " + network.score());

                }
            }

           else
                lineSearchBackProp(eval);
        }

        else
            lineSearchBackProp(eval);





    }


    private void lineSearchBackProp(TrainingEvaluator eval) {
        NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm = network.getDefaultConfiguration().getOptimizationAlgo();
        if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(this,new BackPropStepFunction(network));
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(iterations);
            g.optimize(iterations);

        }

        else if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE) {
            StochasticHessianFree s = new StochasticHessianFree(this,network);
            s.setTrainingEvaluator(eval);
            s.setMaxIterations(iterations);
            s.optimize(iterations);

        }



        else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(this,new BackPropStepFunction(network));
            g.setTrainingEvaluator(eval);
            g.optimize(iterations);

        }

    }


    @Override
    public double getValue() {
        return - (network.score());
    }

    @Override
    public int getNumParameters() {
        if(length < 0)
            length = getParameters().length();
        return length;
    }




    @Override
    public void setParameter(int index, double value) {

    }

    @Override
    public INDArray getParameters() {
        return network.params();
    }

    @Override
    public double getParameter(int index) {
        return 0;
    }

    @Override
    public void setParameters(INDArray params) {
        network.setParameters(params);


    }

    @Override
    public INDArray getValueGradient(int iteration) {
        return network.pack(network.backPropGradient());
    }




}
