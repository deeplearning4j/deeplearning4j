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
import org.nd4j.linalg.dataset.DataSet;
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

    public void optimize(TrainingEvaluator eval) {
        lineSearchBackProp(eval);
        network.getOutputLayer().fit(new DataSet(network.getOutputLayer().getInput(),network.getOutputLayer().getLabels()));



    }


    private void lineSearchBackProp(TrainingEvaluator eval) {
        NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm = network.getDefaultConfiguration().getOptimizationAlgo();
        if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(this,new BackPropStepFunction(network));
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(network.getOutputLayer().conf().getNumIterations());
            g.optimize(network.getOutputLayer().conf().getNumIterations());

        }

        else if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE) {
            StochasticHessianFree s = new StochasticHessianFree(this,network);
            s.setTrainingEvaluator(eval);
            s.setMaxIterations(network.getOutputLayer().conf().getNumIterations());
            s.optimize(network.getOutputLayer().conf().getNumIterations());

        }



        else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(this,new BackPropStepFunction(network));
            g.setTrainingEvaluator(eval);
            g.optimize(network.getOutputLayer().conf().getNumIterations());

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
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray getParameters() {
        return network.params();
    }

    @Override
    public double getParameter(int index) {
        throw new UnsupportedOperationException();
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
