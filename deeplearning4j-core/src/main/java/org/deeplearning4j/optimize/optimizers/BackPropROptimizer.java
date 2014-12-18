package org.deeplearning4j.optimize.optimizers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork;

import org.deeplearning4j.optimize.api.OptimizableByGradientValue;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.optimize.solvers.StochasticHessianFree;
import org.deeplearning4j.optimize.solvers.VectorizedDeepLearningGradientAscent;
import org.deeplearning4j.optimize.solvers.VectorizedNonZeroStoppingConjugateGradient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Optimizes via back prop gradients with
 * the r operator, used in hessian free operators
 * @author Adam Gibson
 */
public class BackPropROptimizer implements Serializable,OptimizableByGradientValue {

    private BaseMultiLayerNetwork network;
    private int length = -1;
    private double lr = 1e-1;
    private int epochs = 1000;
    private static Logger log = LoggerFactory.getLogger(BackPropROptimizer.class);
    private int currentIteration = -1;
    private StochasticHessianFree h;


    public BackPropROptimizer(BaseMultiLayerNetwork network, double lr, int epochs) {
        this.network = network;
        this.lr = lr;
        this.epochs = epochs;
    }

    @Override
    public void setCurrentIteration(int value) {
        this.currentIteration = value;
    }

    public void optimize(TrainingEvaluator eval) {
        NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm = network.getDefaultConfiguration().getOptimizationAlgo();
        if (optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(this);
            g.setTrainingEvaluator(eval);
            g.setMaxIterations(network.getOutputLayer().conf().getNumIterations());
            g.optimize(network.getOutputLayer().conf().getNumIterations());

        } else if (optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE) {
            h = new StochasticHessianFree(this, network);
            h.setTrainingEvaluator(eval);
            h.optimize(network.getOutputLayer().conf().getNumIterations());
        } else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(this);
            g.setTrainingEvaluator(eval);
            g.optimize(network.getOutputLayer().conf().getNumIterations());

        }



    }


    public void getValueGradient(double[] buffer) {
        System.arraycopy(network.getBackPropRGradient(network.params()).data(),0,buffer,0,buffer.length);
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


    public void getParameters(double[] buffer) {
        System.arraycopy(getParameters().data(),0,buffer,0,buffer.length);
    }

    @Override
    public double getParameter(int index) {
        return 0;
    }


    public void setParameters(double[] params) {
        setParameters(Nd4j.create(params));
    }


    @Override
    public INDArray getParameters() {
        return network.params();
    }

    @Override
    public void setParameters(INDArray params) {
        network.setParameters(params);


    }

    @Override
    public void setParameter(int index, double value) {

    }

    @Override
    public INDArray getValueGradient(int iteration) {
        return network.getBackPropRGradient(network.params());
    }



}
