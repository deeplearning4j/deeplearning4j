package org.deeplearning4j.optimize.optimizers;

import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ComposableIterationListener;
import org.deeplearning4j.optimize.solvers.IterationGradientDescent;
import org.deeplearning4j.optimize.stepfunctions.BackPropStepFunction;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;

import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.optimize.api.OptimizableByGradientValue;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.optimize.solvers.StochasticHessianFree;
import org.deeplearning4j.optimize.solvers.VectorizedDeepLearningGradientAscent;
import org.deeplearning4j.optimize.solvers.VectorizedNonZeroStoppingConjugateGradient;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;

/**
 * Optimizes via back prop gradients
 * @author Adam Gibson
 */
public class BackPropOptimizer implements Serializable,OptimizableByGradientValue,IterationListener {

    private BaseMultiLayerNetwork network;
    private int length = -1;
    private double lr  = 1e-1f;
    private int iterations = 1000;
    private static Logger log = LoggerFactory.getLogger(BackPropOptimizer.class);
    private int currentIteration = -1;
    protected NeuralNetPlotter plotter = new NeuralNetPlotter();


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
        network.feedForward();
        Collection<IterationListener> listeners = new ArrayList<>();
        listeners.add(this);
        listeners.addAll(network.getOutputLayer().conf().getListeners());
        IterationListener listener = new ComposableIterationListener(listeners);

        if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(this,new BackPropStepFunction(network),listener);
            g.setTrainingEvaluator(eval);
            if(network.getOutputLayer().conf().getRenderWeightIterations() > 0) {

            }
            g.setMaxIterations(network.getOutputLayer().conf().getNumIterations());
            g.optimize(network.getOutputLayer().conf().getNumIterations());

        }
        else if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT) {
            IterationGradientDescent g = new IterationGradientDescent(this,listener,network.getOutputLayer().conf().getNumIterations());
            g.optimize();
        }
        else if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE) {
            StochasticHessianFree s = new StochasticHessianFree(this,listener,network);
            s.setTrainingEvaluator(eval);
            s.setMaxIterations(network.getOutputLayer().conf().getNumIterations());
            s.optimize(network.getOutputLayer().conf().getNumIterations());

        }



        else {
            VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(this,1e-1,listener,new BackPropStepFunction(network));
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


    @Override
    public void iterationDone(int iteration) {
        int plotIterations = network.getOutputLayer().conf().getRenderWeightIterations();
        if(plotIterations <= 0)
            return;
        if(iteration % plotIterations == 0) {
            plotter.plotNetworkGradient(network.getOutputLayer(),getValueGradient(0),100);
        }
    }
}
