package org.deeplearning4j.optimize.optimizers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.optimize.solvers.IterationGradientDescent;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.OptimizableByGradientValueMatrix;
import org.deeplearning4j.optimize.solvers.VectorizedDeepLearningGradientAscent;
import org.deeplearning4j.optimize.solvers.VectorizedNonZeroStoppingConjugateGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.OptimizerMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Performs basic beam search based on the network's loss function
 * @author Adam Gibson
 *
 */
public abstract class NeuralNetworkOptimizer implements OptimizableByGradientValueMatrix,Serializable,IterationListener {






    private static final long serialVersionUID = 4455143696487934647L;
    protected NeuralNetwork network;
    protected double tolerance = 0.00001f;
    protected static Logger log = LoggerFactory.getLogger(NeuralNetworkOptimizer.class);
    protected List<Double> errors = new ArrayList<>();
    protected transient OptimizerMatrix opt;
    protected  NeuralNetPlotter plotter = new NeuralNetPlotter();
    protected double maxStep = -1;
    protected int currIteration = -1;
    /**
     *
     * @param network
     */
    public NeuralNetworkOptimizer(NeuralNetwork network) {
        this.network = network;
    }

    private void createOptimizationAlgorithm() {
        if(network.conf().getOptimizationAlgo() == OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            opt = new VectorizedNonZeroStoppingConjugateGradient(this,this);
            opt.setTolerance(tolerance);
        }
        else if(network.conf().getOptimizationAlgo() == OptimizationAlgorithm.ITERATION_GRADIENT_DESCENT) {
            opt = new IterationGradientDescent(this,network.conf().getNumIterations());

        }

        else {
            opt = new VectorizedDeepLearningGradientAscent(this,this);
            opt.setTolerance(tolerance);
            if(maxStep > 0)
                ((VectorizedDeepLearningGradientAscent) opt).setMaxStepSize(maxStep);
        }
    }


    @Override
    public INDArray getParameters() {
        return network.params();
    }

    public void train(INDArray x) {
        if(opt == null) {
            createOptimizationAlgorithm();
        }

        network.setInput(x);
        int iterations =  network.conf().getNumIterations();
        opt.setMaxIterations(iterations);
        opt.optimize(iterations);
        network.backProp();



    }

    @Override
    public void iterationDone(int iteration) {
        int plotEpochs = network.conf().getRenderWeightIterations();
        if(plotEpochs <= 0)
            return;
        if(iteration % plotEpochs == 0) {
            plotter.plotNetworkGradient(network,network.getGradient(),100);
        }

    }



    @Override
    public int getNumParameters() {
        return network.numParams();
    }




    @Override
    public double getParameter(int index) {
        throw new UnsupportedOperationException();


    }


    @Override
    public void setParameters(INDArray params) {
        if(network.conf().isConstrainGradientToUnitNorm())
            params.divi(params.normmax(Integer.MAX_VALUE));
        network.setParams(params);
    }

    @Override
    public void setParameter(int index, double value) {
        throw new UnsupportedOperationException();

    }


    private int getAdjustedIndex(int index) {
        int wLength = network.getW().length();
        int vBiasLength = network.getvBias().length();
        if(index < wLength)
            return index;
        else if(index >= wLength + vBiasLength) {
            int hIndex = index - wLength - vBiasLength;
            return hIndex;
        }
        else {
            int vIndex = index - wLength;
            return vIndex;
        }
    }


    @Override
    public INDArray getValueGradient(int iteration) {
        NeuralNetworkGradient g = network.getGradient();
        return Nd4j.toFlattened(Arrays.asList(g.getwGradient(),g.getvBiasGradient(),g.gethBiasGradient()));

    }


    @Override
    public double getValue() {
        return -network.score();

    }

    @Override
    public void setCurrentIteration(int value) {
        if(value < 1) {
            return;
        }

        this.currIteration = value;
    }


}
