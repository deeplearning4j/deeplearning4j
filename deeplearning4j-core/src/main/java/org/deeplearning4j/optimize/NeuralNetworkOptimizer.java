package org.deeplearning4j.optimize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.OptimizerMatrix;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.optimize.Optimizable;

/**
 * Performs basic beam search based on the network's loss function
 * @author Adam Gibson
 *
 */
public abstract class NeuralNetworkOptimizer implements OptimizableByGradientValueMatrix,Serializable,NeuralNetEpochListener {






    private static final long serialVersionUID = 4455143696487934647L;
    protected NeuralNetwork network;
    protected double lr;
    protected Object[] extraParams;
    protected double tolerance = 0.00001;
    protected static Logger log = LoggerFactory.getLogger(NeuralNetworkOptimizer.class);
    protected List<Double> errors = new ArrayList<Double>();
    protected double minLearningRate = 0.001;
    protected transient OptimizerMatrix opt;
    protected OptimizationAlgorithm optimizationAlgorithm;
    protected LossFunction lossFunction;
    protected  NeuralNetPlotter plotter = new NeuralNetPlotter();
    protected double maxStep = -1;
    protected int numParams = -1;
    protected int currIteration = -1;
    /**
     *
     * @param network
     * @param lr
     * @param trainingParams
     */
    public NeuralNetworkOptimizer(NeuralNetwork network,double lr,Object[] trainingParams,OptimizationAlgorithm optimizationAlgorithm,LossFunction lossFunction) {
        this.network = network;
        this.lr = lr;
        //add current iteration as an extra parameter
        this.extraParams = new Object[trainingParams.length + 1];
        System.arraycopy(trainingParams,0,extraParams,0,trainingParams.length);
        this.optimizationAlgorithm = optimizationAlgorithm;
        this.lossFunction = lossFunction;
    }

    private void createOptimizationAlgorithm() {
        if(optimizationAlgorithm == OptimizationAlgorithm.CONJUGATE_GRADIENT) {
            opt = new VectorizedNonZeroStoppingConjugateGradient(this,this);
            ((VectorizedNonZeroStoppingConjugateGradient) opt).setTolerance(tolerance);
        }
        else {
            opt = new VectorizedDeepLearningGradientAscent(this,this);
            ((VectorizedDeepLearningGradientAscent) opt).setTolerance(tolerance);
            if(maxStep > 0)
                ((VectorizedDeepLearningGradientAscent) opt).setMaxStepSize(maxStep);
        }
    }


    @Override
    public DoubleMatrix getParameters() {
        return new DoubleMatrix(ArrayUtil.combine(network.getW().data,network.getvBias().data,network.gethBias().data));
    }

    public void train(DoubleMatrix x) {
        if(opt == null) {
            createOptimizationAlgorithm();
        }

        network.setInput(x);
        int epochs =  extraParams.length < 3 ? 1000 : (int) extraParams[2];
        opt.setMaxIterations(epochs);
        opt.optimize(epochs);
        network.backProp(lr,epochs,extraParams);



    }

    @Override
    public void iterationDone(int iterationDone) {
        int plotEpochs = network.getRenderIterations();
        if(plotEpochs <= 0)
            return;
        if(iterationDone % plotEpochs == 0) {
            plotter.plotNetworkGradient(network,network.getGradient(extraParams),100);
        }

    }



    @Override
    public int getNumParameters() {
        if(numParams < 0) {
            numParams = network.getW().length + network.gethBias().length + network.getvBias().length;
            return numParams;
        }

        return numParams;
    }




    @Override
    public double getParameter(int index) {
        //beyond weight matrix
        if(index >= network.getW().length) {
            int i = getAdjustedIndex(index);
            //beyond visible bias
            if(index >= network.getvBias().length + network.getW().length) {
                return network.gethBias().get(i);
            }
            else
                return network.getvBias().get(i);

        }
        else
            return network.getW().get(index);



    }


    @Override
    public void setParameters(DoubleMatrix params) {
        for(int i = 0; i < params.length; i++)
            setParameter(i, params.get(i));
    }

    @Override
    public void setParameter(int index, double value) {
        //beyond weight matrix
        if(index >= network.getW().length) {
            //beyond visible bias
            if(index >= network.getvBias().length + network.getW().length)  {
                int i = getAdjustedIndex(index);
                network.gethBias().put(i, value);
            }
            else {
                int i = getAdjustedIndex(index);
                network.getvBias().put(i,value);

            }

        }
        else {
            network.getW().put(index,value);
        }
    }


    private int getAdjustedIndex(int index) {
        int wLength = network.getW().length;
        int vBiasLength = network.getvBias().length;
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
    public DoubleMatrix getValueGradient(int iteration) {
        if(iteration >= 1)
            extraParams[extraParams.length - 1] = iteration;
        NeuralNetworkGradient g = network.getGradient(extraParams);
        double[] buffer = new double[getNumParameters()];
        /*
		 * Treat params as linear index. Always:
		 * W
		 * Visible Bias
		 * Hidden Bias
		 */
        int idx = 0;
        for (int i = 0; i < g.getwGradient().length; i++) {
            buffer[idx++] = g.getwGradient().get(i);
        }
        for (int i = 0; i < g.getvBiasGradient().length; i++) {
            buffer[idx++] = g.getvBiasGradient().get(i);
        }
        for (int i = 0; i < g.gethBiasGradient().length; i++) {
            buffer[idx++] = g.gethBiasGradient().get(i);
        }

        return new DoubleMatrix(buffer);
    }


    @Override
    public double getValue() {
        if(this.lossFunction == LossFunction.RECONSTRUCTION_CROSSENTROPY)
            return network.getReConstructionCrossEntropy();
        else if(this.lossFunction == LossFunction.SQUARED_LOSS)
            return - network.squaredLoss();

        else if(this.lossFunction == LossFunction.NEGATIVELOGLIKELIHOOD)
            return -network.negativeLogLikelihood();
        else if(lossFunction == LossFunction.MSE)
            return -network.mse();
        else if(lossFunction == LossFunction.RMSE_XENT)
            return -network.mseRecon();


        return network.getReConstructionCrossEntropy();

    }

    @Override
    public void setCurrentIteration(int value) {
        if(value < 1) {
            log.info("Not setting iteration with value " + value);
            return;
        }

        this.currIteration = value;
    }

    public  double getTolerance() {
        return tolerance;
    }
    public  void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    public double getMaxStep() {
        return maxStep;
    }

    public void setMaxStep(double maxStep) {
        this.maxStep = maxStep;
    }

    public int getCurrIteration() {
        return currIteration;
    }


}
