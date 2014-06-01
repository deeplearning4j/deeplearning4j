package org.deeplearning4j.optimize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.plot.NeuralNetPlotter;
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
public abstract class NeuralNetworkOptimizer implements Optimizable.ByGradientValue,OptimizableByGradientValueMatrix,Serializable,NeuralNetEpochListener {






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
    /**
     *
     * @param network
     * @param lr
     * @param trainingParams
     */
    public NeuralNetworkOptimizer(NeuralNetwork network,double lr,Object[] trainingParams,OptimizationAlgorithm optimizationAlgorithm,LossFunction lossFunction) {
        this.network = network;
        this.lr = lr;
        this.extraParams = trainingParams;
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
    public void epochDone(int epoch) {
        int plotEpochs = network.getRenderEpochs();
        if(plotEpochs <= 0)
            return;
        if(epoch % plotEpochs == 0) {
            plotter.plotNetworkGradient(network,network.getGradient(extraParams),100);
        }

    }

    public List<Double> getErrors() {
        return errors;
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
    public void getParameters(double[] buffer) {
		/*
		 * If we think of the parameters of the model (W,vB,hB)
		 * as a solid line for the optimizer, we get the following:
		 * 
		 */



        for(int i = 0; i < buffer.length; i++)
            buffer[i] = getParameter(i);
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
    public void setParameters(double[] params) {
		/*
		 * If we think of the parameters of the model (W,vB,hB)
		 * as a solid line for the optimizer, we get the following:
		 * 
		 */
        for(int i = 0; i < params.length; i++)
            setParameter(i,params[i]);

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
    public DoubleMatrix getParameters() {
        double[] params = new double[getNumParameters()];
        getParameters(params);
        return new DoubleMatrix(params);
    }
    @Override
    public void setParameters(DoubleMatrix params) {
        setParameters(params.toArray());
    }
    @Override
    public DoubleMatrix getValueGradient() {
        double[] d = new double[getNumParameters()];
        getValueGradient(d);
        return new DoubleMatrix(d);
    }
    @Override
    public abstract void getValueGradient(double[] buffer);


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
}
