package org.deeplearning4j.optimize;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import com.google.common.primitives.Floats;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.OptimizerMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Performs basic beam search based on the network's loss function
 * @author Adam Gibson
 *
 */
public abstract class NeuralNetworkOptimizer implements OptimizableByGradientValueMatrix,Serializable,NeuralNetEpochListener {






    private static final long serialVersionUID = 4455143696487934647L;
    protected NeuralNetwork network;
    protected float lr;
    protected Object[] extraParams;
    protected float tolerance = 0.00001f;
    protected static Logger log = LoggerFactory.getLogger(NeuralNetworkOptimizer.class);
    protected List<Double> errors = new ArrayList<>();
    protected float minLearningRate = 0.001f;
    protected transient OptimizerMatrix opt;
    protected OptimizationAlgorithm optimizationAlgorithm;
    protected LossFunction lossFunction;
    protected  NeuralNetPlotter plotter = new NeuralNetPlotter();
    protected float maxStep = -1;
    protected int numParams = -1;
    protected int currIteration = -1;
    /**
     *
     * @param network
     * @param lr
     * @param trainingParams
     */
    public NeuralNetworkOptimizer(NeuralNetwork network,float lr,Object[] trainingParams,OptimizationAlgorithm optimizationAlgorithm,LossFunction lossFunction) {
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
            opt.setTolerance(tolerance);
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
        return NDArrays.create(Floats.concat(network.getW().data(), network.getvBias().data(), network.gethBias().data()));
    }

    public void train(INDArray x) {
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
            numParams = network.getW().length() + network.gethBias().length() + network.getvBias().length();
            return numParams;
        }

        return numParams;
    }




    @Override
    public float getParameter(int index) {
        //beyond weight matrix
        if(index >= network.getW().length()) {
            int i = getAdjustedIndex(index);
            //beyond visible bias
            if(index >= network.getvBias().length() + network.getW().length()) {
                return (float) network.gethBias().getScalar(i).element();
            }
            else
                return (float) network.getvBias().getScalar(i).element();

        }
        else
            return (float) network.getW().getScalar(index).element();



    }


    @Override
    public void setParameters(INDArray params) {
        if(network.isConstrainGradientToUnitNorm())
            params.divi(params.normmax(Integer.MAX_VALUE));
        for(int i = 0; i < params.length(); i++)
            setParameter(i, (float) params.getScalar(i).element());
    }

    @Override
    public void setParameter(int index, float value) {
        //beyond weight matrix
        if(index >= network.getW().length()) {
            //beyond visible bias
            if(index >= network.getvBias().length() + network.getW().length())  {
                int i = getAdjustedIndex(index);
                network.gethBias().putScalar(i, value);
            }
            else {
                int i = getAdjustedIndex(index);
                network.getvBias().putScalar(i, value);

            }

        }
        else {
            network.getW().ravel().putScalar(index, value);
        }
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
        if(iteration >= 1)
            extraParams[extraParams.length - 1] = iteration;
        NeuralNetworkGradient g = network.getGradient(extraParams);

        if(NDArrays.dataType().equals("float")) {
        /*
		 * Treat params as linear index. Always:
		 * W
		 * Visible Bias
		 * Hidden Bias
		 */


            float[] buffer = Floats.concat(g.getwGradient().data(), g.getvBiasGradient().data(),g.gethBiasGradient().data());

            return NDArrays.create(buffer);
        }
        else {
        /*
		 * Treat params as linear index. Always:
		 * W
		 * Visible Bias
		 * Hidden Bias
		 */

            float[] buffer = Floats.concat(g.getwGradient().floatData(),g.getvBiasGradient().floatData(),g.gethBiasGradient().floatData());
            return NDArrays.create(buffer);
        }

    }


    @Override
    public float getValue() {
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
            return;
        }

        this.currIteration = value;
    }

    public  float getTolerance() {
        return tolerance;
    }
    public  void setTolerance(float tolerance) {
        this.tolerance = tolerance;
    }

    public float getMaxStep() {
        return maxStep;
    }

    public void setMaxStep(float maxStep) {
        this.maxStep = maxStep;
    }

    public int getCurrIteration() {
        return currIteration;
    }


}
