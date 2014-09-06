package org.deeplearning4j.optimize.optimizers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.google.common.primitives.Floats;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.optimize.api.NeuralNetEpochListener;
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
public abstract class NeuralNetworkOptimizer implements OptimizableByGradientValueMatrix,Serializable,NeuralNetEpochListener {






    private static final long serialVersionUID = 4455143696487934647L;
    protected NeuralNetwork network;
    protected float lr;
    protected Object[] extraParams;
    protected float tolerance = 0.00001f;
    protected static Logger log = LoggerFactory.getLogger(NeuralNetworkOptimizer.class);
    protected List<Double> errors = new ArrayList<>();
    protected transient OptimizerMatrix opt;
    protected OptimizationAlgorithm optimizationAlgorithm;
    protected LossFunctions.LossFunction lossFunction;
    protected  NeuralNetPlotter plotter = new NeuralNetPlotter();
    protected float maxStep = -1;
    protected int currIteration = -1;
    /**
     *
     * @param network
     * @param lr
     * @param trainingParams
     */
    public NeuralNetworkOptimizer(NeuralNetwork network,float lr,Object[] trainingParams,OptimizationAlgorithm optimizationAlgorithm,LossFunctions.LossFunction lossFunction) {
        this.network = network;
        this.lr = lr;
        //add current iteration as an extra parameter
        if(trainingParams != null) {
            this.extraParams = new Object[trainingParams.length + 1];
            System.arraycopy(trainingParams,0,extraParams,0,trainingParams.length);
        }
        else
            this.extraParams = new Object[1];

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
        return network.params();
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
        int plotEpochs = network.conf().getRenderWeightsEveryNumEpochs();
        if(plotEpochs <= 0)
            return;
        if(iterationDone % plotEpochs == 0) {
            plotter.plotNetworkGradient(network,network.getGradient(extraParams),100);
        }

    }



    @Override
    public int getNumParameters() {
        return network.numParams();
    }




    @Override
    public float getParameter(int index) {
        throw new UnsupportedOperationException();


    }


    @Override
    public void setParameters(INDArray params) {
        if(network.conf().isConstrainGradientToUnitNorm())
            params.divi(params.normmax(Integer.MAX_VALUE));
        network.setParams(params);
    }

    @Override
    public void setParameter(int index, float value) {
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
        if(iteration >= 1)
            extraParams[extraParams.length - 1] = iteration;
        NeuralNetworkGradient g = network.getGradient(extraParams);
        return Nd4j.toFlattened(Arrays.asList(g.getwGradient(),g.getvBiasGradient(),g.gethBiasGradient()));

    }


    @Override
    public float getValue() {
        return - network.score();

    }

    @Override
    public void setCurrentIteration(int value) {
        if(value < 1) {
            return;
        }

        this.currIteration = value;
    }


}
